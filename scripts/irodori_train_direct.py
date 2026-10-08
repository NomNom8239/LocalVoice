#!/usr/bin/env python3
r"""Build and train one Irodori expressive LoRA from EXISTING LocalVoice audio.

Windows, from F:\AIProjects\LocalVoice:
    .\.venv\Scripts\python.exe .\scripts\irodori_train_direct.py --run

One run: reuse 1,167-item inventory/ASR and speaker bank, select clean
target-speaker speech, include only the previously human-labelled and
tokenizer-checked nonverbal pilot samples, DACVAE encode through the
pinned official Irodori prepare_manifest.py, train an LoRA, and generate
two controlled base/LoRA comparison WAVs. No scraping, ASR reruns, or
per-batch human review. Does not claim any generated audio meets a
subjective voice-similarity or nonverbal-quality threshold.

Only use source material you own or have permission to use.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from lv02_expand_approved import SPEAKER, WORKSPACE, load_authority, sha256, video, write_csv
from lv03_no_codex import CHILD, CHECK
from lv04_no_codex import (
    build_command, official_version, resolve_checkpoint, stream,
    validate_adapter,
)
from lv05_voice_ab import command as inference_command

ROOT = Path(__file__).resolve().parents[1]
VOICE_STYLES = {"breath", "groan", "laugh", "panting"}
TRAIN_COLS = ("audio", "text", "caption", "speaker", "clip_id", "sha256", "kind")
MIN_SPEECH = 24


def output_dir(ws: Path) -> Path:
    for index in range(1, 1000):
        folder = ws / f"direct_lora_{index:03d}"
        if not folder.exists():
            folder.mkdir(exist_ok=False)
            return folder
    raise RuntimeError("No new direct_lora_NNN output directory")


def valid_text(text: str) -> bool:
    return bool(2 <= len(text.strip()) <= 180 and
                re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text) and
                not re.search(r"(.)\1{12,}", text))


def source_rows(state: dict) -> tuple[list[dict], list[dict], Counter]:
    """Frozen approved8 + broader curated candidates; NEVER eval holdout video."""
    retained = [dict(row, kind="previously_approved") for row in state["train_rows"]]
    result = []
    why = Counter()
    for cid, inv in state["inventory"].items():
        if cid in state["train_ids"] or cid in state["eval_ids"]:
            continue
        if video(inv) in state["eval_videos"]:
            why["eval_source_video"] += 1
            continue
        auto, review = state["auto"][cid], state["review"][cid]
        if auto.get("route") != "training_candidate":
            why["not_training_candidate"] += 1
            continue
        if auto.get("source_kind") not in ("my_voice", "review_approved", "review_emotion"):
            why["not_curated_source"] += 1
            continue
        # Only normal speech is auto-staged. Emotion, whisper and all
        # nonverbal styles require human annotation; AST is not a label.
        style = str(auto.get("candidate_style", "")).strip().lower()
        if style != "normal" or auto.get("style_source") not in (
            "curated_source_normal_default", "human_confirmed"
        ):
            why["unverified_style"] += 1
            continue
        text = (review.get("text", "").strip()
                if review.get("decision") == "approved" else "")
        text = text or str(auto.get("text", "")).strip()
        if not valid_text(text):
            why["missing_or_suspicious_asr"] += 1
            continue
        if inv.get("scan_flag") not in ("ok",):
            why["invalid_scan_flag"] += 1
            continue
        seconds = float(inv.get("duration_sec") or 0)
        if not 2.0 <= seconds <= 20.0:
            why["duration_outside_2_to_20s"] += 1
            continue
        result.append({
            "audio": inv["source_path"], "text": text,
            "caption": "", "speaker": SPEAKER, "clip_id": cid,
            "sha256": inv["sha256"],
            "kind": "human_confirmed" if review.get("decision") == "approved"
                    else "curated_asr_suggestion",
        })
    return retained, result, why


def nonverbal_rows(state: dict, source: Path, ws: Path) -> list[dict]:
    """Only exact pilot hypotheses already passed official tokenizer checks.

    No invented ASR text for empty-text nonverbal WAVs.
    """
    pilot_path = ws / "nonverbal_pilot"
    if not all((pilot_path / x).is_file() for x in (
            "hf_audio_dataset_hypothesis.jsonl", "audit.csv", "tokenizer_audit.csv")):
        return []
    from irodori_nonverbal_manifest import validate_sources
    from lv02_expand_approved import load_csv, index
    validated = validate_sources(source, ws, SPEAKER)
    approved = index(load_csv(pilot_path / "audit.csv"), "clip_id", "nonverbal pilot audit")
    results = []
    for row in validated:
        path = Path(row["audio"]).resolve()
        inv = next((r for r in state["inventory"].values()
                    if Path(r["source_path"]).resolve() == path), None)
        if inv is None:
            raise ValueError(f"Nonverbal pilot WAV outside inventory: {path}")
        cid = inv["clip_id"]
        audit = approved.get(cid, {})
        reviewed = state["review"][cid]
        style = audit.get("style")
        if (cid in state["train_ids"] or cid in state["eval_ids"]
                or video(inv) in state["eval_videos"]):
            continue
        if (style not in VOICE_STYLES or
                audit.get("style_source") != "human_confirmed" or
                reviewed.get("speaker_ok") != "yes" or
                reviewed.get("quality") != "good" or
                reviewed.get("decision") not in ("style_confirmed", "approved")):
            continue
        if not row.get("text") or not row.get("caption"):
            continue
        results.append({
            "audio": str(path), "text": row["text"],
            "caption": row["caption"], "speaker": SPEAKER,
            "clip_id": cid, "sha256": inv["sha256"],
            "kind": f"human_nonverbal_{style}",
        })
    return results


def verified_wave_quality(
    rows: list[dict], inv_by_clip: dict, *, strict_acoustic_gate: bool = True,
) -> tuple[list[dict], Counter]:
    """Validate WAV integrity; preserve frozen approvals without a new QC veto.

    All sources are SHA-checked and must decode to finite, valid audio.
    New/unreviewed candidates still pass strict acoustic quality limits.
    Previously approved clips have already passed official LV-03 DACVAE
    and keep their historical acceptance, rather than retroactively failing
    on a new silence/loudness heuristic.
    """
    import numpy as np
    import soundfile as sf
    good, reasons = [], Counter()
    for row in rows:
        path = Path(row["audio"]).resolve()
        inv = inv_by_clip[row["clip_id"]]
        if not path.is_file() or sha256(path) != inv["sha256"]:
            reasons["missing_or_sha_changed"] += 1
            continue
        try:
            signal, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            if signal.size == 0 or sample_rate < 16000 or not np.isfinite(signal).all():
                reasons["bad_wave"] += 1
                continue
            mono = signal.mean(axis=1)
            duration = len(mono) / sample_rate
            rms = float(np.sqrt(np.mean(mono * mono)))
            clipping = float(np.mean(np.abs(signal) >= 0.999))
            dc_offset = float(abs(mono.mean()))
            silent = float(np.mean(np.abs(mono) < 0.002))
            if not (1.5 <= duration <= 25 and rms > 0):
                reasons["invalid_duration_or_silent"] += 1
                continue
            if (strict_acoustic_gate and not (
                    0.003 <= rms <= 0.40 and clipping <= 0.005
                    and dc_offset <= 0.05 and silent < 0.90)):
                reasons["acoustic_quality_gate"] += 1
                continue
        except (OSError, ValueError, RuntimeError):
            reasons["unreadable_wave"] += 1
            continue
        good.append(row)
    return good, reasons


def speaker_screen(rows: list[dict], root: Path, *, minimum: int) -> tuple[list[dict], dict]:
    """Use preexisting target-speaker embeddings as *filter*, never identity proof."""
    if not rows:
        return [], {"evaluated": 0}
    import numpy as np
    import torch
    from common import (embed_audio, embedding_backend, load_audio_mono,
                        load_config, normalize, reference_dir, top_k_score)
    config = load_config(root / "config.toml")
    bank_dir = reference_dir(config, SPEAKER)
    bank_path = bank_dir / "self_reference_bank.npz"
    bank_meta = bank_dir / "self_reference_bank.json"
    if not bank_path.is_file() or not bank_meta.is_file():
        raise FileNotFoundError(
            "Existing Ui_Shigure speaker reference bank missing; do not silently "
            "replace it with a different person's bank")
    data = json.loads(bank_meta.read_text(encoding="utf-8"))
    if data.get("profile") != SPEAKER or data.get("model") != config["models"]["pyannote_pipeline"]:
        raise ValueError("Incorrect or stale voice reference bank model/profile")
    with np.load(bank_path, allow_pickle=False) as saved:
        bank = np.asarray(saved["embeddings"], dtype=np.float32)
    if bank.ndim != 2 or len(bank) < 2:
        raise ValueError("Invalid self-voice reference bank")
    bank = np.stack([normalize(e) for e in bank])
    if data.get("reference_count") != len(bank):
        raise ValueError("Speaker bank reference count mismatch")
    topk = min(int(config["classification"]["top_k"]), len(bank))
    if topk < 1:
        raise ValueError("Invalid speaker top-k")
    thresholds = bank_dir / "thresholds.json"
    calibrated = thresholds.is_file()
    threshold = (float(json.loads(thresholds.read_text(encoding="utf-8"))["accept_threshold"])
                 if calibrated else float(config["classification"]["accept_threshold"]))
    if not 0 <= threshold <= 1:
        raise ValueError("Bad speaker acceptance threshold")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, sr = embedding_backend(config, device)
    passed, counts = [], Counter()
    for i, row in enumerate(rows, 1):
        path = Path(row["audio"])
        try:
            waveform, original_sr = load_audio_mono(path)
            embedding = embed_audio(backend, waveform, original_sr, sr)
            score = top_k_score(embedding, bank, topk)
            if math.isfinite(score) and score >= threshold:
                passed.append(dict(row, speaker_similarity=round(score, 5)))
                counts["speaker_screen_pass"] += 1
            else:
                counts["speaker_screen_review"] += 1
        except (OSError, RuntimeError, ValueError):
            counts["embedding_failed"] += 1
        if i % 50 == 0:
            print(f"[speaker] {i}/{len(rows)} checked", flush=True)
    if len(passed) < minimum:
        raise ValueError(
            f"Only {len(passed)} screened speech candidates passed speaker heuristic "
            f"(need at least {minimum}); inspect existing bank/calibration")
    return passed, {
        "evaluated": len(rows), "threshold": threshold,
        "calibrated_with_negatives": calibrated,
        "by_result": dict(counts),
        "bank_sha256": sha256(bank_path),
        "note": "Embedding match is a heuristic, not proof of voice identity.",
    }


def evenly_sample(rows: list[dict], max_clips: int) -> list[dict]:
    """Prevent a single stream from dominating normal speech."""
    group = defaultdict(list)
    for row in rows:
        group[video({"source_path": row["audio"]})].append(row)
    for g in group.values():
        g.sort(key=lambda item: (item.get("speaker_similarity", 0), item["clip_id"]),
               reverse=True)
    result = []
    while len(result) < max_clips and any(group.values()):
        for vid in sorted(group):
            if group[vid] and len(result) < max_clips:
                result.append(group[vid].pop(0))
    return result


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def command_error(stage: str, log: Path, code: int, detail: str) -> RuntimeError:
    return RuntimeError(f"{stage} failed (exit={code}, {detail}); log: {log}")


def encode(root: Path, upstream: Path, py: Path, folder: Path, csv_file: Path,
           count: int, env: dict) -> Path:
    from lv03_no_codex import execute
    shared = upstream / ".localvoice-toolchain" / "ffmpeg-7.0.2-full-shared" / "bin"
    if os.name == "nt" and not shared.is_dir():
        raise FileNotFoundError("Existing FFmpeg7 shared DLL directory missing")
    manifest = folder / "train_manifest.jsonl"
    args = [
        "--dataset", "csv", "--split", "train",
        "--data-files", "train=" + str(csv_file),
        "--audio-column", "audio", "--text-column", "text",
        "--caption-column", "caption", "--speaker-column", "speaker",
        "--speaker-id-prefix", "LocalVoice-approved",
        "--output-manifest", str(manifest),
        "--latent-dir", str(folder / "latents"),
        "--device", "cuda", "--codec-deterministic-encode",
        "--prefetch", "0", "--flush-every", "1",
        "--log-every", "20", "--max-samples", str(count), "--no-progress",
    ]
    cmd = [str(py), "-u", "-c", CHILD, str(upstream), str(shared), *args]
    save_json(folder / "encode_command.json", {"arguments": args})
    okay, reason = execute(cmd, cwd=upstream,
                           log=folder / "encode.log", env=env,
                           seconds=max(7200, count * 45))
    if not okay:
        raise command_error("DACVAE encode", folder / "encode.log", 2, reason)
    verify = subprocess.run(
        [str(py), "-u", "-c", CHECK, str(manifest), str(count)],
        cwd=upstream, env=env, capture_output=True, text=True,
        timeout=max(300, count * 4), check=False,
    )
    (folder / "encode_validate.log").write_text(
        verify.stdout + "\n" + verify.stderr, encoding="utf-8")
    if verify.returncode:
        raise command_error("Official dataset check",
                            folder / "encode_validate.log", verify.returncode, "invalid manifest")
    return manifest


def weighted_manifest(manifest: Path, folder: Path,
                      normal_count: int, *, repeat: int) -> tuple[Path, dict]:
    original = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if len(original) < normal_count:
        raise ValueError("Not enough rows in encoded official manifest")
    expressive = original[normal_count:]
    if len(original) != normal_count + len(expressive):
        raise RuntimeError("Manifest source mapping changed")
    target = folder / "train_weighted_manifest.jsonl"
    weighted = original + expressive * (repeat - 1)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                              for row in weighted), encoding="utf-8")
    return target, {
        "normal_unique": normal_count,
        "nonverbal_unique": len(expressive),
        "nonverbal_repeat": repeat,
        "effective_training_rows": len(weighted),
        "manifest_sha256": sha256(target),
        "note": "Nonverbal repetition is experimental; duplicate latent paths are intentional."
    }


def infer_pair(root: Path, upstream: Path, py: Path, checkpoint: Path,
               adapter: Path, manifest: Path, output: Path,
               expressive_rows: list[dict], env: dict) -> dict:
    rows = [json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines()
            if s.strip()]
    # More same-speaker references are closer to the previously verified
    # Base-clone setup (11 references, FP32, 40 RF steps, seed 42).
    refs = [Path(item["latent_path"]) for item in rows[:11]]
    refs = [(p if p.is_absolute() else manifest.parent / p).resolve() for p in refs]
    if len(refs) < 11 or not all(p.is_file() for p in refs):
        raise ValueError("Eleven normal speech reference latents required")

    def generate(label: str, *, adapter_path: Path | None, text: str,
                 caption: str | None = None, seconds: float | None = None) -> dict:
        wav = output / f"{label}.wav"
        cmd = inference_command(py, root, upstream, checkpoint, adapter_path,
                                refs, text, wav, seed=42, steps=40,
                                seconds=seconds)
        if "--model-precision" in cmd:
            cmd[cmd.index("--model-precision") + 1] = "fp32"
        if caption:
            cmd += ["--caption", caption]
        log = output / f"{label}.log"
        code, reason = stream(cmd, cwd=upstream, log=log, env=env, timeout=2400)
        if code or not wav.is_file() or wav.stat().st_size < 100:
            raise command_error(f"{label} inference", log, code, reason)
        return {"wav": str(wav), "sha256": sha256(wav)}

    result = {"speech": {}, "expressions": {}}
    normal = "おはようございます。今日はよろしくお願いします。"
    for variant, selected in (("base", None), ("lora", adapter)):
        result["speech"][variant] = generate(
            f"speech_{variant}", adapter_path=selected, text=normal)
    # Expression generation is exploratory and never a correctness/pass gate.
    # Always derive text+caption from actual checked nonverbal examples.
    examples = {}
    for row in expressive_rows:
        style = row["kind"].removeprefix("human_nonverbal_")
        if style not in examples:
            examples[style] = row
    for style, row in sorted(examples.items()):
        result["expressions"][style] = {}
        for variant, selected in (("base", None), ("lora", adapter)):
            result["expressions"][style][variant] = generate(
                f"{style}_{variant}", adapter_path=selected,
                text=row["text"], caption=row["caption"], seconds=3.5)
    return result

def run(args: argparse.Namespace) -> int:
    if not 32 <= args.max_speech <= 900:
        raise ValueError("--max-speech must be 32..900")
    if not 1 <= args.nonverbal_repeat <= 8:
        raise ValueError("--nonverbal-repeat must be 1..8")
    if not 100 <= args.steps <= 10000:
        raise ValueError("--steps must be 100..10000")
    root = ROOT
    source = root / "data" / "training_audio" / SPEAKER / "audio"
    ws = WORKSPACE
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not ws.is_dir():
        raise FileNotFoundError(ws)
    state = load_authority(ws)  # Original 1,167, 8 train and 3 eval; immutable.
    preserved, candidates, discarded = source_rows(state)
    if not candidates:
        raise ValueError("No eligible normal speech in existing 1,167 items")
    curated, qc = verified_wave_quality(candidates, state["inventory"])
    # Existing bank scoring is an input-quality heuristic, not consent or identity proof.
    print(f"[select] {len(candidates)} candidates; {len(curated)} clean WAVs", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Original my_voice/review_approved/review_emotion streams have already
    # passed LocalVoice ingestion/diarization routing. Re-embedding every WAV
    # by default is redundant, slow, and may require unavailable models.
    # An explicit opt-in repeats heuristic speaker screening when desired.
    if args.rescreen_speaker:
        screened, embedding_info = speaker_screen(
            curated, root, minimum=MIN_SPEECH)
    else:
        screened = curated
        embedding_info = {
            "performed": False,
            "reason": "reused existing LocalVoice curated provenance",
            "caveat": "Existing routing does not independently prove identity.",
        }
    speech = evenly_sample(screened, args.max_speech)
    # The 8 original human-approved WAVs already passed official LV-03
    # DACVAE. Preserve that decision, while still rejecting tampered,
    # unreadable, invalid-duration, or completely silent audio.
    approved, bad_approved = verified_wave_quality(
        preserved, state["inventory"], strict_acoustic_gate=False)
    if len(approved) != len(preserved):
        raise ValueError(
            f"Frozen approved WAV integrity failure (not a new QC veto): {bad_approved}")
    included = {r["clip_id"] for r in approved}
    speech = [r for r in speech if r["clip_id"] not in included]
    normal = approved + speech
    expressive = nonverbal_rows(state, source.resolve(), ws)
    expressive, bad_exp = verified_wave_quality(expressive, state["inventory"])
    expressive = [r for r in expressive if r["clip_id"] not in {x["clip_id"] for x in normal}]
    if len(normal) < MIN_SPEECH:
        raise ValueError("Insufficient normal speech for a meaningful LoRA attempt")
    if not expressive:
        raise ValueError(
            "No verified nonverbal pilot speech-compatible samples; "
            "cannot claim an expressive/nonverbal LoRA from this dataset.")
    selection = normal + expressive
    sources = [x["sha256"] for x in selection]
    if len(sources) != len(set(sources)):
        raise ValueError("Duplicate audio SHA in proposed training split")
    if any(video({"source_path": row["audio"]}) in state["eval_videos"] for row in selection):
        raise ValueError("Training split leaks protected evaluation video")
    upstream = root / "Irodori-TTS"
    python = upstream / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError(python)
    official_version(upstream)
    checkpoint, _ = resolve_checkpoint(upstream, None)
    config = upstream / "configs" / "train_v4_small_lora.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    folder = output_dir(ws)
    counts = dict(Counter(item["kind"] for item in selection))
    report = {
        "status": "SELECTED_NOT_TRAINED",
        "source": str(source),
        "input_inventory": len(state["inventory"]),
        "old_approved_train_retained": len(approved),
        "frozen_approved_qc_policy": "integrity_and_decodability_only; LV03 approval retained",
        "external_eval_retained": len(state["eval_ids"]),
        "unique_speech": len(normal),
        "unique_nonverbal": len(expressive),
        "by_kind": counts,
        "candidate_rejected": dict(discarded),
        "acoustic_qc_rejected": dict(qc),
        "nonverbal_qc_rejected": dict(bad_exp),
        "speaker_embedding_filter": embedding_info,
        "base_checkpoint_sha256": sha256(checkpoint),
        "source_sha256": {row["clip_id"]: row["sha256"] for row in selection},
        "model_quality_verified": False,
        "nonverbal_quality_verified": False,
        "target_vocalizations": sorted({x["kind"] for x in expressive}),
        "source_transcript_caveat": (
            "Non-previously-approved normal rows use existing ASR suggestions; "
            "not all texts have a human transcript check."),
    }
    save_json(folder / "result.json", report)
    dataset = folder / "training_sources.csv"
    write_csv(dataset, TRAIN_COLS, selection)
    report["training_sources_sha256"] = sha256(dataset)
    save_json(folder / "result.json", report)
    if not args.run:
        print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"},
                         ensure_ascii=False, indent=2))
        print(f"Selection only: {dataset}. Add --run for LoRA.")
        return 0
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        print(f"[DACVAE] encoding {len(selection)} WAVs", flush=True)
        original = encode(root, upstream, python, folder, dataset, len(selection), env)
        training_manifest, weight_info = weighted_manifest(
            original, folder, len(normal), repeat=args.nonverbal_repeat)
        report["weighted_training"] = weight_info
        report["manifest"] = str(training_manifest)
        report["status"] = "ENCODED_TRAINING_STARTING"
        save_json(folder / "result.json", report)
        command = build_command(
            python, upstream, config, training_manifest, checkpoint,
            folder / "adapter", args.steps, valid_ratio=0.05)
        if args.nonverbal_repeat > 1:
            # Official internal random split cannot group duplicate latent
            # paths. Disable that misleading validation rather than leaking
            # repeated nonverbal clips into both train and validation.
            command[command.index("--valid-ratio") + 1] = "0.0"
            command[command.index("--valid-every") + 1] = "0"
            command[command.index("--checkpoint-best-n") + 1] = "0"
        # Critical LV-04 fix: a tiny corpus had updated the whole duration
        # predictor; keep it frozen in this multi-speaker/expressive baseline.
        command += ["--lora-modules-to-save", "none"]
        command[command.index("--save-every") + 1] = "100"
        if args.nonverbal_repeat == 1:
            command[command.index("--valid-every") + 1] = "50"
        save_json(folder / "training_plan.json", {
            "command": command, "manifest_sha256": sha256(training_manifest),
            "checkpoint_sha256": sha256(checkpoint),
            "duration_predictor_trainable": False,
            "internal_validation": (
                "disabled_to_prevent_duplicate_latent_leakage"
                if args.nonverbal_repeat > 1 else "random_5_percent"),
            "independent_external_eval": "preserved unchanged; not treated as model quality pass",
        })
        print(f"[train] LoRA steps={args.steps} speaker candidates={len(normal)} "
              f"nonverbal unique={len(expressive)} repeat={args.nonverbal_repeat}",
              flush=True)
        code, reason = stream(
            command, cwd=upstream, log=folder / "train.log",
            env=env, timeout=args.train_timeout)
        if code:
            raise command_error("LoRA training", folder / "train.log", code, reason)
        adapter = folder / "adapter" / "checkpoint_final"
        report["adapter"] = validate_adapter(adapter)
        report["status"] = "ADAPTER_READY_INFERENCE_PENDING"
        save_json(folder / "result.json", report)
        print("[infer] base and LoRA paired Japanese speech", flush=True)
        report["comparisons"] = infer_pair(root, upstream, python, checkpoint,
                                          adapter, original, folder, expressive, env)
        report["status"] = "ADAPTER_AND_PAIRED_WAVS_READY_QUALITY_UNVERIFIED"
        save_json(folder / "result.json", report)
        print(f"{report['status']}\nResult: {folder / 'result.json'}", flush=True)
        return 0
    except Exception as err:
        report["status"] = "BLOCKED"
        report["error"] = f"{type(err).__name__}: {err}"
        save_json(folder / "result.json", report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Encode WAVs and train LoRA, then generate paired WAVs")
    parser.add_argument("--max-speech", type=int, default=320)
    parser.add_argument("--nonverbal-repeat", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--rescreen-speaker", action="store_true",
                        help="Optional repeat of speaker embedding checks; normally reuse existing provenance")
    parser.add_argument("--train-timeout", type=int, default=21600)
    args = parser.parse_args()
    if args.train_timeout < 1:
        raise ValueError("Training timeout must be positive")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as err:
        print(f"DIRECT LoRA BLOCKED: {type(err).__name__}: {err}",
              file=sys.stderr)
        raise SystemExit(2)
