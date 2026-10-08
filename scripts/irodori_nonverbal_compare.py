"""Generate paired base-vs-LoRA nonverbal comparisons without manual labeling.

Five prompts: held-out groan/breath conditions both with and without style
captions, plus a normal Japanese speech-retention check. Identical seed,
speaker reference, text, duration and sampler for base and LoRA.

WAV existence and metadata are checks; no audio-quality verdict is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .irodori_dataset import ROOT, audio_info, digest, paths, read_csv
    from .irodori_nonverbal_lora import load_jsonl, checkpoint_path
    from .irodori_nonverbal_manifest import environment
else:
    from irodori_dataset import ROOT, audio_info, digest, paths, read_csv
    from irodori_nonverbal_lora import load_jsonl, checkpoint_path
    from irodori_nonverbal_manifest import environment


def validate_adapter(attempt: Path) -> tuple[Path, list[dict], list[dict], dict]:
    result = attempt / "result.json"
    if not result.is_file():
        raise FileNotFoundError(f"Completed LoRA result not found: {result}")
    info = json.loads(result.read_text(encoding="utf-8"))
    if info.get("status") != "TRAIN_COMMAND_EXITED_ZERO_NOT_QUALITY_VALIDATED":
        raise ValueError("LoRA training did not report successful command completion")
    adapter = attempt / "adapter" / "checkpoint_final"
    if (not (adapter / "adapter_config.json").is_file() or
            not any((adapter / name).is_file()
                    for name in ("adapter_model.safetensors", "adapter_model.bin"))):
        raise FileNotFoundError(
            f"Irodori PEFT adapter checkpoint is missing/incomplete: {adapter}"
        )
    train = load_jsonl(attempt / "train.jsonl")
    holdout = load_jsonl(attempt / "holdout.jsonl")
    if (len(train) != 4 or len(holdout) != 2
            or sorted(Counter(r.get("text") for r in train).values()) != [2, 2]
            or sorted(Counter(r.get("text") for r in holdout).values()) != [1, 1]
            or set(r["text"] for r in holdout) != set(r["text"] for r in train)):
        raise ValueError("LoRA train and holdout stratification differs from experiment")
    trained_latents = {Path(r["latent_path"]).resolve() for r in train}
    holdout_latents = {Path(r["latent_path"]).resolve() for r in holdout}
    if (trained_latents & holdout_latents or len(trained_latents) != 4
            or len(holdout_latents) != 2
            or any(not p.is_file() for p in trained_latents | holdout_latents)):
        raise ValueError("LoRA train/holdout latent overlap or missing latent")
    return adapter, train, holdout, info


def choose_speech_reference(source: Path, out: Path,
                            excluded: set[str]) -> tuple[Path, str, str]:
    """Use authoritative auto-prep provenance; inventory alone is insufficient.

    scan() records source_kind only from manifest.tsv. build() subsequently
    resolves curated provenance via review_origin(), including WAV filename
    markers. The auto-preparation report contains that resolved value, so
    selecting on inventory.source_kind alone incorrectly rejects real clips.
    """
    candidate_path = out / "dataset_for_prepare_manifest_auto.csv"
    report_path = out / "auto_preparation_report.csv"
    if not candidate_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            "Existing curated speech candidates or auto-preparation provenance "
            "is missing; run 'prepare' to recreate reports from existing results"
        )
    candidates = read_csv(candidate_path)
    reports = read_csv(report_path)
    inventory = read_csv(out / "inventory.csv")
    by_path = {r["audio"]: r for r in candidates if r.get("audio")}
    by_id = {r["clip_id"]: r for r in reports if r.get("clip_id")}
    if len(by_path) != len(candidates) or len(by_id) != len(reports):
        raise ValueError("Duplicate or invalid speech candidate / provenance entries")
    # Keep speech conditions distinct from the trial's six nonverbal clips.
    # Do not promote human-unverified acoustic-event suggestions to the
    # reference pool, nor treat unknown origin as proof of curated speech.
    curated_origins = {"my_voice", "review_approved"}
    curated_reasons = {"curated_source_with_asr"}
    scored = []
    inspected = Counter()
    for row in inventory:
        path = row.get("source_path", "")
        if path in excluded or path not in by_path:
            continue
        inspected["candidate_in_inventory"] += 1
        detail = by_id.get(row["clip_id"])
        if (detail is None or detail.get("source_path") != path
                or detail.get("sha256") != row["sha256"]
                or detail.get("route") != "training_candidate"):
            inspected["missing_or_mismatched_provenance"] += 1
            continue
        if (detail.get("source_kind") not in curated_origins
                or detail.get("reason") not in curated_reasons
                or detail.get("candidate_style") != "normal"):
            inspected["not_normal_curated_speech"] += 1
            continue
        if (not by_path[path].get("text", "").strip()
                or by_path[path].get("speaker", "") == ""
                or by_path[path]["text"].strip() != detail.get("text", "").strip()):
            inspected["missing_or_changed_text"] += 1
            continue
        try:
            duration = float(row["duration_sec"])
            rate = int(row["sample_rate"])
        except (ValueError, KeyError, TypeError):
            inspected["invalid_metadata"] += 1
            continue
        if rate < 16000 or row.get("scan_flag") != "ok" or not 1.5 <= duration <= 20:
            inspected["invalid_audio_range"] += 1
            continue
        inspected["eligible_metadata"] += 1
        # Strongly prefer the standard 3..16-second reference window, while
        # permitting a shorter valid clip if that is all that exists.
        penalty = 0 if 3 <= duration <= 16 else 1
        scored.append((penalty, abs(duration - 8.0), row["clip_id"], row, detail))

    # Hash verification is done on the source WAV, not inferred from the CSV.
    for _, _, _, row, detail in sorted(scored, key=lambda x: x[:3]):
        ref = Path(row["source_path"]).resolve()
        if (ref.is_file() and source in ref.parents
                and digest(ref) == row["sha256"]):
            return ref, row["sha256"], detail["reason"]
        inspected["hash_or_path_mismatch"] += 1
    raise ValueError(
        "No SHA-verified, ordinary curated speech reference found in the "
        "existing candidate/provenance tables; held-out nonverbal samples "
        f"cannot substitute for one. Selection audit={dict(inspected)}"
    )


def prompts(holdout: list[dict]) -> list[dict]:
    output = []
    for row in sorted(holdout, key=lambda r: r["text"]):
        label = {"😮‍💨": "breath", "🥵": "groan"}.get(row["text"])
        if label is None:
            raise ValueError("Unexpected nonverbal conditioning cue")
        if not row.get("caption") or not isinstance(row["caption"], str):
            raise ValueError("Missing automated holdout caption")
        output.extend([
            {"id": f"{label}_caption", "text": row["text"],
             "caption": row["caption"], "seconds": 3.0, "seed": 20261008},
            {"id": f"{label}_emoji_only", "text": row["text"],
             "caption": None, "seconds": 3.0, "seed": 20261008},
        ])
    output.append({
        "id": "normal_speech", "text": "こんにちは。今日はいい天気ですね。",
        "caption": None, "seconds": 3.5, "seed": 20261008,
    })
    if len(output) != 5:
        raise ValueError("Unexpected paired inference case count")
    return output


def inference_command(python: Path, upstream: Path, checkpoint: Path,
                      adapter: Path | None, ref: Path, case: dict,
                      output: Path, shared: Path | None) -> list[str]:
    infer_script = upstream / "infer.py"
    if not infer_script.is_file():
        raise FileNotFoundError(f"Irodori inference CLI missing: {infer_script}")
    args = [
        "--checkpoint", str(checkpoint),
        "--text", case["text"], "--output-wav", str(output),
        "--ref-wav", str(ref),
        "--ref-normalize-db", "none",
        "--model-device", "cuda", "--codec-device", "cuda",
        "--model-precision", "bf16", "--codec-precision", "fp32",
        "--num-steps", "8", "--t-schedule-mode", "sway",
        "--seconds", str(case["seconds"]), "--seed", str(case["seed"]),
        "--num-candidates", "1", "--decode-mode", "sequential",
    ]
    if case["caption"] is not None:
        args.extend(["--caption", case["caption"]])
    if adapter is not None:
        args.extend(["--lora-adapter", str(adapter)])
    if shared is not None:
        entry = Path(__file__).with_name("irodori_codec_entry.py").resolve()
        if not entry.is_file():
            raise FileNotFoundError(f"Windows FFmpeg entrypoint missing: {entry}")
        return [str(python), str(entry), str(shared), str(infer_script), *args]
    return [str(python), str(infer_script), *args]


def inspect_wav(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 44:
        raise RuntimeError(f"Inference output WAV absent or empty: {path}")
    # Irodori writes float PCM when torchaudio.save is available.
    # wave.open() does not accept common IEEE FLOAT WAV (format 3) on
    # Python 3.10/3.12; existing audio_info() uses ffprobe first.
    try:
        duration, rate, chans = audio_info(path)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise RuntimeError(f"Invalid inference WAV: {path}") from exc
    if duration <= 0 or rate <= 0 or chans <= 0:
        raise RuntimeError(f"Invalid empty audio output: {path}")
    return {"duration_sec": round(duration, 3),
            "sample_rate": rate, "channels": chans,
            "bytes": path.stat().st_size}


def fresh_compare(attempt: Path) -> Path:
    for i in range(1, 100):
        out = attempt / f"comparison_{i:03d}"
        if not out.exists():
            return out
    raise RuntimeError("No unused inference comparison directory")


def compare(args: argparse.Namespace, source: Path, out: Path, *,
            root: Path = ROOT, runner=subprocess.run,
            windows: bool | None = None) -> dict:
    upstream, python = environment(root)
    attempt = (out / "nonverbal_pilot" / args.lora_attempt).resolve()
    pilot_root = (out / "nonverbal_pilot").resolve()
    if attempt.parent != pilot_root or not attempt.is_dir():
        raise ValueError("LoRA attempt must be an existing isolated pilot directory")
    adapter, train, holdout, prior = validate_adapter(attempt)
    checkpoint = Path(prior.get("checkpoint", "")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Original Irodori base checkpoint missing: {checkpoint}")
    reference, reference_sha, reference_reason = choose_speech_reference(
        source, out, {r["audio"] for r in load_jsonl(
            out / "nonverbal_pilot/hf_audio_dataset_hypothesis.jsonl")}
    )
    codec = out / "nonverbal_pilot" / args.codec_attempt
    meta = json.loads((codec / "result.json").read_text(encoding="utf-8"))
    if meta.get("status") != "PASS_DACVAE_MANIFEST_ONLY":
        raise ValueError("Source DACVAE experiment is not valid")
    shared = None
    if (os.name == "nt" if windows is None else windows):
        raw = meta.get("ffmpeg_shared_bin")
        if not raw or raw == "existing_environment":
            # Existing environment may not require a bundled FFmpeg directory.
            pass
        else:
            shared = Path(raw).resolve()
            if not shared.is_dir():
                raise FileNotFoundError(f"Project-local FFmpeg shared DLL directory missing: {shared}")
    cases = prompts(holdout)
    target = fresh_compare(attempt)
    jobs = []
    for case in cases:
        for variant in ("base", "lora"):
            wav = target / f"{case['id']}_{variant}.wav"
            cmd = inference_command(
                python, upstream, checkpoint,
                adapter if variant == "lora" else None,
                reference, case, wav, shared,
            )
            jobs.append({"case": case, "variant": variant,
                         "wav": wav, "cmd": cmd})
    result = {
        "status": "PLAN_ONLY" if not args.run else "RUNNING",
        "lora_attempt": str(attempt),
        "adapter": str(adapter), "base_checkpoint": str(checkpoint),
        "speech_reference": str(reference),
        "speech_reference_sha256": reference_sha,
        "speech_reference_provenance_reason": reference_reason,
        "heldout_clips": 2, "paired_prompts": len(cases),
        "expected_wavs": len(jobs),
        "output_dir": str(target),
        "manual_classification_required": False,
        "quality_assessed": False,
        "note": "Paired inference verifies behavior only; no claim of acoustic improvement.",
    }
    if not args.run:
        result["planned_files"] = [str(x["wav"]) for x in jobs]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    target.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for job in jobs:
            process = runner(job["cmd"], cwd=str(upstream), text=True,
                             capture_output=True, check=False)
            log = target / (job["wav"].stem + ".log")
            log.write_text(
                (process.stdout or "") + "\n" + (process.stderr or ""),
                encoding="utf-8",
            )
            if process.returncode:
                raise RuntimeError(
                    f"Irodori inference failed for {job['wav'].name}, "
                    f"exit={process.returncode}. See {log}"
                )
            audio_info = inspect_wav(job["wav"])
            results.append({
                "case": job["case"]["id"], "variant": job["variant"],
                "text": job["case"]["text"],
                "caption": job["case"]["caption"],
                "seed": job["case"]["seed"],
                "seconds": job["case"]["seconds"],
                "wav": str(job["wav"]),
                **audio_info,
            })
        result["status"] = "PAIRED_WAVS_READY_NOT_QUALITY_VALIDATED"
        result["completed_wavs"] = len(results)
    except Exception as exc:
        result["status"] = "BLOCK"
        result["error"] = str(exc)
        result["completed_wavs"] = len(results)
        raise
    finally:
        result["wav_metadata"] = results
        (target / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--lora-attempt", default="lora_attempt_001")
    parser.add_argument("--codec-attempt", default="codec_attempt_001")
    parser.add_argument("--run", action="store_true",
                        help="Explicitly generate ten paired WAVs on GPU")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        compare(args, source, out)
        return 0
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
