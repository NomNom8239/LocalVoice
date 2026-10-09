"""Read-only acoustic style *candidate* Pilot; no ASR labels or auto-approval."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from math import gcd
from pathlib import Path
from typing import Any

MODEL_IDS = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "clap": "laion/clap-htsat-fused",
}
# AudioSet labels present in AST; these are acoustic evidence, NOT calibrated
# probabilities for our Japanese reference-WAV taxonomy.
AST_LABELS = {
    "normal_speech": ("Speech", "Conversation", "Narration, monologue"),
    "whisper": ("Whispering",),
    "exhale": ("Sigh",),
    "laugh": ("Laughter", "Giggle", "Snicker", "Chuckle, chortle", "Belly laugh"),
    "moan": ("Wail, moan", "Groan", "Grunt"),
    "pant": ("Pant",),
    "gasp": ("Gasp",),
    "cry": ("Crying, sobbing", "Whimper"),
    "scream": ("Screaming", "Yell", "Shout"),
    "hum": ("Humming",),
}
# English audio event descriptions (not Japanese transcripts or Irodori prompts).
# CLAP scores are softmax RELATIVE to this exact candidate set.
CLAP_CAPTIONS = {
    "normal_speech": "A woman is speaking normally in a conversational voice.",
    "whisper": "A person is quietly whispering words in a soft breathy voice.",
    "exhale": "A person is softly sighing and audibly exhaling a long breath.",
    "laugh": "A person is laughing and giggling.",
    "moan": "A person is moaning and groaning with a sustained voiced sound.",
    "pant": "A person is panting and breathing quickly and heavily.",
    "gasp": "A person sharply gasps and inhales suddenly.",
    "cry": "A person is crying and sobbing audibly.",
    "scream": "A person is screaming or yelling loudly.",
    "yawn": "A person is yawning audibly.",
    "hum": "A person is humming a melody with a closed mouth.",
    "other_overlap": "Two people are speaking over one another.",
    "noise_music": "Music and video game sound effects are playing.",
    "silence": "There is silence with no audible sound.",
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "localvoice.acoustic-pilot.v1"
MAX_CLIPS = 32  # Explicit small Pilot only; no automatic processing of 1,167 WAVs.


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(value, ensure_ascii=False) + "\n")


def validate(manifest: Path) -> list[dict[str, Any]]:
    import soundfile as sf

    with manifest.open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not (1 <= len(rows) <= MAX_CLIPS):
        raise ValueError(f"Expected 1–{MAX_CLIPS} explicit Pilot WAVs, got {len(rows)}")
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    paths: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Manifest entries must be JSON objects")
        source_id = raw.get("source_id", raw.get("sample_id"))
        if not isinstance(source_id, str) or not ID_RE.fullmatch(source_id) or source_id in ids:
            raise ValueError(f"Invalid/duplicate source_id: {source_id!r}")
        ids.add(source_id)
        value = raw.get("source_path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing WAV path for {source_id}")
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise ValueError(f"Not a WAV file: {path}")
        canonical_path = str(path).casefold()
        if canonical_path in paths:
            raise ValueError(f"Duplicate source path: {path}")
        paths.add(canonical_path)
        recorded_sha = raw.get("source_sha256")
        if not isinstance(recorded_sha, str) or not SHA_RE.fullmatch(recorded_sha):
            raise ValueError(f"Invalid source SHA for {source_id}")
        if _sha256(path) != recorded_sha:
            raise ValueError(f"WAV SHA mismatch for {source_id}")
        meta = sf.info(str(path))
        duration_ms = round(meta.frames * 1000 / meta.samplerate)
        if meta.channels not in (1, 2) or meta.samplerate < 8000 or not (250 <= duration_ms <= 30000):
            raise ValueError(f"Unsupported WAV duration/channels/sample rate: {path}")
        if raw.get("duration_ms") is not None and raw["duration_ms"] != duration_ms:
            raise ValueError(f"WAV duration mismatch for {source_id}")
        validated.append({
            "source_id": source_id, "source_path": str(path), "source_sha256": recorded_sha,
            "duration_ms": duration_ms, "profile": raw.get("profile"),
            "identity_review": raw.get("identity_review", "unknown"),
            # Audit only; never provided to the model or treated as ground truth.
            "selection_category_hint": raw.get("category"),
        })
    return validated


def _audio(path: Path, sample_rate: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    samples, old_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("Nonfinite WAV samples")
    is_digital_silence = bool(np.max(np.abs(mono)) == 0.0)
    if old_rate != sample_rate:
        d = gcd(int(old_rate), sample_rate)
        mono = resample_poly(mono, sample_rate // d, int(old_rate) // d).astype("float32")
    return mono, is_digital_silence


def _load_pipeline(model: str, device: str):
    import torch
    from transformers import pipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently fall back to CPU")
    return pipeline(
        "audio-classification" if model == "ast" else "zero-shot-audio-classification",
        model=MODEL_IDS[model], device="cuda:0" if device == "cuda" else "cpu",
    )


def _summarize(model: str, raw: list[dict[str, Any]]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected a nonempty list of acoustic model scores")
    score_map: dict[str, float] = {}
    for item in raw:
        label, score = item.get("label"), item.get("score")
        if not isinstance(label, str) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError("Invalid acoustic score response")
        score_map[label] = float(score)
    if model == "ast":
        # An AudioSet label may be absent from an incompatible checkpoint. Do
        # not convert missing evidence to a fabricated zero-confidence value.
        category_scores = {
            code: max((score_map[x] for x in labels if x in score_map), default=None)
            for code, labels in AST_LABELS.items()
        }
        raw_top = sorted(raw, key=lambda x: -x["score"])[:10]
    else:
        category_scores = {code: score_map.get(caption) for code, caption in CLAP_CAPTIONS.items()}
        raw_top = sorted(raw, key=lambda x: -x["score"])[:10]
        if len(score_map) != len(CLAP_CAPTIONS):
            raise ValueError("CLAP response omitted one or more candidate captions")
    return category_scores, raw_top


def run(manifest: Path, model: str, run_id: str, device: str) -> int:
    if model not in MODEL_IDS or device not in ("cuda", "cpu"):
        raise ValueError("Unknown model/device")
    if not ID_RE.fullmatch(run_id):
        raise ValueError("Unsafe or invalid --run-id")
    rows = validate(manifest)  # Fail before allocating output if manifest is invalid.
    base = PROJECT_ROOT / "work"
    base.mkdir(exist_ok=True)
    target = base / run_id
    target.mkdir(exist_ok=False)  # No deletion/replacement of old runs.
    output = target / "style_predictions.jsonl"
    failures = target / "failures.jsonl"
    status_path = target / "run_manifest.json"
    output.touch(exist_ok=False)
    failures.touch(exist_ok=False)
    config_sha = hashlib.sha256(json.dumps({
        "model_id": MODEL_IDS[model], "ast_labels": AST_LABELS, "clap_captions": CLAP_CAPTIONS,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    state: dict[str, Any] = {
        "schema_version": SCHEMA, "run_id": run_id, "model_key": model, "model_id": MODEL_IDS[model],
        "device_requested": device, "input_manifest": str(manifest.resolve()),
        "input_manifest_sha256": _sha256(manifest), "config_sha256": config_sha,
        "model_revision": None, "samples": len(rows), "completed": 0, "failed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(), "status": "initializing",
        "policy": "candidate scores are not approval, calibrated confidence, or ground truth",
    }

    def save() -> None:
        status_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    save()
    try:
        pipe = _load_pipeline(model, device)
        state["model_revision"] = getattr(getattr(pipe.model, "config", None), "_commit_hash", None)
        rate = int(pipe.feature_extractor.sampling_rate)
        state["sampling_rate"] = rate
        state["status"] = "running"
        save()
        for row in rows:
            record: dict[str, Any] = {
                **row, "schema_version": SCHEMA, "run_id": run_id,
                "model_id": MODEL_IDS[model], "config_sha256": config_sha,
                "segment_id": "source", "start_ms": 0, "end_ms": row["duration_ms"],
                "review_status": "requires_review", "human_approved": False,
            }
            try:
                path = Path(row["source_path"])
                if _sha256(path) != row["source_sha256"]:
                    raise ValueError("WAV changed before inference")
                mono, silent = _audio(path, rate)
                if model == "ast":
                    raw_scores = pipe(mono, top_k=527)
                else:
                    raw_scores = pipe(mono, candidate_labels=list(CLAP_CAPTIONS.values()),
                                      hypothesis_template="{}")
                categories, raw_top = _summarize(model, raw_scores)
                if _sha256(path) != row["source_sha256"]:
                    raise ValueError("WAV changed during inference")
                ranked = sorted(((k, v) for k, v in categories.items() if v is not None),
                                key=lambda pair: -pair[1])
                record.update({
                    "status": "scored", "digital_silence": silent,
                    "candidate_ranking": [{"category": k, "score": v} for k, v in ranked[:5]],
                    "category_scores": categories, "raw_top_audio_labels": raw_top,
                    "score_semantics": "relative_softmax_given_captions" if model == "clap"
                                       else "audioset_class_score_unvalidated",
                    "candidate_is_approved": False,
                    "caution": "digital_silence_models_can_hallucinate" if silent else None,
                })
                state["completed"] += 1
            except Exception as exc:
                state["failed"] += 1
                record.update({"status": "error", "failure_reason": f"{type(exc).__name__}: {exc}"})
                _jsonl(failures, {
                    "source_id": row["source_id"], "failure_kind": type(exc).__name__,
                    "reason": str(exc),
                })
                print(f"FAILED {row['source_id']}: {exc}", file=sys.stderr)
            _jsonl(output, record)
            save()
        state["status"] = "completed" if not state["failed"] else "completed_with_errors"
    except Exception as exc:
        state["status"] = "failed_initialization"
        state["initialization_error"] = f"{type(exc).__name__}: {exc}"
        _jsonl(failures, {"failure_kind": type(exc).__name__, "reason": str(exc)})
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
    print(f"ACOUSTIC PILOT {state['status']}: {target}, completed={state['completed']}, failed={state['failed']}")
    return 0 if state["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only acoustic classification candidates only")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--manifest", type=Path, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            rows = validate(args.manifest)
            print(f"VALID: {len(rows)} acoustic Pilot WAVs; SHA-256 unchanged")
            return 0
        return run(args.manifest, args.model, args.run_id, args.device)
    except (OSError, ValueError, ImportError) as exc:
        print(f"ACOUSTIC PILOT BLOCKED: {exc}", file=sys.stderr)
        return 2
