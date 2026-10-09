"""Read-only, explicit 12-sample Japanese ASR pilot; NOT a style classifier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CATEGORIES = {
    "normal": 3,
    "whisper": 2,
    "mixed": 2,
    "nonverbal": 3,
    "silence_noise": 1,
    "other_overlap": 1,
}
MODEL_IDS = {"kotoba": "kotoba-tech/kotoba-whisper-v2.0", "whisper": "openai/whisper-large-v3"}
IDENTITIES = {"approved_self", "candidate", "rejected_other", "unknown"}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "localvoice.asr-pilot.v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _resolved_source(value: str) -> Path:
    p = Path(value).expanduser()
    p = (p if p.is_absolute() else PROJECT_ROOT / p).resolve(strict=True)
    if not p.is_file() or p.suffix.lower() != ".wav":
        raise ValueError(f"Source must be a WAV file: {p}")
    return p


def _duration_ms(path: Path) -> int:
    import soundfile as sf

    meta = sf.info(str(path))
    if meta.samplerate < 8000 or meta.channels not in (1, 2):
        raise ValueError(f"Unsupported WAV sample rate/channels: {path}")
    dur = 1000 * meta.frames / meta.samplerate
    if not 250 <= dur <= 30000:
        raise ValueError(f"Pilot WAV length must be 0.25–30 seconds: {path} ({dur / 1000:.2f}s)")
    return round(dur)


def _check_selection(rows: list[dict[str, Any]]) -> None:
    counts = Counter(row.get("category") for row in rows)
    if counts != Counter(CATEGORIES):
        raise ValueError(f"Pilot must contain exactly 12 samples by category: expected={CATEGORIES}, actual={dict(counts)}")
    ids: set[str] = set()
    paths: set[str] = set()
    for row in rows:
        name = row.get("sample_id")
        if not isinstance(name, str) or not ID_PATTERN.fullmatch(name) or name in ids:
            raise ValueError(f"Invalid/duplicate sample_id: {name!r}")
        ids.add(name)
        profile = row.get("profile")
        if not isinstance(profile, str) or not ID_PATTERN.fullmatch(profile):
            raise ValueError(f"Invalid profile in {name}: {profile!r}")
        if row.get("identity_review") not in IDENTITIES:
            raise ValueError(f"Missing/invalid identity_review for {name}")
        source = row.get("source_path")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Missing source_path for {name}")
        resolved = str(_resolved_source(source)).casefold()
        if resolved in paths:
            raise ValueError(f"Duplicate audio source: {source}")
        paths.add(resolved)


def seal(csv_path: Path, manifest: Path) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "category", "source_path", "profile", "identity_review"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Selection CSV columns must include {sorted(required)}")
        rows = [{key: str(row[key] or "").strip() for key in required} for row in reader]
    _check_selection(rows)
    sealed = []
    for row in rows:
        path = _resolved_source(row["source_path"])
        sealed.append({**row, "source_path": str(path), "source_sha256": _sha256(path),
                       "duration_ms": _duration_ms(path), "schema_version": SCHEMA})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as f:
        for row in sealed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"SEALED: {manifest} ({len(sealed)} clips; sources unchanged)")


def validate(manifest: Path) -> list[dict[str, Any]]:
    with manifest.open(encoding="utf-8-sig") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    _check_selection(rows)
    for row in rows:
        if row.get("schema_version") != SCHEMA:
            raise ValueError(f"Wrong manifest schema for {row['sample_id']}")
        p = _resolved_source(row["source_path"])
        if _sha256(p) != row.get("source_sha256"):
            raise ValueError(f"WAV integrity failure: {p}")
        if _duration_ms(p) != row.get("duration_ms"):
            raise ValueError(f"WAV duration mismatch: {p}")
    return rows


def _as_float_audio(path: Path) -> Any:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if not np.isfinite(mono).all():
        raise ValueError("Nonfinite WAV samples")
    if sr != 16000:
        divisor = gcd(int(sr), 16000)
        mono = resample_poly(mono, 16000 // divisor, int(sr) // divisor).astype("float32")
    return mono


def _load_pipeline(model_id: str, device: str):
    import torch
    from transformers import pipeline

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Never silently fall back to CPU.")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    return pipeline("automatic-speech-recognition", model=model_id, device="cuda:0" if device == "cuda" else "cpu",
                    torch_dtype=torch_dtype,
                    model_kwargs={"attn_implementation": "sdpa"} if device == "cuda" else {})


def _segment_rows(result: dict[str, Any], source: dict[str, Any], run_id: str, model_id: str) -> list[dict[str, Any]]:
    raw_text = str(result.get("text") or "").strip()
    base = {"schema_version": SCHEMA, "run_id": run_id, "model_id": model_id,
            "source_id": source["sample_id"], "source_path": source["source_path"],
            "source_sha256": source["source_sha256"], "profile": source["profile"],
            "identity_review": source["identity_review"], "pilot_category": source["category"],
            "transcript_checked": None, "failure_reason": None, "asr_diagnostics": None}
    # The whole-source record is always retained, even when all timestamp chunks are absent.
    out = [{**base, "segment_id": "source", "start_ms": 0, "end_ms": source["duration_ms"],
            "asr_status": "speech_candidate" if raw_text else "non_speech_candidate", "transcript_raw": raw_text or None,
            "time_boundary_type": "source_extent_not_voice_detection"}]
    chunks = result.get("chunks")
    if not isinstance(chunks, list):
        return out
    for idx, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            out.append({**base, "segment_id": f"uncertain_{idx}", "start_ms": 0,
                        "end_ms": source["duration_ms"], "asr_status": "uncertain", "transcript_raw": None,
                        "failure_reason": "malformed_timestamp_chunk", "time_boundary_type": "source_extent_fallback"})
            continue
        stamp = chunk.get("timestamp")
        txt = str(chunk.get("text") or "").strip()
        valid = (isinstance(stamp, (tuple, list)) and len(stamp) == 2
                 and all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in stamp))
        start, end = (round(float(stamp[0]) * 1000), round(float(stamp[1]) * 1000)) if valid else (0, source["duration_ms"])
        valid = valid and 0 <= start < source["duration_ms"] and start < end <= source["duration_ms"] + 50
        out.append({**base, "segment_id": f"segment_{idx:04d}", "start_ms": start if valid else 0,
                    "end_ms": min(end, source["duration_ms"]) if valid else source["duration_ms"],
                    "asr_status": ("speech_candidate" if txt else "uncertain") if valid else "uncertain",
                    "transcript_raw": txt or None, "failure_reason": None if valid else "invalid_or_absent_timestamps",
                    "time_boundary_type": "model_estimate" if valid else "source_extent_fallback"})
    return out


def _jsonl_write(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def run(manifest: Path, model: str, run_id: str, device: str) -> int:
    if not ID_PATTERN.fullmatch(run_id):
        raise ValueError("--run-id must be a unique safe name (letters/numbers/._-)")
    rows = validate(manifest)  # fail without creating a run on a bad manifest
    work = PROJECT_ROOT / "work"
    work.mkdir(exist_ok=True)
    target = work / run_id
    target.mkdir(exist_ok=False)  # never overwrite or clean a prior run
    results = target / "transcription.jsonl"
    failures = target / "failures.jsonl"
    runlog = target / "run_manifest.json"
    for path in (results, failures):
        path.touch(exist_ok=False)
    state = {"schema_version": SCHEMA, "run_id": run_id, "model_id": MODEL_IDS[model],
             "device_requested": device, "input_manifest": str(manifest.resolve()),
             "input_manifest_sha256": _sha256(manifest), "samples": len(rows),
             "started_at": datetime.now(timezone.utc).isoformat(), "status": "initializing", "completed": 0, "failed": 0,
             "policy": "ASR candidate text is never a style or speaker approval"}
    def save_state():
        runlog.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    save_state()
    try:
        pipe = _load_pipeline(MODEL_IDS[model], device)
        state["status"] = "running"
        save_state()
        for row in rows:
            try:
                path = _resolved_source(row["source_path"])
                if _sha256(path) != row["source_sha256"]:
                    raise ValueError("WAV changed before inference")
                audio = _as_float_audio(path)
                prediction = pipe(audio, return_timestamps=True, generate_kwargs={"language": "ja", "task": "transcribe"})
                if not isinstance(prediction, dict):
                    raise ValueError("Unexpected ASR output type")
                if _sha256(path) != row["source_sha256"]:
                    raise ValueError("WAV changed during inference")
                for item in _segment_rows(prediction, row, run_id, MODEL_IDS[model]):
                    _jsonl_write(results, item)
                state["completed"] += 1
            except Exception as exc:
                state["failed"] += 1
                record = {"source_id": row["sample_id"], "source_path": row["source_path"],
                          "failure_kind": type(exc).__name__, "reason": str(exc)}
                _jsonl_write(failures, record)
                _jsonl_write(results, {"schema_version": SCHEMA, "run_id": run_id,
                                      "source_id": row["sample_id"], "source_path": row["source_path"],
                                      "source_sha256": row["source_sha256"], "segment_id": "source",
                                      "start_ms": 0, "end_ms": row["duration_ms"], "asr_status": "error",
                                      "transcript_raw": None, "failure_reason": record["reason"]})
                print(f"FAILED {row['sample_id']}: {exc}", file=sys.stderr)
            save_state()
        state["status"] = "completed" if state["failed"] == 0 else "completed_with_errors"
    except Exception as exc:
        state["status"] = "failed_initialization"
        state["initialization_error"] = f"{type(exc).__name__}: {exc}"
        _jsonl_write(failures, {"failure_kind": type(exc).__name__, "reason": str(exc)})
    finally:
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_state()
    print(f"PILOT {state['status']}: {target} {state['completed']}/12 processed, {state['failed']} failed")
    return 0 if state["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Japanese ASR pilot; no style classification")
    root = parser.add_subparsers(dest="command", required=True)
    seal_cmd = root.add_parser("seal", help="Seal a manually curated 12-row CSV into a hashed manifest")
    seal_cmd.add_argument("--csv", required=True, type=Path)
    seal_cmd.add_argument("--manifest", required=True, type=Path)
    check = root.add_parser("validate", help="Validate exactly 12 WAVs and their SHA-256 without model download")
    check.add_argument("--manifest", required=True, type=Path)
    start = root.add_parser("run", help="Run one model over the exact same sealed 12-sample manifest")
    start.add_argument("--manifest", required=True, type=Path)
    start.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    start.add_argument("--run-id", required=True)
    start.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            seal(args.csv, args.manifest)
        elif args.command == "validate":
            samples = validate(args.manifest)
            print(f"VALID: {len(samples)} source WAVs, all SHA-256 verified")
        else:
            return run(args.manifest, args.model, args.run_id, args.device)
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"ASR PILOT BLOCKED: {exc}", file=sys.stderr)
        return 2
