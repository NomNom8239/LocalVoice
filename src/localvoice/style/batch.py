"""Best-effort AST grouping of existing WAVs into an UNVERIFIED browse library.

No ASR, speaker approvals, dataset training, destructive source edits, or silent
promotion into the separately reviewed outputs/datasets tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import pilot

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "localvoice.ast-batch.v1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
JAPANESE_DIRS = {
    "normal_speech": "01_通常会話",
    "whisper": "02_囁き",
    "exhale": "03_吐息",
    "laugh": "04_笑い",
    "moan": "05_喘ぎ_うめき",
    "pant": "06_息切れ_荒い呼吸",
    "gasp": "07_息をのむ",
    "cry": "08_泣き声",
    "scream": "09_悲鳴",
    "hum": "11_鼻歌",
}
UNKNOWN_DIR = "99_未分類_要確認"
DEFAULT_MIN_SCORE = 0.03
DEFAULT_TOP_RATIO = 0.10


def _sha256(path: Path) -> str:
    return pilot._sha256(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _write_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as out:
        out.write(value)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Checkpoint without truncating the last good manifest.

    On Windows a short-lived reader/AV scan can prevent replacement of an
    existing file (WinError 5/32). Retry a bounded number of times. Never
    replace this atomic operation with direct writes to the live manifest.
    """
    temporary = path.with_name(path.name + ".writing")
    if temporary.is_symlink():
        raise ValueError(f"Linked pending state file: {temporary}")
    # A crash may have left our own uncommitted temporary checkpoint.
    # The current run manifest stays authoritative until os.replace succeeds.
    if temporary.exists():
        if not temporary.is_file():
            raise ValueError(f"Unexpected pending state path: {temporary}")
        temporary.unlink()
    _write_new(temporary, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if attempt == 7:
                # Preserve .writing for diagnosis/retry. The previous manifest
                # and durable style_predictions.jsonl are still intact.
                raise PermissionError(
                    f"Checkpoint replace blocked after 8 attempts: {temporary} -> {path}; "
                    "stop other readers/sync/AV processes and resume the same run"
                ) from exc
            time.sleep(min(0.05 * (2 ** attempt), 0.75))


def _stage_text(path: Path, value: str, *, resume: bool, encoding: str = "utf-8") -> None:
    """Metadata only: allow rebuilding our own incomplete stage when resuming."""
    if path.is_symlink() or (path.exists() and (not resume or not path.is_file())):
        raise ValueError(f"Unexpected staged metadata: {path}")
    if path.exists():
        with path.open("w", encoding=encoding, newline="") as out:
            out.write(value)
    else:
        with path.open("x", encoding=encoding, newline="") as out:
            out.write(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    values = [json.loads(s) for s in lines if s.strip()]
    if not all(isinstance(x, dict) for x in values):
        raise ValueError(f"Malformed JSONL object in {path}")
    return values


def _root(profile: str) -> Path:
    if not SAFE_NAME.fullmatch(profile):
        raise ValueError("Unsafe --profile")
    path = PROJECT_ROOT / "data" / "training_audio" / profile / "audio"
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Missing / linked input directory: {path}")
    return path.resolve(strict=True)


def _inventory(profile: str, expected_count: int | None) -> list[dict[str, Any]]:
    source_root = _root(profile)
    files = sorted((p for p in source_root.rglob("*") if p.suffix.lower() == ".wav"),
                   key=lambda p: p.relative_to(source_root).as_posix().casefold())
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(f"Expected {expected_count} WAV files; found {len(files)}. No run created.")
    if not files:
        raise ValueError("No WAV files in the requested profile")
    rows: list[dict[str, Any]] = []
    for number, path in enumerate(files, 1):
        if path.is_symlink() or not path.is_file() or any(p.is_symlink() for p in path.parents if p != source_root):
            raise ValueError(f"Linked or invalid source WAV: {path}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source_root):
            raise ValueError(f"Source escapes audio directory: {path}")
        relative = resolved.relative_to(source_root).as_posix()
        rows.append({
            "source_id": f"c{number:05d}", "profile": profile,
            "source_path": str(resolved), "relative_path": relative,
            "source_sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size,
        })
    return rows


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256((_json(rows) + "\n").encode("utf-8")).hexdigest()


def _work(run_id: str) -> Path:
    if not SAFE_NAME.fullmatch(run_id):
        raise ValueError("Unsafe --run-id")
    return PROJECT_ROOT / "work" / run_id


def _audit(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {r["source_id"]: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate inventory IDs")
    seen: dict[str, dict[str, Any]] = {}
    for result in results:
        source_id = result.get("source_id")
        if source_id not in expected or source_id in seen:
            raise ValueError(f"Unknown/duplicate prediction source_id: {source_id}")
        original = expected[source_id]
        if any(result.get(k) != original[k] for k in
               ("source_path", "source_sha256", "relative_path")):
            raise ValueError(f"Prediction source mismatch: {source_id}")
        if result.get("status") not in ("scored", "unknown", "error"):
            raise ValueError(f"Invalid prediction status: {source_id}")
        if result.get("category_dir") not in set(JAPANESE_DIRS.values()) | {UNKNOWN_DIR}:
            raise ValueError(f"Unknown category output directory: {source_id}")
        seen[source_id] = result
    return seen


def _state(target: Path) -> dict[str, Any]:
    return json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))


def inventory(profile: str, expected_count: int | None) -> None:
    rows = _inventory(profile, expected_count)
    print(f"INVENTORY: profile={profile} count={len(rows)} "
          f"bytes={sum(r['size_bytes'] for r in rows)} sha={_fingerprint(rows)} (read-only)")


def _load_ast(device: str):
    return pilot._load_pipeline("ast", device)


def _excel_safe(value: Any) -> Any:
    """Prevent source-supplied names/error text becoming Excel CSV formulas."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _candidate(raw: list[dict[str, Any]], min_score: float, top_ratio: float
               ) -> tuple[str, str | None, float | None, dict[str, float | None], list[dict[str, Any]]]:
    scores, raw_top = pilot._summarize("ast", raw)
    ranked = sorted(((name, score) for name, score in scores.items() if score is not None),
                    key=lambda item: item[1], reverse=True)
    max_raw = max((item["score"] for item in raw), default=0)
    if not ranked or ranked[0][1] < min_score or (
        max_raw > 0 and ranked[0][1] / max_raw < top_ratio
    ):
        return UNKNOWN_DIR, None, None, scores, raw_top
    category, score = ranked[0]
    return JAPANESE_DIRS[category], category, score, scores, raw_top


def run(profile: str, run_id: str, expected_count: int | None, device: str,
        min_score: float, top_ratio: float, max_new: int | None, resume: bool) -> int:
    if not 0 <= min_score <= 1 or not 0 <= top_ratio <= 1:
        raise ValueError("Scores must be between 0 and 1")
    if max_new is not None and max_new < 1:
        raise ValueError("--max-new must be positive")
    rows = _inventory(profile, expected_count)  # no side effects until complete preflight
    digest = _fingerprint(rows)
    target = _work(run_id)
    if resume:
        if not target.is_dir() or target.is_symlink():
            raise ValueError("No existing run to resume")
        state = _state(target)
        frozen = _read_jsonl(target / "input_manifest.jsonl")
        if (frozen != rows or state.get("input_sha256") != digest or
            state.get("schema_version") != SCHEMA or state.get("profile") != profile or
            state.get("device_requested") != device or
            state.get("min_score") != min_score or state.get("top_ratio") != top_ratio or
            state.get("run_id") != run_id or state.get("model_id") != pilot.MODEL_IDS["ast"] or
            state.get("samples") != len(rows)):
            raise ValueError("Input inventory or configuration changed; resume blocked")
    else:
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Run already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        _write_new(target / "input_manifest.jsonl",
                   "".join(_json(row) + "\n" for row in rows))
        _write_new(target / "style_predictions.jsonl", "")
        _write_new(target / "failures.jsonl", "")
        state = {
            "schema_version": SCHEMA, "run_id": run_id, "profile": profile,
            "input_sha256": digest, "model_id": pilot.MODEL_IDS["ast"],
            "model_revision": None, "device_requested": device,
            "min_score": min_score, "top_ratio": top_ratio,
            "expected_count": expected_count, "samples": len(rows),
            "started_at": _now(), "status": "created",
            "disclaimer": "AST scores are unverified category suggestions; no speaker/QC/style approval",
        }
        _atomic_json(target / "run_manifest.json", state)
    results_path = target / "style_predictions.jsonl"
    recorded = _audit(rows, _read_jsonl(results_path))
    pending = [row for row in rows if row["source_id"] not in recorded]
    if max_new is not None:
        pending = pending[:max_new]
    if not pending:
        state.update({"processed": len(recorded), "remaining": len(rows) - len(recorded),
                      "status": "completed_with_errors" if any(r["status"] == "error"
                         for r in recorded.values()) else
                      ("completed" if len(recorded) == len(rows) else "partial"),
                      "updated_at": _now()})
        _atomic_json(target / "run_manifest.json", state)
        print(f"AST BATCH {state['status']}: {len(recorded)}/{len(rows)}")
        return 0 if len(recorded) == len(rows) else 3
    state["status"] = "loading_model"
    _atomic_json(target / "run_manifest.json", state)
    try:
        pipeline = _load_ast(device)
        state["model_revision"] = getattr(getattr(pipeline.model, "config", None),
                                          "_commit_hash", None)
        sample_rate = int(pipeline.feature_extractor.sampling_rate)
        state["sampling_rate"] = sample_rate
        state["status"] = "running"
        _atomic_json(target / "run_manifest.json", state)
    except Exception as exc:
        state.update({"status": "failed_initialization",
                      "initialization_error": f"{type(exc).__name__}: {exc}",
                      "updated_at": _now()})
        pilot._jsonl(target / "failures.jsonl",
                           {"failure_kind": type(exc).__name__, "reason": str(exc)})
        _atomic_json(target / "run_manifest.json", state)
        print(f"AST BATCH MODEL FAILED: {exc}", file=sys.stderr)
        return 1
    for row in pending:
        result = {**row, "schema_version": SCHEMA, "run_id": run_id,
                  "model_id": state["model_id"], "model_revision": state["model_revision"],
                  "start_ms": 0, "end_ms": None, "status": "unknown",
                  "category_dir": UNKNOWN_DIR, "category_code": None,
                  "category_score": None, "category_scores": None,
                  "review_status": "unverified", "human_approved": False}
        try:
            source = Path(row["source_path"])
            if _sha256(source) != row["source_sha256"]:
                raise ValueError("Source SHA mismatch before inference")
            audio, silent = pilot._audio(source, sample_rate)
            result["digital_silence"] = silent
            result["end_ms"] = round(len(audio) * 1000 / sample_rate)
            if silent:
                result["reason"] = "digital_silence"
            else:
                raw = pipeline(audio, top_k=527)
                directory, code, score, scores, raw_top = _candidate(raw, min_score, top_ratio)
                result.update({"category_dir": directory, "category_code": code,
                               "category_score": score, "category_scores": scores,
                               "raw_top_audio_labels": raw_top,
                               "status": "unknown" if code is None else "scored",
                               "reason": "weak_or_unmapped_ast_signal" if code is None else None})
            if _sha256(source) != row["source_sha256"]:
                raise ValueError("Source SHA mismatch after inference")
        except Exception as exc:
            result.update({"status": "error", "category_dir": UNKNOWN_DIR,
                           "category_code": None, "category_score": None,
                           "failure_reason": f"{type(exc).__name__}: {exc}"})
            pilot._jsonl(target / "failures.jsonl",
                               {"source_id": row["source_id"], "failure_reason": result["failure_reason"]})
        pilot._jsonl(results_path, result)  # durable per-item resume point
        recorded[row["source_id"]] = result
        state.update({"processed": len(recorded), "remaining": len(rows) - len(recorded),
                      "updated_at": _now(), "status": "running"})
        # Each JSONL row is already a closed, durable resume marker. Replacing
        # the Windows state file for every WAV increases lock contention; the
        # summary/resume paths reconcile counts from the JSONL evidence.
        if len(recorded) % 25 == 0 or len(recorded) == len(rows):
            _atomic_json(target / "run_manifest.json", state)
            print(f"AST BATCH: {len(recorded)}/{len(rows)} "
                  f"unknown={sum(r['status'] == 'unknown' for r in recorded.values())} "
                  f"errors={sum(r['status'] == 'error' for r in recorded.values())}",
                  flush=True)
    complete = len(recorded) == len(rows)
    state.update({"status": ("completed_with_errors" if any(r["status"] == "error"
                        for r in recorded.values()) else "completed") if complete else "partial",
                  "processed": len(recorded), "remaining": len(rows) - len(recorded),
                  "updated_at": _now()})
    _atomic_json(target / "run_manifest.json", state)
    print(f"AST BATCH {state['status']}: {len(recorded)}/{len(rows)}; "
          f"run_id={run_id}; sources unchanged")
    return 0 if complete else 3


def summary(run_id: str) -> dict[str, Any]:
    target = _work(run_id)
    state = _state(target)
    rows = _read_jsonl(target / "input_manifest.jsonl")
    result = _audit(rows, _read_jsonl(target / "style_predictions.jsonl"))
    counts = Counter(x["category_dir"] for x in result.values())
    return {"run_id": run_id, "status": state["status"], "total": len(rows),
            "processed": len(result), "remaining": len(rows) - len(result),
            "groups": dict(sorted(counts.items()))}


def _source_guard(row: dict[str, Any], profile: str) -> Path:
    root = _root(profile)
    source = Path(row["source_path"])
    if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(root):
        raise ValueError(f"Unsafe source path at export: {source}")
    if source.resolve().relative_to(root).as_posix() != row["relative_path"]:
        raise ValueError(f"Input relative path mismatch: {source}")
    if _sha256(source) != row["source_sha256"]:
        raise ValueError(f"Source changed before export: {source}")
    return source


def export(run_id: str, version: str, resume: bool) -> Path:
    if not SAFE_NAME.fullmatch(version):
        raise ValueError("Unsafe --version")
    target = _work(run_id)
    state = _state(target)
    if state.get("status") not in ("completed", "completed_with_errors"):
        raise ValueError("Batch not complete. Resume classification before export.")
    profile = state["profile"]
    inventory = _read_jsonl(target / "input_manifest.jsonl")
    results = _audit(inventory, _read_jsonl(target / "style_predictions.jsonl"))
    if len(results) != len(inventory) or _fingerprint(inventory) != state["input_sha256"]:
        raise ValueError("Incomplete / inconsistent batch evidence")
    # Preflight every original BEFORE allocating a publication directory.
    for row in inventory:
        _source_guard(row, profile)
    parent = PROJECT_ROOT / "outputs" / "candidates" / profile
    final = parent / version
    stage = parent / f".building-{version}-{run_id}"
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Immutable candidate library already exists: {final}")
    if resume:
        if not stage.is_dir() or stage.is_symlink():
            raise ValueError("No matching staging export to resume")
    else:
        if stage.exists() or stage.is_symlink():
            raise FileExistsError(f"Staging export exists: {stage}; use --resume-export")
        parent.mkdir(parents=True, exist_ok=True)
        required = sum(row["size_bytes"] for row in inventory)
        available = shutil.disk_usage(parent).free
        if available < required + (32 * 1024 * 1024):
            raise OSError(f"Not enough disk space: need {required} bytes + overhead; free={available}")
        stage.mkdir(exist_ok=False)
    counts: Counter[str] = Counter()
    index_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for row in inventory:
        prediction = results[row["source_id"]]
        group = prediction["category_dir"]
        name = f"{row['source_id']}_{row['source_sha256'][:8]}.wav"
        relative = (Path("audio") / group / name)
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(row["source_path"])
        if destination.exists() or destination.is_symlink():
            if not resume or not destination.is_file() or _sha256(destination) != row["source_sha256"]:
                raise ValueError(f"Unverified existing staged file: {destination}")
        else:
            temporary = destination.with_name(destination.name + ".copying")
            if temporary.exists() or temporary.is_symlink():
                # Only a known, same-run incomplete temp copy may be discarded.
                if not resume or not temporary.is_file():
                    raise ValueError(f"Unexpected incomplete stage file: {temporary}")
                temporary.unlink()
            with source.open("rb") as origin, temporary.open("xb") as dest:
                shutil.copyfileobj(origin, dest, length=1024 * 1024)
            if _sha256(temporary) != row["source_sha256"]:
                raise ValueError(f"Staged WAV hash mismatch: {row['source_id']}")
            temporary.rename(destination)
        counts[group] += 1
        entry = {**prediction, "output_path": relative.as_posix(),
                 "output_sha256": row["source_sha256"], "library_tier": "UNVERIFIED_AST_CANDIDATE"}
        manifests.append(entry)
        index_rows.append({
            "source_id": row["source_id"], "category": group,
            "file": relative.as_posix(), "source_file": row["relative_path"],
            "source_sha256": row["source_sha256"], "status": prediction["status"],
            "ast_category": prediction["category_code"] or "",
            "ast_score": prediction["category_score"] if prediction["category_score"] is not None else "",
            "review": "UNVERIFIED", "reason": prediction.get("failure_reason") or prediction.get("reason") or "",
        })
    _stage_text(stage / "manifest.jsonl",
                "".join(_json(x) + "\n" for x in manifests), resume=resume)
    import io

    index_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(index_buffer, fieldnames=list(index_rows[0]))
    writer.writeheader()
    writer.writerows([{key: _excel_safe(value) for key, value in row.items()}
                      for row in index_rows])
    _stage_text(stage / "index.csv", index_buffer.getvalue(), resume=resume,
                encoding="utf-8-sig")
    readme = (
        "ASTによる暫定分類ライブラリ（未確認）\n"
        "================================\n"
        "注意：分類は機械推定です。本人性・録音品質・カテゴリは未承認。\n"
        "Irodori-TTSに手動で参照WAVを指定する前に必ず試聴してください。\n"
        "『99_未分類_要確認』は、無音・判定不能・推論失敗を含みます。\n"
        "原本はコピーのみ。確認済み音声の正式ライブラリ outputs/datasets/ とは別です。\n"
        f"profile={profile} / version={version} / run_id={run_id} / source_count={len(inventory)}\n"
        f"AST最小スコア={state['min_score']} / 全体Topに対する比={state['top_ratio']}\n"
        f"分類別件数={_json(dict(sorted(counts.items())))}\n"
        "詳細・原本SHA・失敗理由は index.csv と manifest.jsonl を参照。\n"
    )
    _stage_text(stage / "README.txt", readme, resume=resume)
    _stage_text(stage / "summary.json", json.dumps({
        "schema_version": SCHEMA, "tier": "UNVERIFIED_AST_CANDIDATE",
        "profile": profile, "run_id": run_id, "version": version,
        "source_count": len(inventory), "counts": dict(sorted(counts.items())),
        "input_sha256": state["input_sha256"], "model_id": state["model_id"],
        "model_revision": state["model_revision"],
    }, ensure_ascii=False, indent=2) + "\n", resume=resume)
    stage.rename(final)  # atomic publication only after all WAVs + indexes exist
    print(f"EXPORTED UNVERIFIED candidates: {final} ({len(inventory)} WAVs; input files unchanged)")
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AST batch candidate grouping; sources read-only")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("inventory", help="SHA-check full source inventory; no output written")
    check.add_argument("--profile", required=True)
    check.add_argument("--expected-count", type=int)
    execute = sub.add_parser("run", help="Create or resume a per-WAV AST inference run")
    execute.add_argument("--profile", required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--expected-count", type=int)
    execute.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    execute.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    execute.add_argument("--top-ratio", type=float, default=DEFAULT_TOP_RATIO)
    execute.add_argument("--max-new", type=int, help="Only classify N additional WAVs this invocation")
    execute.add_argument("--resume", action="store_true")
    show = sub.add_parser("summary", help="Print grouped input/result counts")
    show.add_argument("--run-id", required=True)
    promote = sub.add_parser("export", help="Publish a *separate unverified* browse library")
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--version", required=True)
    promote.add_argument("--resume-export", action="store_true")
    review = sub.add_parser("apply-review", help="Apply browser review CSV to a new candidate version")
    review.add_argument("--run-id", required=True)
    review.add_argument("--review-csv", required=True, type=Path)
    review.add_argument("--source-version", default="v1")
    review.add_argument("--version", default="v2")
    review.add_argument("--dry-run", action="store_true")
    review.add_argument("--resume-export", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            inventory(args.profile, args.expected_count)
        elif args.command == "run":
            return run(args.profile, args.run_id, args.expected_count, args.device,
                       args.min_score, args.top_ratio, args.max_new, args.resume)
        elif args.command == "summary":
            print(json.dumps(summary(args.run_id), ensure_ascii=False, indent=2))
        elif args.command == "apply-review":
            from .review_export import apply_review

            report = apply_review(args.run_id, args.review_csv, args.source_version,
                                  args.version, args.dry_run, args.resume_export)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not args.dry_run:
                print(f"REVIEW CANDIDATE EXPORTED: {args.version}; originals unchanged")
        else:
            export(args.run_id, args.version, args.resume_export)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"AST BATCH BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
