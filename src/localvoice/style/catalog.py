"""Metadata-only version history for LocalVoice AST classification and browser reviews.

The raw training WAV and the completed AST run are durable dependencies.
Version metadata is immutable; WAV folder materialization is opt-in.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from . import batch, review_export

SCHEMA = "localvoice.audio-metadata-version.v1"
CATEGORIES = review_export.CATEGORIES | {
    review_export.CAUTION, review_export.HELD, review_export.REJECTED
}
INDEX_COLUMNS = [
    "source_id", "category_dir", "effective_category", "ast_category",
    "output_path", "relative_path", "source_sha256", "review_status",
    "review_decision", "review_identity", "review_quality", "manual_category",
    "review_reason", "review_memo", "eligible_for_later_promotion",
]


def _sha_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _root(profile: str) -> Path:
    if not batch.SAFE_NAME.fullmatch(profile):
        raise ValueError("Unsafe profile name")
    return batch.PROJECT_ROOT / "outputs" / "metadata" / profile


def _version_dir(profile: str, version: str) -> Path:
    if not batch.SAFE_NAME.fullmatch(version):
        raise ValueError("Unsafe metadata version")
    return _root(profile) / "versions" / version


def _load_run(run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    state = batch._state(batch._work(run_id))
    if state.get("run_id") != run_id or state.get("status") not in (
        "completed", "completed_with_errors"
    ):
        raise ValueError("AST run must be completed")
    originals = batch._read_jsonl(batch._work(run_id) / "input_manifest.jsonl")
    predictions = batch._audit(
        originals, batch._read_jsonl(batch._work(run_id) / "style_predictions.jsonl")
    )
    if (len(originals) != len(predictions) or
        state.get("samples") != len(originals) or
        state.get("input_sha256") != batch._fingerprint(originals)):
        raise ValueError("AST source manifest fingerprint/count mismatch")
    return state, originals, predictions


def _validate_entries(
    entries: list[dict[str, Any]], originals: list[dict[str, Any]], *,
    profile: str, check_source: bool = True,
) -> None:
    expected = {r["source_id"]: r for r in originals}
    if len(expected) != len(originals) or len(entries) != len(originals):
        raise ValueError("Incomplete/duplicate versioned WAV entries")
    seen: set[str] = set()
    for entry in entries:
        sid = entry.get("source_id")
        if sid not in expected or sid in seen:
            raise ValueError(f"Unknown/duplicate version entry: {sid}")
        seen.add(sid)
        row = expected[sid]
        if any(entry.get(k) != row[k] for k in (
            "source_path", "source_sha256", "relative_path"
        )):
            raise ValueError(f"Version source provenance mismatch: {sid}")
        category = entry.get("category_dir")
        if category not in CATEGORIES or entry.get("human_approved", False):
            raise ValueError(f"Unsafe category/approval in version: {sid}")
        correct = f"audio/{category}/{sid}_{row['source_sha256'][:8]}.wav"
        if entry.get("output_path") != correct:
            raise ValueError(f"Invalid version output path: {sid}")
        if check_source:
            batch._source_guard(row, profile)


def _bytes_optional(path: Path | None, *, json_format: bool = False) -> bytes | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Invalid or linked reviewer input: {path}")
    raw = path.read_bytes()
    if json_format:
        value = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Review JSON must be an object")
    return raw


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _serialize(entries: list[dict[str, Any]]) -> bytes:
    return ("".join(batch._json(row) + "\n" for row in entries)).encode("utf-8")


def _index(entries: list[dict[str, Any]]) -> bytes:
    rows = []
    for entry in entries:
        item = {k: entry.get(k, "") for k in INDEX_COLUMNS}
        item["effective_category"] = entry.get("effective_category", entry["category_dir"])
        item["ast_category"] = entry.get("ast_category", entry["category_dir"])
        rows.append(item)
    return ("\ufeff" + review_export._csv_text(rows, INDEX_COLUMNS)).encode("utf-8")


def _summary(entries: list[dict[str, Any]], state: dict[str, Any], *,
             version: str, parent: str | None, source_library: str | None) -> dict[str, Any]:
    statuses = Counter(e.get("review_status", "unreviewed") for e in entries)
    return {
        "schema_version": SCHEMA, "profile": state["profile"], "run_id": state["run_id"],
        "version": version, "parent_version": parent, "source_library": source_library,
        "source_count": len(entries), "input_sha256": state["input_sha256"],
        "tier": "METADATA_ONLY_UNVERIFIED_CANDIDATE",
        "categories": dict(sorted(Counter(e["category_dir"] for e in entries).items())),
        "review_statuses": dict(sorted(statuses.items())),
        "audio_files_stored": 0,
    }


def _publish(
    entries: list[dict[str, Any]], state: dict[str, Any], *,
    version: str, parent: str | None,
    source_library: str | None, review_csv: bytes | None, review_json: bytes | None,
    dry_run: bool,
) -> dict[str, Any]:
    profile = state["profile"]
    destination = _version_dir(profile, version)
    parent_dir = destination.parent
    stage = parent_dir / f".building-{version}"
    _validate_entries(entries, batch._read_jsonl(
        batch._work(state["run_id"]) / "input_manifest.jsonl"), profile=profile)
    summary = _summary(entries, state, version=version,
                       parent=parent, source_library=source_library)
    manifest = _serialize(entries)
    index = _index(entries)
    summary_blob = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    report = {
        "version": version, "profile": profile, "wav_referenced": len(entries),
        "wav_copies_created": 0, "categories": summary["categories"],
        "review_statuses": summary["review_statuses"],
        "manifest_sha256": _sha_bytes(manifest),
        "review_csv_sha256": _sha_bytes(review_csv) if review_csv is not None else None,
        "review_json_sha256": _sha_bytes(review_json) if review_json is not None else None,
    }
    if dry_run:
        return report
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Immutable metadata version already exists: {destination}")
    if stage.exists() or stage.is_symlink():
        raise FileExistsError(f"Pending metadata version already exists: {stage}")
    if parent_dir.is_symlink() or _root(profile).is_symlink():
        raise ValueError("Linked metadata parent")
    parent_dir.mkdir(parents=True, exist_ok=True)
    stage.mkdir(exist_ok=False)
    receipt = {
        "schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": state["run_id"], "profile": profile, "version": version,
        "parent_version": parent, "source_library": source_library,
        "input_sha256": state["input_sha256"],
        "manifest_sha256": _sha_bytes(manifest),
        "index_sha256": _sha_bytes(index),
        "summary_sha256": _sha_bytes(summary_blob),
        "review_csv_sha256": report["review_csv_sha256"],
        "review_json_sha256": report["review_json_sha256"],
    }
    _write_file(stage / "manifest.jsonl", manifest)
    _write_file(stage / "index.csv", index)
    _write_file(stage / "summary.json", summary_blob)
    _write_file(stage / "receipt.json", (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if review_csv is not None:
        _write_file(stage / "review" / "review_decisions.csv", review_csv)
    if review_json is not None:
        _write_file(stage / "review" / "review_state.json", review_json)
    stage.rename(destination)
    inbox = _root(profile) / "inbox"
    if not inbox.exists() and not inbox.is_symlink():
        inbox.mkdir()
        _write_file(inbox / "README.txt", (
            "LocalVoice review CSV / JSON inbox\n"
            "HTMLからダウンロードしたreview_decisions.csvとreview_state.jsonをここに置けます。\n"
            "各版へ適用すると versions/<version>/review/ にSHA付きの記録として保存します。\n"
            "このフォルダ内のファイルを変更しても作成済みバージョンは変化しません。\n"
        ).encode("utf-8"))
    report["path"] = str(destination)
    return report


def archive(run_id: str, version: str, *, review_csv: Path | None = None,
            review_json: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Archive an existing v1/v2 materialized release as a metadata-only version."""
    state, originals, predictions = _load_run(run_id)
    profile = state["profile"]
    existing = batch.PROJECT_ROOT / "outputs" / "candidates" / profile / version
    if existing.is_symlink() or not existing.is_dir():
        raise ValueError(f"Existing candidate release not found: {existing}")
    src_summary = json.loads((existing / "summary.json").read_text(encoding="utf-8"))
    if (src_summary.get("profile") != profile or
        src_summary.get("run_id") != run_id or
        src_summary.get("version") != version or
        src_summary.get("input_sha256") != state["input_sha256"] or
        src_summary.get("source_count") != len(originals)):
        raise ValueError("Existing candidate release provenance differs from completed AST run")
    entries = batch._read_jsonl(existing / "manifest.jsonl")
    _validate_entries(entries, originals, profile=profile)
    by_id = {x["source_id"]: x for x in entries}
    for sid, record in by_id.items():
        if record.get("ast_category", record["category_dir"]) != predictions[sid]["category_dir"]:
            raise ValueError(f"Existing release baseline AST mismatch: {sid}")
        review_export._safe_audio_file(existing, record)
    with (existing / "index.csv").open(encoding="utf-8-sig", newline="") as f:
        index_ids = [r.get("source_id") for r in csv.DictReader(f)]
    if len(index_ids) != len(originals) or set(index_ids) != set(by_id) or len(set(index_ids)) != len(originals):
        raise ValueError("Existing candidate CSV index does not match manifest")
    csv_bytes = _bytes_optional(review_csv)
    json_bytes = _bytes_optional(review_json, json_format=True)
    if src_summary.get("tier") == "UNVERIFIED_REVIEWED_CANDIDATE":
        if review_csv is None:
            raise ValueError("Reviewed release needs its original reviewer CSV")
        if _sha_bytes(csv_bytes) != src_summary.get("review_csv_sha256"):
            raise ValueError("Reviewed release CSV SHA differs from version evidence")
        review_export._read_csv(review_csv, {
            sid: {"source_sha256": row["source_sha256"],
                  "category_dir": predictions[sid]["category_dir"]}
            for sid, row in by_id.items()
        })
    elif src_summary.get("tier") != "UNVERIFIED_AST_CANDIDATE":
        raise ValueError("Unsupported source library tier")
    elif review_csv is not None:
        raise ValueError("AST baseline version cannot take a review CSV")
    return _publish(
        entries, state, version=version, parent=None if version == "v1" else "v1",
        source_library=f"outputs/candidates/{profile}/{version}",
        review_csv=csv_bytes, review_json=json_bytes, dry_run=dry_run,
    )


def _verify(profile: str, version: str, *, check_audio: bool = True
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = _version_dir(profile, version)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Metadata version not found: {root}")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != SCHEMA or
        receipt.get("profile") != profile or receipt.get("version") != version or
        summary.get("schema_version") != SCHEMA or summary.get("version") != version or
        summary.get("profile") != profile or
        summary.get("tier") != "METADATA_ONLY_UNVERIFIED_CANDIDATE"):
        raise ValueError("Invalid version provenance or schema")
    checks = {
        "manifest.jsonl": receipt.get("manifest_sha256"),
        "index.csv": receipt.get("index_sha256"),
        "summary.json": receipt.get("summary_sha256"),
        "review/review_decisions.csv": receipt.get("review_csv_sha256"),
        "review/review_state.json": receipt.get("review_json_sha256"),
    }
    for name, expected in checks.items():
        path = root / name
        if expected is None:
            if path.exists() or path.is_symlink():
                raise ValueError(f"Unexpected unsourced metadata file: {path}")
        elif path.is_symlink() or not path.is_file() or batch._sha256(path) != expected:
            raise ValueError(f"Metadata version checksum mismatch: {path}")
    state, original, predictions = _load_run(receipt["run_id"])
    if (state["profile"] != profile or
        state["input_sha256"] != receipt.get("input_sha256") or
        summary.get("input_sha256") != state["input_sha256"]):
        raise ValueError("Metadata version AST origin changed")
    entries = batch._read_jsonl(root / "manifest.jsonl")
    _validate_entries(entries, original, profile=profile, check_source=check_audio)
    if summary.get("source_count") != len(entries):
        raise ValueError("Metadata version source count mismatch")
    cats = dict(sorted(Counter(e["category_dir"] for e in entries).items()))
    if cats != summary.get("categories"):
        raise ValueError("Metadata version category counts inconsistent")
    statuses = dict(sorted(Counter(e.get("review_status", "unreviewed") for e in entries).items()))
    if statuses != summary.get("review_statuses"):
        raise ValueError("Metadata version reviewer counts inconsistent")
    for entry in entries:
        sid = entry["source_id"]
        if entry.get("ast_category", entry["category_dir"]) != predictions[sid]["category_dir"]:
            raise ValueError(f"Metadata AST baseline mismatch: {sid}")
    return receipt, entries


def verify(profile: str, version: str) -> dict[str, Any]:
    receipt, entries = _verify(profile, version)
    return {
        "status": "VERIFIED_METADATA_AND_SOURCE_WAV",
        "profile": profile, "version": version, "wav_referenced": len(entries),
        "wav_copies_created": 0,
        "manifest_sha256": receipt["manifest_sha256"],
    }


def revise(profile: str, version: str, parent_version: str, review_csv: Path,
           *, review_json: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Create a new version from the full AST baseline and a cumulative review CSV."""
    if version == parent_version:
        raise ValueError("New version must differ from parent")
    parent_receipt, previous_entries = _verify(profile, parent_version)
    _, baseline = _verify(profile, "v1")
    run_id = parent_receipt["run_id"]
    state, originals, predictions = _load_run(run_id)
    if state["profile"] != profile:
        raise ValueError("Metadata parent belongs to another profile")
    baseline_lookup = {x["source_id"]: x for x in baseline}
    if any(baseline_lookup[sid]["category_dir"] != p["category_dir"] for sid, p in predictions.items()):
        raise ValueError("v1 metadata is not the original AST baseline")
    reviews, _ = review_export._read_csv(review_csv, baseline_lookup)
    previously_reviewed = {
        e["source_id"] for e in previous_entries
        if e.get("review_decision") or e.get("manual_category") or
        e.get("review_memo") or e.get("review_identity", "unverified") != "unverified" or
        e.get("review_quality", "unverified") != "unverified"
    }
    missing = previously_reviewed - reviews.keys()
    if missing:
        raise ValueError(
            f"Review CSV is not cumulative: {len(missing)} earlier reviewer rows missing "
            f"(e.g. {sorted(missing)[0]}). Reload the previous HTML JSON before exporting"
        )
    raw_csv = _bytes_optional(review_csv)
    raw_json = _bytes_optional(review_json, json_format=True)
    entries = []
    for original in originals:
        sid = original["source_id"]
        base = baseline_lookup[sid]
        record = reviews.get(sid)
        group, status = review_export._outcome(base, record)
        effect = (record["manual_category"] or base["category_dir"]) if record else base["category_dir"]
        dest = f"audio/{group}/{sid}_{original['source_sha256'][:8]}.wav"
        entry = {
            **base, "ast_category": base["category_dir"], "effective_category": effect,
            "category_dir": group, "output_path": dest,
            "review_status": status,
            "review_decision": record["decision"] if record else "",
            "review_identity": record["identity"] if record else "unverified",
            "review_quality": record["quality"] if record else "unverified",
            "manual_category": record["manual_category"] if record else "",
            "review_reason": record["reject_reason"] if record else "",
            "review_memo": record["memo"] if record else "",
            "review_updated_at": record["updated_at"] if record else "",
            "eligible_for_later_promotion": status == "reviewed_usable",
            "library_tier": "UNVERIFIED_REVIEWED_CANDIDATE", "human_approved": False,
        }
        entries.append(entry)
    return _publish(
        entries, state, version=version, parent=parent_version, source_library=None,
        review_csv=raw_csv, review_json=raw_json, dry_run=dry_run,
    )


def materialize(profile: str, version: str, output_name: str, *,
                dry_run: bool = False, resume: bool = False) -> dict[str, Any]:
    if not batch.SAFE_NAME.fullmatch(output_name):
        raise ValueError("Unsafe materialization name")
    receipt, entries = _verify(profile, version)
    output_parent = batch.PROJECT_ROOT / "outputs" / "candidates" / profile
    destination = output_parent / output_name
    stage = output_parent / f".building-{output_name}-metadata-{version}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Existing materialized library will not be overwritten: {destination}")
    report = {
        "profile": profile, "version": version, "output_name": output_name,
        "wav_count": len(entries), "manifest_sha256": receipt["manifest_sha256"],
    }
    if dry_run:
        return report
    if not output_parent.is_dir() or output_parent.is_symlink():
        raise ValueError("Candidate library root missing or linked")
    if resume:
        if stage.is_symlink() or not stage.is_dir():
            raise ValueError("No incomplete version materialization to resume")
        stamp = json.loads((stage / "materialization.json").read_text(encoding="utf-8"))
        if stamp != report:
            raise ValueError("Materialization checkpoint does not match version")
    else:
        if stage.exists() or stage.is_symlink():
            raise FileExistsError("Incomplete materialization exists; use --resume")
        available = shutil.disk_usage(output_parent).free
        required = sum(int(e["size_bytes"]) for e in entries)
        if available < required + 32 * 1024 * 1024:
            raise OSError("Insufficient space to materialize the referenced WAVs")
        stage.mkdir(exist_ok=False)
        _write_file(stage / "materialization.json", (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    for entry in entries:
        source = batch._source_guard(entry, profile)
        parts = PurePosixPath(entry["output_path"]).parts
        target = stage.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.parent.is_symlink():
            raise ValueError(f"Unsafe materialization destination: {target}")
        if target.exists():
            if not resume or not target.is_file() or batch._sha256(target) != entry["source_sha256"]:
                raise ValueError(f"Invalid preexisting materialized WAV: {target}")
        else:
            pending = target.with_name(target.name + ".copying")
            if pending.exists() or pending.is_symlink():
                if not resume or not pending.is_file() or pending.is_symlink():
                    raise ValueError(f"Unexpected pending materialization: {pending}")
                pending.unlink()
            with source.open("rb") as input_file, pending.open("xb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            if batch._sha256(pending) != entry["source_sha256"]:
                raise ValueError(f"Materialized WAV checksum mismatch: {entry['source_id']}")
            pending.rename(target)
    version_root = _version_dir(profile, version)
    for filename in ("manifest.jsonl", "index.csv"):
        _write_file(stage / filename, (version_root / filename).read_bytes())
    result_summary = json.loads((version_root / "summary.json").read_text(encoding="utf-8"))
    result_summary.update({
        "tier": "UNVERIFIED_MATERIALIZED_CANDIDATE",
        "audio_files_stored": len(entries),
        "materialized_from_metadata_version": version,
    })
    _write_file(stage / "summary.json", (
        json.dumps(result_summary, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8"))
    _write_file(stage / "version_receipt.json", (version_root / "receipt.json").read_bytes())
    _write_file(stage / "README.txt", (
        "LocalVoice メタデータ履歴から再生成した音声候補フォルダ（未承認）\n"
        f"profile={profile} / version={version} / run={receipt['run_id']}\n"
        f"WAV count={len(entries)} / 元WAVをSHA256で照合してコピー\n"
        "原本・分類履歴は変更せず、正式なoutputs/datasetsへの昇格ではありません。\n"
        "要確認96_・保留97_・除外98_は使用対象ではありません。\n"
    ).encode("utf-8"))
    statuses = Counter(e.get("review_status", "unreviewed") for e in entries)
    if any(k in statuses for k in ("reviewed_usable", "needs_attention", "held", "rejected")):
        for filename, allowed in (
            ("reviewed_usable.csv", {"reviewed_usable"}),
            ("needs_attention.csv", {"needs_attention", "held", "rejected"}),
        ):
            selected = [e for e in entries if e.get("review_status") in allowed]
            rows = [{k: e.get(k, "") for k in INDEX_COLUMNS} for e in selected]
            _write_file(stage / filename, (
                "\ufeff" + review_export._csv_text(rows, INDEX_COLUMNS)
            ).encode("utf-8"))
    for filename in ("review_decisions.csv", "review_state.json"):
        stored = version_root / "review" / filename
        if stored.is_file() and not stored.is_symlink():
            _write_file(stage / "review" / filename, stored.read_bytes())
    stage.rename(destination)
    report["path"] = str(destination)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Versioned CSV/JSON metadata; WAVs stored only on request")
    subs = parser.add_subparsers(dest="command", required=True)
    arc = subs.add_parser("archive", help="Archive an existing v1/v2 release as metadata only")
    arc.add_argument("--run-id", required=True)
    arc.add_argument("--version", required=True)
    arc.add_argument("--review-csv", type=Path)
    arc.add_argument("--review-json", type=Path)
    arc.add_argument("--dry-run", action="store_true")
    rev = subs.add_parser("revise", help="New metadata-only version from cumulative reviewer CSV")
    rev.add_argument("--profile", required=True)
    rev.add_argument("--from-version", required=True)
    rev.add_argument("--version", required=True)
    rev.add_argument("--review-csv", required=True, type=Path)
    rev.add_argument("--review-json", type=Path)
    rev.add_argument("--dry-run", action="store_true")
    ver = subs.add_parser("verify", help="Verify stored metadata and all source WAV hashes")
    ver.add_argument("--profile", required=True)
    ver.add_argument("--version", required=True)
    mat = subs.add_parser("materialize", help="Restore a browsable WAV folder from versioned metadata")
    mat.add_argument("--profile", required=True)
    mat.add_argument("--version", required=True)
    mat.add_argument("--output-name", required=True)
    mat.add_argument("--dry-run", action="store_true")
    mat.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "archive":
            result = archive(args.run_id, args.version, review_csv=args.review_csv,
                             review_json=args.review_json, dry_run=args.dry_run)
        elif args.command == "verify":
            result = verify(args.profile, args.version)
        elif args.command == "revise":
            result = revise(args.profile, args.version, args.from_version,
                            args.review_csv, review_json=args.review_json, dry_run=args.dry_run)
        else:
            result = materialize(args.profile, args.version, args.output_name,
                                 dry_run=args.dry_run, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            UnicodeDecodeError, csv.Error) as exc:
        print(f"METADATA BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
