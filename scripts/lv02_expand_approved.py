#!/usr/bin/env python3
r"""LV-02 expansion: audit auto-candidates, then export only explicitly reviewed clips.

No Codex or new dependencies. From F:\AIProjects\LocalVoice:
  .\.venv\Scripts\python.exe .\scripts\lv02_expand_approved.py audit
  .\.venv\Scripts\python.exe .\scripts\lv02_expand_approved.py export --reviewed PATH

The audit NEVER changes review.csv, automatically approves a speaker, or
promotes ASR suggestions. Export writes a *new* versioned train CSV and
retains the original approved eight, independent evaluation three and
all source files unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "Irodori-TTS" / "outputs" / "localvoice_lora_dataset"
SPEAKER = "Ui_Shigure"
EXPRESSIVE = {"whisper", "breath", "panting", "groan", "laugh", "emotion"}
STYLES = EXPRESSIVE | {"normal", "other"}
REVIEW_COLS = (
    "clip_id", "sha256", "source_video", "suggested_text", "suggested_caption",
    "suggested_style", "source_kind", "text_source", "style_source",
    "action", "speaker_ok", "quality", "text_verified", "text",
    "caption", "style", "evidence",
)
TRAIN_COLS = ("audio", "text", "caption", "speaker", "clip_id", "sha256")


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: tuple[str, ...], items: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index(items: list[dict], column: str, name: str) -> dict[str, dict]:
    result = {}
    for item in items:
        key = item.get(column, "").strip()
        if not key or key in result:
            raise ValueError(f"Missing/duplicate {name} {column}: {key!r}")
        result[key] = item
    return result


def video(row: dict) -> str:
    name = Path(row["source_path"]).name
    if "__" not in name:
        raise ValueError(f"Source video unavailable from filename: {name}")
    return name.split("__", 1)[0]


def identity(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def load_authority(ws: Path) -> dict:
    inv_list = load_csv(ws / "inventory.csv")
    reviews = load_csv(ws / "review.csv")
    auto_list = load_csv(ws / "auto_preparation_report.csv")
    old_train = load_csv(ws / "dataset_for_prepare_manifest_approved_train.csv")
    holdout = load_csv(ws / "lv02_approved_evaluation.csv")
    if (len(inv_list), len(reviews), len(auto_list)) != (1167, 1167, 1167):
        raise ValueError("LV-02 source tables must each contain exactly 1,167 rows")
    if (len(old_train), len(holdout)) != (8, 3):
        raise ValueError("Existing LV-02 freeze must be train8/eval3")
    inventory = index(inv_list, "clip_id", "inventory")
    review = index(reviews, "clip_id", "review")
    auto = index(auto_list, "clip_id", "auto report")
    if set(inventory) != set(review) or set(inventory) != set(auto):
        raise ValueError("LV-02 table clip_id sets differ")
    source_paths = {identity(r["source_path"]): r for r in inv_list}
    if len(source_paths) != len(inv_list):
        raise ValueError("Duplicate source paths in inventory")
    for cid, row in inventory.items():
        for other in (review[cid], auto[cid]):
            if (other.get("sha256") != row.get("sha256")
                    or identity(other["source_path"]) != identity(row["source_path"])):
                raise ValueError(f"Stale review/auto file identity: {cid}")
    train_rows, train_ids = [], set()
    eval_ids, eval_videos = set(), set()
    for label, group in (("train", old_train), ("evaluation", holdout)):
        for row in group:
            source = row.get("audio") or row.get("source_path")
            inv = source_paths.get(identity(source))
            if inv is None:
                raise ValueError(f"{label} CSV source absent from inventory: {source}")
            item_id = inv["clip_id"]
            if item_id in train_ids | eval_ids:
                raise ValueError("Existing train/evaluation clip overlap")
            # Verify original frozen WAVs even in audit-only mode.
            path = Path(inv["source_path"]).resolve()
            if not path.is_file() or sha256(path) != inv["sha256"]:
                raise ValueError(f"{label} source missing/changed: {item_id}")
            if row.get("sha256") and row["sha256"] != inv["sha256"]:
                raise ValueError(f"{label} CSV SHA mismatch: {item_id}")
            if label == "train":
                train_ids.add(item_id)
                if not str(row.get("text", "")).strip():
                    raise ValueError("Frozen training text missing")
                train_rows.append({
                    "audio": str(path), "text": row["text"].strip(),
                    "caption": str(row.get("caption", "")).strip(),
                    "speaker": SPEAKER, "clip_id": item_id,
                    "sha256": inv["sha256"],
                })
            else:
                eval_ids.add(item_id)
                eval_videos.add(video(inv))
    if {video(inventory[cid]) for cid in train_ids} & eval_videos:
        raise ValueError("Existing train/eval share video provenance")
    return {
        "inventory": inventory, "review": review, "auto": auto,
        "train_rows": train_rows, "train_ids": train_ids,
        "eval_ids": eval_ids, "eval_videos": eval_videos,
        "holdout": holdout,
    }


def classify(state: dict) -> tuple[list[dict], dict]:
    rows, counts, videos = [], Counter(), defaultdict(Counter)
    for cid, original in sorted(state["inventory"].items()):
        reviewed = state["review"][cid]
        prepared = state["auto"][cid]
        kind = str(prepared.get("route", ""))
        speaker = str(reviewed.get("speaker_ok", "")).lower()
        quality = str(reviewed.get("quality", "")).lower()
        text = str(reviewed.get("text", "")).strip()
        if cid in state["train_ids"]:
            tier = "EXISTING_TRAIN"
        elif cid in state["eval_ids"]:
            tier = "EXISTING_EVALUATION"
        elif video(original) in state["eval_videos"]:
            tier = "HOLDOUT_VIDEO_EXCLUDED"
        elif (reviewed.get("decision") == "approved"
              and speaker == "yes" and quality == "good" and text):
            tier = "HUMAN_APPROVED_NOT_EXPORTED"
        elif (kind == "training_candidate" and prepared.get("text")
              and str(prepared.get("source_kind", "")) in
              {"my_voice", "review_approved", "review_emotion"}):
            tier = "REVIEW_REQUIRED"
        elif kind == "training_candidate":
            tier = "CANDIDATE_INSUFFICIENT_EVIDENCE"
        else:
            tier = "OTHER_OR_HOLD"
        counts[tier] += 1
        videos[video(original)][tier] += 1
        rows.append({
            "clip_id": cid, "sha256": original["sha256"],
            "source_video": video(original), "tier": tier,
            "source_kind": prepared.get("source_kind", ""),
            "route": kind,
            "route_reason": prepared.get("reason", ""),
            "decision": reviewed.get("decision", ""),
            "speaker_ok": reviewed.get("speaker_ok", ""),
            "quality": reviewed.get("quality", ""),
            "text_source": prepared.get("text_source", ""),
            "style_source": prepared.get("style_source", ""),
            "text_present": str(bool(prepared.get("text", ""))).lower(),
            "candidate_style": prepared.get("candidate_style", ""),
        })
    return rows, {"tiers": dict(counts),
                  "by_source_video": {key: dict(v) for key, v in sorted(videos.items())}}


def new_dir(ws: Path) -> Path:
    for number in range(1, 1000):
        folder = ws / f"lv02_expansion_{number:03d}"
        if not folder.exists():
            folder.mkdir(parents=False, exist_ok=False)
            return folder
    raise RuntimeError("No free LV-02 expansion attempt folder")


def audit(ws: Path, queue_limit: int = 48) -> Path:
    if queue_limit < 1:
        raise ValueError("queue_limit must be positive")
    state = load_authority(ws)
    rows, summary = classify(state)
    dest = new_dir(ws)
    with (dest / "classification.csv").open("w", newline="", encoding="utf-8-sig") as f:
        columns = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # Template is intentionally NOT marked approved. All review decisions blank.
    review = []
    for r in rows:
        if r["tier"] not in {"REVIEW_REQUIRED", "HUMAN_APPROVED_NOT_EXPORTED"}:
            continue
        auto = state["auto"][r["clip_id"]]
        review.append({
            "clip_id": r["clip_id"], "sha256": r["sha256"],
            "source_video": r["source_video"],
            "suggested_text": auto.get("text", ""),
            "suggested_caption": auto.get("caption", ""),
            "suggested_style": auto.get("candidate_style", ""),
            "source_kind": r["source_kind"],
            "text_source": r["text_source"], "style_source": r["style_source"],
            "action": "", "speaker_ok": "", "quality": "",
            "text_verified": "", "text": "", "caption": "", "style": "",
            "evidence": "",
        })
    # Stream-balanced review order, not first-video-only.
    batches = defaultdict(list)
    for row in review:
        batches[row["source_video"]].append(row)
    balanced = []
    while any(batches.values()):
        for key in sorted(batches):
            if batches[key]:
                balanced.append(batches[key].pop(0))
    available = len(balanced)
    balanced = balanced[:queue_limit]
    write_csv(dest / "review_queue.csv", REVIEW_COLS, balanced)
    result = {
        "status": "AUDIT_COMPLETE_NO_AUTO_APPROVAL",
        "input_inventory": 1167, "historical_train": 8,
        "historical_external_eval": 3,
        "review_queue": len(balanced),
        "review_queue_available": available,
        "review_queue_limit": queue_limit,
        "review_queue_order": "balanced round-robin across source video",
        "review_tier_counts": summary["tiers"],
        "by_source_video": summary["by_source_video"],
        "reused_frozen_evaluation": str(ws / "lv02_approved_evaluation.csv"),
        "review_queue_path": str(dest / "review_queue.csv"),
        "note": ("ASR suggestions, source_kind, and auto training_ready are "
                 "not confirmation of speaker identity/transcription. "
                 "A separate explicit review decision is required for export."),
    }
    (dest / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"], "output": str(dest),
        "tiers": summary["tiers"], "review_queue": len(balanced)
    }, ensure_ascii=False, indent=2))
    return dest


def export(ws: Path, reviewed_path: Path) -> Path:
    state = load_authority(ws)
    records = load_csv(reviewed_path.resolve())
    ids = index(records, "clip_id", "review decisions")
    train = list(state["train_rows"])
    added = []
    originals = {row["clip_id"] for row in train}
    source_root = (ROOT / "data" / "training_audio" / SPEAKER / "audio").resolve()
    for cid, row in sorted(ids.items()):
        if str(row.get("action", "")).strip().lower() != "approve":
            continue
        original = state["inventory"].get(cid)
        if original is None or cid in state["eval_ids"] or cid in originals:
            raise ValueError(f"Cannot newly approve unknown/protected/frozen clip: {cid}")
        if video(original) in state["eval_videos"]:
            raise ValueError(f"Video-level holdout leak for clip: {cid}")
        current = state["review"][cid]
        prepared = state["auto"][cid]
        if prepared.get("route") != "training_candidate" and not (
            current.get("decision") == "approved"
        ):
            raise ValueError(f"Not an eligible training candidate: {cid}")
        if row.get("sha256") != original["sha256"]:
            raise ValueError(f"Source fingerprint disagreement: {cid}")
        if (row.get("speaker_ok", "").strip().lower() != "yes"
                or row.get("quality", "").strip().lower() != "good"
                or row.get("text_verified", "").strip().lower() != "yes"
                or not row.get("evidence", "").strip()):
            raise ValueError(f"Missing explicit speaker/audio/transcript review: {cid}")
        style = str(row.get("style", "")).strip().lower()
        text = str(row.get("text", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if style not in STYLES or not text or (style in EXPRESSIVE and not caption):
            raise ValueError(f"Invalid confirmed text/style/caption: {cid}")
        original_path = Path(original["source_path"]).resolve()
        if (source_root not in original_path.parents or not original_path.is_file()
                or sha256(original_path) != original["sha256"]):
            raise ValueError(f"Source path/SHA changed: {cid}")
        added.append({
            "audio": str(original_path), "text": text,
            "caption": caption if style in EXPRESSIVE else "",
            "speaker": SPEAKER, "clip_id": cid,
            "sha256": original["sha256"],
        })
    if not added:
        raise ValueError("No newly confirmed clips: no new version will be written")
    train.extend(added)
    source_shas = [item["sha256"] for item in train]
    if len(source_shas) != len(set(source_shas)):
        raise ValueError("Duplicate SHA in proposed training version")
    train_videos = {video(state["inventory"][row["clip_id"]]) for row in train}
    if train_videos & state["eval_videos"]:
        raise ValueError("External evaluation shares source video with training")
    dest = new_dir(ws)
    write_csv(dest / "dataset_for_prepare_manifest_approved_train.csv",
              TRAIN_COLS, train)
    report = {
        "status": "EXPLICITLY_REVIEWED_TRAIN_VERSION_CREATED",
        "train_total": len(train),
        "historical_train_retained": len(state["train_rows"]),
        "new_approved": len(added),
        "external_evaluation_untouched": len(state["holdout"]),
        "source_videos": sorted(train_videos),
        "decision_file_sha256": sha256(reviewed_path.resolve()),
        "train_csv_sha256": sha256(dest / "dataset_for_prepare_manifest_approved_train.csv"),
        "train_file": str(dest / "dataset_for_prepare_manifest_approved_train.csv"),
        "external_eval_file": str(ws / "lv02_approved_evaluation.csv"),
        "new_clip_ids": [x["clip_id"] for x in added],
        "note": "No existing artifact, review sheet, or source WAV was modified.",
    }
    (dest / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "export"))
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--reviewed", type=Path,
                        help="CSV copied from audit's review_queue.csv; explicit decisions only")
    parser.add_argument("--queue-limit", type=int, default=48,
                        help="Limit a balanced first review wave; full 1167-row audit remains")
    args = parser.parse_args()
    ws = args.workspace.resolve()
    if args.action == "audit":
        audit(ws, queue_limit=args.queue_limit)
    elif args.reviewed is None:
        raise ValueError("export requires --reviewed CSV")
    else:
        export(ws, args.reviewed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"LV-02 EXPANSION BLOCKED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        raise SystemExit(2)
