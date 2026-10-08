"""Automatically stage training candidates; human input only for ambiguous vocal style."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__:
    from .irodori_dataset import digest, normalize_style, paths, read_csv, review_origin, write_csv
else:
    from irodori_dataset import digest, normalize_style, paths, read_csv, review_origin, write_csv

# These are delivery descriptions, NOT transcriptions.
CAPTIONS = {
    "normal": "", "whisper": "囁くような小さな声で話している",
    "laugh": "笑いながら発声している",
    "breath": "息を吐く、吐息を伴う発声",
    "panting": "荒く息をしながら発声している",
    "groan": "うめくような声を出している",
    "emotion": "感情を強く込めて発声している", "other": "",
}
NONVERBAL = {"laugh", "breath", "panting", "groan"}
# README demonstrates these emojis with spoken words, not standalone.
# This is an experiment hint only, never a validated transcript.
EMOJI_HINTS = {"laugh": "🤭", "breath": "😮‍💨"}
DETAIL_COLUMNS = (
    "clip_id", "sha256", "source_path", "source_kind", "decision",
    "route", "reason", "candidate_style", "style_source", "text_source",
    "text", "caption", "training_ready",
)
REVIEW_COLUMNS = (
    "clip_id", "source_path", "duration_sec", "source_kind",
    "asr_status", "candidate_style", "top_events", "reason",
)
EXPERIMENT_COLUMNS = (
    "clip_id", "source_path", "sha256", "style", "style_source",
    "possible_emoji_text", "auto_caption", "reason",
)


def index(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        cid = row.get("clip_id", "")
        if not cid or cid in result:
            raise ValueError(f"Missing or duplicate {name} clip_id: {cid}")
        result[cid] = row
    return result


def sound_hint(hint: dict[str, str]) -> tuple[str, str]:
    if not hint:
        return "", "not_classified"
    if hint.get("status") != "suggested":
        return "", "classification_failed"
    value = hint.get("candidate_style", "")
    if value not in CAPTIONS and value != "unknown":
        return "", "invalid_style_prediction"
    if value == "unknown":
        return "", "model_abstained"
    if hint.get("uncertainty") != "unvalidated":
        return value, "style_prediction_uncertain"
    return value, "style_prediction_candidate"


def build(args: argparse.Namespace, source: Path, out: Path) -> dict[str, object]:
    """Non-interactive assembly; WAV and human judgments remain untouched."""
    reviews = read_csv(out / "review.csv")
    inv = index(read_csv(out / "inventory.csv"), "inventory")
    if len(reviews) != len(inv) or {r.get("clip_id") for r in reviews} != set(inv):
        raise ValueError("Inventory/review identity mismatch")
    apath, spath = out / "asr_suggestions.csv", out / "style_suggestions.csv"
    asr = index(read_csv(apath), "ASR") if apath.is_file() else {}
    hints = index(read_csv(spath), "AST") if spath.is_file() else {}
    details, ambiguous, experiments, ready = [], [], [], []
    summary = Counter()
    for row in reviews:
        cid = row["clip_id"]
        orig = inv[cid]
        if any(orig.get(k) != row.get(k) for k in (
            "sha256", "source_path", "relative_path", "duration_sec",
        )):
            raise ValueError(f"Inventory mismatch: {cid}")
        wav = Path(row["source_path"]).resolve()
        if source not in wav.parents or not wav.is_file() or digest(wav) != row["sha256"]:
            raise ValueError(f"Source WAV missing or changed: {wav}")
        ar, st = asr.get(cid, {}), hints.get(cid, {})
        for name, item in (("ASR", ar), ("AST", st)):
            if item and item.get("sha256") != row["sha256"]:
                raise ValueError(f"Stale {name} result: {cid}")
