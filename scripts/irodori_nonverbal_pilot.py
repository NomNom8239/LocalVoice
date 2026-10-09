"""Prepare isolated, strictly experimental emoji-only nonverbal text hypotheses.

This does not train a model, validate emoji-only conditioning, or alter approved
voice training data. Source WAV paths and hashes are checked before writing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__:
    from .irodori_dataset import digest, paths, read_csv, write_csv
else:
    from irodori_dataset import digest, paths, read_csv, write_csv

# Irodori v4.1-Small official EMOJI_ANNOTATIONS.md; these indicate synthesis
# intentions, not verified transcript text. Bare emojis remain experimental.
# https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/blob/main/EMOJI_ANNOTATIONS.md
STYLE_EMOJI = {
    "laugh": "🤭",
    "breath": "😮‍💨",
    "panting": "🌬️",
    "groan": "🥵",
}
SOURCE_RANK = {"human_confirmed": 0, "human_tentative": 1,
               "ast_experimental": 2}
AUDIT_COLUMNS = (
    "clip_id", "sha256", "source_path", "style", "style_source",
    "status", "reason", "experimental_text", "caption",
    "annotation_reference",
)
CANDIDATE_COLUMNS = ("audio", "text", "caption", "speaker")
ANNOTATIONS = (
    "https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/"
    "blob/main/EMOJI_ANNOTATIONS.md"
)


def build(args: argparse.Namespace, source: Path, out: Path) -> dict[str, object]:
    """Select a balanced small pilot without promoting hypotheses to training."""
    if args.max_per_style < 1:
        raise ValueError("--max-per-style must be >= 1")
    candidates_path = out / "nonverbal_experiments.csv"
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Run prepare first: {candidates_path}")
    items = read_csv(candidates_path)
    inventory = read_csv(out / "inventory.csv")
    identifiers = {r["clip_id"]: r for r in inventory}
    if len(identifiers) != len(inventory):
        raise ValueError("Duplicate inventory identity")
    seen = set()
    for r in items:
        cid = r.get("clip_id", "")
        if not cid or cid in seen:
            raise ValueError(f"Missing/duplicate nonverbal clip ID: {cid}")
        seen.add(cid)
        orig = identifiers.get(cid)
        if not orig or r.get("sha256") != orig.get("sha256"):
            raise ValueError(f"Stale nonverbal input: {cid}")
        wav = Path(r.get("source_path", "")).resolve()
        if (wav != Path(orig["source_path"]).resolve()
                or source not in wav.parents
                or not wav.is_file()
                or digest(wav) != r["sha256"]):
            raise ValueError(f"Changed/missing WAV: {cid}")

    ranked = sorted(items, key=lambda r: (
        SOURCE_RANK.get(r.get("style_source", ""), 99),
        r.get("style", ""), r["clip_id"],
    ))
    counted: Counter[str] = Counter()
    audit = []
    selected = []
    for r in ranked:
        style = r.get("style", "")
        source_kind = r.get("style_source", "")
        emoji = STYLE_EMOJI.get(style)
        status = "held"
        reason = "unmapped_or_broad_style"
        if emoji is not None:
            if source_kind not in SOURCE_RANK:
                reason = "unsupported_classification_provenance"
            elif counted[style] >= args.max_per_style:
                reason = "pilot_style_cap_reached"
            else:
                reason = "emoji_only_hypothesis_unvalidated"
                status = "pilot_hypothesis"
                counted[style] += 1
                selected.append({
                    "audio": r["source_path"], "text": emoji,
                    "caption": r.get("auto_caption", ""),
                    "speaker": args.profile,
                })
        audit.append({
            "clip_id": r["clip_id"], "sha256": r["sha256"],
            "source_path": r["source_path"], "style": style,
            "style_source": source_kind, "status": status,
            "reason": reason, "experimental_text": emoji or "",
            "caption": r.get("auto_caption", ""),
            "annotation_reference": ANNOTATIONS,
        })
    target = out / "nonverbal_pilot"
    write_csv(target / "emoji_only_hypotheses.csv", CANDIDATE_COLUMNS, selected)
    write_csv(target / "audit.csv", AUDIT_COLUMNS, audit)
    result = {
        "source_candidates": len(items),
        "pilot_hypotheses": len(selected),
        "by_style": dict(counted),
        "held_without_manual_work": len(items) - len(selected),
        "status": "HYPOTHESIS_ONLY_NOT_TRAINING_READY",
        "validations_remaining": [
            "tokenizer_retains_emoji_as_conditioning_input",
            "upstream_manifest_encode_accepts_local_audio",
            "paired_pilot_and_control_LoRA_training",
            "generation_comparison_for_nonverbal_quality_and_speech_retention",
        ],
        "files": {
            "hypotheses": str(target / "emoji_only_hypotheses.csv"),
            "audit": str(target / "audit.csv"),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--max-per-style", type=int, default=3)
    args = parser.parse_args()
    try:
        source, out = paths(args)
        build(args, source, out)
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
