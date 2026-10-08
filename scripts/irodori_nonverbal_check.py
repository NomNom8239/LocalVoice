"""Preflight-only: check proposed emoji-only nonverbal examples against Irodori tokenizer.

Does NOT run training, encode audio, or alter existing review/WAV/training files.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

if __package__:
    from .irodori_dataset import ROOT, digest, paths, read_csv, write_csv
else:
    from irodori_dataset import ROOT, digest, paths, read_csv, write_csv

TOKEN_AUDIT_COLUMNS = (
    "clip_id", "style", "emoji", "ids_json", "tokens_json",
    "decoded", "unk_count", "bos_id", "status",
)


def tokenizer_spec(config_path: Path) -> dict[str, object]:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Irodori v4-Small config is required: {config_path}"
        )
    config = config_path.read_text(encoding="utf-8")
    spec = {}
    for key in ("text_tokenizer_repo", "text_encoder_revision", "text_add_bos"):
        match = re.search(r"^\s*" + re.escape(key) + r":\s*([^#\n]+)",
                          config, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing {key} in Irodori config: {config_path}")
        spec[key] = match.group(1).strip().strip("'\"")
    if spec["text_add_bos"].lower() not in {"true", "false"}:
        raise ValueError("Unexpected text_add_bos in model configuration")
    spec["text_add_bos"] = spec["text_add_bos"].lower() == "true"
    if spec["text_encoder_revision"].lower() in {"null", "none", "~"}:
        raise ValueError("Unpinned tokenizer revision; cannot verify reproducibility")
    return spec


def load_tokenizer(spec: dict[str, object]):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers in LocalVoice .venv for tokenizer check") from exc
    return AutoTokenizer.from_pretrained(
        spec["text_tokenizer_repo"],
        revision=spec["text_encoder_revision"],
        trust_remote_code=False,
        use_fast=True,
    )


def atomic_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def build(args: argparse.Namespace, source: Path, out: Path, *,
          tokenizer_loader=load_tokenizer) -> dict[str, object]:
    pilot = out / "nonverbal_pilot"
    hypothesis_path = pilot / "emoji_only_hypotheses.csv"
    audit_path = pilot / "audit.csv"
    hypotheses = read_csv(hypothesis_path)
    audit = read_csv(audit_path)
    inventory = read_csv(out / "inventory.csv")
    inv = {r["source_path"]: r for r in inventory}
    if len(inv) != len(inventory):
        raise ValueError("Duplicate inventory source path")
    picked = [r for r in audit if r.get("status") == "pilot_hypothesis"]
    if not picked or len(picked) != len(hypotheses):
        raise ValueError("Pilot hypotheses/audit mismatch")
    audit_by_path = {r["source_path"]: r for r in picked}
    if len(audit_by_path) != len(picked):
        raise ValueError("Duplicate pilot audit paths")

    # Full source validation before tokenizer work or creating outputs.
    checked = []
    seen = set()
    for row in hypotheses:
        path = Path(row["audio"]).resolve()
        key = str(path)
        match = audit_by_path.get(row["audio"])
        source_info = inv.get(row["audio"])
        if key in seen or not match or not source_info:
            raise ValueError(f"Unmatched/duplicate pilot audio: {path}")
        seen.add(key)
        if (source not in path.parents or not path.is_file()
                or match["sha256"] != source_info["sha256"]
                or match["experimental_text"] != row["text"]
                or match["caption"] != row["caption"]
                or row["speaker"] != args.profile
                or digest(path) != match["sha256"]):
            raise ValueError(f"Pilot audio/evidence mismatch: {path}")
        checked.append((row, match))
    config_path = (Path(args.config).expanduser().resolve()
                   if args.config else ROOT / "Irodori-TTS" / "configs" / "train_v4_small.yaml")
    spec = tokenizer_spec(config_path)
    tokenizer = tokenizer_loader(spec)
    bos = getattr(tokenizer, "bos_token_id", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if spec["text_add_bos"] and bos is None:
        raise ValueError("Tokenizer lacks BOS required by Irodori config")

    token_rows = []
    jsonl_rows = []
    any_failed = False
    for row, evidence in checked:
        raw_ids = tokenizer.encode(row["text"], add_special_tokens=False)
        ids = list(raw_ids)
        if spec["text_add_bos"]:
            ids.insert(0, int(bos))
        tokens = tokenizer.convert_ids_to_tokens(ids)
        decoded = tokenizer.decode(
            raw_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        unknown_count = sum(x == unk for x in raw_ids) if unk is not None else 0
        # Zero-width joiner / emoji-normalization may alter decoded string;
        # fail conservatively and report tokens rather than claiming equivalence.
        restored = unicodedata.normalize("NFC", decoded)
        expected = unicodedata.normalize("NFC", row["text"])
        status = ("pass" if raw_ids and unknown_count == 0
                  and restored == expected else "block")
        any_failed |= status != "pass"
        token_rows.append({
            "clip_id": evidence["clip_id"], "style": evidence["style"],
            "emoji": row["text"], "ids_json": json.dumps(ids),
            "tokens_json": json.dumps(tokens, ensure_ascii=False),
            "decoded": decoded, "unk_count": str(unknown_count),
            "bos_id": str(bos) if spec["text_add_bos"] else "",
            "status": status,
        })
        jsonl_rows.append({
            "audio": str(Path(row["audio"]).resolve()),
            "text": row["text"], "caption": row["caption"],
            "speaker": row["speaker"],
        })

    # Independent inspectable outputs; original pilot and 1,041 speech
    # candidates are never changed. No Irodori latent encoding occurs.
    write_csv(pilot / "tokenizer_audit.csv", TOKEN_AUDIT_COLUMNS, token_rows)
    atomic_jsonl(pilot / "hf_audio_dataset_hypothesis.jsonl", jsonl_rows)
    report = {
        "source_rows": len(checked),
        "tokenizer_repo": spec["text_tokenizer_repo"],
        "tokenizer_revision": spec["text_encoder_revision"],
        "tokenization_pass": sum(r["status"] == "pass" for r in token_rows),
        "tokenization_block": sum(r["status"] == "block" for r in token_rows),
        "tokenizer_gate": "BLOCK" if any_failed else "PASS_TOKENIZATION_ONLY",
        "training_ready": False,
        "upstream_audio_manifest_encoding": "NOT_RUN",
        "lora_training": "NOT_RUN",
        "human_work_required": False,
        "next_gate": "upstream DACVAE manifest preparation on local GPU",
        "outputs": {
            "tokenizer_audit": str(pilot / "tokenizer_audit.csv"),
            "hf_jsonl_hypothesis": str(pilot / "hf_audio_dataset_hypothesis.jsonl"),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--config", help="Path to local Irodori v4-Small YAML config")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        report = build(args, source, out)
        return 2 if report["tokenizer_gate"] == "BLOCK" else 0
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
