"""Build a conservative, auditable LocalVoice naraka Bank-v2 training pilot.

Default is read-only --dry-run. --run writes only a new destination path.
The manually confirmed + QC-passed samples are placed in audio/.
Auto-predicted samples are placed in auto_candidates/ and NEVER in audio/.

Run this script from the LocalVoice project root using its .venv Python.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


RUN = Path("data/runs/naraka/bootstrap_acceptance_001")
EVAL = RUN / "evaluation"
OUTPUT = Path("data/training_audio/naraka_v2_pilot_001")
EXPECTED = {
    "human_verdicts_v1.csv": 19,
    "bank_v2_candidates.csv": 12,
    "v2_threshold049_promoted_audit.csv": 51,
    "v2_threshold049_demoted_audit.csv": 23,
}
VERDICT = {
    "S": "SELF", "SELF": "SELF",
    "O": "OTHER", "OTHER": "OTHER",
    "M": "MIXED", "MIXED": "MIXED",
    "N": "SILENCE", "SILENCE": "SILENCE",
    "?": "UNKNOWN", "UNKNOWN": "UNKNOWN",
}
FIELDS = [
    "file", "v1_class", "v2_class", "v2_score", "in_v2_training",
    "human_truth", "human_sources", "qc_status", "qc_reason",
    "duration_sec", "rms_dbfs", "clipping_ratio", "silence_ratio", "dc_offset",
    "decision", "destination", "sha256",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve())).casefold()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_truth() -> dict[str, dict]:
    known: dict[str, dict] = {}
    for name, expected in EXPECTED.items():
        rows = read_csv(EVAL / name)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} entries, got {len(rows)}")
        for row in rows:
            file = (row.get("file") or "").strip()
            token = (row.get("truth") or "").strip().upper()
            if not file or token not in VERDICT:
                raise ValueError(f"Invalid label in {name}: {row}")
            label = VERDICT[token]
            path_key = key(file)
            if path_key in known:
                if known[path_key]["truth"] != label:
                    raise ValueError(f"Conflicting human verdicts for {file}")
                known[path_key]["sources"].append(name)
            else:
                known[path_key] = {"truth": label, "sources": [name]}
    return known


def audio_qc(path: Path, qc: dict) -> dict:
    try:
        x, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate <= 0 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("empty or invalid audio")
        x = x.mean(axis=1)
        duration = len(x) / rate
        rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))
        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        clipping = float(np.mean(np.abs(x) >= float(qc["clip_level"])))
        silence_threshold = 10.0 ** (float(qc["silence_level_dbfs"]) / 20.0)
        silence = float(np.mean(np.abs(x) < silence_threshold))
        dc = float(abs(np.mean(x)))

        reject = []
        review = []
        if duration < float(qc["min_duration"]):
            reject.append("too_short")
        if rms_db <= float(qc["reject_rms_dbfs"]):
            reject.append("near_silent")
        if clipping >= float(qc["review_clipping_ratio"]):
            review.append("clipping")
        if dc >= float(qc["review_dc_offset"]):
            review.append("dc_offset")
        if silence >= float(qc["review_silence_ratio"]):
            review.append("mostly_silent")
        if rms_db <= float(qc["review_rms_low_dbfs"]):
            review.append("low_rms")
        if rms_db >= float(qc["review_rms_high_dbfs"]):
            review.append("high_rms")

        reason = reject + review
        return {
            "qc_status": "REJECT" if reject else "REVIEW" if review else "PASS",
            "qc_reason": ",".join(reason),
            "duration_sec": f"{duration:.3f}",
            "rms_dbfs": f"{rms_db:.2f}",
            "clipping_ratio": f"{clipping:.6f}",
            "silence_ratio": f"{silence:.6f}",
            "dc_offset": f"{dc:.6f}",
        }
    except Exception as exc:
        return {
            "qc_status": "REJECT", "qc_reason": f"unreadable:{type(exc).__name__}",
            "duration_sec": "", "rms_dbfs": "", "clipping_ratio": "",
            "silence_ratio": "", "dc_offset": "",
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="Create pilot dataset (default: dry-run)")
    args = p.parse_args()

    with Path("config.toml").open("rb") as fp:
        config = tomllib.load(fp)
    qc = config["dataset_qc"]
    truth = load_truth()
    source_rows = read_csv(EVAL / "v2_threshold049_projection.csv")
    if len(source_rows) != 954:
        raise ValueError(f"Expected 954 projection records, got {len(source_rows)}")
    if OUTPUT.exists() or OUTPUT.with_name(OUTPUT.name + ".building").exists():
        raise FileExistsError(f"Output already exists; will not overwrite: {OUTPUT}")

    staging = OUTPUT.with_name(OUTPUT.name + ".building")
    results = []
    counts = Counter()
    seen = set()
    for row in source_rows:
        src = Path(row["file"])
        src_key = key(src)
        if src_key in seen:
            raise ValueError(f"Duplicate source in projection: {src}")
        seen.add(src_key)
        if not src.is_file():
            raise FileNotFoundError(src)
        if row["v2_class_test049"] not in ("SELF", "REVIEW", "OTHER"):
            raise ValueError(f"Unexpected v2 class: {row['v2_class_test049']}")

        human = truth.get(src_key)
        label = human["truth"] if human else "UNREVIEWED"
        metrics = audio_qc(src, qc)
        qc_state = metrics["qc_status"]

        # Human identity labels always override machine classification.
        # Manual SELF is still subject to acoustic quality screening.
        if label in ("OTHER", "MIXED", "SILENCE", "UNKNOWN"):
            decision = "EXCLUDE_HUMAN"
        elif qc_state == "REJECT":
            decision = "EXCLUDE_QC"
        elif qc_state == "REVIEW":
            decision = "QUALITY_REVIEW"
        elif label == "SELF":
            decision = "CONFIRMED_AUDIO"
        elif row["v2_class_test049"] == "SELF":
            decision = "AUTO_CANDIDATE"
        else:
            decision = "HOLD_UNCERTAIN"

        sha = digest(src) if decision in ("CONFIRMED_AUDIO", "AUTO_CANDIDATE") else ""
        dest = ""
        if decision in ("CONFIRMED_AUDIO", "AUTO_CANDIDATE"):
            folder = "audio" if decision == "CONFIRMED_AUDIO" else "auto_candidates"
            dest = str(OUTPUT / folder / f"{src.stem}__{sha[:12]}.wav")

        results.append({
            "file": str(src.resolve()),
            "v1_class": row["v1_class"],
            "v2_class": row["v2_class_test049"],
            "v2_score": row["v2_score"],
            "in_v2_training": row["in_v2_training"],
            "human_truth": label,
            "human_sources": ";".join(human["sources"]) if human else "",
            **metrics,
            "decision": decision,
            "destination": dest,
            "sha256": sha,
        })
        counts[decision] += 1

    print("=== Bank v2 pilot data collection ===")
    print(f"Source classification rows: {len(results)}")
    print(f"Human verdict files: {len(truth)} (includes preview clips outside projection)")
    for choice in (
        "CONFIRMED_AUDIO", "AUTO_CANDIDATE", "QUALITY_REVIEW",
        "EXCLUDE_HUMAN", "EXCLUDE_QC", "HOLD_UNCERTAIN",
    ):
        print(f"  {choice:18s}: {counts[choice]}")
    print(f"Output: {OUTPUT}")
    if not args.run:
        print("DRY RUN: no files were written or copied")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for row in results:
            if row["decision"] not in ("CONFIRMED_AUDIO", "AUTO_CANDIDATE"):
                continue
            final_path = Path(row["destination"])
            relative = final_path.relative_to(OUTPUT)
            staged_file = staging / relative
            staged_file.parent.mkdir(parents=True, exist_ok=True)
            if staged_file.exists():
                raise FileExistsError(f"Name collision: {staged_file}")
            shutil.copy2(row["file"], staged_file)
            if digest(staged_file) != row["sha256"]:
                raise ValueError(f"Copy checksum mismatch: {staged_file}")

        manifest = staging / "pilot_manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(results)
        (staging / "summary.json").write_text(
            json.dumps({
                "pilot": "naraka_v2_threshold049",
                "source_records": len(results),
                "counts": dict(counts),
                "note": "Only audio/ is human-identity-confirmed + automated-QC-passed. "
                        "auto_candidates/ is NOT approved for training.",
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        staging.rename(OUTPUT)
    except BaseException:
        print(f"INCOMPLETE: staging directory retained at {staging}", file=sys.stderr)
        raise

    print(f"CREATED: {OUTPUT / 'pilot_manifest.csv'}")
    print(f"CONFIRMED training audio: {OUTPUT / 'audio'}")
    print(f"Unapproved auto candidates: {OUTPUT / 'auto_candidates'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
