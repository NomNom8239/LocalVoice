"""Read-only-source pilot: contextual rescue of 3,104 naraka clips shorter than 2 s.

Run from F:\AIProjects\LocalVoice with its existing .venv:
    .\.venv\Scripts\python.exe .\scripts\naraka_short_rescue_pilot.py --dry-run
    .\.venv\Scripts\python.exe .\scripts\naraka_short_rescue_pilot.py --run

--dry-run never writes files or loads a model. --run reads existing clips, scores
eligible short+short groups using Bank v2, then writes NEW reports under work/.
Source WAVs, banks, original manifests, and the 954 previous scores are untouched.
No pilot classification is a formal identity approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

RUN = Path("data/runs/naraka/bootstrap_acceptance_001")
PILOT = Path("data/training_audio/naraka_v2_pilot_001/pilot_manifest.csv")
BANK = Path("data/reference_bank/naraka_v2_pilot/self_reference_bank.npz")
DEFAULT_OUTPUT = Path("work/naraka_short_rescue/pilot_001")

EXPECTED_TOTAL = 4058
EXPECTED_SCORED = 954
EXPECTED_SHORT = 3104
EXPECTED_HUMAN_AUDITED = 97
MAX_GAP = 2.0  # trial, not an identity guarantee
MIN_JOIN_VOICE = 2.0
MAX_JOIN_VOICE = 20.0
WINDOW = 5.0
EXPLORATORY_SCORE = 0.49  # NOT an approved identity threshold
NEGATIVES = {"OTHER", "MIXED", "SILENCE"}

FIELDS = [
    "file", "speaker", "start", "end", "duration_sec", "source_turns",
    "original_class", "context_tier", "formal_approval", "human_truth_for_target",
    "near_human_self_count", "near_auto_self_count",
    "nearest_self_distance_sec", "nearest_self_source", "nearest_self_same_label",
    "self_anchor_same_label_count", "self_anchor_cross_label_count",
    "direct_self_anchor", "direct_self_anchor_same_label",
    "near_human_negative_types", "near_human_negative_files",
    "near_v2_other_count", "near_v2_review_count",
    "joined_group_id", "joined_group_clip_count", "joined_group_voice_sec",
    "joined_group_span_sec", "joined_group_max_gap_sec",
    "joined_group_known_negative_near", "joined_bank_v2_score",
    "joined_score_ge_049", "flags",
]
JOIN_FIELDS = [
    "group_id", "start", "end", "speaker", "clip_count", "clip_files",
    "voice_duration_sec", "span_duration_sec", "max_gap_sec",
    "known_negative_near", "known_negative_types", "bank_v2_score",
    "score_ge_049", "status",
]


def canonical(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).resolve())).casefold()


def read_rows(path: Path, sep: str = ",") -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as fp:
        data = list(csv.DictReader(fp, delimiter=sep))
    return data


def require_columns(rows: list[dict], columns: set[str], source: Path) -> None:
    if not rows:
        raise ValueError(f"Empty file: {source}")
    missing = columns - rows[0].keys()
    if missing:
        raise ValueError(f"Missing columns in {source}: {sorted(missing)}")


def temporal_gap(a: dict, b: dict) -> float:
    return max(0.0, a["start"] - b["end"], b["start"] - a["end"])


def load_source() -> tuple[list[dict], list[dict]]:
    source = RUN / "classification.tsv"
    rows = read_rows(source, "\t")
    audits = read_rows(PILOT)
    require_columns(rows, {"file", "start", "end", "classification", "speaker", "source_turns"}, source)
    require_columns(audits, {"file", "v2_class", "v2_score", "human_truth", "qc_status"}, PILOT)

    if len(rows) != EXPECTED_TOTAL or len(audits) != EXPECTED_SCORED:
        raise ValueError(f"Source count drift: classification={len(rows)} audit={len(audits)}")
    nshort = sum(r["classification"] == "UNSCORED" for r in rows)
    if nshort != EXPECTED_SHORT:
        raise ValueError(f"Short count drift: {nshort}, expected {EXPECTED_SHORT}")
    if any(r["classification"] not in {"SELF", "REVIEW", "OTHER", "UNSCORED"} for r in rows):
        raise ValueError("Unrecognized classification")

    byfile = {}
    for a in audits:
        k = canonical(a["file"])
        if k in byfile:
            raise ValueError(f"Duplicate pilot entry: {a['file']}")
        byfile[k] = a

    seen = set()
    matched = 0
    for r in rows:
        k = canonical(r["file"])
        if k in seen:
            raise ValueError(f"Duplicate classification file: {r['file']}")
        seen.add(k)
        r["start"] = float(r["start"])
        r["end"] = float(r["end"])
        if not math.isfinite(r["start"]) or not math.isfinite(r["end"]) or r["end"] <= r["start"]:
            raise ValueError(f"Invalid timeline: {r['file']}")
        r["audit"] = byfile.get(k)
        if r["classification"] == "UNSCORED" and r["audit"] is not None:
            raise ValueError(f"UNSCORED unexpectedly in 954 scored pilot: {r['file']}")
        if r["classification"] != "UNSCORED":
            if r["audit"] is None:
                raise ValueError(f"No matching Bank v2 pilot record: {r['file']}")
            matched += 1

    if matched != EXPECTED_SCORED:
        raise ValueError(f"Pilot match mismatch: {matched}")
    audited = sum(
        a["human_truth"] in {"SELF", "OTHER", "MIXED", "SILENCE"}
        for a in audits
    )
    if audited != EXPECTED_HUMAN_AUDITED:
        raise ValueError(f"Expected {EXPECTED_HUMAN_AUDITED} human audited, found {audited}")
    rows.sort(key=lambda r: (r["start"], r["end"], r["speaker"]))
    return rows, audits


def human_label(r: dict) -> str:
    return r["audit"]["human_truth"] if r.get("audit") else "UNREVIEWED"


def self_source(r: dict) -> str:
    """Do not allow known-human negatives to become SELF anchors."""
    a = r.get("audit")
    if not a or a.get("qc_status") == "REJECT":
        return ""
    if a["human_truth"] in NEGATIVES or a["human_truth"] == "UNKNOWN":
        return ""
    if a["human_truth"] == "SELF":
        return "HUMAN_SELF"
    if a["human_truth"] == "UNREVIEWED" and a["v2_class"] == "SELF":
        return "AUTO_V2_SELF"
    return ""


def build_groups(rows: list[dict]) -> tuple[list[list[dict]], list[list[dict]]]:
    """Reproduce previously tested consecutive same-label UNSCORED grouping."""
    groups: list[list[dict]] = []
    bucket: list[dict] = []
    for r in rows:
        if r["classification"] != "UNSCORED":
            if bucket:
                groups.append(bucket)
            bucket = []
            continue
        if bucket and (
            r["speaker"] != bucket[-1]["speaker"]
            or r["start"] - bucket[-1]["end"] > MAX_GAP
            or r["start"] < bucket[-1]["end"] - 0.005
        ):
            groups.append(bucket)
            bucket = []
        bucket.append(r)
    if bucket:
        groups.append(bucket)
    eligible = [
        g for g in groups
        if len(g) >= 2 and MIN_JOIN_VOICE <= sum(x["end"] - x["start"] for x in g) <= MAX_JOIN_VOICE
    ]
    return groups, eligible


def nearby(target: dict, candidates: list[dict], window: float = WINDOW) -> list[dict]:
    return [r for r in candidates if r is not target and temporal_gap(target, r) <= window]


def group_info(groups: list[list[dict]], negatives: list[dict]) -> list[dict]:
    result = []
    for idx, g in enumerate(groups, 1):
        first, last = g[0], g[-1]
        interval = {"start": first["start"], "end": last["end"]}
        danger = nearby(interval, negatives)
        voice = sum(x["end"] - x["start"] for x in g)
        max_gap = max(g[i]["start"] - g[i - 1]["end"] for i in range(1, len(g)))
        result.append({
            "group_id": f"J{idx:04d}",
            "start": f"{first['start']:.3f}",
            "end": f"{last['end']:.3f}",
            "speaker": first["speaker"],
            "clip_count": len(g),
            "clip_files": ";".join(x["file"] for x in g),
            "voice_duration_sec": f"{voice:.3f}",
            "span_duration_sec": f"{last['end'] - first['start']:.3f}",
            "max_gap_sec": f"{max_gap:.3f}",
            "known_negative_near": bool(danger),
            "known_negative_types": ";".join(sorted({human_label(x) for x in danger})),
            "bank_v2_score": "",
            "score_ge_049": "",
            "status": "UNSCORED_DRY_RUN",
        })
    return result


def score_all(groups: list[list[dict]], infos: list[dict]) -> None:
    """One embedding per *concatenated short-only* group. Never pad with an anchor."""
    import numpy as np
    import soundfile as sf
    import torch

    scripts_dir = str(Path("scripts").resolve())
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from common import embedding_backend, embed_audio, load_config, normalize, top_k_score

    if not BANK.is_file():
        raise FileNotFoundError(BANK)
    config = load_config()
    with np.load(BANK) as data:
        bank = np.stack([normalize(x) for x in np.asarray(data["embeddings"], dtype=np.float32)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, model_rate = embedding_backend(config, device)
    top_k = int(config["classification"]["top_k"])
    print(f"BANK_V2_DEVICE: {device} | GROUPS_TO_SCORE: {len(groups)}", flush=True)
    for n, (group, info) in enumerate(zip(groups, infos, strict=True), 1):
        samples = []
        sample_rate = None
        for r in group:
            src = Path(r["file"])
            with sf.SoundFile(src) as wave:
                rate = wave.samplerate
                audio = wave.read(dtype="float32", always_2d=True)
            if sample_rate is not None and rate != sample_rate:
                raise ValueError(f"Group sample-rate mismatch: {info['group_id']}")
            sample_rate = rate
            if audio.size == 0:
                raise ValueError(f"Empty clip: {src}")
            samples.append(audio.mean(axis=1))
        joined = np.concatenate(samples)
        with torch.inference_mode():
            embedding = embed_audio(backend, joined, sample_rate, model_rate)
            score = top_k_score(embedding, bank, top_k)
        if not np.isfinite(score):
            raise ValueError(f"Nonfinite group score: {info['group_id']}")
        info["bank_v2_score"] = f"{score:.4f}"
        info["score_ge_049"] = score >= EXPLORATORY_SCORE
        info["status"] = "SCORED_EXPLORATORY_ONLY"
        if n % 25 == 0 or n == len(groups):
            print(f"Scored groups: {n}/{len(groups)}", flush=True)


def build_candidates(rows: list[dict], eligible: list[list[dict]], infos: list[dict]) -> list[dict]:
    scored = [r for r in rows if r["classification"] != "UNSCORED"]
    short = [r for r in rows if r["classification"] == "UNSCORED"]
    negatives = [r for r in scored if human_label(r) in NEGATIVES]
    index_by_identity = {id(r): i for i, r in enumerate(rows)}
    group_map: dict[int, dict] = {}
    for group, info in zip(eligible, infos, strict=True):
        for r in group:
            if id(r) in group_map:
                raise ValueError("Overlapping joined groups")
            group_map[id(r)] = info
    output = []

    for r in short:
        near = nearby(r, scored)
        anchors = [(x, self_source(x)) for x in near]
        anchors = [(x, label) for x, label in anchors if label]
        anchors.sort(key=lambda item: (temporal_gap(r, item[0]), 0 if item[1] == "HUMAN_SELF" else 1))
        near_neg = [x for x in near if human_label(x) in NEGATIVES]
        idx = index_by_identity[id(r)]
        immediate = [
            rows[j] for j in (idx - 1, idx + 1)
            if 0 <= j < len(rows)
            and rows[j]["classification"] != "UNSCORED"
            and temporal_gap(r, rows[j]) <= MAX_GAP
            and self_source(rows[j])
        ]
        joined = group_map.get(id(r))
        score_text = (joined or {}).get("bank_v2_score", "")
        score_high = bool(score_text) and float(score_text) >= EXPLORATORY_SCORE
        joined_risk = bool(joined and joined["known_negative_near"])
        has_positive = bool(anchors)
        same_positive = any(x["speaker"] == r["speaker"] for x, _ in anchors)
        cross_positive = any(x["speaker"] != r["speaker"] for x, _ in anchors)
        flags = []
        if near_neg:
            flags.append("HUMAN_NEGATIVE_WITHIN_5S_NOT_TARGET_TRUTH")
        if joined_risk:
            flags.append("JOINED_GROUP_NEAR_HUMAN_NEGATIVE")
        if cross_positive:
            flags.append("SELF_ANCHOR_CROSSES_PYANNOTE_LABEL")
        if has_positive and not same_positive:
            flags.append("SELF_ONLY_ACROSS_LABEL")
        if joined and float(joined["max_gap_sec"]) > 0.8:
            flags.append("JOIN_EXTENDS_ORIGINAL_0P8S_GAP")
        if joined and not score_text:
            flags.append("GROUP_SCORE_PENDING")
        if score_text and not score_high:
            flags.append("JOIN_SCORE_BELOW_EXPLORATORY_049")
        if (near_neg or joined_risk) and score_high:
            flags.append("HIGH_JOIN_SCORE_WITH_KNOWN_NEGATIVE_NEAR")
        if any(x["audit"]["v2_class"] == "OTHER" for x in near):
            flags.append("NEAR_AUTO_V2_OTHER_NOT_PROOF")
        if any(x["audit"]["v2_class"] == "REVIEW" for x in near):
            flags.append("NEAR_AUTO_V2_REVIEW_NOT_PROOF")

        if near_neg or joined_risk:
            tier = "KNOWN_NEGATIVE_CONTEXT_REVIEW"
        elif score_high and has_positive:
            tier = "MULTI_EVIDENCE_SELF_CANDIDATE"
        elif has_positive or score_text:
            tier = "CONTEXT_OR_JOIN_REVIEW"
        else:
            tier = "HOLD_NO_RELIABLE_EVIDENCE"
        best = anchors[0] if anchors else None
        output.append({
            "file": r["file"],
            "speaker": r["speaker"],
            "start": f"{r['start']:.3f}",
            "end": f"{r['end']:.3f}",
            "duration_sec": f"{r['end'] - r['start']:.3f}",
            "source_turns": r["source_turns"],
            "original_class": "UNSCORED",
            "context_tier": tier,
            "formal_approval": "NO",
            "human_truth_for_target": "UNREVIEWED",
            "near_human_self_count": sum(label == "HUMAN_SELF" for _, label in anchors),
            "near_auto_self_count": sum(label == "AUTO_V2_SELF" for _, label in anchors),
            "nearest_self_distance_sec": f"{temporal_gap(r, best[0]):.3f}" if best else "",
            "nearest_self_source": best[1] if best else "",
            "nearest_self_same_label": (best[0]["speaker"] == r["speaker"]) if best else "",
            "self_anchor_same_label_count": sum(x["speaker"] == r["speaker"] for x, _ in anchors),
            "self_anchor_cross_label_count": sum(x["speaker"] != r["speaker"] for x, _ in anchors),
            "direct_self_anchor": bool(immediate),
            "direct_self_anchor_same_label": any(x["speaker"] == r["speaker"] for x in immediate),
            "near_human_negative_types": ";".join(sorted({human_label(x) for x in near_neg})),
            "near_human_negative_files": ";".join(x["file"] for x in near_neg),
            "near_v2_other_count": sum(x["audit"]["v2_class"] == "OTHER" for x in near),
            "near_v2_review_count": sum(x["audit"]["v2_class"] == "REVIEW" for x in near),
            "joined_group_id": joined["group_id"] if joined else "",
            "joined_group_clip_count": joined["clip_count"] if joined else "",
            "joined_group_voice_sec": joined["voice_duration_sec"] if joined else "",
            "joined_group_span_sec": joined["span_duration_sec"],
            "joined_group_max_gap_sec": joined["max_gap_sec"],
            "joined_group_known_negative_near": joined["known_negative_near"],
            "joined_bank_v2_score": score_text,
            "joined_score_ge_049": joined["score_ge_049"],
            "flags": ";".join(flags),
        } if joined else {
            "file": r["file"],
            "speaker": r["speaker"],
            "start": f"{r['start']:.3f}",
            "end": f"{r['end']:.3f}",
            "duration_sec": f"{r['end'] - r['start']:.3f}",
            "source_turns": r["source_turns"],
            "original_class": "UNSCORED",
            "context_tier": tier,
            "formal_approval": "NO",
            "human_truth_for_target": "UNREVIEWED",
            "near_human_self_count": sum(label == "HUMAN_SELF" for _, label in anchors),
            "near_auto_self_count": sum(label == "AUTO_V2_SELF" for _, label in anchors),
            "nearest_self_distance_sec": f"{temporal_gap(r, best[0]):.3f}" if best else "",
            "nearest_self_source": best[1] if best else "",
            "nearest_self_same_label": (best[0]["speaker"] == r["speaker"]) if best else "",
            "self_anchor_same_label_count": sum(x["speaker"] == r["speaker"] for x, _ in anchors),
            "self_anchor_cross_label_count": sum(x["speaker"] != r["speaker"] for x, _ in anchors),
            "direct_self_anchor": bool(immediate),
            "direct_self_anchor_same_label": any(x["speaker"] == r["speaker"] for x in immediate),
            "near_human_negative_types": ";".join(sorted({human_label(x) for x in near_neg})),
            "near_human_negative_files": ";".join(x["file"] for x in near_neg),
            "near_v2_other_count": sum(x["audit"]["v2_class"] == "OTHER" for x in near),
            "near_v2_review_count": sum(x["audit"]["v2_class"] == "REVIEW" for x in near),
            "joined_group_id": "",
            "joined_group_clip_count": "",
            "joined_group_voice_sec": "",
            "joined_group_span_sec": "",
            "joined_group_max_gap_sec": "",
            "joined_group_known_negative_near": "",
            "joined_bank_v2_score": "",
            "joined_score_ge_049": "",
            "flags": ";".join(flags),
        })
    return output


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def execute(*, dry_run: bool, output: Path, assert_expected_join_counts: bool = True) -> dict:
    rows, audits = load_source()
    groups, eligible = build_groups(rows)
    nclips = sum(len(g) for g in eligible)
    if assert_expected_join_counts and (len(groups), len(eligible), nclips) != (2643, 127, 323):
        raise ValueError(
            "Join eligibility differs from the prior verified pilot: "
            f"groups={len(groups)}, eligible={len(eligible)}, eligible clips={nclips}"
        )
    negatives = [r for r in rows if r.get("audit") and human_label(r) in NEGATIVES]
    infos = group_info(eligible, negatives)
    if not dry_run:
        if output.exists() or output.with_name(output.name + ".building").exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
        score_all(eligible, infos)
    candidates = build_candidates(rows, eligible, infos)
    totals = Counter(r["context_tier"] for r in candidates)
    high = sum(x["score_ge_049"] is True for x in infos)
    summary = {
        "status": "DRY_RUN" if dry_run else "CANDIDATE_REPORT_NOT_APPROVED",
        "source_total": len(rows),
        "scored_unchanged": len(audits),
        "unscored_targets": len(candidates),
        "human_audited_reference": 97,
        "unscored_groups": len(groups),
        "join_eligible_groups": len(eligible),
        "join_eligible_short_clips": nclips,
        "groups_near_human_negative": sum(x["known_negative_near"] for x in infos),
        "join_groups_scored": len(infos) if not dry_run else 0,
        "join_scores_ge_exploratory_049": high if not dry_run else None,
        "context_tier_counts": dict(sorted(totals.items())),
        "direct_self_anchor_count": sum(bool(r["direct_self_anchor"]) for r in candidates),
        "same_label_self_5s_count": sum(int(r["self_anchor_same_label_count"]) > 0 for r in candidates),
        "cross_label_only_self_5s_count": sum(
            int(r["self_anchor_same_label_count"]) == 0 and int(r["self_anchor_cross_label_count"]) > 0
            for r in candidates
        ),
        "source_classification_sha256": sha256(RUN / "classification.tsv"),
        "source_pilot_manifest_sha256": sha256(PILOT),
        "reference_bank_sha256": sha256(BANK) if BANK.exists() else "NOT_PRESENT_DRY_RUN",
        "threshold": {"joined_score_ge": EXPLORATORY_SCORE, "status": "EXPLORATORY_ONLY"},
        "boundaries": {
            "all_3104_are_unreviewed": True,
            "no_identity_self_promotions": True,
            "no_formal_approvals": True,
            "no_source_file_modifications": True,
            "negative_context_is_not_target_negative_truth": True,
            "no_join_with_scored_self_anchor": True,
            "joined_audio_not_written": True,
        },
        "notes": [
            "Near SELF means candidate evidence, not speaker identity truth.",
            "Human OTHER/MIXED/SILENCE within 5 seconds is a proximity warning, not a target verdict.",
            "Three previously listened contexts near 1557.880, 1566.841 and 4070.720 seconds contain game voice; exact targets are not individually labeled.",
            "Prior pseudo-short test found bank-only high scores on human MIXED/SILENCE; threshold 0.49 must not auto-approve.",
            "Existing pyannote group labels may split a single narrator or combine different voices.",
            "954 scored records are contextual evidence; none is re-scored here.",
        ],
    }
    print("=== Naraka short rescue pilot ===")
    for field in (
        "source_total", "scored_unchanged", "unscored_targets", "unscored_groups",
        "join_eligible_groups", "join_eligible_short_clips", "groups_near_human_negative",
        "join_groups_scored", "join_scores_ge_exploratory_049",
        "same_label_self_5s_count", "cross_label_only_self_5s_count",
        "direct_self_anchor_count",
    ):
        print(f"{field}: {summary[field]}")
    print("REVIEW_TIERS:", json.dumps(summary["context_tier_counts"], ensure_ascii=False))

    if dry_run:
        print("DRY RUN COMPLETE: no outputs written and no models loaded")
        return summary

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    staging.mkdir(exist_ok=False)
    try:
        save_csv(staging / "short_candidate_manifest.csv", candidates, FIELDS)
        save_csv(staging / "joined_group_scores.csv", infos, JOIN_FIELDS)
        (staging / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    except Exception:
        import shutil
        shutil.rmtree(staging)
        raise
    print(f"OUTPUT: {output.resolve()}")
    print("RUN COMPLETE: no new speaker truth or formal approvals")
    return summary


def self_test() -> None:
    def item(s, e, c="UNSCORED", speaker="A"):
        return {"start": s, "end": e, "classification": c, "speaker": speaker}

    rows = [item(0, 1.2), item(2.2, 3.3), item(3.4, 4.0, "SELF")]
    groups, eligible = build_groups(rows)
    assert len(groups) == 1 and len(eligible) == 1 and len(eligible[0]) == 2
    assert abs(temporal_gap(rows[0], rows[1]) - 1.0) < 0.000001
    assert len(nearby(rows[0], [rows[2]], 2.0)) == 0
    assert len(nearby(rows[0], [rows[2]], 2.3)) == 1
    rows2 = [item(0, 1.1), item(1.6, 2.7, speaker="B"), item(2.8, 4.0)]
    groups2, eligible2 = build_groups(rows2)
    assert len(groups2) == 3 and not eligible2
    print("SELF_TEST: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Check existing manifests and candidate coverage only")
    mode.add_argument("--run", action="store_true", help="Score all 127 joined groups and write new candidate reports")
    mode.add_argument("--self-test", action="store_true", help="Test interval and join logic without input files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="New destination for --run; must not exist")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    execute(dry_run=args.dry_run, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
