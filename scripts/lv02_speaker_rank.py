#!/usr/bin/env python3
r"""LV-02 speaker similarity prioritization (evidence, never automatic approval).

Uses the existing Ui_Shigure reference bank and speaker embedding backend.
Scores the first review queue produced by lv02_expand_approved.py audit.
No retraining, new datasets, auto-ASR, model downloads or WAV modifications.

From LocalVoice root:
    .\.venv\Scripts\python.exe .\scripts\lv02_speaker_rank.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

from lv02_expand_approved import (
    REVIEW_COLS, SPEAKER, WORKSPACE, index, load_authority, load_csv,
    new_dir, sha256, video, write_csv,
)

SCORE_COLS = (
    *REVIEW_COLS,
    "speaker_score", "speaker_priority", "speaker_score_note",
    "rms", "observed_duration",
)


def latest_audit(workspace: Path) -> Path:
    for path in sorted(workspace.glob("lv02_expansion_[0-9][0-9][0-9]"),
                       reverse=True):
        report = path / "audit.json"
        queue = path / "review_queue.csv"
        if report.is_file() and queue.is_file():
            info = json.loads(report.read_text(encoding="utf-8"))
            if info.get("status") == "AUDIT_COMPLETE_NO_AUTO_APPROVAL":
                return path
    raise FileNotFoundError("No valid LV-02 audit directory; first run lv02_expand_approved.py audit")


def validate_queue(state: dict, rows: list[dict]) -> list[dict]:
    if len(rows) < 1:
        raise ValueError("Empty review queue")
    index(rows, "clip_id", "speaker-score review queue")
    checked = []
    for row in rows:
        cid = row["clip_id"]
        inv = state["inventory"].get(cid)
        auto = state["auto"].get(cid)
        if (inv is None or auto is None or
                cid in state["train_ids"] or cid in state["eval_ids"] or
                video(inv) in state["eval_videos"]):
            raise ValueError(f"Not a permitted training review candidate: {cid}")
        if (row.get("sha256") != inv["sha256"] or
                auto["sha256"] != inv["sha256"]):
            raise ValueError(f"Speaker-score queue SHA mismatch: {cid}")
        if (row.get("action") or row.get("speaker_ok") or
                row.get("text_verified")):
            raise ValueError(f"Queue already contains human decisions; do not rerank: {cid}")
        if row.get("source_path") and Path(row["source_path"]).resolve() != Path(inv["source_path"]).resolve():
            raise ValueError(f"Review queue source path mismatch: {cid}")
        path = Path(inv["source_path"]).resolve()
        checked.append({**row, "source_path": str(path),
                        "duration_sec": inv.get("duration_sec", ""),
                        "source_video": video(inv), "scan_flag": inv.get("scan_flag", "")})
    return checked


def balanced_sort(items: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in items:
        groups[row["source_video"]].append(row)
    for group in groups.values():
        group.sort(key=lambda row: (
            row.get("speaker_priority") != "HIGH_SIMILARITY_REVIEW_FIRST",
            row.get("speaker_score") == "",
            -float(row.get("speaker_score") or "-2"),
            row["clip_id"],
        ))
    result = []
    while any(groups.values()):
        for vid in sorted(groups):
            if groups[vid]:
                result.append(groups[vid].pop(0))
    return result


def analyze(ws: Path, audit_dir: Path | None = None) -> Path:
    ws = ws.resolve()
    state = load_authority(ws)
    folder = audit_dir.resolve() if audit_dir else latest_audit(ws)
    if ws not in folder.parents:
        raise ValueError("Audit directory must be inside the verified workspace")
    report = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
    if report.get("status") != "AUDIT_COMPLETE_NO_AUTO_APPROVAL":
        raise ValueError("Input is not a completed LV-02 audit")
    source_queue = folder / "review_queue.csv"
    selected = validate_queue(state, load_csv(source_queue))
    # Heavy imports happen ONLY after checking original evidence and paths.
    # Existing GPU/CPU .venv dependencies from LocalVoice; no reinstall.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import numpy as np
        import torch
        from common import (embedding_backend, embed_audio, load_audio_mono,
                            load_config, load_thresholds, normalize,
                            reference_dir, top_k_score)
    except ImportError as exc:
        raise RuntimeError("Existing LocalVoice speaker-embedding dependencies are missing") from exc

    config = load_config()
    bank_dir = reference_dir(config, SPEAKER)
    bank_path = bank_dir / "self_reference_bank.npz"
    metadata_path = bank_dir / "self_reference_bank.json"
    if not bank_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Speaker reference for {SPEAKER} is not available: {bank_path}. "
            "Do not substitute another speaker bank or claim auto-verification.")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("profile") != SPEAKER:
        raise ValueError("Speaker reference metadata profile is wrong")
    with np.load(bank_path, allow_pickle=False) as saved:
        bank = np.asarray(saved["embeddings"], dtype=np.float32)
    if bank.ndim != 2 or len(bank) < 2:
        raise ValueError("Speaker reference bank must contain two or more embeddings")
    bank = np.stack([normalize(row) for row in bank])
    if int(meta.get("reference_count", -1)) != len(bank):
        raise ValueError("Speaker bank metadata / embedding count mismatch")
    model_id = str(config["models"]["pyannote_pipeline"])
    if meta.get("model") != model_id:
        raise ValueError("Speaker reference model metadata does not match config")

    thresholds_path = bank_dir / "thresholds.json"
    calibrated = thresholds_path.is_file()
    if calibrated:
        accept, review, top_k = load_thresholds(config, SPEAKER)
        if not (-1 <= review <= accept <= 1 and 1 <= top_k <= len(bank)):
            raise ValueError("Invalid saved speaker calibration thresholds")
    else:
        top_k = int(config["classification"]["top_k"])
        accept = review = None
    if top_k < 1:
        raise ValueError("Invalid speaker top-k")

    print(f"Scoring {len(selected)} existing candidate WAVs against {len(bank)} "
          f"references, model={model_id}, calibrated={calibrated}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, target_sr = embedding_backend(config, device)
    scores = []
    for n, row in enumerate(selected, 1):
        clip = row["clip_id"]
        path = Path(row["source_path"])
        value = dict(row)
        value.update({"speaker_score": "", "speaker_priority": "UNSCORED",
                      "speaker_score_note": "", "rms": "", "observed_duration": ""})
        if not path.is_file() or sha256(path) != row["sha256"]:
            value["speaker_priority"] = "INVALID_SOURCE_REVIEW"
            value["speaker_score_note"] = "source missing or SHA changed"
        else:
            try:
                waveform, sample_rate = load_audio_mono(path)
                duration = len(waveform) / sample_rate
                rms = float(np.sqrt(np.mean(np.square(waveform))))
                value["observed_duration"] = f"{duration:.4f}"
                value["rms"] = f"{rms:.6f}"
                if (duration < 2.0 or duration > 20.0 or
                        not np.isfinite(rms) or rms < 1e-4):
                    value["speaker_priority"] = "AUDIO_LENGTH_OR_SILENCE_REVIEW"
                    value["speaker_score_note"] = "outside 2-20s reference limits or nearly silent"
                else:
                    embedding = embed_audio(backend, waveform, sample_rate, target_sr)
                    score = float(top_k_score(embedding, bank, top_k))
                    if not np.isfinite(score):
                        raise ValueError("Non-finite speaker similarity")
                    value["speaker_score"] = f"{score:.6f}"
                    if not calibrated:
                        value["speaker_priority"] = "UNCALIBRATED_SIMILARITY_REVIEW"
                    elif score >= accept:
                        value["speaker_priority"] = "HIGH_SIMILARITY_REVIEW_FIRST"
                    elif score >= review:
                        value["speaker_priority"] = "UNCERTAIN_SIMILARITY_REVIEW"
                    else:
                        value["speaker_priority"] = "LOW_SIMILARITY_REVIEW"
                    value["speaker_score_note"] = (
                        "HEURISTIC ONLY; score and source_kind do not verify target identity"
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                value["speaker_priority"] = "EMBEDDING_FAILURE_REVIEW"
                value["speaker_score_note"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        scores.append(value)
        if n % 10 == 0 or n == len(selected):
            print(f"  {n}/{len(selected)} scored", flush=True)
    ranked = balanced_sort(scores)
    dest = new_dir(ws)
    write_csv(dest / "speaker_ranked_review_queue.csv", SCORE_COLS, ranked)
    summary = {
        "status": "SPEAKER_SIMILARITY_RANKED_NOT_APPROVED",
        "input_audit": str(folder),
        "input_queue_sha256": sha256(source_queue),
        "speaker_reference_bank_sha256": sha256(bank_path),
        "speaker_reference_metadata_sha256": sha256(metadata_path),
        "speaker_reference_count": len(bank),
        "speaker_reference_profile": SPEAKER,
        "speaker_embedding_model": model_id,
        "calibrated_with_negative_examples": calibrated,
        "thresholds_sha256": sha256(thresholds_path) if calibrated else None,
        "scored_rows": len(ranked),
        "priority_counts": dict(Counter(row["speaker_priority"] for row in ranked)),
        "top_k": top_k,
        "output_csv": str(dest / "speaker_ranked_review_queue.csv"),
        "human_approval": 0,
        "training_started": False,
        "caution": ("Speaker embedding is heuristic; may fail for short, "
                    "emotional or noisy clips, and source/profile labels are "
                    "not proof of actual identity. Do not auto-approve."),
    }
    (dest / "speaker_rank_result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(dest),
                      "priorities": summary["priority_counts"]},
                     indent=2, ensure_ascii=False), flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--audit-dir", type=Path,
                        help="Audit directory with review_queue.csv; default latest audit")
    args = parser.parse_args()
    analyze(args.workspace, args.audit_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"LV-02 RANK BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
