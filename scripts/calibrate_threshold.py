from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import (
    embed_audio,
    embedding_backend,
    load_audio_mono,
    load_config,
    normalize,
    project_path,
    reference_dir,
    top_k_score,
)


MIN_DURATION = 2.0
MAX_DURATION = 20.0


def positive_leave_one_out_scores(
    bank: np.ndarray,
    top_k: int,
) -> np.ndarray:
    scores = []
    for i in range(len(bank)):
        references = np.delete(bank, i, axis=0)
        scores.append(top_k_score(bank[i], references, top_k))
    return np.asarray(scores, dtype=np.float32)


def find_balanced_threshold(
    positives: np.ndarray,
    negatives: np.ndarray,
) -> tuple[float, float, float]:
    values = np.unique(np.concatenate([positives, negatives]))
    candidates = [
        (a + b) / 2.0
        for a, b in zip(values[:-1], values[1:])
    ]
    candidates.extend(
        [
            float(values[0]) - 1e-6,
            float(values[-1]) + 1e-6,
        ]
    )

    best = None
    for threshold in candidates:
        tpr = float(np.mean(positives >= threshold))
        fpr = float(np.mean(negatives >= threshold))
        candidate = (tpr - fpr, tpr, -fpr, threshold)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    _, tpr, neg_fpr, threshold = best
    return float(threshold), float(tpr), float(-neg_fpr)


def find_zero_false_accept_threshold(
    positives: np.ndarray,
    negatives: np.ndarray,
) -> tuple[float, float]:
    max_negative = float(np.max(negatives))
    higher = positives[positives > max_negative]

    if len(higher) == 0:
        return float(np.nextafter(max_negative, np.inf)), 0.0

    nearest_positive = float(np.min(higher))
    threshold = (max_negative + nearest_positive) / 2.0
    recall = float(np.mean(positives >= threshold))
    return threshold, recall


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate SELF/REVIEW/OTHER thresholds for a profile."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--negative-source", required=True)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    profile_dir = reference_dir(config, args.profile)
    reference_path = profile_dir / "self_reference_bank.npz"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    negative_root = project_path(args.negative_source)
    if not negative_root.is_dir():
        raise NotADirectoryError(negative_root)

    top_k = (
        args.top_k
        if args.top_k is not None
        else int(config["classification"]["top_k"])
    )
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")

    data = np.load(reference_path)
    bank = np.asarray(data["embeddings"], dtype=np.float32)
    bank = np.stack([normalize(row) for row in bank])

    positive_scores = positive_leave_one_out_scores(bank, top_k)

    negative_files = sorted(
        p for p in negative_root.rglob("*.wav")
        if p.is_file()
    )
    if not negative_files:
        raise RuntimeError(
            f"No negative WAV files found under: {negative_root}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, embedding_sample_rate = embedding_backend(config, device)

    negative_scores = []
    valid_files = []

    print(f"Profile: {args.profile}")
    print(f"Self references: {len(bank)}")
    print(f"Negative WAV files: {len(negative_files)}")
    print(f"Top-K: {top_k}")

    for path in negative_files:
        audio, sample_rate = load_audio_mono(path)
        duration = len(audio) / sample_rate

        if duration < MIN_DURATION or duration > MAX_DURATION:
            print(f"[SKIP] {path.name} {duration:.2f}s")
            continue

        embedding = embed_audio(
            backend,
            audio,
            sample_rate,
            embedding_sample_rate,
        )
        score = top_k_score(embedding, bank, top_k)

        negative_scores.append(score)
        valid_files.append(path)
        print(
            f"{score:.4f}  "
            f"{path.relative_to(negative_root)}"
        )

    if not negative_scores:
        raise RuntimeError("No valid negative samples.")

    negative_scores = np.asarray(negative_scores, dtype=np.float32)

    review_threshold, review_recall, review_fpr = (
        find_balanced_threshold(positive_scores, negative_scores)
    )
    accept_threshold, accept_recall = (
        find_zero_false_accept_threshold(
            positive_scores,
            negative_scores,
        )
    )

    payload = {
        "profile": args.profile,
        "reference_bank": str(reference_path),
        "negative_source": str(negative_root),
        "top_k": top_k,
        "accept_threshold": accept_threshold,
        "review_threshold": review_threshold,
        "positive": {
            "count": int(len(positive_scores)),
            "min": float(positive_scores.min()),
            "median": float(np.median(positive_scores)),
            "max": float(positive_scores.max()),
            "accept_recall": accept_recall,
            "review_recall": review_recall,
        },
        "negative": {
            "count": int(len(negative_scores)),
            "min": float(negative_scores.min()),
            "median": float(np.median(negative_scores)),
            "max": float(negative_scores.max()),
            "review_accept_rate": review_fpr,
            "observed_accept_false_accepts": int(
                np.sum(negative_scores >= accept_threshold)
            ),
        },
    }

    output_path = profile_dir / "thresholds.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("POSITIVE / SELF")
    print(f"count  : {len(positive_scores)}")
    print(f"min    : {positive_scores.min():.4f}")
    print(f"median : {np.median(positive_scores):.4f}")
    print(f"max    : {positive_scores.max():.4f}")

    print("\nNEGATIVE / OTHER")
    print(f"count  : {len(negative_scores)}")
    print(f"min    : {negative_scores.min():.4f}")
    print(f"median : {np.median(negative_scores):.4f}")
    print(f"max    : {negative_scores.max():.4f}")

    print("\nRECOMMENDED THRESHOLDS")
    print(f"accept >= {accept_threshold:.4f}")
    print("  observed negative false accepts: 0")
    print(f"  self recall: {accept_recall:.1%}")
    print(f"review >= {review_threshold:.4f}")
    print(f"  self recall: {review_recall:.1%}")
    print(
        "  observed negative accept rate: "
        f"{review_fpr:.1%}"
    )
    print("=" * 60)

    order = np.argsort(negative_scores)[::-1]
    print("\nMost self-like negative samples:")
    for index in order[:10]:
        print(
            f"{negative_scores[index]:.4f}  "
            f"{valid_files[index]}"
        )

    print(f"Saved thresholds: {output_path}")


if __name__ == "__main__":
    main()
