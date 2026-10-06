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
    project_path,
    reference_dir,
)


MIN_DURATION = 2.0
MAX_DURATION = 20.0
MIN_RMS = 1e-4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a target-speaker embedding reference bank."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    source_root = project_path(args.source)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    files = sorted(
        p for p in source_root.rglob("*.wav")
        if p.is_file()
    )
    if not files:
        raise RuntimeError(f"No WAV files found under: {source_root}")

    output_dir = reference_dir(config, args.profile)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, embedding_sample_rate = embedding_backend(config, device)

    embeddings: list[np.ndarray] = []
    clips: list[dict] = []

    print(f"Device: {device}")
    print(f"Source WAV files: {len(files)}")

    for index, path in enumerate(files, start=1):
        audio, sample_rate = load_audio_mono(path)
        duration = len(audio) / sample_rate

        if duration < MIN_DURATION:
            print(f"[SKIP short] {path.name} ({duration:.2f}s)")
            continue
        if duration > MAX_DURATION:
            print(f"[SKIP long] {path.name} ({duration:.2f}s)")
            continue

        rms = float(np.sqrt(np.mean(np.square(audio))))
        if not np.isfinite(rms) or rms < MIN_RMS:
            print(f"[SKIP silent] {path.name}")
            continue

        embedding = embed_audio(
            backend,
            audio,
            sample_rate,
            embedding_sample_rate,
        )
        embeddings.append(embedding)
        clips.append(
            {
                "file": str(path),
                "relative_file": str(path.relative_to(source_root)),
                "duration_sec": duration,
                "rms": rms,
            }
        )

        print(
            f"[{index:03d}/{len(files):03d}] "
            f"{path.relative_to(source_root)} {duration:.2f}s"
        )

    if len(embeddings) < 3:
        raise RuntimeError(
            f"Too few valid embeddings: {len(embeddings)}"
        )

    bank = np.stack(embeddings).astype(np.float32)
    similarity = bank @ bank.T

    mean_scores = []
    median_scores = []

    for i in range(len(bank)):
        others = np.delete(similarity[i], i)
        mean_scores.append(float(np.mean(others)))
        median_scores.append(float(np.median(others)))

    mean_scores = np.asarray(mean_scores, dtype=np.float32)
    median_scores = np.asarray(median_scores, dtype=np.float32)

    prototype = np.mean(bank, axis=0)
    prototype /= np.linalg.norm(prototype)
    prototype_scores = bank @ prototype

    for i, clip in enumerate(clips):
        clip["mean_similarity_to_others"] = float(mean_scores[i])
        clip["median_similarity_to_others"] = float(median_scores[i])
        clip["similarity_to_prototype"] = float(prototype_scores[i])

    npz_path = output_dir / "self_reference_bank.npz"
    np.savez_compressed(
        npz_path,
        embeddings=bank,
        prototype=prototype.astype(np.float32),
        mean_similarities=mean_scores,
        median_similarities=median_scores,
        prototype_similarities=prototype_scores,
    )

    report = {
        "profile": args.profile,
        "source": str(source_root),
        "model": config["models"]["pyannote_pipeline"],
        "embedding_sample_rate": embedding_sample_rate,
        "reference_count": len(bank),
        "embedding_dimension": int(bank.shape[1]),
        "clips": clips,
    }

    json_path = output_dir / "self_reference_bank.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    order = np.argsort(prototype_scores)
    print("\nLowest similarity clips:")
    for i in order[:10]:
        print(
            f"{prototype_scores[i]:.4f}  "
            f"{clips[i]['relative_file']}"
        )

    print("\nPrototype similarity:")
    print(f"  min    = {prototype_scores.min():.4f}")
    print(f"  median = {np.median(prototype_scores):.4f}")
    print(f"  max    = {prototype_scores.max():.4f}")
    print(f"NPZ : {npz_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
