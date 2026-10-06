from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from common import (
    load_config,
    profile_name,
    run_dir,
    training_audio_dir,
)


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audio_fingerprint(path: Path) -> tuple[str, dict[str, int | float]]:
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0:
        raise ValueError("empty audio")
    if not np.all(np.isfinite(audio)):
        raise ValueError("audio contains NaN or Inf")

    canonical = np.ascontiguousarray(audio.astype("<f4", copy=False))
    frames, channels = canonical.shape

    digest = hashlib.sha256()
    digest.update(b"LocalVoiceAudioFingerprintV1\0")
    digest.update(int(sample_rate).to_bytes(8, "little", signed=False))
    digest.update(int(frames).to_bytes(8, "little", signed=False))
    digest.update(int(channels).to_bytes(4, "little", signed=False))
    digest.update(canonical.tobytes(order="C"))

    return (
        digest.hexdigest(),
        {
            "sample_rate": int(sample_rate),
            "frames": int(frames),
            "channels": int(channels),
            "duration_sec": float(frames / sample_rate),
        },
    )


def sanitize_name(value: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", value).strip("._-")
    return cleaned[:80] or "clip"


def collect_wavs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*.wav") if p.is_file())


def discover_runs(config: dict, profile: str) -> list[Path]:
    root = run_dir(config, profile)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def collect_candidates(
    runs: list[Path],
) -> list[tuple[Path, str, str]]:
    candidates: list[tuple[Path, str, str]] = []

    for run_path in runs:
        run_name = run_path.name

        for path in collect_wavs(run_path / "my_voice"):
            candidates.append((path, "MY_VOICE", run_name))

        for path in collect_wavs(run_path / "review_approved"):
            candidates.append((path, "REVIEW_APPROVED", run_name))

    return candidates


def existing_fingerprints(audio_dir: Path) -> dict[str, Path]:
    fingerprints: dict[str, Path] = {}
    for path in collect_wavs(audio_dir):
        try:
            fingerprint, _ = audio_fingerprint(path)
        except Exception as exc:
            print(f"[WARN existing] {path}: {exc}")
            continue
        fingerprints.setdefault(fingerprint, path)
    return fingerprints


def append_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return

    fieldnames = [
        "added_at_utc",
        "profile",
        "destination",
        "run",
        "source_kind",
        "source",
        "output",
        "sha256",
        "audio_fingerprint",
        "duration_sec",
        "sample_rate",
        "channels",
    ]
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize_training_audio(audio_dir: Path) -> tuple[int, float, int]:
    files = collect_wavs(audio_dir)
    total_duration = 0.0
    unreadable = 0

    for path in files:
        try:
            _, metadata = audio_fingerprint(path)
            total_duration += float(metadata["duration_sec"])
        except Exception:
            unreadable += 1

    return len(files), total_duration, unreadable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect confirmed target-speaker WAVs from every run for one "
            "profile into data/training_audio/<profile>/audio. "
            "my_voice is collected automatically; review is collected only "
            "after manual approval into review_approved."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--destination",
        help=(
            "Destination name under data/training_audio/. "
            "Defaults to --profile."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without copying or updating manifests.",
    )
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    profile = profile_name(args.profile)
    destination = profile_name(args.destination or profile)

    runs = discover_runs(config, profile)
    if not runs:
        raise RuntimeError(
            "No processed runs found for profile "
            f"{profile!r}. Run scripts/localvoice.py with --url, --wav, "
            "or --vocals first."
        )

    candidates = collect_candidates(runs)
    if not candidates:
        raise RuntimeError(
            "Processed runs were found, but no WAV files exist under "
            "my_voice/ or review_approved/. Check the classification results "
            "and manually approve review clips when needed."
        )

    destination_root = training_audio_dir(config, destination)
    audio_dir = destination_root / "audio"
    manifest_path = destination_root / "manifest.tsv"
    summary_path = destination_root / "summary.json"

    if not args.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)

    known_fingerprints = (
        existing_fingerprints(audio_dir)
        if audio_dir.exists()
        else {}
    )
    known = set(known_fingerprints)

    added_rows: list[dict[str, str]] = []
    added = 0
    duplicate = 0
    failed = 0
    added_duration = 0.0
    by_kind = {
        "MY_VOICE": 0,
        "REVIEW_APPROVED": 0,
    }

    print(f"Profile      : {profile}")
    print(f"Runs scanned : {len(runs)}")
    for run_path in runs:
        print(f"  - {run_path.name}")
    print(f"Destination  : {destination_root}")
    print(f"Candidates   : {len(candidates)}")
    print(f"Known audio  : {len(known)}")
    print()

    for source, source_kind, run_name in candidates:
        try:
            file_hash = sha256_file(source)
            fingerprint, metadata = audio_fingerprint(source)
        except Exception as exc:
            failed += 1
            print(f"FAILED     {source} :: {exc}")
            continue

        if fingerprint in known:
            duplicate += 1
            existing = known_fingerprints.get(fingerprint)
            suffix = f" -> {existing}" if existing else ""
            print(
                f"DUPLICATE  {source_kind:16s} "
                f"{run_name}/{source.name}{suffix}"
            )
            continue

        output_name = (
            f"{sanitize_name(run_name)}__"
            f"{source_kind.lower()}__"
            f"{sanitize_name(source.stem)}__"
            f"{fingerprint[:12]}.wav"
        )
        output = audio_dir / output_name

        if not args.dry_run:
            shutil.copy2(source, output)

        added_rows.append(
            {
                "added_at_utc": datetime.now(timezone.utc).isoformat(),
                "profile": profile,
                "destination": destination,
                "run": run_name,
                "source_kind": source_kind,
                "source": str(source),
                "output": str(output),
                "sha256": file_hash,
                "audio_fingerprint": fingerprint,
                "duration_sec": f"{metadata['duration_sec']:.6f}",
                "sample_rate": str(metadata["sample_rate"]),
                "channels": str(metadata["channels"]),
            }
        )

        known.add(fingerprint)
        known_fingerprints[fingerprint] = output
        added += 1
        by_kind[source_kind] += 1
        added_duration += float(metadata["duration_sec"])

        action = "WOULD ADD" if args.dry_run else "ADD"
        print(
            f"{action:10s} {source_kind:16s} "
            f"{metadata['duration_sec']:6.2f}s  "
            f"{run_name}/{source.name}"
        )

    if not args.dry_run:
        append_manifest(manifest_path, added_rows)

        total_files, total_duration, unreadable = summarize_training_audio(
            audio_dir
        )
        summary = {
            "profile": profile,
            "destination": destination,
            "audio_directory": str(audio_dir),
            "total_files": total_files,
            "total_duration_sec": total_duration,
            "total_duration_min": total_duration / 60.0,
            "unreadable_existing_files": unreadable,
            "last_collection": {
                "runs_scanned": [path.name for path in runs],
                "candidate_count": len(candidates),
                "added_count": added,
                "added_my_voice_count": by_kind["MY_VOICE"],
                "added_review_approved_count": by_kind["REVIEW_APPROVED"],
                "added_duration_sec": added_duration,
                "duplicate_count": duplicate,
                "failed_count": failed,
            },
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 64)
    print("TRAINING AUDIO COLLECTION COMPLETE")
    print(f"RUNS      : {len(runs):4d}")
    print(f"ADD       : {added:4d} clips / {added_duration / 60.0:.2f} min")
    print(f"  my_voice        : {by_kind['MY_VOICE']:4d}")
    print(f"  review_approved : {by_kind['REVIEW_APPROVED']:4d}")
    print(f"DUPLICATE : {duplicate:4d} clips")
    print(f"FAILED    : {failed:4d} clips")
    if args.dry_run:
        print("DRY RUN   : no files were copied")
    else:
        print(f"Audio     : {audio_dir}")
        print(f"Manifest  : {manifest_path}")
        print(f"Summary   : {summary_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
