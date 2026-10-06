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
    project_path,
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


def resolve_run_path(
    config: dict,
    profile: str,
    value: str,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        path = candidate.resolve()
    else:
        direct = project_path(candidate)
        if direct.is_dir():
            path = direct
        else:
            path = (run_dir(config, profile) / value).resolve()

    if not path.is_dir():
        profile_runs = run_dir(config, profile)
        available = (
            sorted(p.name for p in profile_runs.iterdir() if p.is_dir())
            if profile_runs.is_dir()
            else []
        )
        detail = (
            "\nAvailable runs: " + ", ".join(available)
            if available
            else "\nNo runs found under: " + str(profile_runs)
        )
        raise NotADirectoryError(
            f"Run directory not found: {path}{detail}"
        )
    return path


def collect_wavs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*.wav") if p.is_file())


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


def candidate_sources(
    config: dict,
    profile: str,
    run_values: list[str],
    source_paths: list[str],
    approved_review_paths: list[str],
) -> list[tuple[Path, str, str]]:
    candidates: list[tuple[Path, str, str]] = []

    for run_value in run_values:
        run_path = resolve_run_path(config, profile, run_value)
        run_name = run_path.name

        my_voice = run_path / "my_voice"
        if not my_voice.is_dir():
            raise NotADirectoryError(
                f"my_voice directory not found: {my_voice}"
            )

        for path in collect_wavs(my_voice):
            candidates.append((path, "MY_VOICE", run_name))

        approved = run_path / "review_approved"
        for path in collect_wavs(approved):
            candidates.append(
                (path, "REVIEW_APPROVED", run_name)
            )

    for value in source_paths:
        root = project_path(value)
        if not root.is_dir():
            raise NotADirectoryError(root)
        label = root.name
        for path in collect_wavs(root):
            candidates.append(
                (path, "CONFIRMED_SOURCE", label)
            )

    for value in approved_review_paths:
        root = project_path(value)
        if not root.is_dir():
            raise NotADirectoryError(root)
        label = root.parent.name or root.name
        for path in collect_wavs(root):
            candidates.append(
                (path, "REVIEW_APPROVED_EXTERNAL", label)
            )

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate confirmed target-speaker WAVs into "
            "data/training_audio/<destination>/audio with "
            "content-based duplicate removal."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Run name under data/runs/<profile>/ or a run directory path. "
            "Repeat for multiple runs."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Directory of already-confirmed target-speaker WAVs to import. "
            "Repeat as needed. Useful for pre-runs data such as "
            "data/reference_bank/<profile>/audio."
        ),
    )
    parser.add_argument(
        "--destination",
        help=(
            "Subdirectory under data/training_audio/. "
            "Defaults to --profile."
        ),
    )
    parser.add_argument(
        "--approved-review",
        action="append",
        default=[],
        help=(
            "Optional directory containing manually approved review WAVs. "
            "Repeat as needed. Each run's review_approved/ is also scanned "
            "automatically when present."
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

    destination_root = training_audio_dir(
        config,
        destination,
    )
    audio_dir = destination_root / "audio"
    manifest_path = destination_root / "manifest.tsv"
    summary_path = destination_root / "summary.json"

    if not args.run and not args.source and not args.approved_review:
        parser.error(
            "at least one of --run, --source, or --approved-review is required"
        )

    candidates = candidate_sources(
        config,
        profile,
        args.run,
        args.source,
        args.approved_review,
    )
    if not candidates:
        raise RuntimeError("No candidate WAV files found.")

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

    print(f"Profile     : {profile}")
    print(f"Destination : {destination_root}")
    print(f"Candidates  : {len(candidates)}")
    print(f"Known audio : {len(known)}")
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
                f"DUPLICATE  {source_kind:24s} "
                f"{source.name}{suffix}"
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

        now = datetime.now(timezone.utc).isoformat()
        added_rows.append(
            {
                "added_at_utc": now,
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
        added_duration += float(metadata["duration_sec"])

        action = "WOULD ADD" if args.dry_run else "ADD"
        print(
            f"{action:10s} {source_kind:24s} "
            f"{metadata['duration_sec']:6.2f}s  {source.name}"
        )

    if not args.dry_run:
        append_manifest(manifest_path, added_rows)

        total_files = collect_wavs(audio_dir)
        total_duration = 0.0
        total_failed = 0
        for path in total_files:
            try:
                _, metadata = audio_fingerprint(path)
                total_duration += float(metadata["duration_sec"])
            except Exception:
                total_failed += 1

        summary = {
            "profile": profile,
            "destination": destination,
            "audio_directory": str(audio_dir),
            "total_files": len(total_files),
            "total_duration_sec": total_duration,
            "total_duration_min": total_duration / 60.0,
            "unreadable_existing_files": total_failed,
            "last_collection": {
                "candidate_count": len(candidates),
                "added_count": added,
                "added_duration_sec": added_duration,
                "duplicate_count": duplicate,
                "failed_count": failed,
                "runs": args.run,
                "source_paths": args.source,
                "approved_review_paths": args.approved_review,
            },
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 64)
    print("TRAINING AUDIO COLLECTION COMPLETE")
    print(f"ADD       : {added:4d} clips / {added_duration / 60.0:.2f} min")
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
