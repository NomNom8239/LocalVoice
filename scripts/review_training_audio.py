from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from common import load_config, profile_name, run_dir


DECISION_FILE = "review_decisions.tsv"
REVIEW_SOURCES = (
    ("review", "REVIEW"),
    ("review_unscored", "UNSCORED"),
)
DECISION_DESTINATIONS = {
    "APPROVE": "review_approved",
    "EMOTION": "review_emotion",
    "REJECT": "review_rejected",
}
SOURCE_KIND_FILTERS = {
    "all": None,
    "review": {"REVIEW"},
    "unscored": {"UNSCORED"},
}


def collect_wavs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*.wav") if p.is_file())


def discover_runs(config: dict, profile: str) -> list[Path]:
    root = run_dir(config, profile)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def load_decided_sources(run_path: Path) -> set[str]:
    path = run_path / DECISION_FILE
    if not path.exists():
        return set()

    decided: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source = (row.get("source") or "").strip()
            if source:
                decided.add(source)
    return decided


def append_decision(
    run_path: Path,
    *,
    source: Path,
    source_kind: str,
    decision: str,
    destination: Path | None,
) -> None:
    path = run_path / DECISION_FILE
    fieldnames = [
        "decided_at_utc",
        "source",
        "source_kind",
        "decision",
        "destination",
    ]
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "decided_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(source.resolve()),
                "source_kind": source_kind,
                "decision": decision,
                "destination": (
                    str(destination.resolve())
                    if destination is not None
                    else ""
                ),
            }
        )


def play_audio(player: str, path: Path) -> None:
    subprocess.run(
        [
            player,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "error",
            str(path),
        ],
        check=False,
    )


def audio_duration(path: Path) -> float:
    info = sf.info(path)
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sample rate: {path}")
    return info.frames / info.samplerate


def pending_candidates(
    runs: list[Path],
    *,
    run_name: str | None,
    source_kind_filter: set[str] | None,
    min_duration: float | None,
    max_duration: float | None,
    limit: int | None,
) -> list[tuple[Path, Path, str, float]]:
    pending: list[tuple[Path, Path, str, float]] = []

    for run_path in runs:
        if run_name and run_path.name != run_name:
            continue

        decided = load_decided_sources(run_path)
        for dirname, source_kind in REVIEW_SOURCES:
            if (
                source_kind_filter is not None
                and source_kind not in source_kind_filter
            ):
                continue

            for path in collect_wavs(run_path / dirname):
                if str(path.resolve()) in decided:
                    continue
                try:
                    duration = audio_duration(path)
                except Exception as exc:
                    print(f"[SKIP unreadable] {path}: {exc}")
                    continue
                if min_duration is not None and duration < min_duration:
                    continue
                if max_duration is not None and duration > max_duration:
                    continue
                pending.append((run_path, path, source_kind, duration))

    pending.sort(key=lambda item: (item[0].name, item[1].name))
    if limit is not None:
        pending = pending[:limit]
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively review uncertain LocalVoice clips. "
            "Approved normal clips go to review_approved/, approved "
            "extreme/emotional clips go to review_emotion/, and rejects "
            "go to review_rejected/. Originals are preserved."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--player",
        default="ffplay",
        help="Audio player command. Defaults to ffplay.",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not launch audio playback automatically.",
    )
    parser.add_argument(
        "--run",
        help="Review only one run directory name.",
    )
    parser.add_argument(
        "--kind",
        choices=sorted(SOURCE_KIND_FILTERS),
        default="all",
        help="Review all uncertain clips, REVIEW only, or UNSCORED only.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        help="Skip clips shorter than this many seconds.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        help="Skip clips longer than this many seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Review at most this many matching pending clips.",
    )
    parser.add_argument(
        "--emotion-only",
        action="store_true",
        help=(
            "Emotion harvesting mode: e=keep as emotion, "
            "n=mark as not-emotion without adding it to training audio."
        ),
    )
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    profile = profile_name(args.profile)
    runs = discover_runs(config, profile)
    if not runs:
        raise RuntimeError(
            f"No processed runs found for profile {profile!r}."
        )

    if not args.no_play and shutil.which(args.player) is None:
        raise RuntimeError(
            f"Audio player not found in PATH: {args.player}. "
            "Install ffplay or use --no-play/--player."
        )

    if args.min_duration is not None and args.min_duration < 0:
        raise ValueError("--min-duration must be >= 0")
    if args.max_duration is not None and args.max_duration <= 0:
        raise ValueError("--max-duration must be > 0")
    if (
        args.min_duration is not None
        and args.max_duration is not None
        and args.min_duration > args.max_duration
    ):
        raise ValueError("--min-duration must be <= --max-duration")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")

    candidates = pending_candidates(
        runs,
        run_name=args.run,
        source_kind_filter=SOURCE_KIND_FILTERS[args.kind],
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        limit=args.limit,
    )
    if not candidates:
        print("No pending review clips matched the filters.")
        return

    totals = {
        "APPROVE": 0,
        "EMOTION": 0,
        "REJECT": 0,
        "NOT_EMOTION": 0,
        "SKIP": 0,
    }

    print(f"Profile : {profile}")
    if args.run:
        print(f"Run     : {args.run}")
    print(f"Kind    : {args.kind}")
    print(f"Pending : {len(candidates)}")
    print()
    print("Keys:")
    if args.emotion_only:
        print("  e = keep emotional/extreme voice")
        print("  n = not emotion (mark reviewed, do not collect)")
    else:
        print("  y = approve normal voice")
        print("  e = approve emotional/extreme voice")
        print("  n = reject")
    print("  r = replay")
    print("  s = skip for now")
    print("  q = quit")
    print()

    for index, (run_path, source, source_kind, duration) in enumerate(
        candidates,
        start=1,
    ):

        print("-" * 72)
        print(
            f"[{index}/{len(candidates)}] "
            f"run={run_path.name} source={source_kind} "
            f"duration={duration:.2f}s"
        )
        print(source)

        while True:
            if not args.no_play:
                play_audio(args.player, source)

            prompt = (
                "[e] emotion / [n] not-emotion / "
                "[r] replay / [s] skip / [q] quit > "
                if args.emotion_only
                else
                "[y] normal / [e] emotion / [n] reject / "
                "[r] replay / [s] skip / [q] quit > "
            )
            choice = input(prompt).strip().lower()

            if choice == "r":
                continue
            if choice == "q":
                print("Review stopped by user.")
                print(
                    "Approved normal: "
                    f"{totals['APPROVE']}, emotion: {totals['EMOTION']}, "
                    f"rejected: {totals['REJECT']}, "
                    f"not-emotion: {totals['NOT_EMOTION']}, "
                    f"skipped: {totals['SKIP']}"
                )
                return
            if choice == "s":
                totals["SKIP"] += 1
                break

            if args.emotion_only:
                decision = {
                    "e": "EMOTION",
                    "n": "NOT_EMOTION",
                }.get(choice)
            else:
                decision = {
                    "y": "APPROVE",
                    "e": "EMOTION",
                    "n": "REJECT",
                }.get(choice)
            if decision is None:
                print("Unknown key.")
                continue

            destination = None
            if decision in DECISION_DESTINATIONS:
                destination_dir = run_path / DECISION_DESTINATIONS[decision]
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / source.name
                shutil.copy2(source, destination)

            append_decision(
                run_path,
                source=source,
                source_kind=source_kind,
                decision=decision,
                destination=destination,
            )
            totals[decision] += 1
            if destination is None:
                print(f"{decision}: marked reviewed")
            else:
                print(f"{decision}: {destination}")
            break

    print("\n" + "=" * 64)
    print("REVIEW COMPLETE")
    print(f"APPROVE : {totals['APPROVE']}")
    print(f"EMOTION : {totals['EMOTION']}")
    print(f"REJECT      : {totals['REJECT']}")
    print(f"NOT_EMOTION : {totals['NOT_EMOTION']}")
    print(f"SKIP        : {totals['SKIP']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
