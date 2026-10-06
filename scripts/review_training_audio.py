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
    destination: Path,
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
                "destination": str(destination.resolve()),
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


def pending_candidates(
    runs: list[Path],
) -> list[tuple[Path, Path, str]]:
    pending: list[tuple[Path, Path, str]] = []

    for run_path in runs:
        decided = load_decided_sources(run_path)
        for dirname, source_kind in REVIEW_SOURCES:
            for path in collect_wavs(run_path / dirname):
                if str(path.resolve()) in decided:
                    continue
                pending.append((run_path, path, source_kind))

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

    candidates = pending_candidates(runs)
    if not candidates:
        print("No pending review clips.")
        return

    totals = {
        "APPROVE": 0,
        "EMOTION": 0,
        "REJECT": 0,
        "SKIP": 0,
    }

    print(f"Profile : {profile}")
    print(f"Pending : {len(candidates)}")
    print()
    print("Keys:")
    print("  y = approve normal voice")
    print("  e = approve emotional/extreme voice")
    print("  n = reject")
    print("  r = replay")
    print("  s = skip for now")
    print("  q = quit")
    print()

    for index, (run_path, source, source_kind) in enumerate(
        candidates,
        start=1,
    ):
        try:
            info = sf.info(source)
            duration = info.frames / info.samplerate
        except Exception:
            duration = 0.0

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

            choice = input(
                "[y] normal / [e] emotion / [n] reject / "
                "[r] replay / [s] skip / [q] quit > "
            ).strip().lower()

            if choice == "r":
                continue
            if choice == "q":
                print("Review stopped by user.")
                print(
                    "Approved normal: "
                    f"{totals['APPROVE']}, emotion: {totals['EMOTION']}, "
                    f"rejected: {totals['REJECT']}, skipped: {totals['SKIP']}"
                )
                return
            if choice == "s":
                totals["SKIP"] += 1
                break

            decision = {
                "y": "APPROVE",
                "e": "EMOTION",
                "n": "REJECT",
            }.get(choice)
            if decision is None:
                print("Unknown key.")
                continue

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
            print(f"{decision}: {destination}")
            break

    print("\n" + "=" * 64)
    print("REVIEW COMPLETE")
    print(f"APPROVE : {totals['APPROVE']}")
    print(f"EMOTION : {totals['EMOTION']}")
    print(f"REJECT  : {totals['REJECT']}")
    print(f"SKIP    : {totals['SKIP']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
