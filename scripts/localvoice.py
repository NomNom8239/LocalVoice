from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from common import (
    data_dir,
    embed_audio,
    embedding_backend,
    load_audio_mono,
    load_config,
    load_thresholds,
    normalize,
    project_path,
    reference_dir,
    run_dir,
    top_k_score,
)


MIN_RMS = 1e-4
MAX_MERGE_GAP_SECONDS = 0.8


def run(cmd: list[str], *, capture: bool = False) -> str:
    print(f"\n> {subprocess.list2cmdline(cmd)}")
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else ""


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found in PATH: {name}")


def youtube_id(url: str) -> str:
    require_command("yt-dlp")
    output = run(
        [
            "yt-dlp",
            "--no-playlist",
            "--print",
            "%(id)s",
            "--skip-download",
            url,
        ],
        capture=True,
    )
    return output.splitlines()[-1].strip()


def download_youtube(config: dict, url: str, video_id: str) -> Path:
    require_command("yt-dlp")

    directory = data_dir(config, "source") / video_id
    directory.mkdir(parents=True, exist_ok=True)

    existing = [
        p
        for p in directory.glob("source.*")
        if p.is_file()
        and not p.name.endswith(".part")
        and not p.name.endswith(".ytdl")
    ]
    if existing:
        print(f"Reusing source: {existing[0]}")
        return existing[0]

    run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-o",
            str(directory / "source.%(ext)s"),
            url,
        ]
    )

    files = [
        p
        for p in directory.glob("source.*")
        if p.is_file()
        and not p.name.endswith(".part")
        and not p.name.endswith(".ytdl")
    ]
    if not files:
        raise RuntimeError("yt-dlp completed but no source audio was found.")

    return files[0]


def make_wav(config: dict, source: Path, job_id: str) -> Path:
    require_command("ffmpeg")

    directory = data_dir(config, "wav_master")
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{job_id}.wav"

    if output.exists():
        print(f"Reusing WAV master: {output}")
        return output

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    return output


def separate_vocals(config: dict, source: Path, directory: Path) -> Path:
    require_command("audio-separator")
    output_dir = directory / "separated"
    output_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            "audio-separator",
            str(source),
            "--model_filename",
            config["models"]["separator"],
            "--output_format",
            "WAV",
            "--output_dir",
            str(output_dir),
        ]
    )

    candidates = sorted(
        [
            p
            for p in output_dir.rglob("*.wav")
            if "vocal" in p.name.lower()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"No Vocals WAV found under: {output_dir}"
        )

    print(f"Vocals: {candidates[0]}")
    return candidates[0]


def has_overlap(
    start: float,
    end: float,
    speaker: str,
    turns: list[tuple[float, float, str]],
) -> bool:
    for other_start, other_end, other_speaker in turns:
        if other_speaker == speaker:
            continue
        if max(start, other_start) < min(end, other_end):
            return True
    return False


def merge_diarization_turns(
    turns: list[tuple[float, float, str]],
    *,
    max_gap: float,
    max_duration: float,
) -> tuple[list[tuple[float, float, str, int]], int]:
    """Merge nearby turns from the same speaker without crossing another speaker.

    Turns that overlap a different speaker are excluded and act as hard
    boundaries. A merged segment never exceeds max_duration.
    """
    ordered = sorted(turns, key=lambda item: (item[0], item[1], item[2]))
    segments: list[tuple[float, float, str, int]] = []
    current: tuple[float, float, str, int] | None = None
    overlap_skipped = 0

    for start, end, speaker in ordered:
        if end <= start:
            continue

        if has_overlap(start, end, speaker, ordered):
            overlap_skipped += 1
            if current is not None:
                segments.append(current)
                current = None
            continue

        if current is None:
            current = (start, end, speaker, 1)
            continue

        current_start, current_end, current_speaker, source_turns = current
        gap = start - current_end
        merged_end = max(current_end, end)
        merged_duration = merged_end - current_start

        if (
            speaker == current_speaker
            and gap <= max_gap
            and merged_duration <= max_duration
        ):
            current = (
                current_start,
                merged_end,
                current_speaker,
                source_turns + 1,
            )
            continue

        segments.append(current)
        current = (start, end, speaker, 1)

    if current is not None:
        segments.append(current)

    return segments, overlap_skipped



BOOTSTRAP_MIN_DURATION = 2.0
BOOTSTRAP_MAX_DURATION = 20.0
BOOTSTRAP_MAX_REFERENCE_CLIPS = 24
BOOTSTRAP_PREVIEWS_PER_SPEAKER = 3
BOOTSTRAP_MAX_CLIPPING_RATIO = 0.005


def diarize_turns(
    config: dict, vocals: Path, audio: np.ndarray, sample_rate: int,
    device: torch.device,
) -> list[tuple[float, float, str]]:
    """Run diarization once; bootstrap and final classification share the turns."""
    pipeline = Pipeline.from_pretrained(
        config["models"]["pyannote_pipeline"], token=True,
    )
    pipeline.to(device)
    print(f"Device: {device}")
    print("Running diarization...")
    with ProgressHook() as hook:
        result = pipeline(
            {
                "waveform": torch.from_numpy(audio.reshape(1, -1).copy()),
                "sample_rate": sample_rate,
                "uri": vocals.stem,
            },
            hook=hook,
        )
    return [
        (float(turn.start), float(turn.end), speaker)
        for turn, speaker in result.speaker_diarization
    ]


def spaced_samples(items: list, count: int) -> list:
    """Pick across the timeline instead of just the beginning of an archive."""
    if len(items) <= count:
        return list(items)
    if count <= 1:
        return [items[len(items) // 2]]
    return [
        items[round(i * (len(items) - 1) / (count - 1))]
        for i in range(count)
    ]


def reference_candidates(
    audio: np.ndarray, sample_rate: int,
    turns: list[tuple[float, float, str]],
) -> dict[str, list[tuple[float, float, str]]]:
    """Select speech-sized non-overlapping turns. This is QC, NOT identity proof."""
    candidates: dict[str, list[tuple[float, float, str]]] = {}
    for start, end, speaker in sorted(turns):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", speaker):
            raise ValueError(f"Unsafe diarized speaker label: {speaker!r}")
        if not (np.isfinite(start) and np.isfinite(end)):
            continue
        duration = end - start
        if not BOOTSTRAP_MIN_DURATION <= duration <= BOOTSTRAP_MAX_DURATION:
            continue
        if any(
            other != speaker and a < end and b > start
            for a, b, other in turns
        ):
            continue
        begin = max(0, int(start * sample_rate))
        finish = min(len(audio), int(end * sample_rate))
        clip = audio[begin:finish]
        if len(clip) < BOOTSTRAP_MIN_DURATION * sample_rate:
            continue
        rms = float(np.sqrt(np.mean(np.square(clip))))
        if not np.isfinite(rms) or rms < MIN_RMS:
            continue
        clipped = float(np.mean(np.abs(clip) >= 0.999))
        if clipped > BOOTSTRAP_MAX_CLIPPING_RATIO:
            continue
        candidates.setdefault(speaker, []).append((start, end, speaker))
    return candidates


def write_bootstrap_clip(
    audio: np.ndarray, sample_rate: int,
    turn: tuple[float, float, str], path: Path,
) -> None:
    start, end, _ = turn
    begin = max(0, int(start * sample_rate))
    finish = min(len(audio), int(end * sample_rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite bootstrap WAV: {path}")
    sf.write(path, audio[begin:finish], sample_rate, subtype="PCM_16")


def choose_bootstrap_speakers(
    candidates: dict[str, list[tuple[float, float, str]]],
    previews: dict[str, list[Path]],
) -> list[str]:
    """Human identifies which pyannote labels belong to the target person."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "New-speaker bootstrap requires an interactive terminal. "
            "Preview WAVs are under the run's bootstrap/previews directory. "
            "Run this command interactively with a new --run-name; "
            "no Reference Bank was created."
        )
    labels = set(candidates)
    print("\nReference Bank bootstrap: identify the TARGET speaker.")
    print("Pyannote labels are anonymous; the same person may have multiple labels.")
    print("Type 'p SPEAKER_00' to play previews, or comma-separated labels to select.")
    print("If no label is reliably the target, type q. Other labels are NOT negatives.")
    while True:
        raw = input("Target speaker label(s) [p <label> / labels / q]: ").strip()
        if raw.lower() in {"q", "quit"}:
            raise RuntimeError("Bootstrap canceled: no Reference Bank was created.")
        if raw.lower().startswith("p "):
            label = raw[2:].strip()
            if label not in previews:
                print("Unknown label:", label)
                continue
            if shutil.which("ffplay") is None:
                print("ffplay is unavailable; open these WAV files manually:")
                for wav in previews[label]:
                    print(" ", wav)
            else:
                for wav in previews[label]:
                    try:
                        run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(wav)])
                    except subprocess.CalledProcessError:
                        print(f"ffplay stopped for {wav}; choose another preview or speaker.")
            continue
        selected = list(dict.fromkeys(v.strip() for v in raw.split(",") if v.strip()))
        if not selected or any(label not in labels for label in selected):
            print("Select one or more displayed labels, or q to cancel.")
            continue
        if sum(len(candidates[label]) for label in selected) < 3:
            print("Fewer than three eligible reference clips; select another archive.")
            continue
        print("Target labels:", ", ".join(selected))
        print("Only select multiple labels if you heard the SAME person in each.")
        if input("Confirm these are the target person's voice [type YES]: ").strip() == "YES":
            return selected
        print("Not confirmed. Review the previews or cancel.")


def bootstrap_reference_bank(
    config: dict, profile: str, vocals: Path, output_dir: Path,
) -> list[tuple[float, float, str]]:
    """Bootstrap a new profile after separation, with one human speaker decision."""
    ref_root = reference_dir(config, profile)
    bank_path = ref_root / "self_reference_bank.npz"
    report_path = ref_root / "self_reference_bank.json"
    if bank_path.exists() or report_path.exists():
        raise RuntimeError(
            f"Reference Bank already exists or is incomplete: {ref_root}. "
            "Refusing to overwrite; inspect it before retrying."
        )
    audio, sample_rate = load_audio_mono(vocals)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    turns = diarize_turns(config, vocals, audio, sample_rate, device)
    candidates = reference_candidates(audio, sample_rate, turns)
    if not candidates:
        raise RuntimeError(
            "No non-overlapping 2–20 second speech turns qualified for bootstrap. "
            "Use a longer/cleaner archive or prepare reference clips manually."
        )
    previews: dict[str, list[Path]] = {}
    for speaker, valid in sorted(candidates.items()):
        print(f"Speaker {speaker}: {len(valid)} qualifying speech clips")
        wavs = []
        for index, turn in enumerate(
            spaced_samples(valid, BOOTSTRAP_PREVIEWS_PER_SPEAKER), 1
        ):
            path = output_dir / "bootstrap" / "previews" / f"{speaker}_{index:02d}.wav"
            write_bootstrap_clip(audio, sample_rate, turn, path)
            wavs.append(path)
            print("  preview:", path)
        previews[speaker] = wavs

    selected = choose_bootstrap_speakers(candidates, previews)
    pool = sorted(
        [turn for label in selected for turn in candidates[label]]
    )
    chosen = spaced_samples(pool, BOOTSTRAP_MAX_REFERENCE_CLIPS)
    if len(chosen) < 3:
        raise RuntimeError("Not enough eligible reference WAVs to build the bank.")
    stage = output_dir / "bootstrap" / "selected_reference_wavs"
    for index, turn in enumerate(chosen, 1):
        write_bootstrap_clip(
            audio, sample_rate, turn, stage / f"ref_{index:03d}.wav"
        )
    print(f"Selected {len(chosen)} reference WAVs in {stage}.")
    print("These are PROVISIONAL references, not a guarantee of speaker identity.")
    if bank_path.exists() or report_path.exists():
        raise RuntimeError("Concurrent or incomplete Reference Bank detected; refusing overwrite.")
    run([
        sys.executable, str(Path(__file__).with_name("build_reference_bank.py")),
        "--profile", profile, "--source", str(stage),
    ])
    if not bank_path.is_file() or not report_path.is_file():
        raise RuntimeError("Reference Bank subprocess ended without complete output.")
    print("Reference Bank created. Inspect 'Lowest similarity clips' for mistakes.")
    print("Thresholds use config.toml defaults until verified negatives are provided.")
    return turns


def classify(
    config: dict,
    profile: str,
    vocals: Path,
    output_dir: Path,
    accept_threshold: float,
    review_threshold: float,
    top_k: int,
    precomputed_turns: list[tuple[float, float, str]] | None = None,
) -> None:
    reference_path = reference_dir(
        config,
        profile,
    ) / "self_reference_bank.npz"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    data = np.load(reference_path)
    bank = np.asarray(data["embeddings"], dtype=np.float32)
    bank = np.stack([normalize(row) for row in bank])

    min_duration = float(
        config["classification"]["min_score_duration"]
    )
    max_duration = float(
        config["classification"]["max_score_duration"]
    )

    audio, sample_rate = load_audio_mono(vocals)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if precomputed_turns is None:
        turns = diarize_turns(config, vocals, audio, sample_rate, device)
    else:
        turns = precomputed_turns
        print("Reusing diarization from Reference Bank bootstrap.")

    segments, overlap_skipped = merge_diarization_turns(
        turns,
        max_gap=MAX_MERGE_GAP_SECONDS,
        max_duration=max_duration,
    )

    backend, embedding_sample_rate = embedding_backend(config, device)

    destinations = {
        "SELF": output_dir / "my_voice",
        "REVIEW": output_dir / "review",
        "OTHER": output_dir / "rejected",
        "UNSCORED": output_dir / "review_unscored",
    }
    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)

    # Manual review destinations. Nothing is copied here automatically.
    for dirname in (
        "review_approved",
        "review_emotion",
        "review_rejected",
    ):
        (output_dir / dirname).mkdir(parents=True, exist_ok=True)

    counts = {key: 0 for key in destinations}
    silent_skipped = 0
    merged_segments = sum(1 for *_, source_turns in segments if source_turns > 1)

    manifest = output_dir / "classification.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "speaker",
                "start",
                "end",
                "duration",
                "source_turns",
                "score",
                "classification",
                "file",
            ]
        )

        written = 0

        for start, end, speaker, source_turns in segments:
            duration = end - start

            start_sample = max(0, int(start * sample_rate))
            end_sample = min(len(audio), int(end * sample_rate))
            clip = audio[start_sample:end_sample]
            if len(clip) == 0:
                continue

            rms = float(np.sqrt(np.mean(np.square(clip))))
            if not np.isfinite(rms) or rms < MIN_RMS:
                silent_skipped += 1
                continue

            written += 1
            filename = (
                f"{written:05d}_{speaker}_"
                f"{start:010.3f}-{end:010.3f}.wav"
            )

            if duration < min_duration or duration > max_duration:
                classification = "UNSCORED"
                score_text = ""
            else:
                embedding = embed_audio(
                    backend,
                    clip,
                    sample_rate,
                    embedding_sample_rate,
                )
                score = top_k_score(embedding, bank, top_k)
                score_text = f"{score:.4f}"

                if score >= accept_threshold:
                    classification = "SELF"
                elif score >= review_threshold:
                    classification = "REVIEW"
                else:
                    classification = "OTHER"

            counts[classification] += 1
            output_path = destinations[classification] / filename

            sf.write(
                output_path,
                clip,
                sample_rate,
                subtype="PCM_16",
            )

            writer.writerow(
                [
                    speaker,
                    f"{start:.3f}",
                    f"{end:.3f}",
                    f"{duration:.3f}",
                    str(source_turns),
                    score_text,
                    classification,
                    str(output_path),
                ]
            )

            print(
                f"{classification:8s} "
                f"{score_text:>6s}  "
                f"{speaker} {start:.3f}-{end:.3f} "
                f"(turns={source_turns})"
            )

    print("\n" + "=" * 60)
    print("Classification complete")
    for key, value in counts.items():
        print(f"{key:16s}: {value}")
    print(f"{'RAW_TURNS':16s}: {len(turns)}")
    print(f"{'SEGMENTS':16s}: {len(segments)}")
    print(f"{'MERGED_SEGMENTS':16s}: {merged_segments}")
    print(f"{'OVERLAP_SKIPPED':16s}: {overlap_skipped}")
    print(f"{'SILENT_SKIPPED':16s}: {silent_skipped}")
    print("=" * 60)
    print(f"Run directory: {output_dir}")
    print(
        "Reference bank is not updated automatically. "
        "Review uncertain clips with scripts/review_training_audio.py "
        "before collecting training audio."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a target speaker from a supported archive URL, source audio, "
            "or Vocals WAV. If the profile has no Reference Bank, use an "
            "interactive first-speaker bootstrap after audio separation."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--wav")
    source.add_argument("--vocals")

    parser.add_argument("--run-name")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--accept-threshold", type=float)
    parser.add_argument("--review-threshold", type=float)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    config = load_config(args.config)

    # Detect a partially-written Bank before any expensive URL/download/separation.
    initial_ref_root = reference_dir(config, args.profile)
    initial_npz = initial_ref_root / "self_reference_bank.npz"
    initial_json = initial_ref_root / "self_reference_bank.json"
    if initial_npz.exists() != initial_json.exists():
        raise RuntimeError(
            f"Incomplete Reference Bank at {initial_ref_root}: "
            "NPZ/JSON must both exist. Inspect rather than overwriting."
        )

    default_accept, default_review, default_top_k = load_thresholds(
        config,
        args.profile,
    )
    accept_threshold = (
        args.accept_threshold
        if args.accept_threshold is not None
        else default_accept
    )
    review_threshold = (
        args.review_threshold
        if args.review_threshold is not None
        else default_review
    )
    top_k = args.top_k if args.top_k is not None else default_top_k

    if review_threshold >= accept_threshold:
        raise ValueError(
            "review threshold must be lower than accept threshold"
        )
    if top_k < 1:
        raise ValueError("--top-k must be >= 1")

    if args.url:
        job_id = youtube_id(args.url)
    elif args.wav:
        job_id = project_path(args.wav).stem
    else:
        job_id = project_path(args.vocals).stem

    name = args.run_name or job_id
    output_dir = run_dir(config, args.profile) / name

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Run directory already exists: {output_dir}\n"
            "Use --force or choose --run-name."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        downloaded = download_youtube(config, args.url, job_id)
        master = make_wav(config, downloaded, job_id)
        vocals = separate_vocals(config, master, output_dir)
    elif args.wav:
        master = project_path(args.wav)
        if not master.exists():
            raise FileNotFoundError(master)
        vocals = separate_vocals(config, master, output_dir)
    else:
        vocals = project_path(args.vocals)
        if not vocals.exists():
            raise FileNotFoundError(vocals)

    reference_root = reference_dir(config, args.profile)
    reference_file = reference_root / "self_reference_bank.npz"
    reference_report = reference_root / "self_reference_bank.json"
    if reference_file.exists() != reference_report.exists():
        raise RuntimeError(
            f"Incomplete Reference Bank at {reference_root}: "
            "NPZ/JSON must both exist. Inspect rather than overwriting."
        )
    bootstrap_turns = None
    if not reference_file.is_file():
        print("Reference Bank missing: starting first-speaker bootstrap.")
        bootstrap_turns = bootstrap_reference_bank(
            config, args.profile, vocals, output_dir,
        )

    classify(
        config,
        args.profile,
        vocals,
        output_dir,
        accept_threshold,
        review_threshold,
        top_k,
        precomputed_turns=bootstrap_turns,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except subprocess.CalledProcessError as exc:
        print(
            f"External command failed with exit code {exc.returncode}.",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode)
