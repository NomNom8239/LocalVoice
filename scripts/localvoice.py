from __future__ import annotations

import argparse
import csv
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


def classify(
    config: dict,
    profile: str,
    vocals: Path,
    output_dir: Path,
    accept_threshold: float,
    review_threshold: float,
    top_k: int,
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
    pipeline = Pipeline.from_pretrained(
        config["models"]["pyannote_pipeline"],
        token=True,
    )
    pipeline.to(device)

    print(f"Device: {device}")
    print("Running diarization...")

    with ProgressHook() as hook:
        result = pipeline(
            {
                "waveform": torch.from_numpy(
                    audio.reshape(1, -1).copy()
                ),
                "sample_rate": sample_rate,
                "uri": vocals.stem,
            },
            hook=hook,
        )

    turns = [
        (float(turn.start), float(turn.end), speaker)
        for turn, speaker in result.speaker_diarization
    ]

    backend, embedding_sample_rate = embedding_backend(config, device)

    destinations = {
        "SELF": output_dir / "my_voice",
        "REVIEW": output_dir / "review",
        "OTHER": output_dir / "rejected",
        "UNSCORED": output_dir / "review_unscored",
    }
    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)

    # Manual acceptance staging area. Nothing is copied here automatically.
    (output_dir / "review_approved").mkdir(parents=True, exist_ok=True)

    counts = {key: 0 for key in destinations}
    overlap_skipped = 0
    silent_skipped = 0

    manifest = output_dir / "classification.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "speaker",
                "start",
                "end",
                "duration",
                "score",
                "classification",
                "file",
            ]
        )

        written = 0

        for start, end, speaker in turns:
            duration = end - start

            if has_overlap(start, end, speaker, turns):
                overlap_skipped += 1
                continue

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
                    score_text,
                    classification,
                    str(output_path),
                ]
            )

            print(
                f"{classification:8s} "
                f"{score_text:>6s}  "
                f"{speaker} {start:.3f}-{end:.3f}"
            )

    print("\n" + "=" * 60)
    print("Classification complete")
    for key, value in counts.items():
        print(f"{key:16s}: {value}")
    print(f"{'OVERLAP_SKIPPED':16s}: {overlap_skipped}")
    print(f"{'SILENT_SKIPPED':16s}: {silent_skipped}")
    print("=" * 60)
    print(f"Run directory: {output_dir}")
    print(
        "Reference bank is not updated automatically. "
        "Review accepted material before adding it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-command target-speaker extraction from YouTube, "
            "mixed WAV, or a Vocals WAV."
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

    classify(
        config,
        args.profile,
        vocals,
        output_dir,
        accept_threshold,
        review_threshold,
        top_k,
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
