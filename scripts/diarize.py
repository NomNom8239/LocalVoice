from __future__ import annotations

import argparse
import csv

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from common import data_dir, load_config, project_path


def overlaps_other(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diarize a Vocals WAV and export clean speaker clips."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--min-duration", type=float, default=1.5)
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = project_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.min_duration <= 0:
        raise ValueError("--min-duration must be > 0")

    output_dir = (
        project_path(args.output)
        if args.output
        else data_dir(config, "diarization") / input_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    audio, sample_rate = sf.read(
        input_path,
        dtype="float32",
        always_2d=True,
    )

    audio_input = {
        "waveform": torch.from_numpy(audio.T.copy()),
        "sample_rate": int(sample_rate),
        "uri": input_path.stem,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.from_pretrained(
        config["models"]["pyannote_pipeline"],
        token=True,
    )
    pipeline.to(device)

    print(f"Device: {device}")
    print("Running diarization...")

    with ProgressHook() as hook:
        result = pipeline(audio_input, hook=hook)

    turns = [
        (float(turn.start), float(turn.end), speaker)
        for turn, speaker in result.speaker_diarization
    ]
    speakers = sorted({speaker for _, _, speaker in turns})
    counts = {speaker: 0 for speaker in speakers}
    durations = {speaker: 0.0 for speaker in speakers}

    manifest = output_dir / "segments.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["speaker", "start", "end", "duration", "file", "classification"]
        )

        for start, end, speaker in turns:
            duration = end - start

            if duration < args.min_duration:
                writer.writerow(
                    [
                        speaker,
                        f"{start:.3f}",
                        f"{end:.3f}",
                        f"{duration:.3f}",
                        "",
                        "SHORT_SKIPPED",
                    ]
                )
                continue

            if overlaps_other(start, end, speaker, turns):
                writer.writerow(
                    [
                        speaker,
                        f"{start:.3f}",
                        f"{end:.3f}",
                        f"{duration:.3f}",
                        "",
                        "OVERLAP_SKIPPED",
                    ]
                )
                continue

            start_sample = max(0, int(start * sample_rate))
            end_sample = min(len(audio), int(end * sample_rate))
            clip = audio[start_sample:end_sample]
            if len(clip) == 0:
                continue

            speaker_dir = output_dir / speaker
            speaker_dir.mkdir(parents=True, exist_ok=True)

            counts[speaker] += 1
            durations[speaker] += duration

            filename = (
                f"{counts[speaker]:04d}_"
                f"{start:09.3f}-{end:09.3f}.wav"
            )
            output_path = speaker_dir / filename

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
                    str(output_path),
                    "EXPORTED",
                ]
            )

    print(f"Detected speakers: {speakers}")
    for speaker in speakers:
        print(
            f"{speaker}: {counts[speaker]} clips / "
            f"{durations[speaker]:.1f} sec"
        )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
