"""Speech-retention ASR probe using the existing isolated .venv-asr.

Invoked as a subprocess by the paired nonverbal evaluation. Never edits
training data or downloads a new Whisper model implicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def transcribe(wavs: list[Path], *, model_factory=None) -> list[dict]:
    if model_factory is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Existing .venv-asr needs faster-whisper") from exc
        model_factory = WhisperModel
    model = model_factory(
        "small", device="cpu", compute_type="int8", local_files_only=True
    )
    result = []
    for path in wavs:
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise FileNotFoundError(f"Speech retention audio not found: {path}")
        segments, info = model.transcribe(
            str(path), language="ja", beam_size=5,
            vad_filter=False, condition_on_previous_text=False,
        )
        text = "".join(s.text.strip() for s in segments).strip()
        result.append({
            "wav": str(path.resolve()), "text": text,
            "language": str(info.language),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs=2)
    args = parser.parse_args()
    try:
        result = transcribe([Path(p).resolve() for p in args.wav])
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
