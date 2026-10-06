from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from common import load_config, project_path, rvc_dir


def dbfs(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return float("-inf")
    return 20.0 * math.log10(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(path: Path, qc: dict) -> dict:
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError("invalid audio metadata")

    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("invalid audio samples")

    mono = np.mean(audio, axis=1, dtype=np.float32)
    absolute = np.abs(mono)

    rms = float(
        np.sqrt(
            np.mean(np.square(mono, dtype=np.float64))
        )
    )
    peak = float(np.max(absolute))
    silence_level = 10.0 ** (
        float(qc["silence_level_dbfs"]) / 20.0
    )

    return {
        "duration_sec": float(info.frames / info.samplerate),
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "clipping_ratio": float(
            np.mean(absolute >= float(qc["clip_level"]))
        ),
        "dc_offset": float(
            abs(np.mean(mono, dtype=np.float64))
        ),
        "silence_ratio": float(
            np.mean(absolute < silence_level)
        ),
    }


def classify(
    metrics: dict,
    qc: dict,
    min_duration: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    duration = metrics["duration_sec"]
    rms = metrics["rms_dbfs"]

    if duration < min_duration:
        return "REJECT", [
            f"duration {duration:.2f}s < {min_duration:.2f}s"
        ]

    if (
        not np.isfinite(rms)
        or rms < float(qc["reject_rms_dbfs"])
    ):
        return "REJECT", [f"effectively silent ({rms:.2f} dBFS)"]

    if rms < float(qc["review_rms_low_dbfs"]):
        reasons.append(f"low RMS ({rms:.2f} dBFS)")
    if rms > float(qc["review_rms_high_dbfs"]):
        reasons.append(f"high RMS ({rms:.2f} dBFS)")
    if metrics["clipping_ratio"] > float(
        qc["review_clipping_ratio"]
    ):
        reasons.append(
            f"possible clipping ({metrics['clipping_ratio']:.2%})"
        )
    if metrics["dc_offset"] > float(qc["review_dc_offset"]):
        reasons.append(
            f"large DC offset ({metrics['dc_offset']:.4f})"
        )
    if metrics["silence_ratio"] > float(
        qc["review_silence_ratio"]
    ):
        reasons.append(
            f"mostly silence ({metrics['silence_ratio']:.1%})"
        )
    if metrics["sample_rate"] < 16000:
        reasons.append(
            f"low sample rate ({metrics['sample_rate']} Hz)"
        )

    return ("REVIEW", reasons) if reasons else ("ACCEPT", [])


def output_name(path: Path, digest: str, index: int) -> str:
    stem = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in path.stem
    ).strip("_")[:80]
    return f"{index:05d}_{stem or 'clip'}_{digest[:10]}.wav"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean single-speaker RVC dataset."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()

    config = load_config(args.config)
    qc = config["dataset_qc"]

    source_root = project_path(args.source)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)

    output_root = (
        project_path(args.output_root)
        if args.output_root
        else rvc_dir(config, args.profile)
    )

    try:
        output_root.resolve().relative_to(source_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError(
            "output-root must not be inside source directory"
        )

    min_duration = (
        args.min_duration
        if args.min_duration is not None
        else float(qc["min_duration"])
    )
    if min_duration <= 0:
        raise ValueError("--min-duration must be > 0")

    files = sorted(
        p for p in source_root.rglob("*.wav")
        if p.is_file()
    )
    if not files:
        raise RuntimeError(
            f"No WAV files found under: {source_root}"
        )

    destinations = {
        "ACCEPT": output_root / "self",
        "REVIEW": output_root / "review",
        "REJECT": output_root / "rejected",
        "DUPLICATE": output_root / "duplicates",
    }

    for path in destinations.values():
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    seen: dict[str, Path] = {}
    rows = []
    stats = {
        "ACCEPT": [0, 0.0],
        "REVIEW": [0, 0.0],
        "REJECT": [0, 0.0],
        "DUPLICATE": [0, 0.0],
    }

    index = 0

    print(f"Profile: {args.profile}")
    print(f"Source WAV files: {len(files)}")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}\n")

    for source in files:
        digest = sha256(source)

        try:
            metrics = analyze(source, qc)
        except Exception as exc:
            classification = "REJECT"
            reasons = [f"decode/analyze error: {exc}"]
            metrics = {
                "duration_sec": 0.0,
                "sample_rate": "",
                "channels": "",
                "rms_dbfs": float("-inf"),
                "peak_dbfs": float("-inf"),
                "clipping_ratio": "",
                "dc_offset": "",
                "silence_ratio": "",
            }
        else:
            if digest in seen:
                classification = "DUPLICATE"
                reasons = [f"exact duplicate of {seen[digest]}"]
            else:
                seen[digest] = source
                classification, reasons = classify(
                    metrics,
                    qc,
                    min_duration,
                )

        duration = float(metrics["duration_sec"])
        stats[classification][0] += 1
        stats[classification][1] += duration

        index += 1
        name = output_name(source, digest, index)

        if classification == "REVIEW" and args.include_review:
            review_copy = destinations["REVIEW"] / name
            shutil.copy2(source, review_copy)
            output_path = destinations["ACCEPT"] / name
            shutil.copy2(source, output_path)
        else:
            output_path = destinations[classification] / name
            shutil.copy2(source, output_path)

        rows.append(
            {
                "source": str(source),
                "output": str(output_path),
                "classification": classification,
                "duration_sec": f"{duration:.3f}",
                "sample_rate": metrics["sample_rate"],
                "channels": metrics["channels"],
                "rms_dbfs": (
                    ""
                    if not np.isfinite(metrics["rms_dbfs"])
                    else f"{metrics['rms_dbfs']:.3f}"
                ),
                "peak_dbfs": (
                    ""
                    if not np.isfinite(metrics["peak_dbfs"])
                    else f"{metrics['peak_dbfs']:.3f}"
                ),
                "clipping_ratio": metrics["clipping_ratio"],
                "dc_offset": metrics["dc_offset"],
                "silence_ratio": metrics["silence_ratio"],
                "sha256": digest,
                "reason": "; ".join(reasons),
            }
        )

        rms_text = (
            f"{metrics['rms_dbfs']:.2f} dBFS"
            if np.isfinite(metrics["rms_dbfs"])
            else "n/a"
        )
        print(
            f"{classification:10s} "
            f"{duration:6.2f}s  "
            f"{rms_text:>12s}  "
            f"{source.name}"
        )

    manifest = output_root / "manifest.tsv"
    fieldnames = list(rows[0].keys())
    with manifest.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "profile": args.profile,
        "source": str(source_root),
        "output_root": str(output_root),
        "source_wav_count": len(files),
        "include_review_in_self": bool(args.include_review),
        "min_duration_sec": min_duration,
        "quality_rules": dict(qc),
        "stats": {
            key: {
                "count": value[0],
                "duration_sec": value[1],
                "duration_min": value[1] / 60.0,
            }
            for key, value in stats.items()
        },
    }

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("RVC DATASET BUILD COMPLETE")
    for key in ("ACCEPT", "REVIEW", "REJECT", "DUPLICATE"):
        count, duration = stats[key]
        print(
            f"{key:10s}: {count:4d} clips / "
            f"{duration / 60.0:.2f} min"
        )
    print("=" * 64)
    print(f"RVC dataset : {destinations['ACCEPT']}")
    print(f"Review      : {destinations['REVIEW']}")
    print(f"Rejected    : {destinations['REJECT']}")
    print(f"Duplicates  : {destinations['DUPLICATE']}")
    print(f"Manifest    : {manifest}")
    print(f"Summary     : {summary_path}")


if __name__ == "__main__":
    main()
