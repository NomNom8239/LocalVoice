"""Automated diagnostic evidence for Irodori base/LoRA paired WAVs.

No evaluation metric proves an emoji-only LoRA produced the intended
nonverbal vocalization. Report waveform sanity, *uncalibrated* AST scores
and Japanese ASR CER for ordinary speech as separate evidence.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

if __package__:
    from .irodori_dataset import ROOT, paths
    from .irodori_style import AudioSetClassifier, evaluate_events, MODEL_ID, MODEL_REVISION
else:
    from irodori_dataset import ROOT, paths
    from irodori_style import AudioSetClassifier, evaluate_events, MODEL_ID, MODEL_REVISION


EXPECTED_CASES = frozenset({
    "breath_caption", "breath_emoji_only",
    "groan_caption", "groan_emoji_only", "normal_speech",
})
EXPECTED_TEXT = "こんにちは。今日はいい天気ですね。"


def verify_comparison(folder: Path) -> list[dict]:
    report_path = folder / "result.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Paired inference result.json absent: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("wav_metadata")
    if (report.get("status") != "PAIRED_WAVS_READY_NOT_QUALITY_VALIDATED"
            or report.get("completed_wavs") != 10
            or not isinstance(rows, list) or len(rows) != 10):
        raise ValueError("Paired inference did not verify exactly 10 WAV outputs")
    grouped = defaultdict(dict)
    for row in rows:
        case, variant = row.get("case"), row.get("variant")
        if case not in EXPECTED_CASES or variant not in {"base", "lora"}:
            raise ValueError(f"Unexpected inference case/variant: {case}/{variant}")
        if variant in grouped[case]:
            raise ValueError(f"Duplicate paired inference variant: {case}/{variant}")
        path = Path(row.get("wav", "")).resolve()
        if (path.parent != folder.resolve()
                or path.name != f"{case}_{variant}.wav"
                or not path.is_file() or path.stat().st_size < 45):
            raise ValueError(f"Missing/out-of-folder paired WAV: {path}")
        if case == "normal_speech":
            if row.get("text") != EXPECTED_TEXT:
                raise ValueError("Normal-speech ASR target was changed")
        elif (case.startswith("breath_") and row.get("text") != "😮‍💨"
              or case.startswith("groan_") and row.get("text") != "🥵"):
            raise ValueError("Nonverbal comparison emoji changed")
        grouped[case][variant] = row
    if set(grouped) != EXPECTED_CASES or any(set(x) != {"base", "lora"} for x in grouped.values()):
        raise ValueError("Five matched base/LoRA pairs are required")
    for case, pair in grouped.items():
        a, b = pair["base"], pair["lora"]
        for key in ("text", "caption", "seed", "seconds"):
            if a.get(key) != b.get(key):
                raise ValueError(f"Unmatched pair conditions for {case}: {key}")
        if ("emoji_only" in case or case == "normal_speech") and a.get("caption") is not None:
            raise ValueError("Emoji-only/speech comparison unexpectedly has a caption")
        if case.endswith("_caption") and not a.get("caption"):
            raise ValueError("Caption comparison lacks conditioning")
    return rows


def waveform_metrics(wav: Path, *, runner=subprocess.run) -> dict:
    """Decode any supported WAV, including IEEE float, without numpy or GPU."""
    process = runner([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(wav),
        "-map", "0:a:0", "-ac", "1", "-ar", "16000",
        "-f", "f32le", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode:
        raise RuntimeError(f"FFmpeg could not decode {wav.name}: "
                           + process.stderr.decode("utf-8", errors="replace")[-450:])
    raw = process.stdout
    if not raw or len(raw) % 4:
        raise RuntimeError(f"Invalid decoded sample stream: {wav}")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError(f"NaN/Infinity in waveform: {wav}")
    count = len(values)
    rms = math.sqrt(sum(x*x for x in values) / count)
    peak = max(abs(x) for x in values)
    quiet_fraction = sum(abs(x) < 0.0031623 for x in values) / count
    clip_fraction = sum(abs(x) >= 0.999 for x in values) / count
    return {
        "decoded_seconds": round(count / 16000, 3),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 2),
        "peak": round(peak, 5),
        "near_silence_fraction": round(quiet_fraction, 4),
        "clip_fraction": round(clip_fraction, 5),
        "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
        "basic_audio_flags": [
            * (["near_silent"] if quiet_fraction > 0.98 else []),
            * (["severe_clipping"] if clip_fraction > 0.01 else []),
        ],
    }


def normalize_japanese(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in text
                   if not ch.isspace() and unicodedata.category(ch)[0] not in {"P", "S"})


def character_error_rate(reference: str, hypothesis: str) -> float:
    src, hyp = normalize_japanese(reference), normalize_japanese(hypothesis)
    if not src:
        raise ValueError("Cannot calculate CER on empty reference")
    previous = list(range(len(hyp) + 1))
    for i, ch in enumerate(src, 1):
        current = [i]
        for j, other in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ch != other)))
        previous = current
    return round(previous[-1] / len(src), 4)


def asr_speech(folder: Path, *, root: Path = ROOT, runner=subprocess.run) -> dict:
    python = root / ".venv-asr" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    script = Path(__file__).with_name("irodori_speech_retention_asr.py")
    if not python.is_file():
        raise FileNotFoundError(f"Existing isolated Faster-Whisper venv not found: {python}")
    args = [
        str(python), str(script),
        str(folder / "normal_speech_base.wav"),
        str(folder / "normal_speech_lora.wav"),
    ]
    process = runner(args, text=True, capture_output=True, check=False,
                     encoding="utf-8")
    if process.returncode:
        raise RuntimeError(
            "Read-only Japanese speech ASR failed; no new models installed: "
            + (process.stderr or process.stdout)[-1000:]
        )
    try:
        rows = json.loads(process.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Faster-Whisper subprocess did not return JSON") from exc
    paths = [str((folder / f"normal_speech_{x}.wav").resolve())
             for x in ("base", "lora")]
    if (not isinstance(rows, list) or len(rows) != 2
            or [r.get("wav") for r in rows] != paths):
        raise ValueError("ASR subprocess returned mismatched output paths")
    output = {}
    for variant, row in zip(("base", "lora"), rows):
        output[variant] = {
            "asr_transcript": row["text"],
            "asr_model": "faster-whisper/small, CPU int8, Japanese",
            "cer_against_fixed_prompt": character_error_rate(
                EXPECTED_TEXT, row["text"]
            ),
        }
    return output


def fresh_analysis(comparison: Path) -> Path:
    for index in range(1, 100):
        attempt = comparison / f"analysis_{index:03d}"
        if not attempt.exists():
            return attempt
    raise RuntimeError("No unused objective analysis directory")


def evaluate(args: argparse.Namespace, source: Path, out: Path, *,
             root: Path = ROOT, metrics=waveform_metrics,
             classifier_factory=AudioSetClassifier,
             speech_probe=asr_speech) -> dict:
    del source
    comparison = (out / "nonverbal_pilot" / args.lora_attempt / args.comparison).resolve()
    pilot = (out / "nonverbal_pilot").resolve()
    if comparison.parent.parent != pilot or not comparison.is_dir():
        raise ValueError("Comparison must be an existing isolated nonverbal LoRA output")
    rows = verify_comparison(comparison)
    dest = fresh_analysis(comparison)
    result = {
        "status": "PLAN_ONLY" if not args.run else "RUNNING",
        "comparison_dir": str(comparison), "output_dir": str(dest),
        "expected_wavs": 10, "matched_pairs": 5,
        "quality_validated": False,
        "no_training_or_dataset_mutation": True,
        "notes": [
            "AST sigmoid scores are uncalibrated and cannot verify actual vocalization.",
            "ASR CER on one short utterance is a diagnostic, not a speech quality verdict.",
            "Speaker identity, audible naturalness and benefit from 24 steps remain unproven.",
        ],
    }
    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    dest.mkdir(parents=True, exist_ok=False)
    observations = []
    try:
        classifier = classifier_factory(device=args.device)
        for row in rows:
            path = Path(row["wav"]).resolve()
            metric = metrics(path)
            events, seconds = classifier.predict(path)
            kind = evaluate_events(
                events, input_seconds=seconds,
                duration_seconds=metric["decoded_seconds"]
            )
            observations.append({
                "case": row["case"], "variant": row["variant"],
                "wav": str(path), "text": row["text"],
                "caption": row.get("caption"),
                "acoustic": metric, "ast_diagnostic": {
                    "suggested_style": kind["candidate_style"],
                    "raw_uncalibrated_style_scores": json.loads(kind["style_scores_json"]),
                    "uncertainty": kind["uncertainty"],
                    "reason": kind["reason"],
                },
            })
        # Separate subprocess/venv: AST and Whisper dependency stacks do not mix.
        speech = speech_probe(comparison, root=root)
        for obs in observations:
            if obs["case"] == "normal_speech":
                obs["speech_asr"] = speech[obs["variant"]]
        grouped = defaultdict(dict)
        for obs in observations:
            grouped[obs["case"]][obs["variant"]] = obs
        comparisons = []
        for case in sorted(grouped):
            a = grouped[case]["base"]
            b = grouped[case]["lora"]
            expected_style = "normal" if case == "normal_speech" else case.split("_", 1)[0]
            raw_a = a["ast_diagnostic"]["raw_uncalibrated_style_scores"][expected_style]
            raw_b = b["ast_diagnostic"]["raw_uncalibrated_style_scores"][expected_style]
            pair = {
                "case": case, "expected_style_hypothesis": expected_style,
                "ast_raw_score_base": raw_a, "ast_raw_score_lora": raw_b,
                "ast_raw_score_delta_lora_minus_base": round(raw_b - raw_a, 5),
                "peak_change_lora_minus_base": round(
                    b["acoustic"]["peak"] - a["acoustic"]["peak"], 5),
                "quality_judgment": "NOT_ESTABLISHED",
            }
            if case == "normal_speech":
                pair["cer_base"] = speech["base"]["cer_against_fixed_prompt"]
                pair["cer_lora"] = speech["lora"]["cer_against_fixed_prompt"]
                pair["cer_change_lora_minus_base"] = round(
                    pair["cer_lora"] - pair["cer_base"], 4)
            comparisons.append(pair)
        result["status"] = "OBJECTIVE_DIAGNOSTICS_ONLY_NOT_QUALITY_VALIDATED"
        result["observations"] = observations
        result["comparison"] = comparisons
    except Exception as exc:
        result["status"] = "BLOCK"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["partial_observations"] = observations
        raise
    finally:
        (dest / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({
        "status": result["status"], "output": str(dest / "result.json"),
        "paired_results": comparisons, "quality_validated": False,
    }, ensure_ascii=False, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--lora-attempt", default="lora_attempt_001")
    parser.add_argument("--comparison", default="comparison_001")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        evaluate(args, source, out)
        return 0
    except (RuntimeError, OSError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
