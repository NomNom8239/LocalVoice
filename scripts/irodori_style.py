"""Experimental, review-only AudioSet event suggestions for Irodori LoRA.

This module NEVER mutates review.csv, the original WAVs, or human judgments.
Scores are uncalibrated sigmoid activations, not probabilities of correctness.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

if __package__:
    from .irodori_dataset import digest, paths, read_csv, write_csv
else:
    from irodori_dataset import digest, paths, read_csv, write_csv

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_REVISION = "f826b80d28226b62986cc218e5cec390b1096902"
STYLE_COLUMNS = (
    "clip_id", "sha256", "model_id", "model_revision", "status",
    "candidate_style", "candidate_score", "runner_up_style",
    "runner_up_score", "uncertainty", "reason", "style_scores_json",
    "top_events_json", "input_seconds", "error_detail",
)

# Exact AudioSet label matches. Generic "breathing" does NOT mean panting,
# nor is "Speech" evidence of a verified transcript or the right speaker.
EVENT_STYLES = {
    "normal": (
        "Speech", "Male speech, man speaking",
        "Female speech, woman speaking", "Child speech, kid speaking",
        "Conversation", "Narration, monologue",
    ),
    "whisper": ("Whispering",),
    "laugh": ("Laughter", "Baby laughter", "Giggle", "Snicker",
              "Belly laugh", "Chuckle, chortle"),
    "breath": ("Breathing", "Sigh", "Gasp", "Wheeze"),
    "panting": ("Pant",),
    "groan": ("Groan", "Grunt", "Wail, moan"),
}

def evaluate_events(events: dict[str, float], *, input_seconds: float,
                    duration_seconds: float) -> dict[str, str]:
    """Rank AudioSet event evidence conservatively; never assert a true style."""
    for key, value in events.items():
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"Invalid sigmoid activation for {key}: {value}")

    scores = {
        style: max((events.get(label, 0.0) for label in labels), default=0.0)
        for style, labels in EVENT_STYLES.items()
    }
    ranking = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    (winner, first), (runner, second) = ranking[:2]
    generic_top = sorted(events.items(), key=lambda item: -item[1])[:5]
    reason = []
    uncertain = False

    if first < 0.20:
        uncertain = True
        reason.append("low_target_evidence")
    if first - second < 0.12:
        uncertain = True
        reason.append("competing_voice_styles")
    # Strong non-voice event evidence means the top voice class cannot be
    # presented confidently as describing the clip.
    nonvoice = [
        (label, score) for label, score in generic_top
        if not any(label in names for names in EVENT_STYLES.values())
    ]
    if nonvoice and nonvoice[0][1] > first + 0.15:
        uncertain = True
        reason.append("other_sound_event_dominates")
    if duration_seconds < 1:
        uncertain = True
        reason.append("subsecond_clip_unvalidated")
    if duration_seconds > input_seconds + 0.1:
        uncertain = True
        reason.append("audio_truncated")

    candidate = winner if first >= 0.20 and "other_sound_event_dominates" not in reason else "unknown"
    return {
        "candidate_style": candidate,
        "candidate_score": f"{first:.5f}",
        "runner_up_style": runner,
        "runner_up_score": f"{second:.5f}",
        "uncertainty": "review_priority" if uncertain else "unvalidated",
        "reason": ";".join(reason) if reason else "model_suggestion_only",
        "style_scores_json": json.dumps(
            {k: round(v, 5) for k, v in scores.items()}, ensure_ascii=False,
        ),
        "top_events_json": json.dumps(
            [{"label": key, "score": round(value, 5)}
             for key, value in generic_top], ensure_ascii=False,
        ),
    }


class AudioSetClassifier:
    """Loads model once, decodes WAV via ffmpeg, and returns raw event scores."""

    def __init__(self, *, device: str = "auto") -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found in PATH")
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise RuntimeError(
                "AudioSet dependencies missing in LocalVoice .venv. "
                "Install with: uv pip install --python .\\.venv\\Scripts\\python.exe "
                "'transformers>=4.53,<5' "
                "(keep the current CUDA-enabled torch)"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA explicitly requested but torch.cuda.is_available() is false")
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.torch = torch
        self.extractor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION,
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True,
        ).to(self.device).eval()
        self.id2label = self.model.config.id2label
        missing = [
            label for labels in EVENT_STYLES.values() for label in labels
            if label not in self.id2label.values()
        ]
        if missing:
            raise RuntimeError(f"AudioSet model labels missing: {missing}")

    def predict(self, wav: Path) -> tuple[dict[str, float], float]:
        # AudioSet AST has a 10.24-second feature window. Limit long audio to
        # the first 10 seconds, record the truncation for human inspection.
        process = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-i", str(wav), "-map", "0:a:0", "-t", "10",
             "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {process.stderr.decode('utf-8', errors='replace')[-400:]}"
            )
        import numpy as np
        samples = np.frombuffer(process.stdout, dtype="<f4")
        if not len(samples):
            raise ValueError("No decoded audio samples")
        seconds = len(samples) / 16000
        inputs = self.extractor(samples, sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs).logits[0].float().sigmoid().cpu().tolist()
        if len(output) != len(self.id2label):
            raise RuntimeError("Unexpected classifier output shape")
        return {self.id2label[i]: float(value) for i, value in enumerate(output)}, seconds


def run(args: argparse.Namespace, *, classifier_factory=AudioSetClassifier) -> None:
    source, out = paths(args)
    review_path = out / "review.csv"
    triage_path = out / "triage.csv"
    if not triage_path.is_file():
        raise FileNotFoundError(f"Run 'triage' before 'classify': {triage_path}")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    reviews = {r["clip_id"]: r for r in read_csv(review_path)}
    queue = read_csv(triage_path)
    target = out / "style_suggestions.csv"
    old = read_csv(target) if target.is_file() else []
    existing = {r["clip_id"]: r for r in old}
    if len(existing) != len(old):
        raise ValueError("Duplicate style suggestion IDs; refusing to modify file")

    work = []
    for queued in queue:
        if args.group and queued.get("review_group") != args.group:
            continue
        cid = queued["clip_id"]
        review = reviews.get(cid)
        if not review:
            raise ValueError(f"Unrecognized queued clip: {cid}")
        if (queued.get("source_path") != review.get("source_path")
                or queued.get("sha256", review.get("sha256")) != review.get("sha256")):
            raise ValueError(f"Stale triage or mismatched WAV identity: {cid}")
        if review.get("decision", "").strip().lower() not in {"pending", "tagged", "needs_text"}:
            # Even already-reviewed clips cannot be inadvertently overwritten.
            continue
        prev = existing.get(cid)
        if prev:
            if (prev.get("sha256") != review.get("sha256")
                    or prev.get("model_id") != MODEL_ID
                    or prev.get("model_revision") != MODEL_REVISION):
                raise ValueError(f"Existing model/clip evidence mismatch: {cid}")
            if prev.get("status") == "suggested" or (
                prev.get("status") == "error" and not args.retry_errors
            ):
                continue
        work.append((queued, review))
        if len(work) >= args.limit:
            break
    if not work:
        print("No unclassified matching clips (no changes made).")
        return

    # All cheap preflight checks happen before loading/downloading the model.
    for _, review in work:
        wav = Path(review["source_path"]).resolve()
        if source not in wav.parents or not wav.is_file() or digest(wav) != review["sha256"]:
            raise ValueError(f"WAV missing, changed, or outside source: {wav}")

    classifier = classifier_factory(device=args.device)
    failures = 0
    count = Counter()
    for index, (_, review) in enumerate(work, start=1):
        cid = review["clip_id"]
        result = {
            "clip_id": cid, "sha256": review["sha256"],
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "status": "error", "candidate_style": "", "candidate_score": "",
            "runner_up_style": "", "runner_up_score": "", "uncertainty": "",
            "reason": "", "style_scores_json": "", "top_events_json": "",
            "input_seconds": "", "error_detail": "",
        }
        try:
            events, input_seconds = classifier.predict(Path(review["source_path"]))
            result.update(evaluate_events(
                events, input_seconds=input_seconds,
                duration_seconds=float(review["duration_sec"]),
            ))
            result.update(status="suggested", input_seconds=f"{input_seconds:.3f}")
            failures = 0
        except Exception as exc:
            failures += 1
            result["error_detail"] = f"{type(exc).__name__}: {exc}"[:1200]
            print(f"[{index}/{len(work)}] ERROR {cid}: {result['error_detail']}",
                  file=sys.stderr)
        existing[cid] = result
        write_csv(target, STYLE_COLUMNS, list(existing.values()))
        count[result["status"]] += 1
        if result["status"] == "suggested":
            print(
                f"[{index}/{len(work)}] {cid}: "
                f"{result['candidate_style']} raw_score={result['candidate_score']} "
                f"({result['uncertainty']})"
            )
        if failures >= 3:
            raise RuntimeError("Three consecutive classification errors; inspect error_detail")
    print(f"Saved experimental style suggestions: {target}")
    print(f"Attempted: {len(work)} / results: {dict(count)}")
    print("Human review.csv, decisions, transcripts, and WAVs were NOT changed.")



EVAL_COLUMNS = (
    "clip_id", "candidate_style", "candidate_score",
    "human_decision", "human_style", "evaluation",
    "uncertainty", "reason",
)


def evaluate_pilot(args: argparse.Namespace) -> None:
    """Join experimental predictions with manually reviewed labels.

    Only explicit speaker/style confirmations (approved, needs_text, needs_caption) count
    as ground truth. Tentative 'tagged' decisions do not count as confirmed.
    """
    _, out = paths(args)
    prediction_path = out / "style_suggestions.csv"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Classify a pilot first: {prediction_path}")
    predictions = read_csv(prediction_path)
    review_rows = read_csv(out / "review.csv")
    reviews = {row["clip_id"]: row for row in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError("Duplicate clip IDs in review.csv")

    result_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    for item in predictions:
        row = reviews.get(item["clip_id"])
        if not row or row.get("sha256") != item.get("sha256"):
            raise ValueError(f"Unmatched or changed source for {item['clip_id']}")
        decision = row.get("decision", "").strip().lower()
        human_style = row.get("style", "").strip().lower()
        candidate = item.get("candidate_style", "")
        status = item.get("status", "")
        evaluation = "pending"
        if status != "suggested":
            evaluation = "classifier_error"
        elif decision == "rejected":
            evaluation = "rejected_audio"
        elif decision == "tagged":
            evaluation = "tentative_match" if candidate == human_style else "tentative_mismatch"
        elif decision == "style_confirmed":
            # Style-only human listening is enough to evaluate style predictions.
            # Text, caption, and speaker/quality are deliberately not required.
            if candidate == "unknown":
                evaluation = "abstained"
            elif candidate == human_style:
                evaluation = "confirmed_match"
            else:
                evaluation = "confirmed_mismatch"
        elif decision in {"approved", "needs_text", "needs_caption"}:
            if (row.get("speaker_ok", "").lower() != "yes" or
                    row.get("quality", "").lower() != "good"):
                evaluation = "unconfirmed_quality"
            elif candidate == "unknown":
                evaluation = "abstained"
            elif candidate == human_style:
                evaluation = "confirmed_match"
            else:
                evaluation = "confirmed_mismatch"
        counts[evaluation] += 1
        result_rows.append({
            "clip_id": item["clip_id"],
            "candidate_style": candidate,
            "candidate_score": item.get("candidate_score", ""),
            "human_decision": decision,
            "human_style": human_style,
            "evaluation": evaluation,
            "uncertainty": item.get("uncertainty", ""),
            "reason": item.get("reason", ""),
        })

    outpath = out / "style_pilot_evaluation.csv"
    write_csv(outpath, EVAL_COLUMNS, result_rows)
    confirmed = counts["confirmed_match"] + counts["confirmed_mismatch"]
    summary = {
        "pilot_predictions": len(result_rows),
        "evaluation_counts": dict(counts),
        "confirmed_comparable": confirmed,
        "confirmed_agreement": (
            f"{counts['confirmed_match']}/{confirmed}" if confirmed else "not_enough_confirmed_labels"
        ),
        "note": (
            "NOT population accuracy. Tentative tags, rejected audio and "
            "model abstentions are not counted as confirmed matches."
        ),
        "evaluation_file": str(outpath),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--group", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Compare model pilot with human labels without inference")
    args = parser.parse_args()
    try:
        if args.evaluate:
            evaluate_pilot(args)
        else:
            run(args)
        return 0
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
