"""Pure and mocked tests for the review-only AudioSet style baseline.

No model downloads, GPU, or classification dependencies are required.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import irodori_dataset as data
from scripts import irodori_style as style


class StyleBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "data" / "training_audio" / "demo" / "audio"
        self.source.mkdir(parents=True)
        self.wav = self.source / "clip.wav"
        with wave.open(str(self.wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\x00\x00" * 32000)
        self.workspace = self.root / "data" / "irodori_lora" / "demo"
        self.options = argparse.Namespace(profile="demo", workspace=None, replace=False)
        self.test_args = argparse.Namespace(
            profile="demo", workspace=None, limit=20, group=None,
            device="cpu", retry_errors=False,
        )
        root_patch = patch.object(data, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        data.scan(self.options, self.source, self.workspace)
        data.triage(self.options, self.source, self.workspace)

    def fake_classifier(self):
        class MockClassifier:
            def __init__(self, *, device: str):
                if device != "cpu":
                    raise AssertionError(f"Unexpected device: {device}")

            def predict(self, wav: Path):
                assert wav.exists()
                return {
                    "Laughter": 0.93, "Speech": 0.04,
                    "Breathing": 0.07, "Pant": 0.01,
                    "Music": 0.02,
                }, 2.0
        return MockClassifier

    def test_rank_laughter_is_unverified_and_preserves_raw_evidence(self):
        result = style.evaluate_events({
            "Laughter": 0.93, "Speech": 0.04,
            "Breathing": 0.07,
        }, input_seconds=2.0, duration_seconds=2.0)
        self.assertEqual(result["candidate_style"], "laugh")
        self.assertEqual(result["uncertainty"], "unvalidated")
        self.assertEqual(result["candidate_score"], "0.93000")
        self.assertEqual(
            json.loads(result["top_events_json"])[0]["label"], "Laughter",
        )

    def test_low_evidence_abstains(self):
        result = style.evaluate_events({
            "Speech": 0.03, "Laughter": 0.02,
            "Music": 0.82,
        }, input_seconds=1.0, duration_seconds=1.0)
        self.assertEqual(result["candidate_style"], "unknown")
        self.assertEqual(result["uncertainty"], "review_priority")

    def test_ambiguous_style_requests_review_and_short_flag(self):
        result = style.evaluate_events({
            "Breathing": 0.55, "Pant": 0.52,
        }, input_seconds=0.5, duration_seconds=0.5)
        self.assertEqual(result["candidate_style"], "breath")
        self.assertEqual(result["uncertainty"], "review_priority")
        self.assertIn("competing_voice_styles", result["reason"])
        self.assertIn("subsecond_clip_unvalidated", result["reason"])

    def test_style_suggestion_never_edits_review_or_wav(self):
        before_review = (self.workspace / "review.csv").read_bytes()
        before_wav = data.digest(self.wav)
        style.run(self.test_args, classifier_factory=self.fake_classifier())
        result = data.read_csv(self.workspace / "style_suggestions.csv")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "suggested")
        self.assertEqual(result[0]["candidate_style"], "laugh")
        self.assertEqual(result[0]["model_revision"], style.MODEL_REVISION)
        self.assertEqual((self.workspace / "review.csv").read_bytes(), before_review)
        self.assertEqual(data.digest(self.wav), before_wav)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            data.status(self.options, self.source, self.workspace)
        summary = json.loads(stream.getvalue())
        self.assertEqual(
            summary["experimental_style_suggestions"]["candidate_styles"]["laugh"], 1,
        )
        self.assertFalse(summary["experimental_style_suggestions"]["human_verified"])

    def test_pilot_is_resumable_without_classifying_same_clip(self):
        style.run(self.test_args, classifier_factory=self.fake_classifier())
        out = self.workspace / "style_suggestions.csv"
        baseline = out.read_bytes()
        def no_model(*args, **kwargs):
            raise AssertionError("Classifier should not initialize for cached clip")
        style.run(self.test_args, classifier_factory=no_model)
        self.assertEqual(out.read_bytes(), baseline)

    def test_hash_mismatch_fails_before_model_initialization(self):
        with self.wav.open("ab") as out:
            out.write(b"changed")
        def no_model(*args, **kwargs):
            raise AssertionError("Model should not load after failed hash preflight")
        with self.assertRaisesRegex(ValueError, "changed"):
            style.run(self.test_args, classifier_factory=no_model)
        self.assertFalse((self.workspace / "style_suggestions.csv").exists())

    def test_review_shows_hint_but_requires_explicit_human_action(self):
        style.run(self.test_args, classifier_factory=self.fake_classifier())
        reviewer = argparse.Namespace(
            kind="all", group=None, candidate="laugh",
            limit=1, no_play=True, include_tagged=False,
            player="ffplay",
        )
        buf = io.StringIO()
        with patch("builtins.input", side_effect=["t", "=", "laughed aloud"]):
            with contextlib.redirect_stdout(buf):
                data.review(reviewer, self.source, self.workspace)
        self.assertIn("Experimental AST", buf.getvalue())
        after = data.read_csv(self.workspace / "review.csv")[0]
        self.assertEqual(after["decision"], "tagged")
        self.assertEqual(after["style"], "laugh")
        self.assertEqual(after["notes"], "laughed aloud")
        self.assertEqual(data.digest(self.wav), data.read_csv(
            self.workspace / "inventory.csv"
        )[0]["sha256"])

    def test_evaluation_separates_tentative_tag_from_verified_label(self):
        style.run(self.test_args, classifier_factory=self.fake_classifier())
        review_path = self.workspace / "review.csv"
        original_review = review_path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            style.evaluate_pilot(self.test_args)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["evaluation_counts"]["pending"], 1)
        self.assertEqual(report["confirmed_comparable"], 0)
        self.assertEqual(review_path.read_bytes(), original_review)

        rows = data.read_csv(review_path)
        rows[0].update({"style": "laugh", "decision": "tagged"})
        data.write_csv(review_path, data.COLUMNS, rows)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            style.evaluate_pilot(self.test_args)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["evaluation_counts"]["tentative_match"], 1)
        self.assertEqual(report["confirmed_comparable"], 0)

        rows = data.read_csv(review_path)
        rows[0].update({
            "style": "groan", "decision": "needs_caption",
            "speaker_ok": "yes", "quality": "good",
            "text": "声を張り上げる", "caption": "",
        })
        data.write_csv(review_path, data.COLUMNS, rows)
        reviewed_bytes = review_path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            style.evaluate_pilot(self.test_args)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["evaluation_counts"]["confirmed_mismatch"], 1)
        self.assertEqual(report["confirmed_comparable"], 1)
        self.assertEqual(report["confirmed_agreement"], "0/1")
        self.assertEqual(review_path.read_bytes(), reviewed_bytes)
        evaluation = data.read_csv(self.workspace / "style_pilot_evaluation.csv")
        self.assertEqual(evaluation[0]["human_style"], "groan")
        self.assertEqual(evaluation[0]["candidate_style"], "laugh")

    def test_style_only_human_decision_evaluates_without_transcript(self):
        style.run(self.test_args, classifier_factory=self.fake_classifier())
        review_file = self.workspace / "review.csv"
        rows = data.read_csv(review_file)
        rows[0].update({"style": "laugh", "decision": "style_confirmed"})
        data.write_csv(review_file, data.COLUMNS, rows)
        original = review_file.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            style.evaluate_pilot(self.test_args)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["confirmed_comparable"], 1)
        self.assertEqual(result["confirmed_agreement"], "1/1")
        self.assertEqual(review_file.read_bytes(), original)

    def test_evaluation_rejects_changed_prediction_hash(self):
        style.run(self.test_args, classifier_factory=self.fake_classifier())
        path = self.workspace / "style_suggestions.csv"
        result = data.read_csv(path)
        result[0]["sha256"] = "incorrect"
        data.write_csv(path, style.STYLE_COLUMNS, result)
        with self.assertRaisesRegex(ValueError, "Unmatched or changed source"):
            style.evaluate_pilot(self.test_args)
        self.assertFalse((self.workspace / "style_pilot_evaluation.csv").exists())

    def test_manual_approved_item_not_classified(self):
        review_path = self.workspace / "review.csv"
        reviews = data.read_csv(review_path)
        reviews[0].update({
            "decision": "approved", "style": "normal",
            "speaker_ok": "yes", "quality": "good", "text": "はい",
        })
        data.write_csv(review_path, data.COLUMNS, reviews)
        # Stale triage queue may contain the clip, but current review state wins.
        def no_model(*args, **kwargs):
            raise AssertionError("Model must not run on approved clip")
        style.run(self.test_args, classifier_factory=no_model)
        self.assertFalse((self.workspace / "style_suggestions.csv").exists())
        self.assertEqual(data.read_csv(review_path)[0]["decision"], "approved")


if __name__ == "__main__":
    unittest.main()
