"""Automatic preparation never requires transcription/caption human prompts."""
import argparse
import contextlib
import io
import unittest
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import irodori_dataset as data
from scripts import irodori_autoprep as auto


class AutoPrepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "data/training_audio/demo/audio"
        self.source.mkdir(parents=True)
        self.wav = self.source / "sample.wav"
        with wave.open(str(self.wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * 32000)
        self.out = self.root / "stage"
        self.args = argparse.Namespace(profile="demo", workspace=str(self.out),
                                       limit=1, no_play=True)
        data.scan(self.args, self.source, self.out)
        self.orig_hash = data.digest(self.wav)
        self.review_before = (self.out / "review.csv").read_bytes()

    def put_asr(self, status="suggested", text="こんにちは"):
        row = data.read_csv(self.out / "inventory.csv")[0]
        data.write_csv(self.out / "asr_suggestions.csv", data.ASR_COLUMNS, [{
            "clip_id": row["clip_id"], "sha256": row["sha256"],
            "asr_status": status, "asr_suggestion": text,
            "language": "ja", "error_detail": "",
        }])

    def put_style(self, candidate="normal", uncertainty="unvalidated"):
        row = data.read_csv(self.out / "inventory.csv")[0]
        from scripts import irodori_style
        data.write_csv(self.out / "style_suggestions.csv",
                       irodori_style.STYLE_COLUMNS, [{
            "clip_id": row["clip_id"], "sha256": row["sha256"],
            "model_id": irodori_style.MODEL_ID,
            "model_revision": irodori_style.MODEL_REVISION,
            "status": "suggested", "candidate_style": candidate,
            "uncertainty": uncertainty, "top_events_json": "[]",
        }])

    def test_no_asr_goes_to_automatic_processing_not_manual_entry(self):
        result = auto.build(self.args, self.source, self.out)
        self.assertEqual(result["routes"]["automatic_processing"], 1)
        self.assertEqual(data.read_csv(self.out / "ambiguous_vocal_review.csv"), [])
        self.assertEqual((self.out / "review.csv").read_bytes(), self.review_before)

    def test_curated_speech_auto_uses_asr_without_manual_caption(self):
        rows = data.read_csv(self.out / "review.csv")
        rows[0]["source_kind"] = "my_voice"
        data.write_csv(self.out / "review.csv", data.COLUMNS, rows)
        original = (self.out / "review.csv").read_bytes()
        self.put_asr()
        report = auto.build(self.args, self.source, self.out)
        self.assertEqual(report["training_candidates"], 1)
        output = data.read_csv(self.out / "dataset_for_prepare_manifest_auto.csv")
        self.assertEqual(output[0]["text"], "こんにちは")
        self.assertEqual(output[0]["caption"], "")
        self.assertEqual((self.out / "review.csv").read_bytes(), original)
        self.assertEqual(data.digest(self.wav), self.orig_hash)

    def test_nonverbal_is_not_exported_with_fabricated_text(self):
        self.put_asr(status="no_detected_text_review_audio", text="")
        self.put_style(candidate="groan")
        report = auto.build(self.args, self.source, self.out)
        self.assertEqual(report["routes"]["nonverbal_experiment"], 1)
        self.assertEqual(data.read_csv(self.out / "dataset_for_prepare_manifest_auto.csv"), [])
        self.assertEqual(len(data.read_csv(self.out / "nonverbal_experiments.csv")), 1)

    def test_ambiguous_uses_only_style_prompt_then_auto_caption(self):
        rows = data.read_csv(self.out / "review.csv")
        rows[0]["source_kind"] = "review_emotion"
        data.write_csv(self.out / "review.csv", data.COLUMNS, rows)
        self.put_asr()
        self.put_style("normal")
        result = auto.build(self.args, self.source, self.out)
        self.assertEqual(result["routes"]["ambiguous_vocal_style"], 1)
        self.assertEqual(result["training_candidates"], 0)
        original_hash = data.digest(self.wav)
        with patch("builtins.input", side_effect=["emotion"]) as prompted:
            auto.resolve(self.args, self.source, self.out)
        self.assertEqual(prompted.call_count, 1)
        self.assertEqual(data.digest(self.wav), original_hash)
        review = data.read_csv(self.out / "review.csv")[0]
        self.assertEqual(review["decision"], "style_confirmed")
        self.assertEqual(review["style"], "emotion")
        self.assertEqual(review["text"], "")
        result = auto.build(self.args, self.source, self.out)
        self.assertEqual(result["training_candidates"], 1)
        output = data.read_csv(self.out / "dataset_for_prepare_manifest_auto.csv")[0]
        self.assertEqual(output["text"], "こんにちは")
        self.assertEqual(output["caption"], auto.CAPTIONS["emotion"])

    def test_mismatched_hash_never_writes_outputs(self):
        self.put_asr()
        file = self.out / "asr_suggestions.csv"
        rows = data.read_csv(file)
        rows[0]["sha256"] = "stale"
        data.write_csv(file, data.ASR_COLUMNS, rows)
        with self.assertRaisesRegex(ValueError, "Stale ASR"):
            auto.build(self.args, self.source, self.out)
        self.assertFalse((self.out / "dataset_for_prepare_manifest_auto.csv").exists())

    def test_rejected_clip_not_exported(self):
        self.put_asr()
        rows = data.read_csv(self.out / "review.csv")
        rows[0].update({"decision": "rejected", "source_kind": "my_voice"})
        data.write_csv(self.out / "review.csv", data.COLUMNS, rows)
        report = auto.build(self.args, self.source, self.out)
        self.assertEqual(report["routes"]["excluded"], 1)
        self.assertEqual(report["training_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
