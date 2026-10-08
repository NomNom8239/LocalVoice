"""Regression checks for the read-only Irodori dataset staging workflow."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from scripts import irodori_dataset as mod


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "data" / "training_audio" / "demo" / "audio"
        self.source.mkdir(parents=True)
        self.wav = self.source / "demo.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 32000)
        self.original = mod.digest(self.wav)
        self.out = self.root / "data" / "irodori_lora" / "demo"
        self.args = argparse.Namespace(profile="demo", workspace=None, replace=False)
        self.patch = patch.object(mod, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_scan_does_not_change_source_and_requires_review(self):
        mod.scan(self.args, self.source, self.out)
        self.assertEqual(mod.digest(self.wav), self.original)
        rows = mod.read_csv(self.out / "review.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "pending")
        self.assertEqual(rows[0]["text"], "")
        with self.assertRaises(ValueError):
            mod.export(self.args, self.source, self.out)

    def approve(self):
        rows = mod.read_csv(self.out / "review.csv")
        row = rows[0]
        row.update({
            "decision": "approved", "speaker_ok": "yes",
            "quality": "good", "style": "normal",
            "text": "こんにちは", "caption": "",
        })
        mod.write_csv(self.out / "review.csv", mod.COLUMNS, rows)

    def test_export_only_after_explicit_approval(self):
        mod.scan(self.args, self.source, self.out)
        self.approve()
        mod.export(self.args, self.source, self.out)
        training = mod.read_csv(self.out / "dataset_for_prepare_manifest.csv")
        self.assertEqual(training[0]["text"], "こんにちは")
        self.assertEqual(training[0]["audio"], str(self.wav))
        self.assertEqual(mod.digest(self.wav), self.original)

    def test_hash_change_fails_closed(self):
        mod.scan(self.args, self.source, self.out)
        self.approve()
        with self.wav.open("ab") as f:
            f.write(b"extra")
        with self.assertRaisesRegex(ValueError, "changed"):
            mod.export(self.args, self.source, self.out)

    def test_asr_merge_never_approves(self):
        mod.scan(self.args, self.source, self.out)
        clip = mod.read_csv(self.out / "inventory.csv")[0]
        mod.write_csv(self.out / "asr_suggestions.csv", mod.ASR_COLUMNS, [{
            "clip_id": clip["clip_id"], "sha256": clip["sha256"],
            "asr_status": "suggested", "asr_suggestion": "テスト",
            "language": "ja", "error_detail": "",
        }])
        mod.merge(self.args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["asr_suggestion"], "テスト")
        self.assertEqual(row["decision"], "pending")
        self.assertFalse(row["text"])

    def test_merge_no_detected_text_is_idempotent(self):
        mod.scan(self.args, self.source, self.out)
        clip = mod.read_csv(self.out / "inventory.csv")[0]
        mod.write_csv(self.out / "asr_suggestions.csv", mod.ASR_COLUMNS, [{
            "clip_id": clip["clip_id"], "sha256": clip["sha256"],
            "asr_status": "no_detected_text_review_audio",
            "asr_suggestion": "", "language": "ja", "error_detail": "",
        }])
        mod.merge(self.args, self.source, self.out)
        review_path = self.out / "review.csv"
        after_first = review_path.read_bytes()
        mod.merge(self.args, self.source, self.out)
        self.assertEqual(review_path.read_bytes(), after_first)
        self.assertEqual(
            mod.read_csv(review_path)[0]["asr_status"],
            "no_detected_text_review_audio",
        )

    def test_merge_later_transcript_updates_previous_no_speech(self):
        mod.scan(self.args, self.source, self.out)
        clip = mod.read_csv(self.out / "inventory.csv")[0]
        path = self.out / "asr_suggestions.csv"
        base = {"clip_id": clip["clip_id"], "sha256": clip["sha256"],
                "language": "ja", "error_detail": ""}
        mod.write_csv(path, mod.ASR_COLUMNS, [{
            **base, "asr_status": "no_detected_text_review_audio",
            "asr_suggestion": "",
        }])
        mod.merge(self.args, self.source, self.out)
        mod.write_csv(path, mod.ASR_COLUMNS, [{
            **base, "asr_status": "suggested",
            "asr_suggestion": "こんにちは",
        }])
        mod.merge(self.args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["asr_status"], "suggested")
        self.assertEqual(row["asr_suggestion"], "こんにちは")
        self.assertEqual(row["decision"], "pending")
        self.assertEqual(row["text"], "")

    def test_status_reports_asr_results_separately_from_review(self):
        mod.scan(self.args, self.source, self.out)
        clip = mod.read_csv(self.out / "inventory.csv")[0]
        mod.write_csv(self.out / "asr_suggestions.csv", mod.ASR_COLUMNS, [{
            "clip_id": clip["clip_id"], "sha256": clip["sha256"],
            "asr_status": "suggested", "asr_suggestion": "テスト",
            "language": "ja", "error_detail": "",
        }])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            mod.status(self.args, self.source, self.out)
        report = json.loads(buffer.getvalue())
        self.assertEqual(report["review_asr_status"][""], 1)
        self.assertEqual(report["asr_suggestions"]["statuses"]["suggested"], 1)
        self.assertEqual(report["asr_suggestions"]["total_attempted"], 1)

    def test_asr_populates_suggestion_not_human_verified_text(self):
        mod.scan(self.args, self.source, self.out)
        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                return ([types.SimpleNamespace(text="こんにちは")],
                        types.SimpleNamespace(language="ja"))

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeModel
        args = argparse.Namespace(
            model="small", device="cpu", compute_type="int8",
            retry_errors=False, limit=1, include_short=False,
        )
        with patch.dict(sys.modules, {"faster_whisper": fake}):
            mod.asr(args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["asr_status"], "suggested")
        self.assertEqual(row["asr_suggestion"], "こんにちは")
        self.assertEqual(row["decision"], "pending")
        self.assertEqual(row["text"], "")

    def test_triage_keeps_review_and_wav_unchanged(self):
        mod.scan(self.args, self.source, self.out)
        review_path = self.out / "review.csv"
        original = review_path.read_bytes()
        inventory = mod.read_csv(self.out / "inventory.csv")
        clip = inventory[0]
        mod.write_csv(self.out / "asr_suggestions.csv", mod.ASR_COLUMNS, [{
            "clip_id": clip["clip_id"], "sha256": clip["sha256"],
            "asr_status": "no_detected_text_review_audio",
            "asr_suggestion": "", "language": "ja", "error_detail": "",
        }])
        mod.triage(self.args, self.source, self.out)
        queue = mod.read_csv(self.out / "triage.csv")
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["review_group"], "no_detected_text")
        self.assertEqual(review_path.read_bytes(), original)
        self.assertEqual(mod.digest(self.wav), self.original)

    def test_triage_flags_repeated_characters_without_auto_approval(self):
        mod.scan(self.args, self.source, self.out)
        clip = mod.read_csv(self.out / "inventory.csv")[0]
        mod.write_csv(self.out / "asr_suggestions.csv", mod.ASR_COLUMNS, [{
            "clip_id": clip["clip_id"], "sha256": clip["sha256"],
            "asr_status": "suggested", "asr_suggestion": "あああああ！",
            "language": "ja", "error_detail": "",
        }])
        mod.triage(self.args, self.source, self.out)
        queue = mod.read_csv(self.out / "triage.csv")
        self.assertEqual(queue[0]["review_group"], "expressive_or_unclear")
        self.assertEqual(mod.read_csv(self.out / "review.csv")[0]["decision"], "pending")

    def test_triage_rejects_mismatched_hash(self):
        mod.scan(self.args, self.source, self.out)
        clip = mod.read_csv(self.out / "inventory.csv")[0]
        mod.write_csv(self.out / "asr_suggestions.csv", mod.ASR_COLUMNS, [{
            "clip_id": clip["clip_id"], "sha256": "bad",
            "asr_status": "suggested", "asr_suggestion": "ああ",
            "language": "ja", "error_detail": "",
        }])
        with self.assertRaisesRegex(ValueError, "ASR hash mismatch"):
            mod.triage(self.args, self.source, self.out)

    def test_interactive_review_approves_only_after_confirmation(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True, player="ffplay", include_tagged=False
        )
        with patch("builtins.input", side_effect=[
            "a", "normal", "こんにちは", "", "y"
        ]):
            mod.review(args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["decision"], "approved")
        self.assertEqual(row["text"], "こんにちは")
        self.assertEqual(row["speaker_ok"], "yes")
        self.assertEqual(mod.digest(self.wav), self.original)

    def test_interactive_review_can_tag_nonverbal_without_approval(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True, player="ffplay", include_tagged=False
        )
        with patch("builtins.input", side_effect=[
            "t", "breath", "Possible breath, confirm later"
        ]):
            mod.review(args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["decision"], "tagged")
        self.assertEqual(row["style"], "breath")
        self.assertEqual(row["text"], "")
        self.assertEqual(mod.digest(self.wav), self.original)
        with self.assertRaises(ValueError):
            mod.export(self.args, self.source, self.out)

    def test_tagged_item_is_not_represented_as_pending_approval(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True,
            player="ffplay", include_tagged=False
        )
        with patch("builtins.input", side_effect=["t", "groan", "Needs review"]):
            mod.review(args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        queue = mod.read_csv(self.out / "triage.csv")
        self.assertEqual(queue[0]["review_group"], "tagged_nonverbal")
        self.assertEqual(mod.read_csv(self.out / "review.csv")[0]["decision"], "tagged")

    def test_laugh_style_accepts_common_names(self):
        for label in ("laugh", "laughter", "laughing", "笑い声", "笑い"):
            self.assertEqual(mod.normalize_style(label), "laugh")
        self.assertIn("laugh", mod.VALID_STYLES)
        self.assertIn("laugh", mod.SPECIAL_STYLES)

    def test_laugh_only_tag_is_saved_but_not_approved(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True,
            player="ffplay", include_tagged=False
        )
        with patch("builtins.input", side_effect=["t", "laugh", "laugh only"]):
            mod.review(args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["style"], "laugh")
        self.assertEqual(row["decision"], "tagged")
        self.assertEqual(row["notes"], "laugh only")
        self.assertEqual(mod.digest(self.wav), self.original)
        with self.assertRaises(ValueError):
            mod.export(self.args, self.source, self.out)

    def test_laughing_with_verified_speech_can_be_approved(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True,
            player="ffplay", include_tagged=False
        )
        with patch("builtins.input", side_effect=[
            "a", "笑い声", "こんにちは", "笑いながら話す", "y"
        ]):
            mod.review(args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["style"], "laugh")
        self.assertEqual(row["decision"], "approved")
        mod.export(self.args, self.source, self.out)
        self.assertEqual(
            mod.read_csv(self.out / "dataset_for_prepare_manifest.csv")[0]["caption"],
            "笑いながら話す",
        )

    def test_normal_speech_review_without_text_is_saved_for_later(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        review_args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True,
            player="ffplay", include_tagged=False,
        )
        with patch("builtins.input", side_effect=["a", "normal", "", "y"]):
            mod.review(review_args, self.source, self.out)
        row = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(row["decision"], "needs_text")
        self.assertEqual(row["style"], "normal")
        self.assertEqual(row["speaker_ok"], "yes")
        self.assertEqual(row["quality"], "good")
        self.assertEqual(row["text"], "")
        self.assertEqual(mod.digest(self.wav), self.original)
        with self.assertRaises(ValueError):
            mod.export(self.args, self.source, self.out)

        mod.triage(self.args, self.source, self.out)
        queue = mod.read_csv(self.out / "triage.csv")
        self.assertEqual(queue[0]["review_group"], "needs_transcript")

        review_args.group = "needs_transcript"
        with patch("builtins.input", side_effect=[
            "a", "normal", "こんにちは", "", "y"
        ]):
            mod.review(review_args, self.source, self.out)
        self.assertEqual(
            mod.read_csv(self.out / "review.csv")[0]["decision"], "approved",
        )
        mod.export(self.args, self.source, self.out)

    def test_short_wavs_are_not_asr_transcribed_without_opt_in(self):
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 8000)
        mod.scan(self.args, self.source, self.out)
        row = mod.read_csv(self.out / "inventory.csv")[0]
        self.assertEqual(row["scan_flag"], "short")
        model_args = argparse.Namespace(
            model="small", device="cpu", compute_type="int8",
            retry_errors=False, limit=1, include_short=False,
        )
        mod.asr(model_args, self.source, self.out)
        self.assertFalse((self.out / "asr_suggestions.csv").exists())

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, *args, **kwargs):
                return ([types.SimpleNamespace(text="はい")],
                        types.SimpleNamespace(language="ja"))

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeModel
        model_args.include_short = True
        with patch.dict(sys.modules, {"faster_whisper": fake}):
            mod.asr(model_args, self.source, self.out)
        updated = mod.read_csv(self.out / "review.csv")[0]
        self.assertEqual(updated["asr_suggestion"], "はい")
        self.assertEqual(updated["decision"], "pending")

    def test_interactive_review_fails_if_audio_changed(self):
        mod.scan(self.args, self.source, self.out)
        mod.triage(self.args, self.source, self.out)
        with self.wav.open("ab") as stream:
            stream.write(b"changed")
        args = argparse.Namespace(
            kind="all", group=None, limit=1, no_play=True, player="ffplay", include_tagged=False
        )
        with self.assertRaisesRegex(ValueError, "WAV missing/changed"):
            mod.review(args, self.source, self.out)

    def test_existing_workspace_is_reused(self):
        legacy = self.root / "Irodori-TTS" / "outputs" / "localvoice_lora_dataset"
        legacy.mkdir(parents=True)
        (legacy / "review.csv").write_text("clip_id\n", encoding="utf-8")
        source, out = mod.paths(self.args)
        self.assertEqual(source, self.source)
        self.assertEqual(out, legacy)


if __name__ == "__main__":
    unittest.main()
