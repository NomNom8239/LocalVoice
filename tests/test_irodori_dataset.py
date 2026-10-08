"""Regression checks for the read-only Irodori dataset staging workflow."""
import argparse
import csv
import tempfile
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

    def test_existing_workspace_is_reused(self):
        legacy = self.root / "Irodori-TTS" / "outputs" / "localvoice_lora_dataset"
        legacy.mkdir(parents=True)
        (legacy / "review.csv").write_text("clip_id\n", encoding="utf-8")
        source, out = mod.paths(self.args)
        self.assertEqual(source, self.source)
        self.assertEqual(out, legacy)


if __name__ == "__main__":
    unittest.main()
