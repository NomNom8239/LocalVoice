"""CPU-only checks for LV-05 controlled paired inference."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lv05_voice_ab as ab


class LV05PairedVoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "train_manifest.jsonl"
        rows = []
        for i in range(8):
            latent = self.root / f"latent{i}.pt"
            latent.write_bytes(f"reference {i}".encode())
            rows.append({
                "text": "これは学習素材です。",
                "speaker_id": "LocalVoice-approved:Ui_Shigure",
                "num_frames": 50 + i,
                "latent_path": str(latent),
            })
        self.manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_only_approved_manifest_refs_selected(self):
        single, multiple = ab.latent_references(self.manifest)
        self.assertEqual(len(multiple), 4)
        self.assertEqual(single, multiple[0])
        self.assertTrue(all(path.is_file() for path in multiple))

    def test_build_matched_commands_without_duration_cap(self):
        upstream = self.root / "Irodori-TTS"
        shared = upstream / ".localvoice-toolchain" / "ffmpeg-7.0.2-full-shared" / "bin"
        shared.mkdir(parents=True)
        script_dir = self.root / "scripts"
        script_dir.mkdir()
        (script_dir / "irodori_codec_entry.py").write_text("# test wrapper")
        checkpoint = upstream / "model.safetensors"
        checkpoint.write_bytes(b"placeholder")
        adapter = self.root / "adapter"
        adapter.mkdir()
        refs = [self.root / "latent0.pt", self.root / "latent1.pt",
                self.root / "latent2.pt", self.root / "latent3.pt"]
        variants = []
        for selected in (None, adapter):
            cmd = ab.command(
                upstream / ".venv" / "Scripts" / "python.exe",
                self.root, upstream, checkpoint, selected, refs,
                "おはようございます。", self.root / "out.wav",
                seed=7, steps=40,
            )
            variants.append(cmd)
            self.assertNotIn("--seconds", cmd)
            self.assertIn("--ref-latents", cmd)
            self.assertEqual(cmd[cmd.index("--seed") + 1], "7")
            self.assertEqual(cmd[cmd.index("--num-steps") + 1], "40")
        self.assertNotIn("--lora-adapter", variants[0])
        self.assertIn("--lora-adapter", variants[1])
        self.assertEqual(
            variants[0],
            variants[1][:len(variants[0])],
        )

    def test_missing_lv04_success_evidence_is_rejected(self):
        attempt = self.root / ab.LV04_DIR
        attempt.mkdir()
        (attempt / "result.json").write_text(
            json.dumps({"status": "TRAIN_FAILED"}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "completion evidence"):
            ab.verified_adapter(self.root, self.manifest)

    def test_fresh_comparison_never_overwrites(self):
        (self.root / "lv05_voice_ab_001").mkdir()
        candidate = ab.fresh_attempt(self.root, create=False)
        self.assertEqual(candidate.name, "lv05_voice_ab_002")
        self.assertFalse(candidate.exists())
        created = ab.fresh_attempt(self.root, create=True)
        self.assertEqual(created, candidate)
        self.assertTrue(created.is_dir())


if __name__ == "__main__":
    unittest.main()
