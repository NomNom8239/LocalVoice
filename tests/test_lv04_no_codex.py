"""CPU-only tests for the LV-04 official Irodori LoRA runner."""
from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

# Scripts are imported through the same mechanism as direct CLI execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lv04_no_codex as lv04


class LV04Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "train_manifest.jsonl"
        self.latents = []
        self.rows = []
        for idx in range(8):
            latent = self.root / f"latent{idx}.pt"
            latent.write_bytes(b"test latent placeholder")
            self.latents.append(latent)
            self.rows.append({
                "text": "こんにちは",
                "speaker_id": lv04.SPEAKER_ID,
                "num_frames": 30,
                "latent_path": latent.name,
            })
        self.manifest.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows),
            encoding="utf-8",
        )

    def test_manifest_eight(self):
        self.assertEqual(len(lv04.parse_manifest(self.manifest)), 8)

    def test_missing_latent_is_rejected(self):
        self.latents[2].unlink()
        with self.assertRaisesRegex(ValueError, "Missing or duplicate latent"):
            lv04.parse_manifest(self.manifest)

    def test_wrong_speaker_is_rejected(self):
        self.rows[2]["speaker_id"] = "other"
        self.manifest.write_text(
            "".join(json.dumps(r) + "\n" for r in self.rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Unexpected speaker_id"):
            lv04.parse_manifest(self.manifest)

    def test_completion_result_mismatch_fails_closed(self):
        codec = self.root / lv04.MANIFEST_DIR
        codec.mkdir()
        source_sha = {f"clip-{idx}": f"hash-{idx}" for idx in range(11)}
        result = {
            "status": "PASS_LV03_DACVAE_AND_DATASET",
            "train_rows": 8,
            "evaluation_rows": 3,
            "training_started": False,
            "input_sha256_by_clip": source_sha,
            "manifest": str(codec / "train_manifest.jsonl"),
        }
        (codec / "lv03_final_result.json").write_text(json.dumps(result))
        self.assertEqual(
            lv04.read_lv03_result(self.root, source_sha),
            codec / "train_manifest.jsonl",
        )
        source_sha["clip-0"] = "wrong"
        with self.assertRaisesRegex(ValueError, "SHA evidence"):
            lv04.read_lv03_result(self.root, source_sha)

    def test_command_is_safe_and_validation_is_not_external_eval(self):
        upstream = self.root / "Irodori-TTS"
        cmd = lv04.build_command(
            upstream / ".venv" / "Scripts" / "python.exe",
            upstream, upstream / "config.yaml", self.manifest,
            upstream / "model.safetensors", self.root / "out", 120,
        )
        self.assertIn("--valid-ratio", cmd)
        self.assertEqual(cmd[cmd.index("--valid-ratio") + 1], "0.125")
        self.assertIn("--init-checkpoint", cmd)
        self.assertNotIn("--resume", cmd)
        self.assertIn("--gradient-checkpointing", cmd)
        self.assertIn("--no-wandb", cmd)
        self.assertNotIn("lv02_approved_evaluation.csv", " ".join(cmd))

    def test_resume_requires_adapter_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "checkpoint directory"):
            lv04.build_command(
                Path("python"), Path("."), Path("c.yaml"), self.manifest,
                Path("model.safetensors"), self.root / "out", 120,
                resume=self.root / "missing",
            )

    def test_attempts_never_reuse_old_directory(self):
        for i in range(1, 4):
            (self.root / f"lv04_lora_{i:03d}").mkdir()
        self.assertEqual(lv04.fresh_attempt(self.root).name, "lv04_lora_004")

    def test_adapter_validation(self):
        folder = self.root / "checkpoint_final"
        folder.mkdir()
        (folder / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        (folder / "adapter_model.safetensors").write_bytes(b"not-empty")
        self.assertIn("adapter_sha256", lv04.validate_adapter(folder))


if __name__ == "__main__":
    unittest.main()
