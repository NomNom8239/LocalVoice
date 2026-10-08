"""Model-free pilot planning/execute simulations for nonverbal LoRA."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest
import wave

from scripts import irodori_dataset as ds
from scripts import irodori_autoprep as auto
from scripts import irodori_nonverbal_pilot as pilot
from scripts import irodori_nonverbal_lora as lora


class NonverbalLoRATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "data/training_audio/demo/audio"
        self.source.mkdir(parents=True)
        for i in range(6):
            p = self.source / f"{i:02d}.wav"
            with wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((i + 1).to_bytes(2, "little", signed=True) * 16000)
        self.out = self.root / "stage"
        self.args = argparse.Namespace(
            profile="demo", workspace=str(self.out),
            codec_attempt="codec_attempt_001",
            checkpoint=None, steps=24, run=False, fetch_checkpoint=False,
            max_per_style=3,
        )
        ds.scan(self.args, self.source, self.out)
        inv = ds.read_csv(self.out / "inventory.csv")
        expr = []
        for i, row in enumerate(inv):
            style = "breath" if i < 3 else "groan"
            expr.append({
                "clip_id": row["clip_id"], "sha256": row["sha256"],
                "source_path": row["source_path"], "style": style,
                "style_source": "human_tentative",
                "auto_caption": f"caption_{style}",
            })
        ds.write_csv(self.out / "nonverbal_experiments.csv",
                     auto.EXPERIMENT_COLUMNS, expr)
        with contextlib.redirect_stdout(io.StringIO()):
            pilot.build(self.args, self.source, self.out)
        hypothesis = ds.read_csv(
            self.out / "nonverbal_pilot/emoji_only_hypotheses.csv"
        )
        audit = ds.read_csv(self.out / "nonverbal_pilot/audit.csv")
        ds.write_csv(
            self.out / "nonverbal_pilot/tokenizer_audit.csv",
            ("clip_id", "style", "emoji", "status"),
            [{"clip_id": row["clip_id"], "style": row["style"],
              "emoji": row["experimental_text"], "status": "pass"}
             for row in audit if row["status"] == "pilot_hypothesis"],
        )
        jsonl = self.out / "nonverbal_pilot/hf_audio_dataset_hypothesis.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            for row in hypothesis:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        codec = self.out / "nonverbal_pilot/codec_attempt_001"
        latents = codec / "latents"
        latents.mkdir(parents=True)
        manifest = []
        for i, row in enumerate(hypothesis):
            name = f"{i:03d}.pt"
            (latents / name).write_bytes(b"fake")
            manifest.append({
                "text": row["text"], "caption": row["caption"],
                "speaker_id": "demo:voice",
                "latent_path": "latents/" + name, "num_frames": 20,
            })
        (codec / "train_manifest.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in manifest), encoding="utf-8")
        (codec / "result.json").write_text(
            json.dumps({"status": "PASS_DACVAE_MANIFEST_ONLY"}), encoding="utf-8"
        )
        upstream = self.root / "Irodori-TTS"
        (upstream / "configs").mkdir(parents=True)
        (upstream / ".venv" / ("Scripts" if os.name == "nt" else "bin")).mkdir(
            parents=True
        )
        (upstream / ".venv" / ("Scripts/python.exe" if os.name == "nt"
                              else "bin/python")).touch()
        (upstream / "prepare_manifest.py").write_text("# test")
        (upstream / "train.py").write_text("# test")
        (upstream / "configs/train_v4_small_lora.yaml").write_text(
            "model:\n  text_tokenizer_repo: sbintuitions/modernbert-ja-310m\n"
            "train:\n  lora_enabled: true\n",
            encoding="utf-8",
        )
        self.review = (self.out / "review.csv").read_bytes()

    def test_plan_only_stratifies_without_writes_or_training(self):
        with contextlib.redirect_stdout(io.StringIO()):
            report = lora.plan(self.args, self.source, self.out, root=self.root)
        self.assertEqual(report["status"], "PLAN_ONLY")
        self.assertEqual((report["train_clips"], report["holdout_clips"]), (4, 2))
        self.assertEqual(set(report["by_style_holdout"]), {"🥵", "😮‍💨"})
        self.assertEqual(report["max_steps"], 24)
        self.assertFalse(report["checkpoint_exists"])
        self.assertFalse(Path(report["output"]).exists())
        self.assertEqual((self.out / "review.csv").read_bytes(), self.review)

    def test_fetch_and_training_are_mutually_exclusive(self):
        self.args.run = True
        self.args.fetch_checkpoint = True
        with self.assertRaisesRegex(ValueError, "separate commands"):
            lora.plan(self.args, self.source, self.out, root=self.root)

    def test_codec_path_rebase_refuses_missing_latents(self):
        codec = self.out / "nonverbal_pilot/codec_attempt_001"
        src = lora.load_jsonl(codec / "train_manifest.jsonl")
        latent = codec / src[0]["latent_path"]
        latent.unlink()
        with self.assertRaisesRegex(RuntimeError, "manifest row"):
            lora.plan(self.args, self.source, self.out, root=self.root)

    def test_checkpoint_explicit_fetch_is_sha_verified_and_does_not_train(self):
        self.args.fetch_checkpoint = True
        checkpoint = self.root / "downloaded-model.safetensors"
        checkpoint.write_bytes(b"fake-official-weights")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        calls = []
        mocked = ModuleType("huggingface_hub")
        mocked.try_to_load_from_cache = lambda *a, **kw: None

        def download(*a, **kw):
            calls.append((a, kw))
            return str(checkpoint)

        mocked.hf_hub_download = download
        with patch.dict(sys.modules, {"huggingface_hub": mocked}), patch.object(
            lora, "CHECKPOINT_SHA256", digest
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                report = lora.plan(self.args, self.source, self.out, root=self.root)
        self.assertEqual(report["status"], "PLAN_ONLY")
        self.assertTrue(report["checkpoint_exists"])
        self.assertEqual(report["checkpoint_source"], "official_hf_sha256_verified")
        self.assertEqual(len(calls), 1)
        self.assertFalse(Path(report["output"]).exists())

    def test_checkpoint_bad_hash_fails_before_training(self):
        self.args.fetch_checkpoint = True
        checkpoint = self.root / "bad.safetensors"
        checkpoint.write_bytes(b"wrong-model")
        mocked = ModuleType("huggingface_hub")
        mocked.try_to_load_from_cache = lambda *a, **kw: None
        mocked.hf_hub_download = lambda **kw: str(checkpoint)
        with patch.dict(sys.modules, {"huggingface_hub": mocked}):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                lora.plan(self.args, self.source, self.out, root=self.root)
        self.assertFalse(list((self.out / "nonverbal_pilot").glob("lora_attempt_*")))

    def test_no_checkpoint_blocks_opted_in_training(self):
        self.args.run = True
        with self.assertRaisesRegex(FileNotFoundError, "Full-precision"):
            lora.plan(self.args, self.source, self.out, root=self.root)
        self.assertFalse(list((self.out / "nonverbal_pilot").glob("lora_attempt_*")))

    def test_explicit_run_uses_isolated_four_clip_manifest(self):
        self.args.run = True
        model = self.root / "Irodori-TTS/model.safetensors"
        model.write_bytes(b"test-weights")
        invoked = []
        def runner(argv, **kwargs):
            invoked.append(argv)
            manifest = Path(argv[argv.index("--manifest") + 1])
            rows = lora.load_jsonl(manifest)
            self.assertEqual(len(rows), 4)
            # Regression: codec manifest's relative latent_path must be
            # relocated before the split is written in lora_attempt.
            self.assertTrue(all(Path(r["latent_path"]).is_absolute() for r in rows))
            self.assertTrue(all(Path(r["latent_path"]).is_file() for r in rows))
            self.assertIn("--gradient-checkpointing", argv)
            self.assertIn("--no-wandb", argv)
            self.assertEqual(argv[argv.index("--max-steps") + 1], "24")
            self.assertIn("--lora", argv)
            return SimpleNamespace(returncode=0)
        with contextlib.redirect_stdout(io.StringIO()):
            report = lora.plan(
                self.args, self.source, self.out,
                root=self.root, runner=runner
            )
        self.assertEqual(report["status"], "TRAIN_COMMAND_EXITED_ZERO_NOT_QUALITY_VALIDATED")
        self.assertEqual(len(invoked), 1)
        p = Path(report["output"])
        holdout = lora.load_jsonl(p / "holdout.jsonl")
        self.assertEqual(len(holdout), 2)
        self.assertTrue(all(Path(r["latent_path"]).is_file() for r in holdout))
        self.assertEqual((p / "upstream_config.yaml").read_text(),
                         (self.root / "Irodori-TTS/configs/train_v4_small_lora.yaml").read_text())
        self.assertEqual((self.out / "review.csv").read_bytes(), self.review)

    def test_wrong_dacvae_status_blocks(self):
        p = self.out / "nonverbal_pilot/codec_attempt_001/result.json"
        p.write_text(json.dumps({"status": "BLOCK"}))
        with self.assertRaisesRegex(ValueError, "has not passed"):
            lora.plan(self.args, self.source, self.out, root=self.root)

    def test_single_class_cannot_fake_both_holdouts(self):
        manifest = [{"text": "🥵", "caption": "groan", "speaker_id": "demo"}] * 6
        with self.assertRaisesRegex(ValueError, "exactly two"):
            lora.group_holdout(manifest)


if __name__ == "__main__":
    unittest.main()
