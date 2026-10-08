"""Model-free DACVAE manifest smoke tests; no training or HF downloads."""
import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from scripts import irodori_dataset as ds
from scripts import irodori_nonverbal_pilot as pilot
from scripts import irodori_autoprep as autoprep
from scripts import irodori_nonverbal_manifest as stage


class NonverbalManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "data/training_audio/demo/audio"
        self.source.mkdir(parents=True)
        self.wav = self.source / "sample.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        self.out = self.root / "stage"
        self.args = argparse.Namespace(profile="demo", workspace=str(self.out),
                                       dry_run=False, device="cuda",
                                       max_per_style=1)
        ds.scan(self.args, self.source, self.out)
        inv = ds.read_csv(self.out / "inventory.csv")[0]
        ds.write_csv(self.out / "nonverbal_experiments.csv",
                     autoprep.EXPERIMENT_COLUMNS, [{
            "clip_id": inv["clip_id"], "sha256": inv["sha256"],
            "source_path": inv["source_path"], "style": "groan",
            "style_source": "human_tentative",
            "auto_caption": "うめき声",
        }])
        with contextlib.redirect_stdout(io.StringIO()):
            pilot.build(self.args, self.source, self.out)
        selected = ds.read_csv(self.out / "nonverbal_pilot/audit.csv")
        ds.write_csv(
            self.out / "nonverbal_pilot/tokenizer_audit.csv",
            ("clip_id", "style", "emoji", "status"),
            [{"clip_id": x["clip_id"], "style": x["style"],
              "emoji": x["experimental_text"], "status": "pass"}
             for x in selected if x["status"] == "pilot_hypothesis"]
        )
        jsonfile = self.out / "nonverbal_pilot/hf_audio_dataset_hypothesis.jsonl"
        candidates = ds.read_csv(self.out / "nonverbal_pilot/emoji_only_hypotheses.csv")
        jsonfile.write_text("".join(
            json.dumps({"audio": r["audio"], "text": r["text"],
                        "caption": r["caption"], "speaker": r["speaker"]},
                       ensure_ascii=False) + "\n" for r in candidates
        ), encoding="utf-8")
        self.irodori = self.root / "Irodori-TTS"
        self.irodori.mkdir()
        (self.irodori / "prepare_manifest.py").write_text(
            "# upstream placeholder\n", encoding="utf-8"
        )
        interpreter = self.irodori / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        interpreter.parent.mkdir(parents=True)
        interpreter.touch()
        self.initial_review = (self.out / "review.csv").read_bytes()
        self.initial_wav = ds.digest(self.wav)

    def fake_runner(self, argv, **kwargs):
        if argv[1] == "-c":
            return SimpleNamespace(returncode=0, stdout="cuda_available=True",
                                   stderr="")
        man = Path(argv[argv.index("--output-manifest") + 1])
        latents = Path(argv[argv.index("--latent-dir") + 1])
        latents.mkdir(parents=True, exist_ok=True)
        latent = latents / "00000000_00000000.pt"
        latent.write_bytes(b"fake-dacvae-latent")
        man.write_text(json.dumps({
            "text": "🥵", "latent_path": os.path.relpath(latent, man.parent),
            "num_frames": 10, "speaker_id": "pilot:demo",
            "caption": "うめき声",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    def test_success_requires_real_upstream_manifest_shape(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = stage.smoke(self.args, self.source, self.out,
                                 runner=self.fake_runner, root=self.root)
        self.assertEqual(result["status"], "PASS_DACVAE_MANIFEST_ONLY")
        self.assertEqual(result["verified_latents"], 1)
        self.assertEqual(result["lora_training"], "NOT_RUN")
        self.assertFalse(result["training_ready"])
        self.assertEqual((self.out / "review.csv").read_bytes(), self.initial_review)
        self.assertEqual(ds.digest(self.wav), self.initial_wav)

    def test_dry_run_creates_no_output_directory(self):
        self.args.dry_run = True
        with contextlib.redirect_stdout(io.StringIO()):
            result = stage.smoke(self.args, self.source, self.out,
                                 runner=self.fake_runner, root=self.root)
        self.assertEqual(result["status"], "PREFLIGHT_ONLY")
        self.assertFalse(Path(result["proposed_output"]).exists())

    def test_missing_latent_rows_block_and_keep_attempt_isolated(self):
        def no_rows(argv, **kwargs):
            if argv[1] == "-c":
                return SimpleNamespace(returncode=0, stdout="cuda_available=True",
                                       stderr="")
            Path(argv[argv.index("--output-manifest") + 1]).write_text("",
                                                                       encoding="utf-8")
            return SimpleNamespace(returncode=0)
        with self.assertRaisesRegex(RuntimeError, "0/1"):
            stage.smoke(self.args, self.source, self.out,
                        runner=no_rows, root=self.root)
        failures = list((self.out / "nonverbal_pilot").glob("codec_attempt_*/result.json"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0].read_text())["status"], "BLOCK")

    def test_stale_audio_blocks_before_environment_or_codec(self):
        with self.wav.open("ab") as fh:
            fh.write(b"changed")
        with self.assertRaisesRegex(ValueError, "removed or changed"):
            stage.smoke(self.args, self.source, self.out,
                        runner=self.fake_runner, root=self.root)
        self.assertFalse(list((self.out / "nonverbal_pilot").glob("codec_attempt_*")))

    def test_failed_tokenizer_gate_blocks_before_codec(self):
        target = self.out / "nonverbal_pilot/tokenizer_audit.csv"
        rows = ds.read_csv(target)
        rows[0]["status"] = "block"
        ds.write_csv(target, ("clip_id", "style", "emoji", "status"), rows)
        with self.assertRaisesRegex(ValueError, "Tokenizer gate"):
            stage.smoke(self.args, self.source, self.out,
                        runner=self.fake_runner, root=self.root)

    def test_win32_torchcodec_dll_error_bootstraps_only_compatible_ffmpeg(self):
        calls = []
        shared = self.root / "private-ffmpeg" / "bin"
        shared.mkdir(parents=True)

        def simulated_win(argv, **kwargs):
            calls.append(argv)
            if argv[1] == "-c":
                if "from importlib.metadata" in argv[2]:
                    return SimpleNamespace(returncode=0, stdout=json.dumps({
                        "torch": "2.10.0+cu128", "torchcodec": "0.10.0",
                    }), stderr="")
                if "add_dll_directory" in argv[2]:
                    return SimpleNamespace(returncode=0, stdout="cuda_available=True",
                                           stderr="")
                return SimpleNamespace(returncode=1, stdout="",
                                       stderr="Could not load libtorchcodec_core7.dll")
            return self.fake_runner(argv, **kwargs)

        with contextlib.redirect_stdout(io.StringIO()):
            result = stage.smoke(
                self.args, self.source, self.out,
                runner=simulated_win, root=self.root,
                windows=True, shared_resolver=lambda upstream, explicit: shared
            )
        self.assertEqual(result["status"], "PASS_DACVAE_MANIFEST_ONLY")
        self.assertEqual(result["ffmpeg_shared_bin"], str(shared))
        self.assertTrue(any("irodori_codec_entry.py" in item[1] for item in calls
                            if item[1] != "-c"))
        self.assertEqual((self.out / "review.csv").read_bytes(), self.initial_review)

    def test_win32_version_mismatch_blocks_without_installation(self):
        called = []

        def wrong_pair(argv, **kwargs):
            if "from importlib.metadata" in argv[2]:
                return SimpleNamespace(returncode=0, stdout=json.dumps({
                    "torch": "2.8.0+cu128", "torchcodec": "0.10.0",
                }), stderr="")
            return SimpleNamespace(returncode=1, stdout="",
                                   stderr="Could not load libtorchcodec")
        with self.assertRaisesRegex(RuntimeError, "Incompatible Torch/TorchCodec"):
            stage.smoke(
                self.args, self.source, self.out, runner=wrong_pair,
                root=self.root, windows=True,
                shared_resolver=lambda upstream, explicit: called.append("install")
            )
        self.assertEqual(called, [])

    def test_upstream_cuda_unavailable_block(self):
        def no_cuda(argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout="cuda_available=False",
                                   stderr="")
        with self.assertRaisesRegex(RuntimeError, "cannot use CUDA"):
            stage.smoke(self.args, self.source, self.out,
                        runner=no_cuda, root=self.root)


if __name__ == "__main__":
    unittest.main()
