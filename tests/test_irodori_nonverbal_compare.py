"""Irodori paired comparison tests use fake WAVs; no model downloads or GPU."""
import argparse
import contextlib
import io
import json
import shutil
import struct
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

from scripts import irodori_dataset as ds
from scripts import irodori_nonverbal_compare as compare


class PairedInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "data/training_audio/demo/audio"
        self.source.mkdir(parents=True)
        for i in range(7):
            path = self.source / f"{i:02d}.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes((i + 1).to_bytes(2, "little") * 16000 * 7)
        self.out = self.root / "workspace"
        self.args = argparse.Namespace(
            profile="demo", workspace=str(self.out), run=False,
            lora_attempt="lora_attempt_001", codec_attempt="codec_attempt_001",
        )
        ds.scan(self.args, self.source, self.out)
        inventory = ds.read_csv(self.out / "inventory.csv")
        speech = [r for r in inventory if Path(r["source_path"]).stem == "06"][0]
        # Regression: scan inventory may have no source_kind because the
        # WAV filename marker (not manifest.tsv) is what build() later used
        # to resolve curated origin. Use the derived auto-prep report.
        self.assertEqual(speech["source_kind"], "")
        ds.write_csv(
            self.out / "auto_preparation_report.csv",
            ("clip_id", "source_path", "sha256", "source_kind", "route",
             "reason", "candidate_style", "text", "text_source"),
            [{"clip_id": speech["clip_id"], "source_path": speech["source_path"],
              "sha256": speech["sha256"], "source_kind": "my_voice",
              "route": "training_candidate", "reason": "curated_source_with_asr",
              "candidate_style": "normal",
              "text": "こんにちは。今日はいい天気ですね。",
              "text_source": "asr_unverified"}],
        )
        ds.write_csv(
            self.out / "dataset_for_prepare_manifest_auto.csv",
            ("audio", "text", "caption", "speaker"),
            [{"audio": speech["source_path"], "text": "こんにちは。今日はいい天気ですね。",
              "caption": "普通の話し方", "speaker": "demo"}],
        )
        voice = sorted([x for x in inventory if x["clip_id"] != speech["clip_id"]],
                       key=lambda r: r["source_path"])
        pilot = self.out / "nonverbal_pilot"
        pilot.mkdir()
        (pilot / "hf_audio_dataset_hypothesis.jsonl").write_text(
            "".join(json.dumps({"audio": r["source_path"]}) + "\n" for r in voice),
            encoding="utf-8",
        )
        codec = pilot / "codec_attempt_001"
        codec.mkdir()
        (codec / "result.json").write_text(json.dumps({
            "status": "PASS_DACVAE_MANIFEST_ONLY",
            "ffmpeg_shared_bin": "existing_environment",
        }))
        lora = pilot / "lora_attempt_001"
        lora.mkdir()
        adapter = lora / "adapter/checkpoint_final"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (adapter / "adapter_model.safetensors").write_bytes(b"fake-adapter")
        self.checkpoint = self.root / "base-model.safetensors"
        self.checkpoint.write_bytes(b"model")
        (lora / "result.json").write_text(json.dumps({
            "status": "TRAIN_COMMAND_EXITED_ZERO_NOT_QUALITY_VALIDATED",
            "checkpoint": str(self.checkpoint),
            "lora_training": "STARTED",  # original report had stale marker
        }))
        rows = []
        for i, r in enumerate(voice):
            tensor_path = codec / f"latent-{i:02d}.pt"
            tensor_path.write_bytes(b"tensor")
            rows.append({
                "text": "😮‍💨" if i < 3 else "🥵",
                "caption": "自動キャプション",
                "latent_path": str(tensor_path),
                "speaker_id": "pilot:demo",
                "num_frames": 70,
            })
        def store(name, group):
            (lora / name).write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in group),
                encoding="utf-8",
            )
        store("train.jsonl", rows[:2] + rows[3:5])
        store("holdout.jsonl", [rows[2], rows[5]])
        upstream = self.root / "Irodori-TTS"
        upstream.mkdir()
        (upstream / "prepare_manifest.py").touch()
        (upstream / "infer.py").touch()
        import os
        py = upstream / ".venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        py.parent.mkdir(parents=True)
        py.touch()
        self.runs = []
        self.review = (self.out / "review.csv").read_bytes()

    def fake_runner(self, argv, **kwargs):
        self.runs.append(argv)
        dest = Path(argv[argv.index("--output-wav") + 1])
        dest.parent.mkdir(exist_ok=True)
        with wave.open(str(dest), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x01\x00" * 16000)
        return SimpleNamespace(returncode=0, stdout="Saved", stderr="")

    def test_plan_read_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = compare.compare(
                self.args, self.source, self.out, root=self.root,
                runner=self.fake_runner
            )
        self.assertEqual(result["status"], "PLAN_ONLY")
        self.assertEqual(result["expected_wavs"], 10)
        self.assertEqual(result["paired_prompts"], 5)
        self.assertEqual(len(self.runs), 0)
        self.assertFalse(Path(result["output_dir"]).exists())

    def test_render_five_matched_base_and_adapter_pairs(self):
        self.args.run = True
        with contextlib.redirect_stdout(io.StringIO()):
            result = compare.compare(
                self.args, self.source, self.out, root=self.root,
                runner=self.fake_runner, windows=False
            )
        self.assertEqual(result["status"], "PAIRED_WAVS_READY_NOT_QUALITY_VALIDATED")
        self.assertEqual(result["completed_wavs"], 10)
        self.assertEqual(len(self.runs), 10)
        for index in range(0, 10, 2):
            base, adapter = self.runs[index:index + 2]
            self.assertNotIn("--lora-adapter", base)
            self.assertIn("--lora-adapter", adapter)
            for flag in ("--text", "--seed", "--seconds", "--ref-wav"):
                self.assertEqual(base[base.index(flag) + 1],
                                 adapter[adapter.index(flag) + 1])
        self.assertEqual(
            len([j for j in self.runs if "--caption" not in j]), 6
        )  # emoji-only 2 cues * 2 + normal speech * 2
        self.assertEqual(len(result["wav_metadata"]), 10)
        self.assertEqual((self.out / "review.csv").read_bytes(), self.review)
        self.assertFalse(result["quality_assessed"])
        self.assertTrue((Path(result["output_dir"]) / "result.json").is_file())

    def test_missing_adapter_fails_before_files(self):
        (self.out / "nonverbal_pilot/lora_attempt_001/adapter/"
                    "checkpoint_final/adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "missing/incomplete"):
            compare.compare(
                self.args, self.source, self.out, root=self.root,
                runner=self.fake_runner
            )
        self.assertEqual(len(self.runs), 0)

    def test_inspect_ieee_float_wav_using_ffprobe(self):
        if not shutil.which("ffprobe"):
            self.skipTest("ffprobe needed for float WAV inspection")
        # Minimal IEEE float WAV. Python 3.10 wave.open rejects format=3.
        audio = struct.pack("<f", 0.15) * 1600
        riff = (
            b"WAVE"
            + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000,
                                      64000, 4, 32)
            + b"data" + struct.pack("<I", len(audio)) + audio
        )
        path = self.root / "float32.wav"
        path.write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)
        metadata = compare.inspect_wav(path)
        self.assertEqual(metadata["sample_rate"], 16000)
        self.assertAlmostEqual(metadata["duration_sec"], 0.1, places=2)

    def test_reference_uses_resolved_provenance_not_empty_inventory_column(self):
        reference, digest, reason = compare.choose_speech_reference(
            self.source, self.out, set()
        )
        self.assertEqual(reference, (self.source / "06.wav").resolve())
        self.assertEqual(digest, ds.digest(reference))
        self.assertEqual(reason, "curated_source_with_asr")

    def test_reference_rejects_uncurated_ast_suggestions(self):
        provenance = self.out / "auto_preparation_report.csv"
        columns = ("clip_id", "source_path", "sha256", "source_kind", "route",
                   "reason", "candidate_style", "text", "text_source")
        rows = ds.read_csv(provenance)
        rows[0]["reason"] = "strong_non_normal_event_candidate"
        rows[0]["candidate_style"] = "groan"
        ds.write_csv(provenance, columns, rows)
        with self.assertRaisesRegex(ValueError, "not_normal_curated_speech"):
            compare.choose_speech_reference(self.source, self.out, set())

    def test_reference_exclusion_cannot_choose_pilot_audio(self):
        with self.assertRaisesRegex(ValueError, "Selection audit"):
            compare.choose_speech_reference(
                self.source, self.out, {str(self.source / "06.wav")}
            )

    def test_reference_rejects_stale_provenance_sha(self):
        provenance = self.out / "auto_preparation_report.csv"
        columns = ("clip_id", "source_path", "sha256", "source_kind", "route",
                   "reason", "candidate_style", "text", "text_source")
        rows = ds.read_csv(provenance)
        rows[0]["sha256"] = "0" * 64
        ds.write_csv(provenance, columns, rows)
        with self.assertRaisesRegex(ValueError, "missing_or_mismatched_provenance"):
            compare.choose_speech_reference(self.source, self.out, set())

    def test_missing_speech_reference_is_not_replaced_by_holdout(self):
        target = self.out / "dataset_for_prepare_manifest_auto.csv"
        ds.write_csv(target, ("audio", "text", "caption", "speaker"), [])
        with self.assertRaisesRegex(ValueError, "separately curated"):
            compare.compare(self.args, self.source, self.out, root=self.root)
        self.assertEqual(len(self.runs), 0)

    def test_partial_failure_reports_block_without_reusing_output(self):
        self.args.run = True
        def fail_second(argv, **kwargs):
            if len(self.runs) == 1:
                return SimpleNamespace(returncode=10, stdout="", stderr="error")
            return self.fake_runner(argv, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "exit=10"):
            compare.compare(
                self.args, self.source, self.out, root=self.root,
                runner=fail_second, windows=False
            )
        p = self.out / "nonverbal_pilot/lora_attempt_001/comparison_001/result.json"
        result = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "BLOCK")
        self.assertEqual(result["completed_wavs"], 1)


if __name__ == "__main__":
    unittest.main()
