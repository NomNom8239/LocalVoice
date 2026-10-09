"""Model-free audio diagnostic regression tests; no Whisper/AST downloads."""
import argparse
from array import array
import contextlib
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import wave

from scripts import irodori_nonverbal_evaluate as ev


class FakeAST:
    def __init__(self, device="auto"):
        self.device = device
        self.calls = []

    def predict(self, wav):
        self.calls.append(wav)
        if "breath" in wav.name:
            return {"Breathing": .7, "Speech": .05}, 3.0
        if "groan" in wav.name:
            return {"Groan": .65, "Speech": .1}, 3.0
        return {"Speech": .9}, 3.5


class NonverbalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "workspace"
        self.folder = (
            self.out / "nonverbal_pilot/lora_attempt_001/comparison_001"
        )
        self.folder.mkdir(parents=True)
        self.args = argparse.Namespace(
            profile="demo", workspace=str(self.out), run=False,
            lora_attempt="lora_attempt_001", comparison="comparison_001",
            device="cpu"
        )
        rows = []
        self.cases = {
            "breath_caption": ("😮‍💨", "息を吐いている"),
            "breath_emoji_only": ("😮‍💨", None),
            "groan_caption": ("🥵", "うめくような声"),
            "groan_emoji_only": ("🥵", None),
            "normal_speech": (ev.EXPECTED_TEXT, None),
        }
        for case, (text, caption) in self.cases.items():
            for variant in ("base", "lora"):
                wav = self.folder / f"{case}_{variant}.wav"
                self._wav(wav)
                rows.append({
                    "case": case, "variant": variant,
                    "wav": str(wav), "text": text, "caption": caption,
                    "seed": 20261008, "seconds": 3.5 if case == "normal_speech" else 3.0,
                })
        (self.folder / "result.json").write_text(
            json.dumps({
                "status": "PAIRED_WAVS_READY_NOT_QUALITY_VALIDATED",
                "completed_wavs": 10,
                "wav_metadata": rows,
            }, ensure_ascii=False), encoding="utf-8"
        )
        self.ast = FakeAST()

    @staticmethod
    def _wav(path):
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\x04\x00" * 1600)

    def _metrics(self, wav):
        return {
            "decoded_seconds": 3.0,
            "peak": .33, "sha256": "fake", "rms_dbfs": -20.0,
            "near_silence_fraction": .1, "clip_fraction": 0.0,
            "basic_audio_flags": [],
        }

    def _asr(self, folder, root):
        return {
            "base": {"asr_transcript": ev.EXPECTED_TEXT,
                     "cer_against_fixed_prompt": 0.0},
            "lora": {"asr_transcript": "こんにちは。今日は晴れです。",
                     "cer_against_fixed_prompt": .2},
        }

    def test_plan_is_read_only_and_verified(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = ev.evaluate(
                self.args, self.root, self.out,
                root=self.root, metrics=self._metrics,
                classifier_factory=lambda **kw: self.ast,
                speech_probe=self._asr
            )
        self.assertEqual(result["status"], "PLAN_ONLY")
        self.assertEqual(result["matched_pairs"], 5)
        self.assertFalse(Path(result["output_dir"]).exists())

    def test_evaluate_all_ten_and_preserve_comparison(self):
        self.args.run = True
        with contextlib.redirect_stdout(io.StringIO()):
            result = ev.evaluate(
                self.args, self.root, self.out,
                root=self.root, metrics=self._metrics,
                classifier_factory=lambda **kw: self.ast,
                speech_probe=self._asr
            )
        self.assertEqual(result["status"], "OBJECTIVE_DIAGNOSTICS_ONLY_NOT_QUALITY_VALIDATED")
        self.assertFalse(result["quality_validated"])
        self.assertEqual(len(result["observations"]), 10)
        self.assertEqual(len(self.ast.calls), 10)
        self.assertEqual(len(result["comparison"]), 5)
        speech = next(p for p in result["comparison"] if p["case"] == "normal_speech")
        self.assertEqual(speech["cer_base"], 0.0)
        self.assertEqual(speech["cer_lora"], .2)
        self.assertTrue((Path(result["output_dir"]) / "result.json").is_file())
        self.assertTrue((self.folder / "normal_speech_base.wav").is_file())

    def test_deny_missing_pair_or_unmatched_conditions(self):
        rows = json.loads((self.folder / "result.json").read_text(encoding="utf-8"))
        rows["wav_metadata"][0]["seed"] = 999
        (self.folder / "result.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Unmatched pair conditions"):
            ev.evaluate(self.args, self.root, self.out, root=self.root)
        self.assertFalse((self.folder / "analysis_001").exists())

    def test_fail_closed_without_reusing_or_overwriting_files(self):
        self.args.run = True
        def bad(*args, **kwargs):
            raise RuntimeError("model inference failed")
        with self.assertRaisesRegex(RuntimeError, "model inference failed"):
            ev.evaluate(self.args, self.root, self.out, root=self.root,
                        metrics=self._metrics, classifier_factory=bad)
        summary = json.loads((self.folder / "analysis_001/result.json").read_text())
        self.assertEqual(summary["status"], "BLOCK")
        self.assertFalse(summary["quality_validated"])
        self.assertFalse((self.folder / "analysis_002").exists())

    def test_ja_character_error_rate(self):
        self.assertEqual(ev.character_error_rate(
            "こんにちは。", "こんにちは"), 0.0)
        self.assertAlmostEqual(ev.character_error_rate("ありがとう", "ありがと"), .2)
        self.assertEqual(ev.normalize_japanese("ＡＢＣ。"), "abc")

    def test_float_wav_metrics_with_ffmpeg(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg needed for IEEE WAV decoding")
        body = struct.pack("<f", .15) * 16000
        data = b"WAVE" + b"fmt " + struct.pack(
            "<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32
        ) + b"data" + struct.pack("<I", len(body)) + body
        filename = self.root / "float.wav"
        filename.write_bytes(
            b"RIFF" + struct.pack("<I", len(data)) + data
        )
        metrics = ev.waveform_metrics(filename)
        self.assertAlmostEqual(metrics["decoded_seconds"], 1.0)
        self.assertAlmostEqual(metrics["peak"], .15, places=3)
        self.assertEqual(metrics["basic_audio_flags"], [])


if __name__ == "__main__":
    unittest.main()
