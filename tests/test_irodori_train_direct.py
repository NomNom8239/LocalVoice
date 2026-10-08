"""CPU-only checks for the one-command Irodori expressive LoRA pipeline."""
from __future__ import annotations

import csv
import json
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import irodori_train_direct as direct
import lv04_no_codex as lv04


class DirectLoraTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_unverified_expression_and_evaluation_video_are_not_training(self):
        def src(cid, vid):
            return {"clip_id": cid, "source_path": f"{vid}__{cid}.wav",
                    "sha256": cid, "scan_flag": "ok", "duration_sec": "6.5"}
        state = {
            "train_rows": [{"audio": "train_video__existing.wav",
                            "text": "人間確認済み", "caption": "",
                            "speaker": direct.SPEAKER, "clip_id": "old",
                            "sha256": "old"}],
            "train_ids": {"old"}, "eval_ids": {"eval"},
            "eval_videos": {"eval_video"},
            "inventory": {
                "old": src("old", "train_video"),
                "eval": src("eval", "eval_video"),
                "a": src("a", "stream_a"),
                "b": src("b", "stream_b"),
                "c": src("c", "eval_video"),
                "d": src("d", "stream_c"),
            },
            "auto": {
                "a": {"route": "training_candidate", "source_kind": "my_voice",
                      "candidate_style": "normal",
                      "style_source": "curated_source_normal_default",
                      "text": "これは候補です。"},
                "b": {"route": "training_candidate", "source_kind": "my_voice",
                      "candidate_style": "groan",
                      "style_source": "ast_strong_heuristic_unverified",
                      "text": "これはうめき声です。"},
                "c": {"route": "training_candidate", "source_kind": "my_voice",
                      "candidate_style": "normal",
                      "style_source": "curated_source_normal_default",
                      "text": "評価動画と同じです。"},
                "d": {"route": "training_candidate", "source_kind": "unknown",
                      "candidate_style": "normal",
                      "style_source": "curated_source_normal_default",
                      "text": "来歴不明です。"},
            },
            "review": {cid: {"decision": "pending"} for cid in "abcd"},
        }
        approved, candidates, reason = direct.source_rows(state)
        self.assertEqual(len(approved), 1)
        self.assertEqual([row["clip_id"] for row in candidates], ["a"])
        self.assertEqual(candidates[0]["kind"], "curated_asr_suggestion")
        self.assertEqual(reason["eval_source_video"], 1)
        self.assertEqual(reason["unverified_style"], 1)
        self.assertEqual(reason["not_curated_source"], 1)

    def test_duplicate_latent_weighting_only_affects_training_manifest(self):
        manifest = self.root / "train_manifest.jsonl"
        rows = [
            {"text": "おはよう", "latent_path": "a.pt", "speaker_id": "voice", "num_frames": 10},
            {"text": "こんにちは", "latent_path": "b.pt", "speaker_id": "voice", "num_frames": 12},
            {"text": "😮‍💨", "caption": "吐息", "latent_path": "c.pt", "speaker_id": "voice", "num_frames": 9},
        ]
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        weighted, info = direct.weighted_manifest(manifest, self.root,
                                                   2, repeat=4)
        produced = [json.loads(line) for line in weighted.read_text().splitlines()]
        self.assertEqual(len(produced), 6)
        self.assertEqual([x["text"] for x in produced].count("😮‍💨"), 4)
        self.assertEqual(info["normal_unique"], 2)
        self.assertEqual(info["nonverbal_unique"], 1)
        self.assertEqual(len(manifest.read_text().splitlines()), 3)
        self.assertNotEqual(weighted, manifest)

    def test_frozen_approved_quiet_wav_kept_without_weakening_new_gate(self):
        # This previously approved clip is structurally valid and unchanged.
        # Its low RMS fails the NEW provisional acoustic heuristic only.
        wav = self.root / "old__quiet.wav"
        rate = 16000
        with wave.open(str(wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            frames = (int(32767 * 0.001 * math.sin(2 * math.pi * 220 * n / rate))
                      for n in range(rate * 2))
            out.writeframes(b"".join(struct.pack("<h", x) for x in frames))
        original = direct.sha256(wav)
        row = {"audio": str(wav), "clip_id": "old", "sha256": original}
        inventory = {"old": {"sha256": original}}
        strict, rejection = direct.verified_wave_quality([row], inventory)
        self.assertEqual(strict, [])
        self.assertEqual(rejection["acoustic_quality_gate"], 1)
        frozen, failure = direct.verified_wave_quality(
            [row], inventory, strict_acoustic_gate=False)
        self.assertEqual(frozen, [row])
        self.assertEqual(failure, {})
        wav.write_bytes(b"changed")
        tampered, rejection = direct.verified_wave_quality(
            [row], inventory, strict_acoustic_gate=False)
        self.assertEqual(tampered, [])
        self.assertEqual(rejection["missing_or_sha_changed"], 1)

    def test_prior_dacvae_approved_very_short_quiet_wav_retained(self):
        # Regression for real failure: prior official LV03 accepted 8 clips,
        # of which 5 later failed invented 1.5-second/RMS constraints.
        wav = self.root / "short__approved.wav"
        rate = 16000
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            samples = (
                int(32767 * 0.0001 * math.sin(2 * math.pi * 220 * n / rate))
                for n in range(rate // 2)
            )
            output.writeframes(
                b"".join(struct.pack("<h", sample) for sample in samples))
        original_sha = direct.sha256(wav)
        row = {"audio": str(wav), "clip_id": "frozen",
               "sha256": original_sha}
        inventory = {"frozen": {"sha256": original_sha}}
        strict, reasons = direct.verified_wave_quality([row], inventory)
        self.assertEqual(strict, [])
        self.assertEqual(reasons["acoustic_quality_gate"], 1)
        frozen, reasons = direct.verified_wave_quality(
            [row], inventory, strict_acoustic_gate=False)
        self.assertEqual(frozen, [row])
        self.assertFalse(reasons)

    def test_frozen_approved_corrupted_or_empty_audio_still_rejected(self):
        wav = self.root / "empty__approved.wav"
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(b"")
        actual_sha = direct.sha256(wav)
        row = {"audio": str(wav), "clip_id": "frozen",
               "sha256": actual_sha}
        valid, reasons = direct.verified_wave_quality(
            [row], {"frozen": {"sha256": actual_sha}},
            strict_acoustic_gate=False)
        self.assertEqual(valid, [])
        self.assertEqual(reasons["bad_wave"], 1)

    def test_existing_tokenizer_checked_nonverbal_pilot_is_experimental(self):
        import irodori_nonverbal_manifest

        pilot = self.root / "nonverbal_pilot"
        pilot.mkdir()
        (pilot / "hf_audio_dataset_hypothesis.jsonl").write_text("")
        (pilot / "tokenizer_audit.csv").write_text("")
        audio = self.root / "voice_video__breath.wav"
        audio.write_bytes(b"synthetic fixture, not for DACVAE")
        with (pilot / "audit.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "clip_id", "style", "style_source"])
            writer.writeheader()
            writer.writerow({
                "clip_id": "breath1", "style": "breath",
                "style_source": "ast_experimental",
            })
        state = {
            "inventory": {
                "breath1": {
                    "clip_id": "breath1", "source_path": str(audio),
                    "sha256": "fixed-source-hash",
                },
            },
            "auto": {"breath1": {"source_kind": "my_voice"}},
            "review": {"breath1": {"decision": "pending"}},
            "train_ids": set(), "eval_ids": set(), "eval_videos": set(),
        }
        pilot_rows = [{
            "audio": str(audio), "text": "😮‍💨",
            "caption": "息を吐く、吐息を伴う発声",
        }]
        with patch.object(irodori_nonverbal_manifest, "validate_sources",
                          return_value=pilot_rows):
            rows = direct.nonverbal_rows(state, self.root, self.root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["kind"], "experimental_nonverbal_breath")
            state["auto"]["breath1"]["source_kind"] = "unknown"
            self.assertEqual(
                direct.nonverbal_rows(state, self.root, self.root), []
            )
            state["auto"]["breath1"]["source_kind"] = "my_voice"
            state["eval_videos"] = {"voice_video"}
            self.assertEqual(
                direct.nonverbal_rows(state, self.root, self.root), []
            )

    def test_round_robin_balances_video_source(self):
        clips = [
            {"audio": "A__a.wav", "clip_id": "a1", "speaker_similarity": 0.9},
            {"audio": "A__b.wav", "clip_id": "a2", "speaker_similarity": 0.8},
            {"audio": "B__c.wav", "clip_id": "b1", "speaker_similarity": 0.5},
        ]
        selected = direct.evenly_sample(clips, 3)
        self.assertEqual([v["clip_id"] for v in selected], ["a1", "b1", "a2"])

    def test_invalid_asr_is_not_silently_selected(self):
        self.assertFalse(direct.valid_text(""))
        self.assertFalse(direct.valid_text("abc"))
        self.assertFalse(direct.valid_text("あ" * 220))
        self.assertFalse(direct.valid_text("あ" * 14))
        self.assertTrue(direct.valid_text("よろしくお願いします。"))

    def test_official_train_command_supports_freezing_duration_predictor(self):
        upstream = self.root / "Irodori-TTS"
        cmd = lv04.build_command(
            Path("python"), upstream, upstream / "config.yaml",
            self.root / "manifest.jsonl", self.root / "model.safetensors",
            self.root / "adapter", 600, valid_ratio=0.05,
        )
        cmd += ["--lora-modules-to-save", "none"]
        self.assertEqual(cmd[cmd.index("--lora-modules-to-save") + 1], "none")
        self.assertEqual(cmd[cmd.index("--max-steps") + 1], "600")
        self.assertNotIn("--resume", cmd)


if __name__ == "__main__":
    unittest.main()
