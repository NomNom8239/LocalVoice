"""Irodori emoji-only conditioning preflight: model-free regression."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path

from scripts import irodori_dataset as ds
from scripts import irodori_nonverbal_check as check
from scripts import irodori_nonverbal_pilot as pilot


class FakeTokenizer:
    bos_token_id = 1
    unk_token_id = 0

    def __init__(self, fail=False):
        self.fail = fail

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return [0] if self.fail else [100 + ord(ch) for ch in text]

    def convert_ids_to_tokens(self, ids):
        return [str(i) for i in ids]

    def decode(self, ids, **kwargs):
        if self.fail:
            return "?"
        return "".join(chr(i - 100) for i in ids)


class NonverbalCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.wav = self.source / "clip.wav"
        with wave.open(str(self.wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        self.out = self.root / "out"
        self.args = argparse.Namespace(profile="demo", workspace=str(self.out),
                                       max_per_style=1, config=None)
        ds.scan(self.args, self.source, self.out)
        inv = ds.read_csv(self.out / "inventory.csv")[0]
        ds.write_csv(self.out / "nonverbal_experiments.csv",
                     pilot.EXPERIMENT_COLUMNS, [{
            "clip_id": inv["clip_id"], "source_path": inv["source_path"],
            "sha256": inv["sha256"], "style": "groan",
            "style_source": "human_tentative",
            "possible_emoji_text": "🥵", "auto_caption": "うめき",
            "reason": "test",
        }])
        with contextlib.redirect_stdout(io.StringIO()):
            pilot.build(self.args, self.source, self.out)
        self.initial = (self.out / "nonverbal_experiments.csv").read_bytes()
        self.review = (self.out / "review.csv").read_bytes()
        self.config = self.root / "train_v4_small.yaml"
        self.config.write_text(
            "model:\n"
            "  text_tokenizer_repo: sbintuitions/modernbert-ja-310m\n"
            "  text_encoder_revision: 77675fc96a7e445e982e2ba90246b816efc74ec6\n"
            "  text_add_bos: true\n", encoding="utf-8"
        )
        self.args.config = str(self.config)

    def test_real_manifest_hypothesis_staged_only_when_tokenizer_roundtrips(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = check.build(self.args, self.source, self.out,
                                 tokenizer_loader=lambda spec: FakeTokenizer())
        self.assertEqual(result["tokenizer_gate"], "PASS_TOKENIZATION_ONLY")
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["lora_training"], "NOT_RUN")
        path = self.out / "nonverbal_pilot" / "hf_audio_dataset_hypothesis.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "🥵")
        self.assertEqual(rows[0]["audio"], str(self.wav))
        audit = ds.read_csv(self.out / "nonverbal_pilot/tokenizer_audit.csv")
        self.assertEqual(audit[0]["status"], "pass")
        self.assertEqual(audit[0]["bos_id"], "1")
        self.assertEqual((self.out / "review.csv").read_bytes(), self.review)
        self.assertEqual((self.out / "nonverbal_experiments.csv").read_bytes(), self.initial)

    def test_unknown_token_blocks_gate_even_when_text_nonempty(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = check.build(
                self.args, self.source, self.out,
                tokenizer_loader=lambda spec: FakeTokenizer(fail=True),
            )
        self.assertEqual(result["tokenizer_gate"], "BLOCK")
        self.assertEqual(result["tokenization_block"], 1)
        self.assertFalse(result["training_ready"])

    def test_wrong_audio_hash_refuses_to_create_preflight_outputs(self):
        with self.wav.open("ab") as out:
            out.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            check.build(self.args, self.source, self.out,
                        tokenizer_loader=lambda spec: FakeTokenizer())
        self.assertFalse((self.out / "nonverbal_pilot/tokenizer_audit.csv").exists())

    def test_model_config_must_be_pinned(self):
        self.config.write_text(
            "model:\n  text_tokenizer_repo: model-a\n"
            "  text_encoder_revision: null\n  text_add_bos: true\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Unpinned"):
            check.build(self.args, self.source, self.out,
                        tokenizer_loader=lambda spec: FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
