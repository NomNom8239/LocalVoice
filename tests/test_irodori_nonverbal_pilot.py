"""Emoji-only nonverbal pilot is isolated and never auto-promoted."""
import argparse
import contextlib
import io
import tempfile
import unittest
import wave
from pathlib import Path

from scripts import irodori_dataset as data
from scripts import irodori_nonverbal_pilot as pilot


class NonverbalPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "data" / "training_audio" / "demo" / "audio"
        self.source.mkdir(parents=True)
        for number, frames in enumerate((32000, 24000)):
            with wave.open(str(self.source / f"voice_{number}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x00\x00" * frames)
        self.out = self.root / "output"
        self.args = argparse.Namespace(profile="demo", workspace=str(self.out),
                                       max_per_style=3)
        data.scan(self.args, self.source, self.out)
        self.inventory = data.read_csv(self.out / "inventory.csv")
        self.before = (self.out / "review.csv").read_bytes()

    def stage(self, styles=("groan", "emotion"), sources=("human_tentative", "human_confirmed")):
        items = []
        for row, style, source in zip(self.inventory, styles, sources):
            items.append({
                "clip_id": row["clip_id"],
                "sha256": row["sha256"],
                "source_path": row["source_path"],
                "style": style,
                "style_source": source,
                "auto_caption": f"自動説明 {style}",
            })
        data.write_csv(self.out / "nonverbal_experiments.csv",
                       ("clip_id", "sha256", "source_path",
                        "style", "style_source", "auto_caption"), items)

    def test_creates_hypothesis_without_touching_existing_data(self):
        self.stage()
        with contextlib.redirect_stdout(io.StringIO()):
            result = pilot.build(self.args, self.source, self.out)
        self.assertEqual(result["source_candidates"], 2)
        self.assertEqual(result["pilot_hypotheses"], 1)
        self.assertEqual(result["held_without_manual_work"], 1)
        self.assertEqual(result["status"], "HYPOTHESIS_ONLY_NOT_TRAINING_READY")
        hypothesis = data.read_csv(self.out / "nonverbal_pilot" / "emoji_only_hypotheses.csv")
        self.assertEqual(hypothesis[0]["text"], "🥵")
        self.assertEqual(hypothesis[0]["caption"], "自動説明 groan")
        self.assertEqual(len(hypothesis), 1)
        audit = data.read_csv(self.out / "nonverbal_pilot" / "audit.csv")
        self.assertEqual({r["status"] for r in audit}, {"held", "pilot_hypothesis"})
        self.assertEqual((self.out / "review.csv").read_bytes(), self.before)

    def test_style_cap_prevents_unbounded_pilot(self):
        self.stage(styles=("breath", "breath"),
                   sources=("human_tentative", "ast_experimental"))
        self.args.max_per_style = 1
        with contextlib.redirect_stdout(io.StringIO()):
            result = pilot.build(self.args, self.source, self.out)
        self.assertEqual(result["pilot_hypotheses"], 1)
        audit = data.read_csv(self.out / "nonverbal_pilot" / "audit.csv")
        self.assertEqual(sum(x["reason"] == "pilot_style_cap_reached" for x in audit), 1)

    def test_source_change_blocks_outputs(self):
        self.stage()
        target = self.source / "voice_0.wav"
        with target.open("ab") as handle:
            handle.write(b"extra")
        with self.assertRaisesRegex(ValueError, "Changed/missing WAV"):
            pilot.build(self.args, self.source, self.out)
        self.assertFalse((self.out / "nonverbal_pilot" / "emoji_only_hypotheses.csv").exists())

    def test_unrecognized_style_is_held_not_fabricated(self):
        self.stage(styles=("emotion", "other"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = pilot.build(self.args, self.source, self.out)
        self.assertEqual(result["pilot_hypotheses"], 0)
        self.assertEqual(result["held_without_manual_work"], 2)
        self.assertEqual(data.read_csv(
            self.out / "nonverbal_pilot" / "emoji_only_hypotheses.csv"
        ), [])


if __name__ == "__main__":
    unittest.main()
