"""Safety regression: LV-02 audit never turns ASR guesses into training approval."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lv02_expand_approved as exp


def write(path: Path, keys: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class LV02ExpansionTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.root_patch = patch.object(exp, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.ws = self.root / "Irodori-TTS" / "outputs" / "localvoice_lora_dataset"
        self.ws.mkdir(parents=True)
        self.source = self.root / "data" / "training_audio" / "Ui_Shigure" / "audio"
        self.source.mkdir(parents=True)
        inv, reviews, auto = [], [], []
        self.old_train, self.holdout = [], []
        for n in range(1167):
            if n < 8:
                video = "train_video"
            elif n < 11:
                video = "eval_video"
            else:
                video = f"candidate_video_{(n % 3) + 1}"
            path = self.source / f"{video}__clip{n:04d}.wav"
            digest = hashlib.sha256(f"WAV clip {n}".encode()).hexdigest()
            # Materialize frozen train/eval WAVs AND all four review candidates.
            # Export validates each approved candidate against on-disk SHA-256.
            if n < 15:
                path.write_bytes(f"WAV clip {n}".encode())
            inv.append({
                "clip_id": f"clip{n:04d}", "source_path": str(path),
                "sha256": digest,
            })
            reviews.append({
                **inv[-1], "decision": "pending",
                "speaker_ok": "", "quality": "", "text": "",
            })
            auto.append({
                **inv[-1],
                "route": "training_candidate" if n >= 11 and n < 15 else "deferred_unresolved_audio",
                "reason": "curated_source_with_asr" if n < 15 else "uncertain",
                "source_kind": "my_voice" if n >= 11 and n < 15 else "unknown",
                "text_source": "asr_unverified",
                "style_source": "curated_source_normal_default",
                "text": f"未検証の音声{n}" if n < 15 else "",
                "caption": "", "candidate_style": "normal",
            })
            if n < 8:
                self.old_train.append({
                    "audio": str(path), "text": f"元の確認済みテキスト{n}",
                    "caption": "", "speaker": "Ui_Shigure",
                })
            elif n < 11:
                self.holdout.append({
                    "audio": str(path), "text": f"評価の文{n}",
                    "caption": "", "speaker": "Ui_Shigure",
                })
        write(self.ws / "inventory.csv", list(inv[0]), inv)
        write(self.ws / "review.csv", list(reviews[0]), reviews)
        write(self.ws / "auto_preparation_report.csv", list(auto[0]), auto)
        write(self.ws / "dataset_for_prepare_manifest_approved_train.csv",
              list(self.old_train[0]), self.old_train)
        write(self.ws / "lv02_approved_evaluation.csv", list(self.holdout[0]),
              self.holdout)

    def audit_queue(self):
        folder = exp.audit(self.ws)
        return folder, exp.load_csv(folder / "review_queue.csv")

    def test_audit_requires_human_confirmation(self) -> None:
        folder, queue = self.audit_queue()
        self.assertEqual(len(queue), 4)
        self.assertTrue(all(row["action"] == "" for row in queue))
        self.assertTrue(all(Path(row["source_path"]).name.endswith(".wav") for row in queue))
        self.assertTrue(all(row["source_video"] in row["source_path"] for row in queue))
        self.assertTrue(all("duration_sec" in row and "scan_flag" in row for row in queue))
        report = json.loads((folder / "audit.json").read_text())
        self.assertEqual(report["status"], "AUDIT_COMPLETE_NO_AUTO_APPROVAL")
        self.assertEqual(report["review_tier_counts"]["EXISTING_TRAIN"], 8)
        self.assertEqual(report["review_tier_counts"]["EXISTING_EVALUATION"], 3)
        self.assertFalse((folder / "dataset_for_prepare_manifest_approved_train.csv").exists())

    def test_explicit_confirmed_export_creates_version_without_overwrite(self) -> None:
        folder, queue = self.audit_queue()
        original_bytes = (self.ws / "dataset_for_prepare_manifest_approved_train.csv").read_bytes()
        eval_bytes = (self.ws / "lv02_approved_evaluation.csv").read_bytes()
        chosen = queue[0]
        chosen.update({
            "action": "approve", "speaker_ok": "yes",
            "quality": "good", "text_verified": "yes",
            "text": "改めて音声を確認しました。",
            "style": "normal", "evidence": "human: listened and checked transcript",
        })
        exp.write_csv(folder / "review_queue.csv", exp.REVIEW_COLS, queue)
        result = exp.export(self.ws, folder / "review_queue.csv")
        dataset = exp.load_csv(result / "dataset_for_prepare_manifest_approved_train.csv")
        self.assertEqual(len(dataset), 9)
        self.assertEqual(dataset[-1]["text"], "改めて音声を確認しました。")
        self.assertEqual(
            (self.ws / "dataset_for_prepare_manifest_approved_train.csv").read_bytes(),
            original_bytes,
        )
        self.assertEqual((self.ws / "lv02_approved_evaluation.csv").read_bytes(), eval_bytes)
        self.assertEqual(json.loads((result / "selection.json").read_text())["new_approved"], 1)

    def test_auto_asr_without_confirmation_rejected(self) -> None:
        folder, queue = self.audit_queue()
        queue[0]["action"] = "approve"
        exp.write_csv(folder / "review_queue.csv", exp.REVIEW_COLS, queue)
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            exp.export(self.ws, folder / "review_queue.csv")
        self.assertFalse((self.ws / "lv02_expansion_002").exists())

    def test_excluded_eval_video_cannot_be_added(self) -> None:
        folder, queue = self.audit_queue()
        queue[0].update({
            "action": "approve", "speaker_ok": "yes", "quality": "good",
            "text_verified": "yes", "text": "確認", "style": "normal",
            "evidence": "I listened",
            "clip_id": "clip0008", "sha256": self.holdout[0].get("sha256", ""),
        })
        # The holdout identity remains prohibited independent of review notes.
        queue[0]["sha256"] = exp.load_csv(self.ws / "inventory.csv")[8]["sha256"]
        exp.write_csv(folder / "review_queue.csv", exp.REVIEW_COLS, queue)
        with self.assertRaisesRegex(ValueError, "protected/frozen"):
            exp.export(self.ws, folder / "review_queue.csv")

    def test_hashed_wav_tampering_rejected(self) -> None:
        folder, queue = self.audit_queue()
        chosen = queue[0]
        chosen.update({
            "action": "approve", "speaker_ok": "yes", "quality": "good",
            "text_verified": "yes", "text": "確認", "style": "normal",
            "evidence": "I listened",
        })
        exp.write_csv(folder / "review_queue.csv", exp.REVIEW_COLS, queue)
        clipnum = int(chosen["clip_id"][4:])
        source = self.source / f"candidate_video_{(clipnum % 3) + 1}__clip{clipnum:04d}.wav"
        source.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "path/SHA changed"):
            exp.export(self.ws, folder / "review_queue.csv")


if __name__ == "__main__":
    unittest.main()
