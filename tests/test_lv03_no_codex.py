"""Smoke and safety tests for the Codex-free LV-03 runner (no CUDA needed)."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "lv03_no_codex.py"
spec = importlib.util.spec_from_file_location("lv03_no_codex", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(runner)


def write_rows(path: Path, headers: list[str], records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


class LV03NoCodexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.trains = []
        self.evals = []
        inventory = []
        for i in range(11):
            name = ("video_train" if i < 8 else "video_eval") + f"__clip{i:03}.wav"
            path = self.workspace / name
            path.write_bytes(("fake WAV source " + str(i)).encode())
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {"clip_id": f"clip-{i}", "source_path": str(path), "sha256": sha}
            inventory.append(record)
            selected = {"clip_id": f"clip-{i}", "audio": str(path),
                        "sha256": sha, "speaker": "Ui_Shigure",
                        "text": "てすと", "caption": ""}
            (self.trains if i < 8 else self.evals).append(selected)
        for i in range(11, 1167):
            inventory.append({"clip_id": f"unused-{i}",
                              "source_path": str(self.workspace / f"unused_{i}.wav"),
                              "sha256": "ignored by this focused test"})
        write_rows(self.workspace / "inventory.csv",
                   ["clip_id", "source_path", "sha256"], inventory)
        self.write_approved()

    def write_approved(self) -> None:
        columns = ["clip_id", "audio", "sha256", "speaker", "text", "caption"]
        write_rows(self.workspace / "dataset_for_prepare_manifest_approved_train.csv",
                   columns, self.trains)
        write_rows(self.workspace / "lv02_approved_evaluation.csv",
                   columns, self.evals)

    def test_expected_splits_and_source_hashes(self) -> None:
        source, all_ids = runner.verify_sources(self.workspace)
        self.assertEqual(source.name, "dataset_for_prepare_manifest_approved_train.csv")
        self.assertEqual(len(all_ids), 11)

    def test_versioned_train_nine_is_accepted_without_changing_legacy(self) -> None:
        path = self.workspace / "video_train__clip011.wav"
        path.write_bytes(b"approved additional reference")
        new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        inv = runner.rows(self.workspace / "inventory.csv")
        inv[11] = {"clip_id": "new-approved", "source_path": str(path),
                   "sha256": new_sha}
        write_rows(self.workspace / "inventory.csv",
                   ["clip_id", "source_path", "sha256"], inv)
        version = self.workspace / "lv02_expansion_001"
        version.mkdir()
        candidate = version / "dataset_for_prepare_manifest_approved_train.csv"
        extra = {"clip_id": "new-approved", "audio": str(path),
                 "sha256": new_sha, "speaker": "Ui_Shigure",
                 "text": "追加承認済みです。", "caption": ""}
        write_rows(candidate, ["clip_id", "audio", "sha256", "speaker",
                               "text", "caption"], [*self.trains, extra])
        selected, hashes = runner.verify_sources(self.workspace, candidate)
        self.assertEqual(selected, candidate)
        self.assertEqual(len(hashes), 12)
        legacy, old = runner.verify_sources(self.workspace)
        self.assertEqual(len(old), 11)
        self.assertEqual(legacy.name, "dataset_for_prepare_manifest_approved_train.csv")

    def test_train_eval_same_video_fails_closed(self) -> None:
        old = Path(self.evals[0]["audio"])
        renamed = old.with_name(old.name.replace("video_eval__", "video_train__"))
        old.rename(renamed)
        inventory_path = self.workspace / "inventory.csv"
        inventory = runner.rows(inventory_path)
        for row in inventory:
            if row["source_path"] == str(old):
                row["source_path"] = str(renamed)
        write_rows(inventory_path, ["clip_id", "source_path", "sha256"], inventory)
        self.evals[0]["audio"] = str(renamed)
        self.write_approved()
        with self.assertRaisesRegex(ValueError, "leakage: video"):
            runner.verify_sources(self.workspace)

    def test_duplicate_train_eval_clip_fails_closed(self) -> None:
        self.evals[0]["audio"] = self.trains[0]["audio"]
        self.evals[0]["sha256"] = self.trains[0]["sha256"]
        self.evals[0]["clip_id"] = self.trains[0]["clip_id"]
        self.write_approved()
        with self.assertRaisesRegex(ValueError, "leakage"):
            runner.verify_sources(self.workspace)

    def test_modified_wav_is_rejected(self) -> None:
        Path(self.trains[0]["audio"]).write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            runner.verify_sources(self.workspace)

    def test_existing_failed_attempts_are_not_reused(self) -> None:
        for i in range(1, 4):
            (self.workspace / f"lv03_dacvae_{i:03}").mkdir()
        new_path = runner.new_attempt(self.workspace)
        self.assertEqual(new_path.name, "lv03_dacvae_004")

    def test_embedded_python_entrypoints_compile(self) -> None:
        compile(runner.CHILD, "<codec-child>", "exec")
        compile(runner.CHECK, "<latent-validator>", "exec")


if __name__ == "__main__":
    unittest.main()
