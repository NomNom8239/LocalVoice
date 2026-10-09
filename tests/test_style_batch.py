"""Offline AST batch + provisional library acceptance; no user WAVs or model downloads."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localvoice.style import batch


class FakeAST:
    feature_extractor = SimpleNamespace(sampling_rate=16000)
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="mock-ast-revision"))

    def __call__(self, audio, **kwargs):
        assert kwargs == {"top_k": 527}
        assert audio.ndim == 1
        if np.max(np.abs(audio)) > 0.15:
            return [
                {"label": "Whispering", "score": 0.68},
                {"label": "Speech", "score": 0.20},
                {"label": "Laughter", "score": 0.12},
            ]
        return [
            {"label": "Thunder", "score": 0.94},
            {"label": "Speech", "score": 0.006},
            {"label": "Whispering", "score": 0.004},
        ]


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "PROJECT_ROOT", tmp_path)
    original = tmp_path / "data" / "training_audio" / "Ui_Shigure" / "audio"
    original.mkdir(parents=True)
    for i in range(3):
        audio = np.zeros(16000, dtype=np.float32)
        if i != 1:
            audio = (0.20 if i == 0 else 0.08) * np.sin(
                2 * np.pi * 100 * np.arange(16000) / 16000
            ).astype(np.float32)
        sf.write(original / f"{i:02}.wav", audio, 16000)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.iterdir()}
    monkeypatch.setattr(batch, "_load_ast", lambda device: FakeAST())
    return tmp_path, original, before


def _run(*, max_new=None, resume=False, run_id="ast_run"):
    return batch.run("Ui_Shigure", run_id, 3, "cuda", 0.03, 0.10, max_new, resume)


def test_inventory_is_readonly_and_expected_count_fail_closed(library):
    root, original, before = library
    batch.inventory("Ui_Shigure", 3)
    with pytest.raises(ValueError, match="Expected 4 WAV"):
        batch.inventory("Ui_Shigure", 4)
    with pytest.raises(ValueError, match="Expected 4 WAV"):
        batch.run("Ui_Shigure", "bad_count", 4, "cuda", 0.03, 0.10, None, False)
    assert not (root / "work").exists()
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.iterdir()}


def test_partial_resume_then_export_unverified_categories(library):
    root, original, before = library
    assert _run(max_new=1) == 3
    partial = batch.summary("ast_run")
    assert partial["status"] == "partial"
    assert partial["processed"] == 1 and partial["remaining"] == 2
    with pytest.raises(ValueError, match="not complete"):
        batch.export("ast_run", "v1", False)
    assert _run(resume=True) == 0
    current = batch.summary("ast_run")
    assert current["status"] == "completed"
    assert current["processed"] == 3 and current["remaining"] == 0
    published = batch.export("ast_run", "v1", False)
    assert published == root / "outputs" / "candidates" / "Ui_Shigure" / "v1"
    assert (published / "README.txt").read_text(encoding="utf-8").startswith(
        "ASTによる暫定分類ライブラリ（未確認）"
    )
    manifest = [
        json.loads(x) for x in (published / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 3
    assert {x["category_dir"] for x in manifest} == {
        "02_囁き", "99_未分類_要確認"
    }
    assert all(x["library_tier"] == "UNVERIFIED_AST_CANDIDATE" for x in manifest)
    assert all(not x["human_approved"] for x in manifest)
    assert all((published / x["output_path"]).is_file() for x in manifest)
    with (published / "index.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and all(x["review"] == "UNVERIFIED" for x in rows)
    assert not (root / "outputs" / "datasets").exists()
    with pytest.raises(FileExistsError):
        batch.export("ast_run", "v1", False)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.iterdir()}


def test_resume_blocks_changed_inventory_and_scoring_rules(library):
    root, original, before = library
    assert _run(max_new=1) == 3
    with pytest.raises(ValueError, match="configuration changed"):
        batch.run("Ui_Shigure", "ast_run", 3, "cuda", 0.07, 0.10, None, True)
    with (original / "00.wav").open("ab") as f:
        f.write(b"manually changed input in synthetic test")
    with pytest.raises(ValueError, match="Input inventory"):
        _run(resume=True)
    assert batch.summary("ast_run")["processed"] == 1


def test_bad_source_after_run_cannot_be_exported(library):
    root, original, before = library
    assert _run() == 0
    with (original / "02.wav").open("ab") as f:
        f.write(b"changed after inference")
    with pytest.raises(ValueError, match="Source changed before export"):
        batch.export("ast_run", "v1", False)
    assert not (root / "outputs" / "candidates").exists()


def test_model_inference_failure_is_retained_as_unknown_not_silent_drop(library, monkeypatch):
    root, original, before = library

    class ErrorAST(FakeAST):
        def __call__(self, audio, **kwargs):
            raise RuntimeError("Test inference failure")

    monkeypatch.setattr(batch, "_load_ast", lambda device: ErrorAST())
    assert _run(run_id="failures") == 0
    assert batch.summary("failures")["status"] == "completed_with_errors"
    result = batch._read_jsonl(root / "work" / "failures" / "style_predictions.jsonl")
    assert len(result) == 3
    assert {x["status"] for x in result} == {"unknown", "error"}
    assert all(x["category_dir"] == "99_未分類_要確認" for x in result)
    assert all(x["human_approved"] is False for x in result)
    published = batch.export("failures", "v1", False)
    assert len(list((published / "audio" / "99_未分類_要確認").glob("*.wav"))) == 3
    assert (root / "work" / "failures" / "failures.jsonl").read_text(encoding="utf-8")
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.iterdir()}


def test_interrupted_stage_export_can_resume_without_overwriting_final(library, monkeypatch):
    root, original, before = library
    assert _run() == 0
    real_copy = batch.shutil.copyfileobj
    invoked = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal invoked
        invoked += 1
        if invoked == 2:
            raise RuntimeError("Disk copy interrupted in staging")
        return real_copy(*args, **kwargs)

    monkeypatch.setattr(batch.shutil, "copyfileobj", interrupt_after_first)
    with pytest.raises(RuntimeError, match="interrupted"):
        batch.export("ast_run", "v2", False)
    stage = root / "outputs" / "candidates" / "Ui_Shigure" / ".building-v2-ast_run"
    assert stage.is_dir()
    monkeypatch.setattr(batch.shutil, "copyfileobj", real_copy)
    with pytest.raises(FileExistsError, match="Staging export exists"):
        batch.export("ast_run", "v2", False)
    final = batch.export("ast_run", "v2", True)
    assert final.is_dir() and not stage.exists()
    assert len(list(final.rglob("*.wav"))) == 3
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.iterdir()}


def test_weak_unmapped_acoustic_scores_abstain():
    label, category, score, scores, raw = batch._candidate(
        [{"label": "Thunder", "score": 0.94},
         {"label": "Speech", "score": 0.001}],
        min_score=0.03, top_ratio=0.10,
    )
    assert label == "99_未分類_要確認"
    assert category is None and score is None
    assert scores["normal_speech"] == 0.001

def test_windows_transient_permission_error_retries_atomic_checkpoint(tmp_path, monkeypatch):
    state_path = tmp_path / "run_manifest.json"
    state_path.write_text('{"status": "old"}', encoding="utf-8")
    actual_replace = batch.os.replace
    attempts = []

    def transient_lock(source, dest):
        attempts.append((source, dest))
        if len(attempts) <= 2:
            raise PermissionError("simulated transient Windows sharing restriction")
        return actual_replace(source, dest)

    monkeypatch.setattr(batch.os, "replace", transient_lock)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    batch._atomic_json(state_path, {"status": "updated"})
    assert len(attempts) == 3
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"status": "updated"}
    assert not (tmp_path / "run_manifest.json.writing").exists()


def test_checkpoint_persistent_permission_error_preserves_previous_state(tmp_path, monkeypatch):
    state_path = tmp_path / "run_manifest.json"
    state_path.write_text('{"status": "old"}', encoding="utf-8")
    actual_replace = batch.os.replace
    attempts = []

    def permanent_lock(source, dest):
        attempts.append((source, dest))
        raise PermissionError("simulated persistent Windows lock")

    monkeypatch.setattr(batch.os, "replace", permanent_lock)
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: None)
    with pytest.raises(PermissionError, match="after 8 attempts"):
        batch._atomic_json(state_path, {"status": "new"})
    assert len(attempts) == 8
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"status": "old"}
    assert (tmp_path / "run_manifest.json.writing").exists()
    monkeypatch.setattr(batch.os, "replace", actual_replace)
    batch._atomic_json(state_path, {"status": "recovered"})
    assert json.loads(state_path.read_text(encoding="utf-8")) == {"status": "recovered"}
    assert not (tmp_path / "run_manifest.json.writing").exists()


def test_resume_after_final_state_checkpoint_failure_reuses_saved_results(library, monkeypatch):
    root, original, before = library
    original_atomic = batch._atomic_json

    def blocked_last_checkpoint(path, state):
        if state.get("status") == "completed":
            raise PermissionError("simulated state checkpoint lock after JSONL persisted")
        return original_atomic(path, state)

    monkeypatch.setattr(batch, "_atomic_json", blocked_last_checkpoint)
    with pytest.raises(PermissionError, match="checkpoint lock"):
        _run(run_id="checkpoint_lock")
    results_path = root / "work" / "checkpoint_lock" / "style_predictions.jsonl"
    assert len(batch._read_jsonl(results_path)) == 3
    monkeypatch.setattr(batch, "_atomic_json", original_atomic)
    assert _run(resume=True, run_id="checkpoint_lock") == 0
    assert len(batch._read_jsonl(results_path)) == 3  # not recomputed or duplicated
    assert batch.summary("checkpoint_lock")["processed"] == 3
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in original.iterdir()}
