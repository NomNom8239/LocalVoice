"""No model downloads or user voice files in these regression tests."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localvoice.transcription import pilot


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "PROJECT_ROOT", tmp_path)
    data = tmp_path / "data" / "originals"
    data.mkdir(parents=True)
    records = []
    i = 0
    for category, num in pilot.CATEGORIES.items():
        for _ in range(num):
            i += 1
            wav = data / f"{i:02}.wav"
            sf.write(wav, np.zeros(16000, dtype=np.float32), 16000)
            records.append({"sample_id": f"sample_{i:02}", "category": category,
                            "source_path": str(wav), "profile": "Ui_Shigure",
                            "identity_review": "rejected_other" if category == "other_overlap" else "candidate"})
    csv_path = tmp_path / "selection.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    return records, csv_path, tmp_path / "work" / "selection.jsonl", tmp_path


def test_seal_validate_and_hash_mismatch(selected):
    records, csv_path, manifest, root = selected
    assert pilot.main(["seal", "--csv", str(csv_path), "--manifest", str(manifest)]) == 0
    assert pilot.main(["validate", "--manifest", str(manifest)]) == 0
    assert len(pilot.validate(manifest)) == 12
    assert sorted(p.name for p in (root / "data" / "originals").iterdir()) == [f"{i:02}.wav" for i in range(1, 13)]
    with pytest.raises(FileExistsError):
        pilot.seal(csv_path, manifest)
    with (root / "data" / "originals" / "01.wav").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        pilot.validate(manifest)


def test_reject_wrong_category_counts(selected):
    rows, csv_path, manifest, root = selected
    rows[0]["category"] = "nonverbal"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match="exactly 12"):
        pilot.seal(csv_path, manifest)
    assert not manifest.exists()


def test_output_source_record_always_survives_missing_timestamps(selected):
    rows, csv_path, manifest, root = selected
    pilot.seal(csv_path, manifest)
    row = pilot.validate(manifest)[0]
    result = pilot._segment_rows({"text": "はい", "chunks": [{"text": "はい", "timestamp": (None, None)}]}, row,
                                 "mock_run", pilot.MODEL_IDS["kotoba"])
    assert len(result) == 2
    assert result[0]["time_boundary_type"] == "source_extent_not_voice_detection"
    assert result[1]["asr_status"] == "uncertain"
    assert result[1]["failure_reason"] == "invalid_or_absent_timestamps"
    assert result[1]["end_ms"] == 1000


def test_run_both_models_immutable_and_honest_labels(selected, monkeypatch):
    rows, csv_path, manifest, root = selected
    pilot.seal(csv_path, manifest)
    seen_models = []
    def loader(model_id, device):
        seen_models.append((model_id, device))
        def fake_pipe(audio, **kwargs):
            assert kwargs["return_timestamps"] is True
            assert kwargs["generate_kwargs"] == {"language": "ja", "task": "transcribe"}
            assert len(audio) == 16000
            return {"text": "こんにちは", "chunks": [{"text": "こんにちは", "timestamp": (0.1, 0.9)}]}
        return fake_pipe
    monkeypatch.setattr(pilot, "_load_pipeline", loader)
    before = {path.name: pilot._sha256(path) for path in (root / "data" / "originals").iterdir()}
    assert pilot.run(manifest, "kotoba", "run_kotoba", "cuda") == 0
    assert pilot.run(manifest, "whisper", "run_whisper", "cuda") == 0
    assert len(seen_models) == 2
    out = root / "work" / "run_kotoba"
    records = [json.loads(l) for l in (out / "transcription.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(records) == 24
    assert all(row["asr_status"] == "speech_candidate" for row in records)
    assert all("style" not in row and "emoji" not in row for row in records)
    assert all(row["identity_review"] != "approved_self" for row in records)
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf8"))["status"] == "completed"
    with pytest.raises(FileExistsError):
        pilot.run(manifest, "kotoba", "run_kotoba", "cuda")
    after = {path.name: pilot._sha256(path) for path in (root / "data" / "originals").iterdir()}
    assert before == after


def test_failed_model_load_keeps_failure_evidence(selected, monkeypatch):
    rows, csv_path, manifest, root = selected
    pilot.seal(csv_path, manifest)
    def fail(*args): raise RuntimeError("GPU missing")
    monkeypatch.setattr(pilot, "_load_pipeline", fail)
    assert pilot.run(manifest, "kotoba", "initialization_failed", "cuda") == 1
    out = root / "work" / "initialization_failed"
    state = json.loads((out / "run_manifest.json").read_text(encoding="utf8"))
    assert state["status"] == "failed_initialization"
    assert "GPU missing" in state["initialization_error"]
    assert (out / "failures.jsonl").read_text(encoding="utf8")
