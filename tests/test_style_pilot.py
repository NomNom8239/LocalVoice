"""Synthetic, offline regression checks; never open a user's private audio or download models."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from localvoice.style import pilot


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "PROJECT_ROOT", tmp_path)
    originals = tmp_path / "data" / "untouched"
    originals.mkdir(parents=True)
    rows = []
    for index, (voice, hint) in enumerate(((True, "whisper"), (False, "silence_noise")), 1):
        wav = originals / f"{index:02}.wav"
        audio = np.zeros(44100, dtype=np.float32)
        if voice:
            audio = 0.15 * np.sin(2 * np.pi * 440 * np.arange(len(audio)) / 44100)
        sf.write(wav, audio, 44100)
        rows.append({
            "schema_version": "localvoice.asr-pilot.v1",
            "sample_id": f"sample_{index:02}", "source_path": str(wav),
            "source_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            "duration_ms": 1000, "category": hint,
            "profile": "Ui_Shigure", "identity_review": "candidate",
        })
    manifest = tmp_path / "pilot.sealed.jsonl"
    manifest.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    return manifest, originals, rows


def test_validate_read_only_and_fail_closed(selection):
    manifest, originals, rows = selection
    assert len(pilot.validate(manifest)) == 2
    assert pilot.main(["validate", "--manifest", str(manifest)]) == 0
    duplicates = [rows[0], rows[0]]
    manifest.write_text("".join(json.dumps(x) + "\n" for x in duplicates), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source_id"):
        pilot.validate(manifest)
    assert not (originals.parent.parent / "work").exists()


def test_hash_mismatch_and_no_output(selection):
    manifest, originals, rows = selection
    with (originals / "01.wav").open("ab") as out:
        out.write(b"change")
    with pytest.raises(ValueError, match="SHA mismatch"):
        pilot.validate(manifest)
    assert pilot.main(["run", "--manifest", str(manifest), "--model", "ast", "--run-id", "invalid"]) == 2
    assert not (originals.parent.parent / "work").exists()


class FakePipeline:
    def __init__(self, model):
        self.model_key = model
        self.feature_extractor = SimpleNamespace(sampling_rate=16000 if model == "ast" else 48000)
        self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash="mock-revision"))
        self.calls = []

    def __call__(self, audio, **kwargs):
        assert audio.ndim == 1
        assert len(audio) == self.feature_extractor.sampling_rate
        self.calls.append(kwargs)
        if self.model_key == "ast":
            assert kwargs == {"top_k": 527}
            return [{"label": "Whispering", "score": 0.72},
                    {"label": "Speech", "score": 0.16},
                    {"label": "Groan", "score": 0.12}]
        assert kwargs["hypothesis_template"] == "{}"
        assert kwargs["candidate_labels"] == list(pilot.CLAP_CAPTIONS.values())
        return [{"label": desc, "score": 1.0 / len(pilot.CLAP_CAPTIONS)}
                for desc in pilot.CLAP_CAPTIONS.values()]


def test_both_models_are_candidates_only_with_preserved_sources(selection, monkeypatch):
    manifest, originals, rows = selection
    seen = []
    def load(model, device):
        assert device == "cuda"
        pipe = FakePipeline(model)
        seen.append(pipe)
        return pipe
    monkeypatch.setattr(pilot, "_load_pipeline", load)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals.iterdir()}
    assert pilot.run(manifest, "ast", "ast_001", "cuda") == 0
    assert pilot.run(manifest, "clap", "clap_001", "cuda") == 0
    for model, name in (("ast", "ast_001"), ("clap", "clap_001")):
        out = originals.parent.parent / "work" / name
        records = [json.loads(x) for x in (out / "style_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(records) == 2
        assert [r["source_id"] for r in records] == ["sample_01", "sample_02"]
        assert all(r["status"] == "scored" for r in records)
        assert all(r["review_status"] == "requires_review" and not r["human_approved"] for r in records)
        assert records[0]["selection_category_hint"] == "whisper"  # Audit only.
        assert records[1]["digital_silence"] is True
        assert records[1]["caution"] == "digital_silence_models_can_hallucinate"
        assert records[0]["candidate_ranking"]
        if model == "ast":
            assert "yawn" not in records[0]["category_scores"]  # Unsupported AudioSet label.
            assert records[0]["category_scores"]["whisper"] == 0.72
            assert records[0]["category_scores"]["pant"] is None
        else:
            assert records[0]["score_semantics"] == "relative_softmax_given_captions"
            assert len(records[0]["category_scores"]) == len(pilot.CLAP_CAPTIONS)
        state = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
        assert state["status"] == "completed"
        assert state["completed"] == 2 and state["failed"] == 0
        assert state["model_revision"] == "mock-revision"
        assert (out / "failures.jsonl").read_text(encoding="utf-8") == ""
    assert len(seen) == 2
    with pytest.raises(FileExistsError):
        pilot.run(manifest, "ast", "ast_001", "cuda")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals.iterdir()}
    assert before == after


def test_initialization_failure_persists_manifest(selection, monkeypatch):
    manifest, originals, rows = selection
    def fail(model, device):
        raise RuntimeError("CUDA unavailable")
    monkeypatch.setattr(pilot, "_load_pipeline", fail)
    assert pilot.run(manifest, "ast", "fail_init", "cuda") == 1
    out = originals.parent.parent / "work" / "fail_init"
    run = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed_initialization"
    assert "CUDA unavailable" in run["initialization_error"]
    assert (out / "failures.jsonl").read_text(encoding="utf-8")


def test_sample_failure_does_not_approve(selection, monkeypatch):
    manifest, originals, rows = selection
    def bad_load(model, device):
        obj = FakePipeline(model)
        def fail(*args, **kwargs):
            raise ValueError("Model failure")
        obj.__call__ = fail  # handled below using proxy to simulate call failure
        class Broken:
            feature_extractor = obj.feature_extractor
            model = obj.model
            def __call__(self, *args, **kwargs):
                return fail(*args, **kwargs)
        return Broken()
    monkeypatch.setattr(pilot, "_load_pipeline", bad_load)
    assert pilot.run(manifest, "ast", "fail_samples", "cuda") == 1
    out = originals.parent.parent / "work" / "fail_samples"
    records = [json.loads(x) for x in (out / "style_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert all(x["status"] == "error" and not x["human_approved"] for x in records)
    assert json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["failed"] == 2
