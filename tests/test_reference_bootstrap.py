"""Offline first-profile Reference Bank bootstrap: no network or model downloads."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("localvoice_legacy_acquisition", SCRIPTS / "localvoice.py")
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_spaced_samples_across_archive():
    assert module.spaced_samples(list(range(15)), 4) == [0, 5, 9, 14]
    assert module.spaced_samples([1, 2], 5) == [1, 2]
    assert module.spaced_samples([], 3) == []


def test_reference_qc_excludes_overlap_clipping_silence_and_bad_durations():
    rate = 16000
    audio = np.full(40 * rate, 0.18, dtype=np.float32)
    audio[20 * rate:24 * rate] = 1.0  # clipped
    audio[24 * rate:28 * rate] = 0  # silent
    turns = [
        (0, 4, "SPEAKER_00"),
        (4, 8, "SPEAKER_00"),
        (8, 12, "SPEAKER_00"),
        (12, 16, "SPEAKER_00"),   # overlaps speaker_01
        (14, 17, "SPEAKER_01"),
        (20, 24, "SPEAKER_00"),
        (24, 28, "SPEAKER_00"),
        (30, 31, "SPEAKER_00"),  # too short
    ]
    result = module.reference_candidates(audio, rate, turns)
    assert result == {
        "SPEAKER_00": [(0, 4, "SPEAKER_00"), (4, 8, "SPEAKER_00"),
                       (8, 12, "SPEAKER_00")]
    }


def test_bootstrap_uses_previewed_speaker_and_staged_references(tmp_path, monkeypatch):
    rate = 16000
    audio = np.full(rate * 40, 0.1, dtype=np.float32)
    vocals = tmp_path / "Vocals.wav"
    sf.write(vocals, audio, rate)
    turns = [(i * 4.0, (i + 1) * 4.0, "SPEAKER_00") for i in range(6)]
    turns += [(24, 28, "SPEAKER_01"), (28, 32, "SPEAKER_01")]
    output = tmp_path / "run"
    root = tmp_path / "bank"
    called = []

    monkeypatch.setattr(module, "reference_dir", lambda config, profile: root)
    monkeypatch.setattr(module, "diarize_turns", lambda *args: turns)
    monkeypatch.setattr(
        module, "choose_bootstrap_speakers",
        lambda candidates, previews: ["SPEAKER_00"],
    )

    def fake_run(args, **kwargs):
        called.append(args)
        assert args[-4:] == ["--profile", "Test_Speaker", "--source", str(output / "bootstrap" / "selected_reference_wavs")]
        assert not (root / "self_reference_bank.npz").exists()
        assert len(list((output / "bootstrap" / "selected_reference_wavs").glob("*.wav"))) == 6
        root.mkdir(parents=True)
        (root / "self_reference_bank.npz").write_bytes(b"FAKE")
        (root / "self_reference_bank.json").write_text("{}")
        return ""

    monkeypatch.setattr(module, "run", fake_run)
    returned = module.bootstrap_reference_bank({}, "Test_Speaker", vocals, output)
    assert returned == turns
    assert len(called) == 1
    assert len(list((output / "bootstrap" / "previews").glob("*.wav"))) == 5
    assert len(list((output / "bootstrap" / "selected_reference_wavs").glob("*.wav"))) == 6


def test_existing_bank_not_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "bank"
    root.mkdir()
    existing = root / "self_reference_bank.npz"
    existing.write_bytes(b"OLD")
    monkeypatch.setattr(module, "reference_dir", lambda *args: root)
    with pytest.raises(RuntimeError, match="already exists"):
        module.bootstrap_reference_bank({}, "Test", tmp_path / "x.wav", tmp_path / "run")
    assert existing.read_bytes() == b"OLD"
    assert not (tmp_path / "run").exists()


def test_bootstrap_requires_confirmed_identity_and_supports_multiple_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(module.sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    responses = iter(["SPEAKER_00,SPEAKER_01", "YES"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    paths = {"SPEAKER_00": [tmp_path / "a.wav"], "SPEAKER_01": [tmp_path / "b.wav"]}
    sources = {
        "SPEAKER_00": [(1, 5, "SPEAKER_00"), (10, 14, "SPEAKER_00")],
        "SPEAKER_01": [(20, 24, "SPEAKER_01")],
    }
    assert module.choose_bootstrap_speakers(sources, paths) == ["SPEAKER_00", "SPEAKER_01"]


def test_noninteractive_selection_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(module.sys, "stdin", type("Pipe", (), {"isatty": lambda self: False})())
    with pytest.raises(RuntimeError, match="interactive terminal"):
        module.choose_bootstrap_speakers({"S0": [(0, 3, "S0")]}, {"S0": [tmp_path / "a.wav"]})


def test_run_once_reuses_diarization_after_bootstrap(tmp_path, monkeypatch):
    vocals = tmp_path / "vocals.wav"
    sf.write(vocals, np.full(16000 * 5, 0.1, dtype=np.float32), 16000)
    reference = tmp_path / "ref"
    reference.mkdir()
    np.savez(reference / "self_reference_bank.npz", embeddings=np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(module, "reference_dir", lambda *_: reference)
    monkeypatch.setattr(module, "load_audio_mono", lambda p: (np.full(16000 * 5, 0.1, dtype=np.float32), 16000))
    monkeypatch.setattr(module, "embedding_backend", lambda *_: (object(), 16000))
    monkeypatch.setattr(module, "embed_audio", lambda *args: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(module, "top_k_score", lambda *args: 0.9)
    monkeypatch.setattr(module, "diarize_turns", lambda *args: pytest.fail("diarization must not rerun"))
    config = {"classification": {"min_score_duration": 2.0, "max_score_duration": 20.0}}
    module.classify(
        config, "Test", vocals, tmp_path / "run", 0.5, 0.3, 1,
        precomputed_turns=[(0.0, 5.0, "SPEAKER_00")],
    )
    assert len(list((tmp_path / "run" / "my_voice").glob("*.wav"))) == 1


def test_main_without_bank_invokes_bootstrap_for_vocals(tmp_path, monkeypatch):
    vocals = tmp_path / "vocals.wav"
    vocals.touch()
    root = tmp_path / "runs"
    bank = tmp_path / "ref"
    monkeypatch.setattr(module, "load_config", lambda *_: {})
    monkeypatch.setattr(module, "load_thresholds", lambda *args: (0.5, 0.3, 1))
    monkeypatch.setattr(module, "run_dir", lambda *args: root)
    monkeypatch.setattr(module, "reference_dir", lambda *args: bank)
    monkeypatch.setattr(module, "project_path", lambda value: Path(value))
    seen = []
    monkeypatch.setattr(module, "bootstrap_reference_bank", lambda *args: [(0.0, 4.0, "SPEAKER_00")])
    monkeypatch.setattr(module, "classify", lambda *args, **kwargs: seen.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["localvoice.py", "--profile", "New_Speaker", "--vocals", str(vocals)])
    module.main()
    assert len(seen) == 1
    assert seen[0]["precomputed_turns"] == [(0.0, 4.0, "SPEAKER_00")]
    assert (root / "New_Speaker" / "vocals").is_dir()
