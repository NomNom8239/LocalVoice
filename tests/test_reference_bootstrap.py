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
        assert len(list((output / "bootstrap" / "selected_reference_wavs").glob("*.wav"))) == 3
        root.mkdir(parents=True)
        (root / "self_reference_bank.npz").write_bytes(b"FAKE")
        (root / "self_reference_bank.json").write_text("{}")
        return ""

    monkeypatch.setattr(module, "run", fake_run)
    returned = module.bootstrap_reference_bank({}, "Test_Speaker", vocals, output)
    assert returned == turns
    assert len(called) == 1
    assert len(list((output / "bootstrap" / "previews").glob("*.wav"))) == 5
    assert len(list((output / "bootstrap" / "selected_reference_wavs").glob("*.wav"))) == 3


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
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).touch()
    paths = {
        "SPEAKER_00": [tmp_path / "a.wav", tmp_path / "c.wav"],
        "SPEAKER_01": [tmp_path / "b.wav"],
    }
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
    monkeypatch.setattr(module, "run_dir", lambda config, profile: root / profile)
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


def test_main_existing_bank_skips_bootstrap(tmp_path, monkeypatch):
    vocals = tmp_path / "existing.wav"
    vocals.touch()
    root = tmp_path / "runs"
    reference = tmp_path / "ref"
    reference.mkdir()
    (reference / "self_reference_bank.npz").write_bytes(b"EXISTING")
    (reference / "self_reference_bank.json").write_text("{}")
    monkeypatch.setattr(module, "load_config", lambda *_: {})
    monkeypatch.setattr(module, "load_thresholds", lambda *args: (0.5, 0.3, 1))
    monkeypatch.setattr(module, "run_dir", lambda config, profile: root / profile)
    monkeypatch.setattr(module, "reference_dir", lambda *args: reference)
    monkeypatch.setattr(module, "project_path", lambda value: Path(value))
    monkeypatch.setattr(
        module, "bootstrap_reference_bank",
        lambda *args: pytest.fail("should not bootstrap existing reference"),
    )
    called = []
    monkeypatch.setattr(module, "classify", lambda *args, **kwargs: called.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["localvoice.py", "--profile", "Existing", "--vocals", str(vocals)])
    module.main()
    assert len(called) == 1
    assert called[0]["precomputed_turns"] is None
    assert (reference / "self_reference_bank.npz").read_bytes() == b"EXISTING"
    assert (root / "Existing" / "existing").is_dir()
    assert not (root / "Existing" / "existing" / "bootstrap").exists()


def test_preview_requires_confirmed_person_and_never_guesses_label(monkeypatch, tmp_path):
    monkeypatch.setattr(module.sys, "stdin", type("TTY", (), {"isatty": lambda self: True})())
    inputs = iter(["SPEAKER_00", "NO", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    with pytest.raises(RuntimeError, match="canceled"):
        wavs = [tmp_path / f"clip{i}.wav" for i in range(3)]
        for wav in wavs:
            wav.touch()
        module.choose_bootstrap_speakers(
            {"SPEAKER_00": [(0, 5, "SPEAKER_00")] * 3},
            {"SPEAKER_00": wavs},
        )


def test_main_incomplete_bank_refuses_classification(tmp_path, monkeypatch):
    vocals = tmp_path / "sample.wav"
    vocals.touch()
    ref = tmp_path / "reference"
    ref.mkdir()
    (ref / "self_reference_bank.npz").write_bytes(b"PARTIAL")
    monkeypatch.setattr(module, "load_config", lambda *_: {})
    monkeypatch.setattr(module, "load_thresholds", lambda *args: (0.5, 0.3, 1))
    monkeypatch.setattr(module, "run_dir", lambda config, profile: tmp_path / "runs" / profile)
    monkeypatch.setattr(module, "reference_dir", lambda *args: ref)
    monkeypatch.setattr(module, "project_path", lambda path: Path(path))
    monkeypatch.setattr(module, "classify", lambda *args, **kwargs: pytest.fail("must fail before scoring"))
    monkeypatch.setattr(sys, "argv", ["localvoice.py", "--profile", "Partial", "--vocals", str(vocals)])
    with pytest.raises(RuntimeError, match="Incomplete Reference Bank"):
        module.main()
    assert (ref / "self_reference_bank.npz").read_bytes() == b"PARTIAL"
    assert not (tmp_path / "runs").exists()


def test_mixed_speaker_samples_select_exact_human_confirmed_preview_files(tmp_path):
    root = tmp_path / "bootstrap" / "previews"
    root.mkdir(parents=True)
    previews = {}
    for speaker in ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]:
        files = []
        for index in (1, 2, 3):
            path = root / f"{speaker}_{index:02d}.wav"
            path.write_bytes(f"{speaker}: {index}".encode())
            files.append(path)
        previews[speaker] = files

    approved = module.resolve_bootstrap_preview_selection(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02_02", "SPEAKER_02_03"],
        previews,
    )
    assert len(approved) == 8
    assert set(approved) == (
        set(previews["SPEAKER_00"])
        | set(previews["SPEAKER_01"])
        | set(previews["SPEAKER_02"][1:])
    )
    assert previews["SPEAKER_02"][0] not in approved
    assert not any(x in approved for x in previews["SPEAKER_03"])
    with pytest.raises(ValueError, match="Unknown preview"):
        module.resolve_bootstrap_preview_selection(["SPEAKER_04"], previews)


def test_preview_only_reference_generation_does_not_promote_unreviewed_turns(
    tmp_path, monkeypatch,
):
    profile = "Fresh"
    run_root = tmp_path / "run"
    preview_root = run_root / "bootstrap" / "previews"
    preview_root.mkdir(parents=True)
    speaker_files = []
    for i in (1, 2, 3):
        file = preview_root / f"SPEAKER_00_{i:02d}.wav"
        file.write_bytes(f"approved-{i}".encode())
        speaker_files.append(file)
    another = preview_root / "SPEAKER_01_01.wav"
    another.write_bytes(b"other-speaker")
    root = tmp_path / "ref"
    monkeypatch.setattr(module, "reference_dir", lambda *_: root)
    monkeypatch.setattr(
        module, "choose_bootstrap_speakers",
        lambda _candidates, _previews: ["SPEAKER_00"],
    )

    def fake_run(args, **kwargs):
        stage = Path(args[-1])
        staged = sorted(stage.glob("*.wav"))
        assert len(staged) == 3
        assert sorted(p.read_bytes() for p in staged) == sorted(p.read_bytes() for p in speaker_files)
        root.mkdir()
        (root / "self_reference_bank.npz").write_bytes(b"synthetic")
        (root / "self_reference_bank.json").write_text("{}")
        return ""

    monkeypatch.setattr(module, "run", fake_run)
    module.build_approved_bootstrap_bank(
        {}, profile, run_root,
        {"SPEAKER_00": speaker_files, "SPEAKER_01": [another]},
    )


def test_resume_bootstrap_uses_existing_previews_and_vocals_without_separator(
    tmp_path, monkeypatch,
):
    output = tmp_path / "runs" / "Fresh" / "previous_run"
    preview_root = output / "bootstrap" / "previews"
    preview_root.mkdir(parents=True)
    for i in (1, 2, 3):
        (preview_root / f"SPEAKER_00_{i:02d}.wav").touch()
    separated = output / "separated"
    separated.mkdir()
    vocals = separated / "audio_Vocals.wav"
    vocals.touch()
    ref = tmp_path / "ref"
    monkeypatch.setattr(module, "reference_dir", lambda *_: ref)
    seen = []
    monkeypatch.setattr(
        module, "build_approved_bootstrap_bank",
        lambda _config, _profile, _output, preview: seen.append(("bank", preview)),
    )
    monkeypatch.setattr(
        module, "classify",
        lambda _config, _profile, source, _output, *_: seen.append(("classify", source)),
    )
    module.resume_bootstrap({}, "Fresh", output, None, 0.7, 0.4, 3)
    assert [entry[0] for entry in seen] == ["bank", "classify"]
    assert len(seen[0][1]["SPEAKER_00"]) == 3
    assert seen[1][1] == vocals


def test_resume_refuses_previous_classification_without_touching_it(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    manifest = output / "classification.tsv"
    manifest.write_bytes(b"ORIGINAL")
    monkeypatch.setattr(module, "reference_dir", lambda *_: tmp_path / "bank")
    with pytest.raises(RuntimeError, match="already classified"):
        module.resume_bootstrap({}, "Fresh", output, None, 0.7, 0.4, 3)
    assert manifest.read_bytes() == b"ORIGINAL"


def test_resume_previews_loader_preserves_speaker_boundaries(tmp_path):
    base = tmp_path / "run" / "bootstrap" / "previews"
    base.mkdir(parents=True)
    for name in ["SPEAKER_00_01.wav", "SPEAKER_01_01.wav", "SPEAKER_02_02.wav"]:
        (base / name).touch()
    previews = module.existing_bootstrap_previews(tmp_path / "run")
    assert {key: len(value) for key, value in previews.items()} == {
        "SPEAKER_00": 1, "SPEAKER_01": 1, "SPEAKER_02": 1,
    }
