from __future__ import annotations

import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.toml"
PROFILE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = (
        Path(path).expanduser().resolve()
        if path
        else DEFAULT_CONFIG
    )
    with config_path.open("rb") as f:
        return tomllib.load(f)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def data_dir(config: dict[str, Any], key: str) -> Path:
    root = project_path(config["paths"]["data_root"])
    return (root / config["paths"][key]).resolve()


def profile_name(value: str) -> str:
    if not PROFILE_RE.fullmatch(value):
        raise ValueError(
            "profile must contain only letters, numbers, '.', '_' or '-'"
        )
    return value


def reference_dir(config: dict[str, Any], profile: str) -> Path:
    return data_dir(config, "reference_bank") / profile_name(profile)


def rvc_dir(config: dict[str, Any], profile: str) -> Path:
    return data_dir(config, "rvc_dataset") / profile_name(profile)


def run_dir(config: dict[str, Any], profile: str) -> Path:
    return data_dir(config, "runs") / profile_name(profile)


def load_thresholds(
    config: dict[str, Any],
    profile: str,
) -> tuple[float, float, int]:
    fallback = (
        float(config["classification"]["accept_threshold"]),
        float(config["classification"]["review_threshold"]),
        int(config["classification"]["top_k"]),
    )
    path = reference_dir(config, profile) / "thresholds.json"
    if not path.exists():
        return fallback

    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        float(payload.get("accept_threshold", fallback[0])),
        float(payload.get("review_threshold", fallback[1])),
        int(payload.get("top_k", fallback[2])),
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid embedding")
    return vector / norm


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(
        path,
        dtype="float32",
        always_2d=True,
    )
    audio = (
        np.mean(audio, axis=1, dtype=np.float32)
        if audio.shape[1] > 1
        else audio[:, 0]
    )
    return audio.astype(np.float32, copy=False), int(sample_rate)


def resample(
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    gcd = math.gcd(sample_rate, target_sample_rate)
    return resample_poly(
        audio,
        target_sample_rate // gcd,
        sample_rate // gcd,
    ).astype(np.float32)


def embedding_backend(config: dict[str, Any], device: torch.device):
    model = config["models"]["pyannote_pipeline"]
    backend = PretrainedSpeakerEmbedding(
        {"checkpoint": model, "subfolder": "embedding"},
        device=device,
        token=True,
    )
    return backend, int(backend.sample_rate)


def embed_audio(
    backend,
    audio: np.ndarray,
    sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    audio = resample(audio, sample_rate, target_sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
    return normalize(np.asarray(backend(waveform), dtype=np.float32))


def top_k_score(
    candidate: np.ndarray,
    reference_bank: np.ndarray,
    k: int,
) -> float:
    similarities = reference_bank @ normalize(candidate)
    k = min(k, len(similarities))
    return float(np.mean(np.partition(similarities, -k)[-k:]))
