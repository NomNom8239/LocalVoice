"""Reproducible project-local FFmpeg 7 shared runtime for TorchCodec 0.10.

No system PATH changes, PyTorch reinstalls, or writes to the voice dataset.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

# Pinned Gyan.D FFmpeg shared archive, from the WinGet
# Gyan.FFmpeg.Shared 7.0.2 manifest. Only the DLL-bearing bin/ is retained.
URL = ("https://github.com/GyanD/codexffmpeg/releases/download/7.0.2/"
       "ffmpeg-7.0.2-full_build-shared.zip")
SHA256 = "216feea30258c17628f666e20a2536da7bb50171e6e583a768968e11ceafc8a3"
VERSION = "7.0.2"
REQUIRED = ("avcodec-61.dll", "avformat-61.dll", "avutil-59.dll",
            "swresample-5.dll")


def validate_bin(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"FFmpeg shared bin directory missing: {path}")
    missing = [name for name in REQUIRED if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Not an FFmpeg 7 full-shared bin: missing {missing} in {path}")
    return path


def install(install_root: Path, *, opener=urlopen, sha256: str = SHA256,
            url: str = URL) -> Path:
    """SHA-256 verified and ZIP-slip-safe extraction; no system installation."""
    install_root = install_root.resolve()
    target = install_root / "ffmpeg-7.0.2-full-shared"
    marker = target / "source.json"
    if marker.is_file():
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            if metadata.get("sha256") == sha256:
                return validate_bin(target / "bin")
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        raise RuntimeError(
            f"Existing local FFmpeg runtime is unverified: {target}. "
            "Do not overwrite it automatically."
        )
    if target.exists():
        raise RuntimeError(f"Existing incomplete FFmpeg runtime: {target}")
    install_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".ffmpeg-verify-", dir=install_root))
    archive = scratch / "ffmpeg-shared.zip"
    try:
        h = hashlib.sha256()
        count = 0
        with opener(url, timeout=120) as resp, archive.open("wb") as output:
            while True:
                block = resp.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > 500 * 1024 * 1024:
                    raise RuntimeError("FFmpeg archive exceeds 500 MiB limit")
                output.write(block)
                h.update(block)
        if h.hexdigest() != sha256:
            raise RuntimeError("Pinned FFmpeg shared archive SHA-256 mismatch")
        staged_bin = scratch / "staged" / "bin"
        staged_bin.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            extracted = set()
            for info in z.infolist():
                # Only the top-level upstream bin files; never extract paths.
                segments = Path(info.filename.replace("\\", "/")).parts
                if (not info.is_dir() and len(segments) == 3
                        and segments[1] == "bin"
                        and (segments[2].lower().endswith(".dll")
                             or segments[2].lower() in {"ffmpeg.exe", "ffprobe.exe"})):
                    name = segments[2]
                    if name in extracted:
                        raise RuntimeError("Duplicate FFmpeg binary in archive")
                    extracted.add(name)
                    with z.open(info) as inp, (staged_bin / name).open("wb") as out:
                        shutil.copyfileobj(inp, out)
        validate_bin(staged_bin)
        (scratch / "staged" / "source.json").write_text(
            json.dumps({"url": url, "sha256": sha256, "version": VERSION}),
            encoding="utf-8",
        )
        (scratch / "staged").rename(target)
        return validate_bin(target / "bin")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def resolve(upstream: Path, explicit: str | None = None) -> Path:
    """Use a supplied shared bin, otherwise bootstrap pinned private runtime."""
    override = explicit or os.environ.get("LOCALVOICE_FFMPEG_SHARED_BIN")
    if override:
        return validate_bin(Path(override))
    return install(upstream / ".localvoice-toolchain")
