"""Execute upstream Irodori with an isolated Windows FFmpeg DLL directory."""
from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: irodori_codec_entry.py FFMPEG_SHARED_BIN PREPARE_MANIFEST.py [args...]")
    shared_bin = Path(sys.argv[1]).resolve()
    script = Path(sys.argv[2]).resolve()
    if not shared_bin.is_dir() or not script.is_file():
        raise RuntimeError("FFmpeg shared bin or upstream script missing")
    if os.name != "nt":
        raise RuntimeError("The isolated FFmpeg DLL entrypoint is Windows-only")
    # Keep handle alive until runpy returns / process exits. The parent
    # PATH is never changed, avoiding global FFmpeg 9 vs 7 conflicts.
    dll_handle = os.add_dll_directory(str(shared_bin))
    os.environ["PATH"] = str(shared_bin) + os.pathsep + os.environ.get("PATH", "")
    sys.argv = [str(script), *sys.argv[3:]]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        dll_handle.close()


if __name__ == "__main__":
    main()
