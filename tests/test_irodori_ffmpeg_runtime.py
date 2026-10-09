"""Regression tests for pinned, project-local FFmpeg shared runtime."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import irodori_ffmpeg_runtime as ff


class FFmpegRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as archive:
            for name in [*ff.REQUIRED, "ffprobe.exe", "ffmpeg.exe"]:
                archive.writestr(f"ffmpeg-7.0.2-full_build-shared/bin/{name}",
                                 (name + "-fixture").encode("ascii"))
            # Archive paths must never be extracted outside the staging bin.
            archive.writestr("../../system32/evil.dll", b"unexpected")
            archive.writestr("ffmpeg-7.0.2-full_build-shared/README.txt",
                             b"not-needed")
        self.archive = buffer.getvalue()
        self.hash = hashlib.sha256(self.archive).hexdigest()

    def test_download_extract_validate_and_reuse_without_network(self):
        counter = [0]

        def opener(url, timeout):
            counter[0] += 1
            self.assertEqual(timeout, 120)
            return io.BytesIO(self.archive)
        output = ff.install(self.root, opener=opener,
                            sha256=self.hash, url="https://example.test/file.zip")
        self.assertEqual(output.parent.name, "ffmpeg-7.0.2-full-shared")
        self.assertTrue(all((output / name).is_file() for name in ff.REQUIRED))
        self.assertFalse((self.root / "system32" / "evil.dll").exists())
        marker = json.loads((output.parent / "source.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["sha256"], self.hash)
        self.assertEqual(ff.install(self.root, opener=opener,
                                    sha256=self.hash), output)
        self.assertEqual(counter[0], 1)

    def test_wrong_hash_does_not_install_dlls(self):
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            ff.install(self.root, opener=lambda url, timeout: io.BytesIO(self.archive),
                       sha256="0" * 64)
        self.assertFalse((self.root / "ffmpeg-7.0.2-full-shared").exists())

    def test_explicit_full_shared_directory_must_contain_dlls(self):
        empty = self.root / "bin"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "missing"):
            ff.validate_bin(empty)

    def test_unverified_existing_folder_is_not_deleted_or_overwritten(self):
        target = self.root / "ffmpeg-7.0.2-full-shared"
        target.mkdir()
        (target / "my-data.txt").write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            ff.install(self.root, opener=lambda url, timeout: io.BytesIO(self.archive),
                       sha256=self.hash)
        self.assertEqual((target / "my-data.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
