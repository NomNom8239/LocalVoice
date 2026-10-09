"""Validate the optional labelled Irodori emoji-palette UI overlay.

The external Irodori-TTS checkout is ignored by LocalVoice and is never changed
by these tests. Git patch application is validated in temporary directories.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from html import escape
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "irodori" / "gradio_emoji_palette-labelled-grid.patch"
TARGET = ROOT / "Irodori-TTS" / "irodori_tts" / "gradio_emoji_palette.py"


def _git_apply(directory: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "apply", *options, str(PATCH)],
        cwd=directory,
        text=True,
        capture_output=True,
        check=False,
    )


def test_patch_targets_only_shared_palette_and_preserves_insertion() -> None:
    patch = PATCH.read_text(encoding="utf-8")
    assert patch.count("diff --git ") == 1
    assert "a/irodori_tts/gradio_emoji_palette.py" in patch
    assert "b/irodori_tts/gradio_emoji_palette.py" in patch
    assert 'grid-template-columns: repeat(auto-fit, minmax(min(100%, 230px), 1fr));' in patch
    assert 'max-height: min(56vh, 420px);' in patch
    assert 'class="emoji-palette-glyph"' in patch
    assert 'class="emoji-palette-label"' in patch
    assert 'class="emoji-palette-description"' in patch
    assert 'data-emoji="{emoji}"' in patch
    assert 'onpointerdown="{handler}"' in patch
    assert 'aria-label="{title}"' in patch
    # No JS insertion handler or app's TTS inference functions may be edited.
    assert "_INSERT_EMOJI_ON_POINTER_DOWN =" not in patch
    assert "_run_generation" not in patch


def _synthetic_context_from_patch() -> str:
    """Rebuild exactly the old-side unified diff contexts, without upstream checkout."""
    text = PATCH.read_text(encoding="utf-8")
    lines = text.splitlines()
    hunks: list[tuple[int, int, list[str]]] = []
    old_line = None
    count = 0
    collected: list[str] = []
    for line in lines:
        m = re.match(r"^@@ -(\d+),(\d+) \+\d+,\d+ @@", line)
        if m:
            if old_line is not None:
                assert len(collected) == count
                hunks.append((old_line, count, collected))
            old_line, count = int(m.group(1)), int(m.group(2))
            collected = []
        elif old_line is not None and line.startswith((" ", "-")):
            collected.append(line[1:])
    assert old_line is not None
    assert len(collected) == count
    hunks.append((old_line, count, collected))
    assert len(hunks) >= 2
    old = [f"# filler line {i}" for i in range(max(x + n for x, n, _ in hunks) + 3)]
    for start, size, context in hunks:
        assert all(v.startswith("# filler line ") for v in old[start - 1 : start - 1 + size])
        old[start - 1 : start - 1 + size] = context
    return "\n".join(old) + "\n"


def test_patch_apply_reverse_round_trip(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    target = tmp_path / "irodori_tts" / "gradio_emoji_palette.py"
    target.parent.mkdir(parents=True)
    initial = _synthetic_context_from_patch()
    target.write_text(initial, encoding="utf-8")
    for args in (("--check",), (), ("--reverse", "--check"), ("--reverse",)):
        result = _git_apply(tmp_path, *args)
        assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8") == initial


def _patched_source(tmp_path: Path) -> str:
    if not TARGET.is_file():
        pytest.skip("existing Irodori-TTS checkout is not mounted")
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    working = tmp_path / "irodori_tts"
    working.mkdir()
    temp_target = working / "gradio_emoji_palette.py"
    shutil.copy2(TARGET, temp_target)
    if _git_apply(tmp_path, "--reverse", "--check").returncode == 0:
        # Already applied on the user's existing checkout.
        return temp_target.read_text(encoding="utf-8")
    check = _git_apply(tmp_path, "--check")
    assert check.returncode == 0, (
        "Existing Irodori palette differs from pinned upstream. "
        "Inspect the patch instead of forcing it: " + check.stderr
    )
    applied = _git_apply(tmp_path)
    assert applied.returncode == 0, applied.stderr
    return temp_target.read_text(encoding="utf-8")


def test_rendered_html_has_all_labels_and_descriptions(tmp_path: Path) -> None:
    code = _patched_source(tmp_path)
    tree = ast.parse(code)
    required = {
        "EmojiPaletteItem", "EMOJI_PALETTE_CSS", "EMOJI_PALETTE_ITEMS",
        "_INSERT_EMOJI_ON_POINTER_DOWN", "_textbox_selector", "_emoji_palette_html",
    }
    selected = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in required)
        or (isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(name.id in required for name in ast.walk(node) if isinstance(name, ast.Name)))
    ]
    assert selected
    scope = {
        "dataclass": dataclass,
        "escape": escape,
        "gr": SimpleNamespace(Textbox=object),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<palette>", "exec"), scope)
    html = scope["_emoji_palette_html"](SimpleNamespace(elem_id="test-input"))
    items = scope["EMOJI_PALETTE_ITEMS"]
    assert len(items) == 45
    assert html.count('<button type="button"') == 45
    assert html.count('class="emoji-palette-glyph"') == 45
    assert html.count('class="emoji-palette-label"') == 45
    assert html.count('class="emoji-palette-description"') == 45
    assert html.count('onpointerdown=') == 45
    for item in items:
        assert escape(item.label) in html
        assert escape(item.description) in html
        assert escape(item.emoji, quote=True) in html
    css = scope["EMOJI_PALETTE_CSS"]
    assert "display: grid;" in css
    assert "focus-visible" in css
    assert "overflow-y: auto" in css
