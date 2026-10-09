"""Contract tests for LocalVoice-owned Irodori VoiceDesign UI patch.

Do not modify the ignored external Irodori-TTS checkout during tests.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "irodori" / "gradio_app_voicedesign-caption-presets.patch"
UPSTREAM = ROOT / "Irodori-TTS" / "gradio_app_voicedesign.py"


def _patch() -> str:
    return PATCH.read_text(encoding="utf-8")


def _added_python() -> str:
    # Grab only additions in the top-level definitions block (not the UI hooks).
    additions = [
        line[1:] for line in _patch().splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    start = additions.index("# LOCALVOICE_CAPTION_PRESETS_BEGIN")
    end = additions.index("# LOCALVOICE_CAPTION_PRESETS_END")
    return "\n".join(additions[start : end + 1]) + "\n"


def test_eight_unique_caption_templates_and_selection_behavior() -> None:
    source = _added_python()
    ast.parse(source)
    scope: dict[str, object] = {}
    exec(compile(source, "<caption-presets>", "exec"), scope)
    items = scope["CAPTION_PRESET_ITEMS"]
    assert isinstance(items, tuple) and len(items) == 8
    labels = [label for label, _ in items]
    assert labels == [
        "01｜消耗と微かな反抗",
        "02｜諦めと虚脱",
        "03｜屈辱と反抗心",
        "04｜感情の崩壊",
        "05｜虚勢と脆さ",
        "06｜陶然と混乱",
        "07｜静かな絶望と冷笑",
        "08｜限界寸前の緊張と震え",
    ]
    assert len(set(labels)) == 8
    assert all(isinstance(caption, str) and len(caption) >= 50 for _, caption in items)
    callback = scope["_select_caption_preset"]
    for label, caption in items:
        assert callback(label, "custom text") == caption
    assert callback(None, "custom text") == "custom text"
    assert callback("", "custom text") == "custom text"


def test_patch_adds_only_ui_selection_without_touching_inference() -> None:
    patch = _patch()
    assert patch.count("@@ -") == 3
    assert not any(line.startswith("-") and not line.startswith("---") for line in patch.splitlines())
    assert 'caption_preset = gr.Dropdown(' in patch
    assert 'choices=[label for label, _ in CAPTION_PRESET_ITEMS]' in patch
    assert 'inputs=[caption_preset, caption]' in patch
    assert 'outputs=[caption]' in patch
    assert '_run_generation' in patch
    assert 'gradio_app_voicedesign.py' in patch


def test_git_patch_applies_and_reverses_on_pinned_context(tmp_path: Path) -> None:
    # Reconstruct exact 3-line context for each hunk at its original line numbers.
    # This checks the actual diff format without vendoring the whole upstream app.
    patch_lines = _patch().splitlines()
    hunks: list[tuple[int, int, list[str]]] = []
    current: tuple[int, int, list[str]] | None = None
    for line in patch_lines:
        match = re.match(r"@@ -(\d+),(\d+) \+\d+,\d+ @@", line)
        if match:
            current = (int(match.group(1)), int(match.group(2)), [])
            hunks.append(current)
        elif current is not None and line.startswith((" ", "-")):
            current[2].append(line[1:])
    assert len(hunks) == 3
    total = max(start + count for start, count, _ in hunks) + 4
    base = [f"# filler {i}" for i in range(total)]
    for start, count, before in hunks:
        assert len(before) == count
        base[start - 1 : start - 1 + count] = before
    upstream = tmp_path / "gradio_app_voicedesign.py"
    original = "\n".join(base) + "\n"
    upstream.write_text(original, encoding="utf-8")
    for args in [
        ["git", "apply", "--check", str(PATCH)],
        ["git", "apply", str(PATCH)],
        ["git", "apply", "--reverse", "--check", str(PATCH)],
        ["git", "apply", "--reverse", str(PATCH)],
    ]:
        process = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
    assert upstream.read_text(encoding="utf-8") == original


def test_checkout_is_applicable_or_already_applied_when_present() -> None:
    if not UPSTREAM.is_file():
        pytest.skip("external Irodori-TTS checkout not available")
    for args in (["git", "apply", "--check", str(PATCH)],
                 ["git", "apply", "--reverse", "--check", str(PATCH)]):
        process = subprocess.run(args, cwd=UPSTREAM.parent, capture_output=True, text=True)
        if process.returncode == 0:
            return
    pytest.fail("External VoiceDesign UI differs from pinned upstream; reconcile before application")
