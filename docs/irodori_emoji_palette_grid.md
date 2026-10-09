# Irodori Emoji Palette: labelled responsive grid

The VoiceDesign UI already uses upstream `irodori_tts/gradio_emoji_palette.py`.
This change updates **only the shared palette UI**, with 45 emoji entries.

- Show each entry as a card: emoji + Japanese effect name + a short explanation.
- Align cards in a responsive multi-column grid; stack on narrow displays.
- Allow vertical scrolling when many cards are visible.
- Preserve existing emoji insertion at the text cursor / selection, `data-emoji`,
  tooltip, accessible label, keyboard focus, and HTML escaping.
- Translate ten short English effect explanations into Japanese.
- Does **not** modify existing Caption templates, Gradio Text inputs, model
  loading, inference, voice references, or LoRA.
- Because the palette is shared, the layout appears in **both** Irodori Gradio
  UIs that call `build_emoji_palette`.

## Repository boundary

`F:\AIProjects\LocalVoice\Irodori-TTS` is a **separate, gitignored
Irodori-TTS checkout**. The LocalVoice repository owns only a minimal patch:

```text
patches/irodori/gradio_emoji_palette-labelled-grid.patch
```

The baseline is official `Aratako/Irodori-TTS@89f9d8fbd4d51ea019867ee1197725ede1df13c5`
and the target is **only** `irodori_tts/gradio_emoji_palette.py`.

Merging the LocalVoice PR records the patch, but it does not automatically
modify the separate local Irodori-TTS checkout.

## Apply to the existing checkout without switching or merging branches

From `F:\AIProjects\LocalVoice` in PowerShell. This command retrieves **only
the patch contents** from the merged `feature/phase4-acoustic-pilot`
branch, checks them without changing either working-tree branch, and applies
to the **existing** checkout. The Python subprocess passes bytes without
PowerShell text re-encoding:

```powershell
git fetch origin feature/phase4-acoustic-pilot
git -C Irodori-TTS status --short -- irodori_tts/gradio_emoji_palette.py

@'
import subprocess

def git(*args, input=None, check=True):
    return subprocess.run(["git", *args], input=input, capture_output=True, check=check)

target = "irodori_tts/gradio_emoji_palette.py"
dirty = git("-C", "Irodori-TTS", "status", "--porcelain", "--", target).stdout
if dirty:
    raise SystemExit("Palette has local modifications. Inspect before applying; nothing changed.")
patch = git("show", "origin/feature/phase4-acoustic-pilot:patches/irodori/gradio_emoji_palette-labelled-grid.patch").stdout
def probe(*args):
    return git("-C", "Irodori-TTS", "apply", *args, "-", input=patch, check=False)
if probe("--reverse", "--check").returncode == 0:
    print("Emoji palette overlay is already applied.")
elif probe("--check").returncode == 0:
    git("-C", "Irodori-TTS", "apply", "-", input=patch)
    print("Emoji palette overlay applied.")
else:
    raise SystemExit("Palette differs from supported baseline. No changes made.")
'@ | .\.venv\Scripts\python.exe -

git -C Irodori-TTS diff -- irodori_tts/gradio_emoji_palette.py
```

**Important:** If the target file is already modified locally, inspect the
diff and do not blindly overwrite it. Since `git apply` only changes the
targeted palette file, the previously applied Caption preset change in
`gradio_app_voicedesign.py` is untouched.

Restart the existing VoiceDesign Web UI to pick up the CSS/HTML change.

## Verification

For a LocalVoice worktree containing this branch's tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_irodori_emoji_palette_patch.py -q
```

Tests cover patch target and insertion attributes, git apply/reverse in a
temporary directory, and (when the existing external checkout is present)
the 45 entries in rendered HTML with effect labels and descriptions. The
Windows Gradio browser click / caret placement is still a runtime acceptance
check. Reversing the patch should use the exact recorded patch, and only
after reviewing any later edits to that target file.
