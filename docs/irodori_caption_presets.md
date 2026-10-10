# VoiceDesign Inference: Caption templates (LocalVoice overlay)

This integration deliberately preserves the external `Irodori-TTS/` checkout:
it is excluded by LocalVoice's `.gitignore` and must **not** be vendored or
silently treated as tracked LocalVoice source.

## Source and scope

- Target: `Irodori-TTS/gradio_app_voicedesign.py`
- Upstream baseline: `Aratako/Irodori-TTS@89f9d8fbd4d51ea019867ee1197725ede1df13c5`
- Repository-owned change: `patches/irodori/gradio_app_voicedesign-caption-presets.patch`
- Behavior: eight named Japanese Caption presets in a Gradio dropdown immediately
  above the existing Caption textbox. Selecting one replaces the textbox value;
  the value remains manually editable. Clearing the dropdown preserves manual text.
- No modifications to `_run_generation`, `SamplingRequest`, Text, reference audio,
  LoRA, model loading, or sampling defaults.

## Verify and apply on the existing Windows checkout

From `F:\AIProjects\LocalVoice`, after pulling the merged LocalVoice branch:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_irodori_caption_presets_patch.py -q

# Inspect the external checkout state first (including any local edits).
git -C Irodori-TTS status --short

# Dry-run: refuse incompatible local upstream changes.
git -C Irodori-TTS apply --check ../patches/irodori/gradio_app_voicedesign-caption-presets.patch

# Apply only after the check succeeds.
git -C Irodori-TTS apply ../patches/irodori/gradio_app_voicedesign-caption-presets.patch

# Verify what changed inside the external Irodori-TTS checkout.
git -C Irodori-TTS diff -- gradio_app_voicedesign.py
```

Restart the VoiceDesign Gradio process to display the dropdown.

To undo **only this overlay** if the checkout contains that patch:

```powershell
git -C Irodori-TTS apply --reverse --check ../patches/irodori/gradio_app_voicedesign-caption-presets.patch
git -C Irodori-TTS apply --reverse ../patches/irodori/gradio_app_voicedesign-caption-presets.patch
```

Do not run `git reset --hard` or replace the whole upstream file: either would
risk destroying local changes. If the `--check` fails because the upstream app
has drifted or a previous local patch was already applied, reconcile its diff
instead of forcing the patch.

## Verification boundaries

- `pytest` validates the 8 complete captions, UI event arguments, and the
  exact git patch's apply / reverse round-trip.
- When `Irodori-TTS/` is present, tests accept only an applicable or
  already-applied patch; upstream drift blocks the test.
- This is a UI-only change; a real Gradio launch and selection interaction on
  the user's Windows install remain a separate runtime acceptance check.
- A LocalVoice PR merge only adds the **tracked overlay** to LocalVoice.
  It cannot automatically update the separately checked-out, ignored upstream
  repository. That checkout must be updated with the commands above, or the
  upstream repository must first be forked and managed separately.
