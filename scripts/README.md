# Script index

All tracked executable scripts live in this directory. Run them from the
repository root so relative paths and imports remain compatible.
The repository already uses a single scripts/ folder; do not create a
parallel script/ folder or relocate existing pipelines without updating
their imports, PowerShell entrypoints and tests.

## Voice collection / RVC dataset

| Purpose | Entry point |
| --- | --- |
| End-to-end extraction and speaker routing | localvoice.py |
| Diarization | diarize.py |
| Reference voice embeddings | build_reference_bank.py |
| Speaker threshold calibration | calibrate_threshold.py |
| Manual clip review | review_training_audio.py |
| Accumulate accepted audio | collect_training_audio.py |
| Build RVC dataset | build_rvc_dataset.py |
| Shared helpers (module, not standalone) | common.py |

## Irodori dataset and audio experiments

| Purpose | Entry point |
| --- | --- |
| PowerShell dispatcher | irodori.ps1 |
| Dataset scan / ASR suggestions / manual review / export | irodori_dataset.py |
| Automatic candidate routing (not final approval) | irodori_autoprep.py |
| Style suggestions | irodori_style.py |
| Nonverbal candidate pilot | irodori_nonverbal_pilot.py |
| Dataset/tokenization preflight | irodori_nonverbal_check.py |
| Isolated codec smoke test | irodori_nonverbal_manifest.py |
| Codec DLL wrapper (internal) | irodori_codec_entry.py |
| FFmpeg 7 shared-DLL runtime (internal) | irodori_ffmpeg_runtime.py |
| Nonverbal LoRA pilot | irodori_nonverbal_lora.py |
| Base vs LoRA comparison | irodori_nonverbal_compare.py |
| Audio/ASR comparison | irodori_nonverbal_evaluate.py |
| Separate speech-retention ASR | irodori_speech_retention_asr.py |
| **LV-03 DACVAE recovery without Codex** | **lv03_no_codex.py** |

Existing script paths are preserved because the dispatcher and tests refer to
them. Documentation and tests stay in docs/ and tests/, not in scripts/.

### LV-03: no Codex credits needed

From F:\AIProjects\LocalVoice:

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv03_no_codex.py
~~~

Uses existing Irodori-TTS/.venv and FFmpeg 7 shared libraries.
Checks eight approved training clips and three independent evaluation clips.
Runs a one-clip DACVAE diagnostic before attempting all eight approved
training clips. Does not train LoRA or generate synthetic audio.

Every run creates unused output directories under
Irodori-TTS/outputs/localvoice_lora_dataset/lv03_dacvae_NNN/.
The failed old attempts, source WAVs and previous model outputs are retained.
Logs include process.log, invocation.json and validation.log.

To stop after the single diagnostic:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\lv03_no_codex.py --diagnose-only
~~~

For other Irodori commands, see [dataset documentation](../docs/irodori_dataset.md).
