# Irodori-TTS dataset preparation

LocalVoice owns collection, speaker review and provenance. Irodori-TTS remains a separate training/inference engine.

## Windows PowerShell commands

From the LocalVoice repository root:

```powershell
.\scripts\irodori.ps1 status -Speaker Toto_Kogara
.\scripts\irodori.ps1 asr -Speaker Toto_Kogara -RetryErrors -Limit 1
.\scripts\irodori.ps1 asr -Speaker Toto_Kogara -RetryErrors -Limit 30
.\scripts\irodori.ps1 merge -Speaker Toto_Kogara
.\scripts\irodori.ps1 export -Speaker Toto_Kogara
```

Use `scan` only when no review workspace exists. An existing `Irodori-TTS/outputs/localvoice_lora_dataset/review.csv` is reused automatically; otherwise the workspace is `data/irodori_lora/Toto_Kogara/`.

The input WAV directory is `data/training_audio/Toto_Kogara/audio/` and is never changed. Git excludes `data/`. Existing `training_audio/Toto_Kogara/manifest.tsv` supplies optional provenance. Never copy material back into `reference_bank` automatically.

### ASR environment

The first `asr` invocation creates a separate `LocalVoice/.venv-asr` with `uv` and installs `faster-whisper`. Later runs reuse it. CPU/int8 is the default, preserving the existing RVC and Irodori CUDA environments. On a failure, `asr_suggestions.csv` retains `error_detail`, and processing stops after three consecutive errors. `-RetryErrors` retries only previously failed suggestions; try `-Limit 1` first.

### Human review gate

For each approved clip, listen to the audio and enter a verified `text`, `style`, and a `caption` for special delivery. Set `speaker_ok=yes`, `quality=good`, `decision=approved` only after checking the content.

ASR writes suggestions, never verified text or approval. Short clips remain candidates for manual review. No algorithm is claimed to reliably label breaths or groans without listening.

`export` rechecks file hashes and outputs `dataset_for_prepare_manifest.csv` with columns `audio,text,caption,speaker`. This is an intermediate CSV, not the Irodori latent JSONL manifest. Upstream `prepare_manifest.py` remains a separate validated step.

### Regression checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_irodori_dataset.py"
```

Tests cover non-destructive scanning, explicit approval, immutable source check, manual ASR promotion, and workspace reuse. Windows runtime acceptance is still required.

### Role boundaries

- `collect_training_audio.py` remains the collector for human-approved `my_voice`, `review_approved`, and `review_emotion` outputs.
- `irodori_dataset.py` only stages text/style-reviewed data, not diarization or RVC dataset QC.
- The Irodori upstream repository owns codec conversion and LoRA training.