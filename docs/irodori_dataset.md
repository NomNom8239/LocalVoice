# Irodori-TTS dataset preparation

LocalVoice owns collection, speaker review and provenance. Irodori-TTS remains a separate training/inference engine.

## Windows PowerShell commands

From the LocalVoice repository root:

```powershell
.\scripts\irodori.ps1 status -Speaker Toto_Kogara
.\scripts\irodori.ps1 asr -Speaker Toto_Kogara -RetryErrors -Limit 1
.\scripts\irodori.ps1 asr -Speaker Toto_Kogara -RetryErrors -Limit 30
.\scripts\irodori.ps1 merge -Speaker Toto_Kogara
.\scripts\irodori.ps1 triage -Speaker Toto_Kogara
.\scripts\irodori.ps1 export -Speaker Toto_Kogara
```

Use `scan` only when no review workspace exists. An existing `Irodori-TTS/outputs/localvoice_lora_dataset/review.csv` is reused automatically; otherwise the workspace is `data/irodori_lora/Toto_Kogara/`.

The input WAV directory is `data/training_audio/Toto_Kogara/audio/` and is never changed. Git excludes `data/`. Existing `training_audio/Toto_Kogara/manifest.tsv` supplies optional provenance. Never copy material back into `reference_bank` automatically.

### ASR environment

The first `asr` invocation creates a separate `LocalVoice/.venv-asr` with `uv`. It installs a compatible pair (`faster-whisper==1.2.1`, `av==18.1.0`) and **repairs an existing ASR environment** that has incompatible PyAV 19 installed. PyAV 19 removed `metadata_errors` from `av.open`, which faster-whisper 1.2.1 still uses. Later runs reuse this environment. CPU/int8 is the default, preserving the existing RVC and Irodori CUDA environments. On a failure, `asr_suggestions.csv` retains `error_detail`, and processing stops after three consecutive errors. `-RetryErrors` retries only previously failed suggestions; try `-Limit 1` first.

### Priority review queue

Run `triage` after ASR to create `triage.csv` in the existing workspace.
It reads the 1,167-row review table and ASR suggestions without changing
either, then groups **pending** clips by review priority: invalid audio,
short audio without an ASR transcript, speech not detected, very brief/repeated
utterances, short transcripts and ordinary candidates. Short clips successfully
processed via `asr -IncludeShort` are subsequently sorted by their ASR status
or transcript characteristics, not automatically left in `short_audio`.
These are still review priorities, **not** confirmed laughter/breath/groan labels.

These groups are *inspection hints*, not automatic confirmations of
breathing, groaning, whispering or the correct speaker. Listen and edit
`review.csv` (not `triage.csv`) to approve/reject clips.

### Interactive LoRA review

Use the existing `review_emotion` provenance first. These were already
curated as emotional/extreme voice by the LocalVoice source review, though
the Irodori transcription/style/quality still need human verification.

```powershell
.\scripts\irodori.ps1 review -Speaker Ui_Shigure -Kind emotion -Group short_audio -Limit 20
```

Audio plays through `ffplay` (as in the existing `review_training_audio.py`).
Controls: `a` = confirm speaker/quality/style; `t` = save only a
tentative nonverbal style tag; `n` = reject; `r` = replay; `s` = skip;
`q` = quit.

**No transcript is required during listening review.** With `a`, press
Enter at the verified transcription prompt to save `decision=needs_text`
(speaker and style checked, *not yet eligible for LoRA export*). If you
can verify spoken words, enter them to save `decision=approved`.
For an ASR suggestion, `=` explicitly confirms the displayed suggestion;
simply pressing Enter defers rather than silently adopting ASR output.
Nonverbal laughter, breathing or groaning remains `t` tagged; never invent
speech to satisfy training.

Later, re-run `triage` and use this to complete deferred transcripts:

```powershell
.\scripts\irodori.ps1 review -Speaker Ui_Shigure -Kind emotion -Group needs_transcript -Limit 20
```

Optional ASR for the 208 short clips can produce **suggestions** (including
possible hallucinations), but never approves them:

```powershell
.\scripts\irodori.ps1 asr -Speaker Ui_Shigure -IncludeShort -Limit 20
.\scripts\irodori.ps1 triage -Speaker Ui_Shigure
```

Without `-IncludeShort`, the original 959 normal-length candidates are the
only ASR targets; successful existing ASR results are retained without rerun.


For laughter, the dedicated style is `laugh`. Both `laugh` and `laughter`
(and the Japanese input `笑い声`) normalize to `laugh`. Select `t` + `laugh`
for laughter alone, or `a` + `laugh` for verified spoken words delivered
while laughing. `t` does not automatically approve audio for training.
After each action, changes are atomically persisted to `review.csv`,
without touching the WAV. An item tagged with `t` gets `decision=tagged`,
**not** `approved`; it is ignored by export and by subsequent default
review sessions. To revisit previously tagged clips, regenerate the queue
with `triage`, then use `-IncludeTagged` (and optionally
`-Group tagged_nonverbal`).

```powershell
.\scripts\irodori.ps1 triage -Speaker Ui_Shigure
.\scripts\irodori.ps1 review -Speaker Ui_Shigure -Kind emotion -Group tagged_nonverbal -IncludeTagged -Limit 20
```

Clips without a verified transcript can be classified and confirmed as
`needs_text`, but cannot enter the current intermediate export. In particular, do **not** invent utterance text for
breath/groan-only clips. Tag them until the upstream Irodori representation
for nonverbal data has been validated.

### Human review gate

For each approved clip, listen to the audio and enter a verified `text`, `style`, and a `caption` for special delivery. Set `speaker_ok=yes`, `quality=good`, `decision=approved` only after checking the content.

ASR writes suggestions, never verified text or approval. A successful ASR run automatically merges only new/changed suggestion fields into `review.csv`; a separate `merge` command is no longer needed. The merge is idempotent: a prior `no_detected_text_review_audio` result with empty text is not counted again on every run. `ASR suggestion file: ... (N attempted)` is the number processed during that invocation, not the merge count. `status` shows the live status counts from both the review sheet and the ASR suggestion file, so older errors without details cannot obscure new successes. Short clips remain candidates for manual review. No algorithm is claimed to reliably label breaths or groans without listening.

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