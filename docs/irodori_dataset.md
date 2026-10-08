# Irodori-TTS: automated voice dataset preparation

## Default: manual work only for ambiguous vocal-style identification

Do **not** write transcripts or captions for every clip. Auto-preparation
reuses existing ASR and acoustic classification predictions and synthesizes
deterministic style captions. It does not edit WAVs or overwrite prior human
judgments.

From the LocalVoice root (Windows PowerShell):

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\scripts\irodori.ps1 auto -Speaker Ui_Shigure
~~~

The single **auto** action resumes ASR, including the short files, refreshes
triage, completes missing AST inference, and writes all preparation outputs.
By default each ASR and AST pass can process up to 2000 files. Existing
successful results are reused. Prerequisites: ffmpeg, uv for the isolated
ASR environment, and transformers plus torch in LocalVoice .venv for AST.

To reuse existing ASR and AST data without executing models:

~~~powershell
.\scripts\irodori.ps1 prepare -Speaker Ui_Shigure
~~~

Outputs in the existing dataset workspace:

- **dataset_for_prepare_manifest_auto.csv**: Audio, ASR or verified text,
  automatic style Caption, speaker; these are TRAINING CANDIDATES, not
  a verified final training manifest.
- **auto_preparation_report.csv**: Per-file provenance, classification,
  text source (including asr_unverified), reason, and export eligibility.
- **ambiguous_vocal_review.csv**: Only clips for which a human vocal
  style decision is still needed.
- **nonverbal_experiments.csv**: Nonverbal vocalizations, held separately
  from the normal text-conditioned training candidates.

ASR output is automatic and **may be wrong**, so the candidate report is
honest about what is not human verified. No individual manual transcript
or caption entry is required to run this pipeline.

Curated normal voice with ASR enters the candidate set if no acoustic
evidence contradicts the default. Review-emotion provenance is **not**
treated as a precise vocal style; its unclear cases go to the exception
queue. Strong contradictions, uncertain acoustic events, and unknown
styles are excluded pending classification. Invalid/rejected recordings
are excluded and stale file hashes stop processing before output writes.

## The only interactive workflow

~~~powershell
.\scripts\irodori.ps1 resolve -Speaker Ui_Shigure -Limit 20
.\scripts\irodori.ps1 prepare -Speaker Ui_Shigure
~~~

For each ambiguous clip, listen and type only one vocal style:
normal, whisper, laugh, breath, panting, groan, emotion or other.
Enter = to accept the displayed AST suggestion after listening;
r replays, s skips, x rejects, q exits. It never asks for a transcript,
Caption, or extra approval. A style-only confirmation is recorded
without implying a transcription or speaker verification.

The legacy review command remains available for existing workflows, but
its per-clip transcription/Caption path is no longer the default.

## Important nonverbal-training constraint

Upstream Irodori prepare_manifest.py excludes empty text; captions are
optional. The official README illustrates emojis interspersed with
spoken words (such as 🤭 and 😮‍💨), but has not established whether an
emoji-only text field works as a nonverbal training transcript.
Never fabricate words just to satisfy the required text column.
Nonverbal examples and optional emoji hypotheses stay in the separate
experiment file until a small-scale conditioning experiment establishes
the intended behavior. That validation is an engineering task, not manual
transcription for the user.

The generated training CSV is an intermediate proposal, not the final
latent manifest and not evidence that LoRA learning has succeeded.

## Tests

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_irodori*.py" -v
~~~

Model-free regression runs in GitHub Actions; Windows inference and LoRA
learning must be separately validated.
