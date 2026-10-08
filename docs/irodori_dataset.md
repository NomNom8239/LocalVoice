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

**Updated routing after full-corpus AST inference:** AudioSet AST values
are uncalibrated. They must **not** trigger a human listening task just
because a raw score is low or the top label is unknown. For ASR-detected
speech, curated `my_voice`/`review_approved` defaults to `normal`;
curated `review_emotion` defaults to a broad `emotion` caption. Only
very strong, distinct non-normal acoustic evidence overrides that default,
and its source stays labeled `ast_strong_heuristic_unverified`. These
defaults are inferred from *collection provenance*, not presented as
new human verification.

Without detected speech, a supported nonverbal prediction is retained
in the separate experimental dataset; unknown / weak evidence goes to
`deferred_unresolved_audio`, **not** the user's manual queue. Only
conflicting nonverbal sound evidence goes to
`ambiguous_vocal_review.csv` for optional targeted listening.
Previously entered tentative human tags (`tagged`) that disagree with AST
are preserved and routed to `deferred_unresolved_audio`, not back into
human listening or directly into training. This keeps model disagreement
from creating repeat manual work.
The JSON summary now contains `manual_style_review_cases` and
`reason_counts`, so an unexpectedly large human queue is visible.
Rejected/invalid recordings stay excluded; stale source hashes stop
processing before output writes.

Run `prepare` again after updating the code; existing AST/ASR
predictions are reused without GPU inference.

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

## Nonverbal conditioning feasibility: isolated pilot (next phase)

The official Irodori-v4.1-Small
[EMOJI_ANNOTATIONS.md](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small/blob/main/EMOJI_ANNOTATIONS.md)
documents `🤭` for chuckles, `😮‍💨` for breath/sigh,
`🌬️` for heavy breathing, and `🥵` for panting/moans/groans.
**This does not prove standalone emoji works as a LoRA training text.**

Generate a **separate**, tiny, balanced experiment input from the existing
nonverbal candidates. This command does no inference/training, reads no ASR,
does not call ffplay, and never alters WAVs or human decisions:

~~~powershell
.\scripts\irodori.ps1 nonverbal-pilot -Speaker Ui_Shigure -MaxPerStyle 3
~~~

The outputs are:
- `nonverbal_pilot/emoji_only_hypotheses.csv`: deliberately experimental
  `audio,text,caption,speaker` rows using *one unverified emoji* per style.
- `nonverbal_pilot/audit.csv`: all original candidate IDs and source hashes,
  selection versus hold rationale and the official mapping reference.

No unsupported generic emotion label is mapped to a specific emoji, because
its exact vocal event cannot be inferred from source category alone.
All original 15 nonverbal items remain present in the audit; at most three
per supported style enter this **hypothesis-only** pilot.
The `nonverbal_experiments.csv` source is unchanged.

## Tokenizer compatibility check (no training)

Use the *actual v4-Small model config* at
`Irodori-TTS/configs/train_v4_small.yaml`. It identifies the
`sbintuitions/modernbert-ja-310m` tokenizer and a specific immutable
revision. Check its emoji encoding rather than guessing from the displayed
characters. With the already generated six hypotheses:

~~~powershell
.\scripts\irodori.ps1 nonverbal-check -Speaker Ui_Shigure
~~~

The script reads the original nonverbal candidate audit and inventory,
verifies original WAV SHA-256 values and the chosen subset, loads **only**
the pinned HF tokenizer (not the model weights), checks unknown tokens,
BOS handling and emoji roundtrip, and writes:

- `nonverbal_pilot/tokenizer_audit.csv`: IDs, token representations,
  unknown-token counts, roundtrip and pass/block status.
- `nonverbal_pilot/hf_audio_dataset_hypothesis.jsonl`: local audio
  paths in a Hugging Face JSON layout matching the Irodori
  `--dataset json --data-files` input route.

`PASS_TOKENIZATION_ONLY` **does not mean the emoji controls a
generated vocalization**, that DACVAE has encoded the WAV, or that
LoRA training has worked. A `BLOCK` return prevents continuing the
experiment with unsupported emoji representations. The JSONL stays in
`nonverbal_pilot`, isolated from the speech training dataset.
No manual text entry, Caption entry or sound playback is required.

**Validation gates before actual training:**
1. Check tokenizer acceptance and whether standalone emoji is preserved as a
   meaningful text condition (not just a nonempty string).
2. Check the upstream `prepare_manifest.py` DACVAE path with locally readable
   audio; do not confuse this with LoRA success.
3. Run a paired **tiny** LoRA feasibility experiment versus a speech-control
   set, keeping normal speech performance and nonverbal quality separate.
4. Only if the conditioning works, consider scaling to additional recordings.
   Never mix all hypothesis rows into the 1,041 speech candidates by default.

No manual transcription/caption writing or five-clip classification is needed
for this stage. Actual CUDA model training and audible validation must run
on the local PC. Treat the model's explicit-consent voice-use restrictions
as prerequisites for any identifiable-person voice model training.

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
