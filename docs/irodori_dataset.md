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

The v4-Small tokenizer uses SentencePiece. The PowerShell runner checks
`LocalVoice/.venv` for the `sentencepiece` package and installs it
automatically via `uv` only if it is missing, before loading the tokenizer.
This fixes the Hugging Face error "Cannot instantiate this tokenizer from a
slow version. If it's based on sentencepiece, make sure you have
sentencepiece installed." The downloaded tokenizer files and pilot CSVs
are reused; neither the original WAVs nor review decisions are touched.
When starting the Python checker directly, an actionable dependency error
is produced instead of a vague conversion failure.

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

## DACVAE six-clip manifest smoke test (no LoRA training)

After `nonverbal-check` reports `PASS_TOKENIZATION_ONLY`, verify
that the six hypothesized local WAV examples can be encoded into actual
DACVAE latents by **upstream Irodori's prepare_manifest.py**, without
recreating ASR/style predictions or editing the 1,041 speech examples.

The runner requires a separate `Irodori-TTS/.venv`, configured using
Irodori's own dependency lock and CUDA tooling. It deliberately does not
install large model/toolchain dependencies into the LocalVoice `.venv`.

**Windows TorchCodec DLL compatibility:** the pinned Irodori
`torchcodec 0.10` requires `torch 2.10` and FFmpeg *shared* libraries
from a supported major version (4–8). A working standalone `ffmpeg.exe`,
especially version 9, does **not** supply these DLLs. If the first native
probe fails specifically while importing TorchCodec, the runner first
checks the installed Torch/TorchCodec versions. If they do not match the
supported pair, it stops **without reinstalling CUDA or PyTorch**.
If they match, it uses a project-local, pinned FFmpeg **7.0.2 full-shared**
archive under `Irodori-TTS/.localvoice-toolchain/`, downloaded from
Gyan.D's GitHub release and verified against WinGet's SHA-256
before extraction. This changes *only the Irodori child process* DLL
search path via `os.add_dll_directory` and never alters global PATH,
system FFmpeg, or the other Python venvs. The archive is reused
on subsequent runs; a corrupt/partial existing installation fails closed.
The first `-DryRun` may download this one pinned portable runtime.

For machines that already have a working FFmpeg 7 full-shared directory,
override the automatic download explicitly:

~~~powershell
.\scripts\irodori.ps1 nonverbal-manifest -Speaker Ui_Shigure -DryRun -FFmpegSharedBin 'C:\ffmpeg-7.0.2-full_build-shared\bin'
~~~

Only a valid directory with the FFmpeg 7 shared DLLs is accepted.
Any other native dependency failure is reported without silently
changing the Irodori environment.

Preflight only (verify inputs and Irodori environment, no encoding):

~~~powershell
.\scripts\irodori.ps1 nonverbal-manifest -Speaker Ui_Shigure -DryRun
~~~

Run the isolated six-clip DACVAE conversion only after the preflight:

~~~powershell
.\scripts\irodori.ps1 nonverbal-manifest -Speaker Ui_Shigure
~~~

It executes the upstream CLI using `--dataset json`,
`--data-files train=<absolute-jsonl>`, `--audio-column audio`,
`--text-column text`, `--caption-column caption`,
`--speaker-column speaker` and `--device cuda`.
`--normalize-db none` is used **only** to avoid loudness normalization
dependencies and confounds in this smoke test, not as a production setting.
No LoRA training or inference is performed.

Each invocation gets a fresh, isolated
`nonverbal_pilot/codec_attempt_NNN/` directory. The tool validates that
all six expected `train_manifest.jsonl` entries refer to existing
positive-length latent tensors under its own `latents/` directory.
Failed/partial attempts remain traceable with `result.json`;
they are never silently treated as passing results.
A PASS proves dataset decoding and DACVAE manifest generation **only**,
not whether emoji-only text can teach nonverbal speech styles.

## Controlled LoRA feasibility plan after six DACVAE latents

The six DACVAE latents are enough to test *pipeline feasibility*, not model
accuracy. Do not mix these emoji-only hypotheses into the 1,041 general
speech candidates. The official v4-Small LoRA config defaults to batch 40
and 30,000 steps, which is far too large for this pilot.

Stage a read-only **plan** (default, no GPU training, no new files):

~~~powershell
.\scripts\irodori.ps1 nonverbal-lora -Speaker Ui_Shigure
~~~

The planner verifies `codec_attempt_001/result.json` reports full DACVAE
success and reads the actual six-entry latent manifest. It requires three
instances for each of exactly two emoji cues. It deterministically selects
**two train and one held-out latent per cue** (four train, two holdout)
without altering the original data. This is a deliberately small software
smoke test, not a statistically valid sample or performance measurement.
`PLAN_ONLY` prints the proposed paths, model checkpoint status, and
upstream `train.py` command.

The full-precision **unquantized** Irodori v4.1-Small
`model.safetensors` must be available locally. The planner checks
`Irodori-TTS/model.safetensors` and an existing Hugging Face cache copy
without network traffic. You can provide an explicit path with
`-CheckpointPath`.

If the planner reports `checkpoint_exists=false`, explicitly fetch
the **official 3.06 GB** checkpoint to the Hugging Face model cache
(without training) via:

~~~powershell
.\scripts\irodori.ps1 nonverbal-lora -Speaker Ui_Shigure -FetchCheckpoint
~~~

The command reuses any cached copy and SHA-256 checks a new download
against the official v4.1-Small model file. This does not duplicate the
large model into the repository. The subsequent `nonverbal-lora` plan
discovers the cached file automatically. A missing file is never silently
replaced with a quantized checkpoint; no CUDA dependencies are rebuilt.
Fetching and training cannot be performed in one command.

**Manifest path correction:** Upstream Irodori resolves
`latent_path` relative to the location of its containing manifest.
The pilot plan rebases every train/holdout latent path to the existing
absolute `codec_attempt_001/latents/*.pt` file before writing split
manifests, and re-verifies that those files are present. It never
moves/re-encodes any of the six latents.

The generated train invocation is capped at
24 steps, batch size 1, single-process Windows dataloader, AdamW,
gradient checkpointing and no W&B. These are conservative *initial*
memory settings; RTX 5060 Ti variants have different VRAM sizes and a
real pilot may still fail with CUDA OOM. It produces output only under a
new `nonverbal_pilot/lora_attempt_NNN/` and never touches the 1,041
speech samples.

Only after checkpoint/version/VRAM checks and by explicit choice can
a real training smoke run be launched with:

~~~powershell
.\scripts\irodori.ps1 nonverbal-lora -Speaker Ui_Shigure -CheckpointPath 'F:\path\to\unquantized\model.safetensors' -Run
~~~

`-Run` is intentional; without it there is no training. Completing
`train.py` is **not proof** that emoji-only text learns useful
nonverbal generation. A later inference experiment must compare the
unaltered base checkpoint against the pilot adapter with identical
prompts/seeds/reference conditions, testing both nonverbal sound
and preservation of normal speech. Never promote this smoke adapter
to production solely because training finishes.

## Paired nonverbal generation comparison (base vs 24-step LoRA)

After `nonverbal-lora -Run` finishes and creates the Irodori PEFT adapter
under `lora_attempt_001/adapter/checkpoint_final/`, compare **the
unaltered base and experimental adapter** using the same model checkpoint,
speaker reference, prompts, captions, sampling seed, duration and
inference settings. Start with a read-only plan:

~~~powershell
.\scripts\irodori.ps1 nonverbal-compare -Speaker Ui_Shigure
~~~

Explicitly generate the comparison WAVs:

~~~powershell
.\scripts\irodori.ps1 nonverbal-compare -Speaker Ui_Shigure -Run
~~~

The comparison runner loads the same base checkpoint path recorded in the
training attempt; it refuses a missing or incomplete PEFT adapter and
verifies the 4+2 train/holdout separation. It **automatically chooses
one SHA-256-verified, previously curated ordinary speech clip** (3–16s)
from the existing 1,041 candidate table as a speaker reference, outside
the six nonverbal candidates. There is no manual audio classification,
Caption editing or ASR step; no held-out nonverbal WAV is used as
speaker-reference audio.

Five prompts are tested, each with **base** and **LoRA** output WAV
under an isolated `lora_attempt_001/comparison_NNN`:

- `breath_caption`: `😮‍💨` + the held-out automatic caption.
- `breath_emoji_only`: `😮‍💨` without caption.
- `groan_caption`: `🥵` + the held-out automatic caption.
- `groan_emoji_only`: `🥵` without caption.
- `normal_speech`: fixed Japanese conversational sentence, without caption.

Each pair uses the **same** seed (`20261008`), reference WAV, output
duration (3.0s for nonverbal, 3.5s for speech), 8-step Sway sampling
and bf16 model / FP32 codec inference to limit VRAM demand. Results:
10 WAVs, individual `.log` files and `result.json` with duration,
format, completion count and exact paths. Existing data and checkpoints
are not modified. Windows inference reuses the same project-local
FFmpeg 7 shared DLLs used by DACVAE, when required.

The script validates **successful audio generation only**: it does not
claim a useful LoRA, target-speaker quality, intelligible speech,
or nonverbal correctness. The test set is only 1 held-out sound per
class; acoustic efficacy and normal-speech preservation remain
**unvalidated**. In particular, `🥵`/ `😮‍💨` standalone emojis
as training text are hypotheses, not an officially verified technique.
Human listening is optional, *not* a prerequisite to automate the
dataset; only irreducibly ambiguous source labels are queued for
manual classification.

The older LoRA script could report `lora_training: STARTED` after an
exit-zero completion: the training **status** is authoritative for older
runs, and the comparison runner also verifies the actual adapter
checkpoint files. New runs mark completion explicitly and fail if
the checkpoint files are missing.

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
