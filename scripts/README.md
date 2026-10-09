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
| **LV-02 dataset expansion and review evidence gate** | **lv02_expand_approved.py** |
| **LV-02 speaker similarity prioritization (never auto-approve)** | **lv02_speaker_rank.py** |
| **LV-03 DACVAE recovery without Codex** | **lv03_no_codex.py** |
| **LV-04 official LoRA baseline without Codex** | **lv04_no_codex.py** |
| **LV-05 voice A/B, reference-condition diagnosis** | **lv05_voice_ab.py** |

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


### LV-04: controlled official LoRA baseline (no Codex)

LV-03 must already have passed in lv03_dacvae_005 with exactly 8 approved
source clips and 3 independent evaluation clips. The user has confirmed
permission for local, private use. The approved 8 are split **7 train /
1 internal validation**, while the 3 external evaluations remain outside
the train.py manifest. With only eight approved examples, this is a
small-data baseline and not a voice-quality guarantee.

From the root of LocalVoice:

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py
~~~

The first command is a **read-only preflight** checking official Irodori
revision, source identity/SHA, LV-03 result, train manifest, config,
and the full-precision unquantized base-model SHA256. It does not download
weights, touch output attempts, train or infer.

If preflight passes, run the one baseline:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py --run
~~~

This runs the *official* Irodori train.py using existing Irodori-TTS/.venv,
BF16, batch 1, accumulation 2, grad checkpointing, AdamW, 120 steps,
cosine LR with 8 warmup steps, validation every 20 steps, checkpoints every
30 steps, and no WandB. VRAM usage is not guaranteed: on an OOM, check
the captured train.log and avoid silently replacing core components.

The first free lv04_lora_NNN directory receives the full immutable plan,
upstream YAML copy, training log and an adapter under
adapter/checkpoint_final/. If training succeeds, the script validates
the adapter's PEFT files and uses upstream infer.py to attempt a short,
non-explicit Japanese inference smoke test. result.json distinguishes
adapter success from inference success; it never claims audio quality was
human validated. The script does not auto-resume an interrupted run or
reuse prior failed output directories.

Only pass --resume PATH when explicitly continuing a compatible
official LoRA checkpoint directory; otherwise a new run always starts
from the SHA-verified full base checkpoint.


### LV-05: voice does not match the speaker

LV-04 being able to train and synthesize is **not** evidence that the
generated voice matches the source. The LV-04 smoke test used just the
first training latent as reference and hard-coded 3 seconds for the text.

The reported LV-04 smoke voice was perceptually different from the
user-provided example, and that example's exact-file SHA did not match
the eight officially trained WAV SHA values in the LV-04 report. It may
still be the same voice/recording after processing, or another sample
among the candidate set; do not treat it as proof of an exact training
WAV identity.

The LV-05 comparison is built from the existing, SHA-verified
lv03_dacvae_005 train manifest and lv04_lora_001/adapter/checkpoint_final.
It creates two matched base-vs-LoRA pairs:

* single reference latent (reproduces smoke's limited reference length)
* concatenated first 4 approved training latents (stronger reference context)

All four outputs use the SAME Japanese prompt, seed, 40 RF steps,
checkpoint, guidance defaults, and model/codec precision. Duration is
predicted by the official model rather than forced to 3 seconds.
Independent evaluation clips are not fed to the trainer or A/B inputs.

From the repository root:

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv05_voice_ab.py
.\.venv\Scripts\python.exe .\scripts\lv05_voice_ab.py --run
~~~

The first command is read-only preflight. The second generates four WAVs
in a fresh lv05_voice_ab_NNN directory alongside individual inference
logs and a result.json file. No new LoRA learning or DACVAE encoding is
performed; the LV-04 adapter and original files remain unchanged.

Interpretation must be based on listening:
* If base is close but LoRA is far: investigate LoRA training/loss/overfit.
* If both base and LoRA are far with one reference but improve with four:
  improve reference selection/length before modifying training.
* If all four are far: inspect reference provenance, voice sample quality,
  text conditioning, and original data coverage before retraining.
  Pitch differences alone are insufficient to verify voice identity.

Quality/voice similarity stays unvalidated until heard. Use the actual
A/B outputs and their logs; do not infer success from exit code zero.


#### LV-05 follow-up: duration-control diagnostic

The first LV-05 comparison (lv05_voice_ab_001) returned all four WAVs,
but the same Japanese prompt had significantly different model-selected
durations (base single 4.20s, LoRA single 2.84s, base multi4 6.20s,
LoRA multi4 3.00s). The official LoRA YAML uses lora_modules_to_save:
auto, which includes the duration_predictor when active. The observed
duration difference is consistent with changed duration prediction but
does not alone prove overfitting or voice identity.

To isolate timbre while keeping the four approved reference latents,
run only the next *two* 4-second paired samples:

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv05_voice_ab.py --ref-mode multi4 --fixed-seconds 4 --run
~~~

The first comparison directory stays unchanged. The next free
lv05_voice_ab_NNN receives multi4_base.wav and multi4_lora.wav with the
same text, seed, references and duration, plus logs and result.json.
This is diagnostic inference, not a replacement for the official full
quality review or a request to retrain.


### LV-02 reopened: expand train data rather than freeze at 8

The previous LV-02 freeze verified 1,167 audio files and clean ledger
partitioning but **did not promote** 1,029 automatically generated,
unverified training candidates. That 8-item approval freeze was suitable
for proving the pipeline could run, not a sufficient sample of the target
voice for practical LoRA quality. Reevaluate before any further training.

The expansion script reads existing review, inventory and automatic
preparation reports without recomputing ASR or modifying WAVs:

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv02_expand_approved.py audit
~~~

A new lv02_expansion_NNN directory contains:
- audit.json: totals grouped by evidence classification and source video
- classification.csv: all 1,167 rows classified
- review_queue.csv: first 48 review candidates round-robin by source video

The first review wave is intentionally bounded. ASR text is copied to the
suggested_text column as a **suggestion**, NEVER into text or an
approval flag. Candidate provenance (my_voice) is not independent
speaker verification. Fill the review queue only for genuinely checked
clips: action=approve, speaker_ok=yes, quality=good,
text_verified=yes, actual checked text, style, any needed caption,
and non-empty evidence explaining the human check. Leave uncertain
clips blank; leave evaluation holdout alone.

To export a *versioned* approved CSV after confirmations:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\lv02_expand_approved.py export --reviewed "F:\AIProjects\LocalVoice\Irodori-TTS\outputs\localvoice_lora_dataset\lv02_expansion_NNN\review_queue.csv"
~~~

The export retains the prior 8 training clips, appends only explicitly
confirmed and SHA-checked new clips, and references the independent
evaluation 3 unchanged. It creates a separate versioned train CSV and
selection.json with all added IDs; old LV-02, LV-03 and LV-04
artifacts are untouched. No automatic promotion without voice/text
evidence, and no full-corpus manual transcription is requested.

When *and only when* expanded approval has been completed, a separate
new LV-03 attempt can process the new CSV (replace NNN below by the
actual saved folder). These commands are **not** part of the initial
audit and should not be run just to generate more model artifacts:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\lv03_no_codex.py --train-csv "F:\AIProjects\LocalVoice\Irodori-TTS\outputs\localvoice_lora_dataset\lv02_expansion_NNN\dataset_for_prepare_manifest_approved_train.csv"
.\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py --lv03-attempt lv03_dacvae_NNN
~~~

The LV-04 invocation above is a read-only preflight; --run is a
separate deliberate training action. Original LV-03 _005 and LV-04
_001 are preserved as reproducible **technical** baselines.


### LV-02 after actual audit 001: source-video holdout and speaker ranking

Audit 001 resulted in 543 REVIEW_REQUIRED, 531 HOLDOUT_VIDEO_EXCLUDED,
82 OTHER_OR_HOLD, 8 EXISTING_TRAIN and 3 EXISTING_EVALUATION. The 531
holdout-video items are NOT failed-quality files; they share the source
video of the three evaluation examples. Do not silently move them into
training or abandon the source-video isolation gate.

The 543 REVIEW_REQUIRED are candidates, **not approved clips**.
Rather than manually confirming all 48 first-wave candidates up front,
rank the wave with the existing profile speaker reference bank (only
if the pre-existing Ui_Shigure bank and embedding model are installed):

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv02_speaker_rank.py
~~~

The program uses the most recent completed LV-02 audit, including
lv02_expansion_001, and finds original source WAVs via the validated
inventory, even if the older audit queue lacks source_path. It checks
SHA256 and protected evaluation video isolation. It does not change
review decisions or WAVs. It writes a *new* lv02_expansion_NNN folder
with speaker_ranked_review_queue.csv and speaker_rank_result.json.

IMPORTANT: This score is a heuristic to prioritize which candidates
to listen to, NOT a speaker-identity verdict or automatic training
approval. The bank must be the correct profile and model; filename
and source_kind alone do not prove identity. A missing bank/model fails
closed without a download or new model installation. If the bank has
calibrated negative examples, saved thresholds may label the higher
priorities; without them the output remains UNCALIBRATED_SIMILARITY_REVIEW.
Review only promising and uncertain clips first; manually verify actual
speaker, audio quality, text, and expressive caption before promoting.
The original first-wave 48 need not all be manually transcribed.

The new audit command now includes the source WAV path, duration and
scan flag in review_queue.csv, so future queues are directly listenable.
The old audit 001 remains immutable. The score command can resolve
the old queue's paths using inventory.csv without asking to rerun audit.
