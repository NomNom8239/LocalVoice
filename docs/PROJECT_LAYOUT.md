# LocalVoice directory ownership and change rules

Status: adopted directory-ownership rules; current implementation status reconciled 2026-10-09 on `feature/phase4-acoustic-pilot`.
Scope: directory and non-destructive write contracts. **Ui_Shigure best-effort AST classification and targeted human review are complete as unverified candidate v2**, with metadata-only v1/v2 history verified. Formal QC / approved `outputs/datasets/` remains separate and pending. ASR is optional metadata and does not gate acoustic classification or review.

## Ownership boundaries

| Path | Owner / responsibility | Git policy | Write policy |
| --- | --- | --- | --- |
| `data/` | Existing acquired audio, review decisions, reference banks and legacy outputs | Ignored; preserve in place | **No writes by newly developed ASR/style/dataset stages.** Existing legacy scripts have established writes here; do not alter their contracts without a separate, tested migration. Never bulk-clean this directory. |
| `scripts/` | Existing archive retrieval, separation, diarization, identity verification, review, and RVC dataset commands | Tracked | Maintain the existing entry points. Do not add new ASR/style experiment scripts here. |
| `src/localvoice/` | First-party application code (ASR Pilot, AST full batch, review CSV application, metadata-only history CLI implemented; comprehensive QC and approved dataset promotion pending) | Tracked | Place new acoustic classification, QC/review and export code here under one CLI; do not modify legacy acquisition scripts solely for these stages. |
| `tests/` | Tests for first-party code | Tracked | Tests reflect public behavior and IO boundaries; fixtures cannot silently depend on private `data/`. |
| `docs/` | Stable layout, design decisions and operating instructions | Tracked | Keep stable rules in this document; put significant architecture decisions in `docs/decisions/` when they occur. |
| `Irodori-TTS/` | **Upstream** Irodori-TTS standalone Git clone and runtime | Ignored by LocalVoice | Do not add LocalVoice code, patches, tools, LoRA experiments, or custom datasets inside. Normal upstream-managed `.venv/`, `gradio_outputs_voicedesign/`, and caches may appear. |
| `.venv/` | Existing LocalVoice acquisition/processing Python runtime | Ignored | No extra environments without a documented dependency conflict. Irodori uses its own `Irodori-TTS/.venv/`. |
| `work/` | Per-run intermediate outputs, ASR Pilot evidence, classification/QC/review candidates | Ignored | Create a fresh run ID; never overwrite another run or `data/`. Save failure/unknown/review records; do not automatically wipe contents. |
| `outputs/` | Versioned unverified candidate-browsing library **and separate** human-approved reference WAV libraries | Ignored | Keep `outputs/candidates/` visibly UNVERIFIED and `outputs/datasets/` approved only. Never overwrite versions, alter original WAVs, or silently relabel unverified outputs as approved. |
| `outputs/metadata/<profile>/` | Immutable **CSV/JSON/manifest-only** version history and reviewer input inbox | Ignored | Metadata-only v1/v2/v3 history in `versions/`; exported HTML CSV/JSON in `inbox/` and immutable `versions/<version>/review/`. No WAV copies here. `outputs/candidates/` becomes an on-demand materialized view; legacy v1/v2 media stay untouched until explicitly retired after verification. |
| `cache/` | **Optional** disposable model/tool caches explicitly configured for this project | Ignored | Prefer tool defaults outside the source tree; never place cache files in `scripts/`, `src/`, or the upstream repository as custom additions. |
| `start_irodori_voicedesign.bat` | LocalVoice launch entry point for upstream WebUI | Currently local/untracked | May be versioned after checking for machine-specific paths; never store the launcher within upstream. |
| `config.toml` | Existing pipeline settings and legacy data-root mapping | Tracked | Do not repurpose legacy keys or change their meanings to implement new stages. |

`archive/` is an ignored, optional manual retirement area, not a new mandatory workflow or a location for active code. Old LoRA experiment code/artifacts are not reinstated.

## Intended layout (not a claim about the current filesystem)

```text
LocalVoice/
├── .git/
├── .gitignore
├── README.md
├── config.toml                     # legacy acquisition pipeline config
├── pyproject.toml                  # current editable package configuration
├── start_irodori_voicedesign.bat   # current local launcher
├── scripts/                        # existing acquisition/review commands ONLY
│   ├── localvoice.py
│   ├── diarize.py
│   ├── build_reference_bank.py
│   ├── calibrate_threshold.py
│   ├── review_training_audio.py
│   ├── collect_training_audio.py
│   ├── build_rvc_dataset.py
│   └── common.py
├── src/localvoice/                 # first-party ASR and AST/classification/review code
│   ├── __main__.py                 # ONE CLI (python -m localvoice ...)
│   ├── transcription/             # existing asr-pilot; full ASR deferred
│   ├── style/                     # AST pilot/full batch, CSV apply-review, metadata versioning
│   ├── quality/                   # proposed owner for identity/QC/review; create if needed
│   └── dataset/                   # planned versioned Japanese-category export
├── tests/                          # tests organized by feature
├── docs/
│   └── PROJECT_LAYOUT.md          # this document
├── data/                           # existing data; DO NOT MIGRATE/DELETE
├── work/<run-id>/                  # per-run inputs, predictions, QC, review, manifests, logs
├── outputs/metadata/<profile>/     # versions/vN metadata; inbox for review CSV/JSON
├── outputs/tts/                    # future: retained copies of WebUI audio
├── cache/                          # optional: explicitly configured cache
├── .venv/                          # existing LocalVoice runtime
└── Irodori-TTS/                    # upstream Git clone only
    ├── .git/
    ├── .venv/
    ├── gradio_app_voicedesign.py
    └── gradio_outputs_voicedesign/ # upstream runtime output, if generated
```

Do not create placeholder directories or files merely to match the diagram. On the active `feature/phase4-acoustic-pilot` branch, `src/localvoice/style/batch.py`, `review_export.py`, `catalog.py`, `localvoice_ast_candidate_reviewer.html`, `work/` (local), `tests/`, and `outputs/metadata/` (local) have concrete roles. `quality/`, approved `dataset/` promotion and final `outputs/datasets/` remain future capabilities. AST candidate output and the reviewer do **not** approve training audio.

## First-profile Reference Bank bootstrap (2026-10-09)

The existing tracked `scripts/localvoice.py` is responsible for **both** existing-bank
archive collection and first-speaker bootstrap. The new design does not create a
second acquisition CLI or copy LocalVoice code into upstream `Irodori-TTS/`.

- **Bank present:** legacy acquire → Vocals separation → diarization → speaker
  embedding comparison and SELF/REVIEW/OTHER flow continues.
- **Bank absent:** download / Vocals first, diarization once, create speaker
  preview snippets inside the **new run only**, ask the user to listen and select
  the target speaker label(s), produce up to 24 screened 2–20-second candidate
  clips, call the existing `scripts/build_reference_bank.py`, then classify
  using the same diarization evidence. Do not assume the longest/loudest
  speaker or a different diarization label is another human.
- The run stores `bootstrap/previews/` and `bootstrap/selected_reference_wavs/`.
  Bank NPZ/JSON go to `data/reference_bank/<profile>/`. The input source,
  existing run directories and established Bank are never overwritten in this
  workflow. `--force` retains its **legacy destructive run-dir behavior**
  and must not be used to recover an interrupted bootstrap casually.
- Preliminary automated quality gates do not replace speaker identity
  confirmation, final recording quality QC, or human dataset approval.
- **Runtime acceptance pending:** tests added, but the actual user Windows
  E2E and existing Ui_Shigure reference-bank regression have not been run.
- Cross-site yt-dlp source-ID namespacing, fully automated new archive → AST
  incremental integration, and bootstrap interruption resume are separate
  LV-R07 follow-ups, not silently claimed here.

See [operational design and acceptance](REFERENCE_BANK_BOOTSTRAP.md).

## Existing acquisition compatibility: exception, not the new pattern

`config.toml` currently points existing code to `data/source`, `data/wav_master`, `data/diarization`, `data/reference_bank`, `data/runs`, `data/training_audio`, and `data/rvc_dataset`.

The existing `scripts/localvoice.py --force` deletes a selected `data/runs/<profile>/<run-name>` folder before recreating it. **Never use `--force` against an existing run without a separate, explicit backup/review.** Phase 3+ code must not reuse that destructive execution pattern.

Do not claim that the old acquisition pipeline is read-only, or move existing `data/` items as part of new classification/export work. New Phase 3+ capabilities must treat `data/` as read-only. Any future migration of the legacy pipeline's output root requires its own scoped design and regression tests.

## Phase 3+ creation rules

1. Use the existing user-facing command interface (`python -m localvoice ...`) and a reusable, import-safe code path per capability. The ASR Pilot already uses this CLI; extend it rather than creating parallel `*_v2.py`, `*_final.py`, or `experiment_*.py` implementations.
2. ASR (when invoked) consumes explicitly selected audio **by absolute/validated input path** from `data/` (or another chosen source) and writes only to new `work/<run-id>/`. It does **not** train models or update speaker reference banks. **ASR output is optional** for acoustic classification, QC, review and library export.
3. Style assessment is **acoustic** and separate from speech recognition, speaker identity and recording QC. The existing 1,167 WAVs are candidates, not all human-approved/clean/classified. Each gets a classification/QC outcome or explicit failure/unknown; nonverbal, ambiguous and unsupported evidence stays `unknown`/`requires_review`. ASR text is not proof of a vocal style.
4. Each run records a manifest with run ID, original clip/segment identity and offsets, source path/SHA, acquisition provenance, identity decision, QC result, acoustic-style candidates, any reviewer approval, model/config/taxonomy versions, failures and outputs. No invented confidence or implicit approval. Final export is a separate promotion under `outputs/datasets/<profile>/<version>/`. **Folders must be browsable and playable without opening metadata files**, with `README.txt`, `index.csv` and `manifest.jsonl`: see [Emoji Palette-aligned audio library specification](EMOJI_AUDIO_DATASET.md).
5. No implicit recursive cleanup, auto-approval, destructive overwrite, or silent modification of existing audio/reference banks. New operations fail closed on destination conflicts. Data operations and tests must not depend on the user's private recordings being present.
6. Never modify upstream Irodori-TTS files to compensate for a LocalVoice issue. Keep launch/integration glue in LocalVoice, and use the upstream UI as shipped. The official UI may generate `Irodori-TTS/gradio_outputs_voicedesign/`; copy any audio worth retaining into `outputs/tts/` before a future reinstall.
7. A proposed new module, CLI entry point, environment, or output directory needs a documented purpose and an owner in this table. If an existing location owns the responsibility, extend that implementation rather than adding another file.
8. **Library v1 acceptance:** verify a real existing WAV subset from candidate → acoustic classification → independent identity/QC → targeted human review → versioned Japanese category folder → playback and manual selection in upstream VoiceDesign. Confirm read-only `data/`, preserved unknown/rejections, manifest provenance and immutable outputs. Full new archive-to-dataset E2E is a **later LV-R07 task**, not the current library v1 gate. Isolated module tests do not substitute for this existing-WAV E2E.

## Metadata-first versioning update (2026-10-09)

The metadata-first design and implementation live in
[VERSIONED_AUDIO_METADATA.md](VERSIONED_AUDIO_METADATA.md).
Archive the existing candidate releases **without copying audio** into
`outputs/metadata/<profile>/versions/v1,v2`. CSV/JSON inputs are retained per
version under `review/`. New versions need a full cumulative reviewer CSV and
are stored as metadata only. A browsable WAV folder is rebuilt from the original
source plus the version manifest **only when requested**. Do not delete legacy
v1/v2 materializations automatically; verify old/new counts, hashes, ability
to restore, and backups before any manual retirement. An approved
`outputs/datasets` library remains a separate workflow.

## Current v2 checkpoint (2026-10-09)

- `Ui_Shigure` AST batch `ast_batch_20261009_163035451`: 1,167/1,167 WAV, unknown 5, errors 0. v1 unverified candidate output complete; targeted reviewer CSV 105 entries produced v2 candidate with reviewed usable 100, caution 5, unreviewed 1,062.
- `outputs/metadata/<profile>/versions/v1,v2,...` is the **immutable classification and review history**. Existing v1/v2 metadata verified against original WAVs on Windows (1,167 referenced each, no WAV copying). Browser CSV/JSON drop area is `outputs/metadata/<profile>/inbox/`.
- A **single tracked HTML** `localvoice_ast_candidate_reviewer.html` handles first human review of classified v1 and subsequent reviews of v2/v3/etc. It accepts either candidate WAV folders or metadata version + original WAV folder, and exports cumulative review CSV/JSON. Mock JS integration passed; **Windows browser acceptance is still pending**.
- For later revisions, use `style-batch metadata revise` instead of copying all 1,167 WAVs. `style-batch apply-review` remains an older copy-producing workflow.
- Source WAVs, completed AST run evidence and metadata must be retained/backed up; a metadata JSON alone does not contain WAV bytes. Old candidate audio folders are not automatically deleted. No actual copy-based restoration test was run.
- Notion handoff: LV-R04 / R05 / R08 Done, LV-R09 Chrome acceptance Ready; LV-R06 formal QC/backlog, LV-R03 ASR comparison backlog, LV-R07 ingestion E2E backlog.

## Implementation checkpoint — AST full-batch code added (2026-10-09)

`src/localvoice/style/batch.py` and the existing single CLI now own read-only AST batch inventory, incremental predictions, checkpoint/resume, and staged `outputs/candidates/<profile>/<version>/` browse export. Offline mock regressions are in `tests/test_style_batch.py`; [Windows runbook](AST_BATCH_RUNBOOK.md) is available. **Windows実機のAST全1,167件処理は完了（unknown=5、errors=0）し、v1候補・v2レビュー反映版も作成済み。** ただし別途承認済み`outputs/datasets/`、全件独立identity/QCと新規取り込みE2Eは未完了。

## Execution priority clarification: AST best-effort browse first (2026-10-09)

AST is the sole required model for the first batch. The **first useful output** is an explicitly **unverified** category-browsing set for all 1,167 existing WAVs; no dependency on CLAP comparisons, perfect whisper/sigh/moan distinction, all-item human approval, full ASR or fine segmentation. If a class is unsupported/ambiguous, place it under `99_未分類_要確認` and keep the evidence.

- Provisional candidate output owner: `outputs/candidates/<profile>/<version>/`, **marked unverified** with an Excel-compatible index and provenance. It may contain unchecked/low-quality samples and must never be presented as reviewed or clean. It is still read-only-copy output; it must not touch `data/` or the upstream Irodori checkout.
- Human-approved output owner (separate): `outputs/datasets/<profile>/<version>/`. Identity/QC/style reviewer gates remain mandatory **only for this certified area**.
- All 1,167 candidate sources must produce a category/unknown/error audit row before the initial candidate-library run is considered complete, but uncertain source WAVs need not be listened to before the preliminary category folders are browsable.
- Source integrity, collision protection, explicit failure evidence, and honest candidate/approved labeling are always required. Model-performance perfection is **not** a blocking gate.

This supersedes any below wording that implies the **initial candidate-browse library** must wait for full reviewer approval. See [two-tier output contract](EMOJI_AUDIO_DATASET.md).

## Historical MVP plan (superseded by the v2 checkpoint above)

**Historical planning notes below:** Earlier v1 acceptance described full QC plus approved `outputs/datasets/` as the initial gate. That is **not** what R04/R05/R08 marked Done: those tasks delivered the best-effort AST candidate v1 and reviewed candidate v2. Comprehensive QC and approved-dataset promotion remain LV-R06 Backlog; Chrome reviewer acceptance is LV-R09 Ready. The original rationale is retained for traceability.


**MVP / current priority:** Generate a versioned, human-browsable reference WAV library from the existing **1,167** `data/training_audio/Ui_Shigure/audio/` candidate WAVs. This includes candidate classification/QC coverage for all inputs, but **does not require that every recording be automatically approved**. Unknown/foreign speaker/QC failures stay outside the promoted library. Only reviewed and approved clean references are copied. Users manually choose a file in Explorer and attach it to upstream Irodori-TTS VoiceDesign; fully automated synthesis is not an MVP requirement.

- **LV-R00 / Phase 1 (done):** preserve legacy acquisition/diarization/speaker processing and clarify ownership; old independent LoRA experiments are excluded.
- **LV-R02 / Phase 2 (done):** upstream Irodori-TTS installed independently, GPU works, and manual reference-WAV VoiceDesign generation has been tried. This does **not** imply automatic selection or style QC.
- **LV-R03 / ASR auxiliary (Pilot done, full adoption deferred):** 12-sample Kotoba-Whisper v2.0 and Whisper large-v3 CUDA runs each processed 12/12; both hallucinated on artificial silence and Whisper repeated text on nonverbal clips. Detailed evaluation/48-sample ASR comparison is optional later. **It is not a prerequisite for LV-R04, LV-R06, or LV-R05.** See [ASR contract](PHASE3_JAPANESE_ASR.md).
- **LV-R04 / Acoustic style classification (first priority):** approve audibly distinguishable primary Japanese category labels and optional secondary tags; use human-ground-truth real audio to evaluate candidate classifiers; process all 1,167 files to classification candidates/unknown/failure evidence. ASR transcripts are **not** ground-truth style labels. Do not generate a forced definitive label for uncertain or mixed sounds.
- **LV-R06 / Speaker identity + recording QC + targeted human review (parallel with LV-R04):** keep identity, quality and style distinct; preserve existing manual approvals; screen for other speakers/game voice, clipping/BGM/noise and unusable material; present only ambiguous/high-risk cases for replay and approve/reject/defer with durable provenance. Mechanical WAV-format acceptance is not quality or speaker approval.
- **LV-R05 / Category library export (can prototype with a small explicitly approved subset):** combine LV-R04 style and LV-R06 QC/reviewer records; copy **only approved** WAVs/segments into `outputs/datasets/<profile>/<version>/audio/<Japanese-category>/` plus human README, Excel-compatible index and manifest. Verify real existing-WAV end-to-end and manual upstream VoiceDesign upload. Do not overwrite `data/`, prior releases, or upstream checkout.
- **LV-R07 / Future archive ingestion E2E:** hook new streaming archives into the library workflow only after v1. Do not make it an acceptance gate for v1.

The 48-sample **ASR** comparison and automated Irodori control are out of scope for library v1. Acoustic **style** validation with separately human-labeled audio is still mandatory. Change these boundaries in GitHub docs before expanding implementation, not by silently relaxing QC.

If requirements change, change this layout/decision first; do not place additional scripts in whichever directory is convenient.
