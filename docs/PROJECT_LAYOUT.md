# LocalVoice directory ownership and change rules

Status: adopted for the clean-baseline branch (2026-10-09).
Scope: development layout and file-placement rules only. This document does not claim that future directories or features already exist.

## Ownership boundaries

| Path | Owner / responsibility | Git policy | Write policy |
| --- | --- | --- | --- |
| `data/` | Existing acquired audio, review decisions, reference banks and legacy outputs | Ignored; preserve in place | **No writes by newly developed ASR/style/dataset stages.** Existing legacy scripts have established writes here; do not alter their contracts without a separate, tested migration. Never bulk-clean this directory. |
| `scripts/` | Existing archive retrieval, separation, diarization, identity verification, review, and RVC dataset commands | Tracked | Maintain the existing entry points. Do not add new ASR/style experiment scripts here. |
| `src/localvoice/` | **Future** first-party application code for transcription, style assessment, dataset export and orchestration | Tracked | All new Phase 3+ application code goes here. Create only modules that the approved implementation actually needs. |
| `tests/` | Tests for first-party code | Tracked | Tests reflect public behavior and IO boundaries; fixtures cannot silently depend on private `data/`. |
| `docs/` | Stable layout, design decisions and operating instructions | Tracked | Keep stable rules in this document; put significant architecture decisions in `docs/decisions/` when they occur. |
| `Irodori-TTS/` | **Upstream** Irodori-TTS standalone Git clone and runtime | Ignored by LocalVoice | Do not add LocalVoice code, patches, tools, LoRA experiments, or custom datasets inside. Normal upstream-managed `.venv/`, `gradio_outputs_voicedesign/`, and caches may appear. |
| `.venv/` | Existing LocalVoice acquisition/processing Python runtime | Ignored | No extra environments without a documented dependency conflict. Irodori uses its own `Irodori-TTS/.venv/`. |
| `work/` | **Future** disposable, per-run intermediate outputs | Ignored | New operations create a fresh run ID; never overwrite another run or `data/`. Safe to regenerate, but do not automatically wipe unknown contents. |
| `outputs/` | **Future** promoted, reviewed deliverables (datasets, transcripts if exported, selected synthesized audio) | Ignored | Immutable versioned deliveries; never silently replace or delete. |
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
├── pyproject.toml                  # future: create when packaging Phase 3
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
├── src/localvoice/                 # future: new application modules
│   ├── __main__.py                 # future: ONE CLI (python -m localvoice ...)
│   ├── transcription/             # Phase 3 only
│   ├── style/                     # Phase 4 only
│   └── dataset/                   # Phase 5 integration only
├── tests/                          # future: tests organized by feature
├── docs/
│   └── PROJECT_LAYOUT.md          # this document
├── data/                           # existing data; DO NOT MIGRATE/DELETE
├── work/<run-id>/                  # future: intermediates, manifests, logs
├── outputs/datasets/<profile>/<version>/   # future: human-browsable, approved Emoji Palette-aligned audio library
├── outputs/tts/                    # future: retained copies of WebUI audio
├── cache/                          # optional: explicitly configured cache
├── .venv/                          # existing LocalVoice runtime
└── Irodori-TTS/                    # upstream Git clone only
    ├── .git/
    ├── .venv/
    ├── gradio_app_voicedesign.py
    └── gradio_outputs_voicedesign/ # upstream runtime output, if generated
```

Do not create placeholder directories or files just to match the diagram. In particular, `src/`, `work/`, `outputs/`, `tests/`, and `pyproject.toml` need not exist until a task requires them.

## Existing acquisition compatibility: exception, not the new pattern

`config.toml` currently points existing code to `data/source`, `data/wav_master`, `data/diarization`, `data/reference_bank`, `data/runs`, `data/training_audio`, and `data/rvc_dataset`.

The existing `scripts/localvoice.py --force` deletes a selected `data/runs/<profile>/<run-name>` folder before recreating it. **Never use `--force` against an existing run without a separate, explicit backup/review.** Phase 3+ code must not reuse that destructive execution pattern.

Do not claim that the old acquisition pipeline is read-only, or move existing `data/` items as part of Phase 3. Any future migration of the legacy pipeline's output root requires its own scoped design and regression tests.

## Phase 3+ creation rules

1. Start with one user-facing command interface (`python -m localvoice ...`) and one reusable, import-safe code path per capability. Build its editable-package configuration when Phase 3 is implemented. Never create `*_v2.py`, `*_final.py`, `experiment_*.py`, and similar parallel implementations in the source tree.
2. Transcription consumes approved audio **by absolute/validated input path** from `data/` (or another explicitly chosen source); it writes only to a fresh `work/<run-id>/`. It does **not** train models or auto-update speaker reference banks.
3. Style assessment is separate from speech recognition. Missing, nonverbal, ambiguous or unsupported evidence remains explicitly `unknown`/`requires_review`; ASR text is not proof of a vocal style.
4. Each run records a manifest with run ID, source identity/path and fingerprint, selected models/config versions, stage statuses, and output paths. Save evidence and failures, not just success files. A final reviewed dataset export is a separate, explicit promotion to a new version under `outputs/datasets/`. **The promoted folders must be usable by people without opening metadata files:** see [Emoji Palette-aligned audio library specification](EMOJI_AUDIO_DATASET.md).
5. No implicit recursive cleanup, auto-approval, destructive overwrite, or silent modification of existing audio/reference banks. New operations fail closed on destination conflicts. Data operations and tests must not depend on the user's private recordings being present.
6. Never modify upstream Irodori-TTS files to compensate for a LocalVoice issue. Keep launch/integration glue in LocalVoice, and use the upstream UI as shipped. The official UI may generate `Irodori-TTS/gradio_outputs_voicedesign/`; copy any audio worth retaining into `outputs/tts/` before a future reinstall.
7. A proposed new module, CLI entry point, environment, or output directory needs a documented purpose and an owner in this table. If an existing location owns the responsibility, extend that implementation rather than adding another file.
8. Verify the entire archive-to-dataset flow with an actual sample before declaring Phase 5 complete. A test pass for isolated modules is not an integration pass.

## Phase gates

- **Phase 1:** baseline code retains archive acquisition and speaker/dataset processing; old independent LoRA files are excluded.
- **Phase 2:** official Irodori-TTS installed independently, GPU works, and reference-audio VoiceDesign generation succeeds.
- **Phase 3:** approve transcription IO schema and model choices first; implement **only** `src/localvoice/transcription/`, a single CLI integration, and matching tests. Distinguish speech/non-speech/uncertain events for later classification; **do not** classify Emoji Palette styles at this stage. Existing `data/` is read-only to this stage.
- **Phase 4:** approve style-label taxonomy (including distinguishable Emoji Palette-aligned categories and human review) separately; implement `src/localvoice/style/` and corresponding tests.
- **Phase 5:** export reviewed clips into versioned **human-browsable Japanese category folders** under `outputs/datasets/<profile>/<version>/audio/` (normal conversation, whisper, laughter, moans, breathlessness, etc.) with CSV/JSONL index. Verify real end-to-end provenance, error handling, ease of manual use and non-modification of existing inputs.

If requirements change, change this layout/decision first; do not place additional scripts in whichever directory is convenient.
