# Human-browsable audio library aligned with Irodori Emoji Palette

Status (2026-10-09): **Priority MVP design requirement for LV-R04/06/05; category library not implemented.** ASR 12-sample CUDA Pilot is complete but full ASR adoption deferred. Preserve the existing archive/RVC code; do not restore legacy Irodori ASR/style/LoRA experiments.

## User-facing goal and MVP scope

**Goal:** Take the existing **1,167** candidate WAVs under `data/training_audio/Ui_Shigure/audio/` and create a **human-browsable Japanese-category reference audio library**. People must be able to open `通常会話`, `囁き`, `吐息`, `笑い`, `喘ぎ_うめき`, `息切れ_荒い呼吸` and other populated categories, listen to real recordings, and manually pick clean, verified reference WAVs in upstream Irodori VoiceDesign **without opening metadata or running Python**.

The processing workflow is: **input WAV inventory → acoustic style candidate assessment (LV-R04) + independent identity/recording-quality QC (LV-R06) → targeted replay/reviewer approvals → immutable Japanese-category export (LV-R05)**. Work with real existing candidate WAVs first; **automatic new-stream ingestion is later LV-R07**, and controlling upstream VoiceDesign automatically is out of scope. All 1,167 inputs must have an evidence-backed processing status, but not all must be approved or placed in final folders.

**Speech-to-text is optional metadata.** It is not a prerequisite for classification, QC, human approval or v1 export. The 12-sample Kotoba/Whisper ASR Pilot found hallucinations even on artificial silence, and extreme repetition on nonverbal input, so text/caption must never be treated as the source-ground-truth acoustic label. The 48-sample **ASR** model comparison belongs to later optional ASR formal selection; acoustic **style** validation against human-labeled WAVs remains necessary.

Category names are aligned to the [official Irodori Emoji Palette implementation](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/gradio_emoji_palette.py), but **Emoji Palette entries are synthesis-time controls, not an audio classifier or ground-truth recording labels**. Track palette revision when freezing the taxonomy. Actual reference conditioning also depends on recording quality; a category alone does not guarantee the corresponding style will be synthesized.

## Current implementation checkpoint

AST's read-only batch classifier and separate **UNVERIFIED** candidate folder export are implemented under `src/localvoice/style/batch.py` with CLI `style-batch inventory|run|summary|export`. This implementation does not imply that the real 1,167 files have been processed: **real Windows/GPU batch and manual folder-browsing acceptance are still pending**. Use the [AST batch Windows runbook](AST_BATCH_RUNBOOK.md). The reviewed `outputs/datasets/` export and LV-R06 identity/recording QC remain separate unfinished work.

## Pragmatic two-tier delivery (2026-10-09 decision)

**Do not block the user's first useful library on perfect discrimination among whisper, sigh, pant and moan.** The near-term priority is to process the existing 1,167 WAVs with **AST alone** and create a human-browsable, **explicitly unverified candidate category library**. CLAP, 48-sample ASR accuracy evaluation, class-by-class human reference annotation, fine-grained segmentation, and calibrated confidence/recall benchmarks are **not prerequisites** to this first batch/browse deliverable.

- **Candidate/browse stage (first delivery):** AST assigns best-effort **suggested** Japanese categories when supported by its available AudioSet label set. No supported label, weak/ambiguous/overlapping evidence, or unsupported expressions are stored under `99_未分類_要確認` (or a clearly marked mixed category), not forced into a false claim. Retain raw AST scores, label mapping version, input SHA, review status and any QC warnings. Every one of the 1,167 inputs must remain accounted for, including prediction failures. Keep original WAVs unchanged.
- **Candidate output:** `outputs/candidates/<profile>/<version>/audio/<Japanese-category>/` plus `README.txt`, `index.csv` and `manifest.jsonl`. Mark the entire candidate version **UNVERIFIED / AST suggested grouping** prominently; allow manual browsing without Notion/Python. Provisional file placement does **not** assert a human-approved category or clean audio. It is a separate delivery surface from approved libraries.
- **Approved/curated stage (later or parallel):** LV-R06 identity/QC plus human-reviewed style approval may promote selected clips to the existing **`outputs/datasets/<profile>/<version>/`** structure. Only this verified area may be called human-approved and clean. Prior manually approved provenance remains intact, but do not mark automatically collected `my_voice` as explicitly human-reviewed.
- Start at **WAV-level classification**, even if some short WAVs contain mixed events. Fine-grained segmentation, model ensembling and retraining are subsequent improvements only when the candidate library's actual use reveals a need.
- A quick AST technical smoke test is enough to proceed to batch **candidate grouping**; do not demand calibrated distinction among all acoustic classes before moving on. Critical technical blockers (cannot load model, corrupt input evidence, unsafe writes, missing per-input status) still stop the run.
- Avoid a pipeline that sets `requires_review` on every file **and then refuses to make anything browsable**. `requires_review` must block **verified approval**, not the clearly labeled unverified browsing version. Unknown-category material remains accessible as unknown, not deleted.

This deliberately separates **useful organization now** from **quality certification later**, without silently relaxing the integrity checks on final approved exports.

## Initial category taxonomy

| Stable code | Export directory | Corresponding Palette entry | Label evidence |
| --- | --- | --- | --- |
| `normal_speech` | `01_通常会話` | None (LocalVoice baseline) | Predominantly ordinary, intelligible speech |
| `whisper` | `02_囁き` | 👂 囁き | Audible whisper-like delivery |
| `exhale` | `03_吐息` | 😮‍💨 吐息 | Exhalation, sigh, sleep breath |
| `laugh` | `04_笑い` | 🤭 笑い | Giggle or laughter |
| `moan` | `05_喘ぎ_うめき` | 🥵 喘ぎ | Moaning, groaning, vocal panting-like moans |
| `pant` | `06_息切れ_荒い呼吸` | 🌬️ 息切れ | Rapid/heavy breathing and breathlessness |
| `gasp` | `07_息をのむ` | 😮 息をのむ | Short audible sharp inhale/gasp |
| `cry` | `08_泣き声` | 😭 泣き声 | Sobbing / crying |
| `scream` | `09_悲鳴` | 😱 悲鳴 | Screams or yells |
| `yawn` | `10_あくび` | 🥱 あくび | Yawning |
| `hum` | `11_鼻歌` | 🎵 鼻歌 | Humming |
| `other_approved` | `90_その他_承認済み` | None (fallback) | Usable clip manually approved but not a listed category |

Only create populated categories; do not create 40+ empty folders to mirror every palette button. Labels such as `😏 からかう`, `🫶 優しく`, `😠 怒り`, `😪 眠そう`, etc. may be proposed as **secondary tags** until their audible boundaries and human-labeled evaluation data support a separate folder. `⏸️ 間` is not a standalone reference voice class; `📢 エコー` and `📞 電話越し` describe recording/effect conditions and are not separate clean reference voice classes. Their presence should be tracked and may be a QC reason to exclude the clip.

## Workflow ownership and gates

- **LV-R04 / acoustic style:** Use features **of the WAV audio** to assess ordinary speech and distinguishable vocal events/delivery. Keep speaker identity and recording QC separate. Human-annotate a small, representative set of real clips/segments, freeze a label/uncertainty evaluation protocol, test candidate model/thresholds, then batch-generate **candidates/unknown/failures** for all 1,167 existing WAVs. An ASR transcript and a Pilot selection category are **not** acoustic gold labels. For uncertain, mixed and unsupported evidence, set `unknown` or `requires_review`; do not force a class.
- **LV-R06 / independent identity/QC and review:** Preserve the distinction between automatically collected `my_voice` and human-reviewed `review_approved`/`review_emotion`. Run recording QC (BGM/effects, other speakers, game voices, clipping/noise, silence, too-short/poor-quality clips), use known approval provenance, prioritize ambiguous/high-risk clips for simple replay, and save approve/reject/defer/edit-label decisions with reviewer/provenance and input fingerprint. **File format eligibility and classification confidence are not human approval.** Do not require a person to listen to every one of the 1,167 files before candidate processing.
- **LV-R05 / versioned reference library export:** Combine **approved speaker identity + quality acceptance + human-approved primary style**; copy clean clips to one category each in a **new** immutable version under `outputs/datasets/<profile>/<version>/`. Produce an Explorer-friendly index and provenance; keep unapproved/failed material in review/evidence, not the promoted library. Prove the full *existing WAV subset → candidate classification/QC → review → category output → manual VoiceDesign reference selection* integration before publishing v1.
- **LV-R03 / optional ASR auxiliary:** ASR is a separate, non-blocking source of text/time estimates, not a required stage of acoustic classification or export. Its 12-sample CUDA Pilot passed technically; 48-sample formal comparison and full-ASR adoption are deferred. See [ASR status/contract](PHASE3_JAPANESE_ASR.md).
- **LV-R07 / later ingestion E2E:** New streaming archive acquisition through processing → library may be implemented after the existing-WAV v1 is usable. Not a v1 completion gate.

Both unverified candidate and approved output folders are **manual lookup tools**, not evidence that emoji-only conditioning or LoRA training works. Upstream VoiceDesign UI remains untouched. Do not confuse a candidate's display location with an approval.

## Intended on-disk output

```text
LocalVoice/
├─ data/                              # Existing assets: NEVER migrate or modify
├─ work/<run-id>/
│  ├─ transcription.jsonl             # Optional ASR evidence when requested
│  ├─ style_predictions.jsonl         # LV-R04 candidates/unknown + offsets, model versions
│  ├─ qc_results.jsonl                # LV-R06 identity / recording quality evidence
│  ├─ review_queue.csv                # Pending decisions, not final exports
│  └─ review_decisions.jsonl          # Human approvals/rejections/deferred (proposed contract)
└─ outputs/datasets/<profile>/<version>/
   ├─ README.txt                       # Japanese guide, label/emoji mapping, selection tips
   ├─ index.csv                        # Human-readable search/index (UTF-8 with BOM for Excel)
   ├─ manifest.jsonl                   # Machine-readable provenance and labels
   └─ audio/
      ├─ 01_通常会話/
      ├─ 02_囁き/
      ├─ 03_吐息/
      ├─ 04_笑い/
      ├─ 05_喘ぎ_うめき/
      ├─ 06_息切れ_荒い呼吸/
      ├─ 07_息をのむ/
      ├─ 08_泣き声/
      ├─ 09_悲鳴/
      ├─ 10_あくび/
      ├─ 11_鼻歌/
      └─ 90_その他_承認済み/
```

These directories are examples of the *final schema*, not directories to create yet; omit empty classes.

## Clip and review rules

1. Classify **real WAV segments**, not entire multi-minute streams or mixed-event WAVs as a single expression. Preserve the original audio, source SHA, segment IDs and start/end offsets. If clipping would remove essential vocal/breath context, retain the intact clip with tags or queue for review rather than cutting blindly. All 1,167 candidates must receive a classification/QC/unknown/failure evidence row, not forced approval.
2. Each **promoted** clip has one human-approved primary category for directory placement and zero or more secondary labels. For mixed ordinary speech/laughter, choose a dominant class **only with audible evidence and approval**; otherwise queue for review. Do not silently duplicate the same WAV across several folders.
3. Acoustic candidate labels, model confidence and mechanical WAV validity are **not approvals**. Missing/unsupported confidence is `null`, never fabricated. Identity (speaker), QC (recording quality), style and reviewer decision are separate fields. Keep `unknown`, `requires_review`, rejected, QC-failed and non-target-speaker audio **out of promoted folders**, with review/evidence and the reason retained. Preserve existing genuine human approval provenance; do not infer it for automatically collected `my_voice`.
4. Each exported audio file has a stable neutral filename (not a generated descriptive claim). `index.csv` (UTF-8 with BOM for Excel) provides original source, optional Japanese transcript **only when available**, primary Japanese label/emoji, secondary tags, duration, identity/QC/review status and listening notes. A missing transcript never excludes an otherwise approved nonverbal reference WAV. Avoid raw emoji in filenames for cross-tool compatibility.
5. `manifest.jsonl` records source WAV path and SHA-256, original acquisition/run/provenance, original clip/segment offsets, output SHA-256/path, independent speaker-identity decision, QC outcomes, primary/secondary acoustic labels, mapped emoji, model/config/taxonomy revisions, reviewer decision and approval provenance; optional ASR speech/transcript fields may be null. Keep failure and unresolved evidence in `work/<run-id>/`, with every input accounted for.
6. Do not change files under `data/`; published versions are immutable. On collision, refuse to overwrite and require a new version. Do not put LocalVoice scripts or exported library under `Irodori-TTS/`.
7. Allow manual playback and category browsing **without dependency on Notion, model downloads, or a custom browser UI**. `README.txt` explains that Irodori Palette controls inference, not category selection.
8. **Library v1 acceptance** walks a real existing-WAV subset → acoustic classification candidates → separate identity/QC review → reviewer approval → new Japanese-category folder/index/manifest → Explorer playback and manual reference upload in VoiceDesign. Verify provenance, per-input processing states, non-destructive reads, and that ambiguous/unapproved material is excluded. A real new-archive ingestion E2E belongs to **later LV-R07**; full ASR/CER testing or Irodori automation is not a v1 gate.

## Library v1 acceptance gate and non-scope

- **Input coverage:** inventory all existing 1,167 candidate WAVs in `data/training_audio/Ui_Shigure/audio/` without editing them; emit a versioned, auditable classification/QC/unknown/failure record for each.
- **Acoustic validation:** freeze primary taxonomy and evaluate on a small independently **human-annotated real audio** set before batch classification. Track class confusions, unclear mixed events and conservative abstentions. An ASR transcript cannot substitute for this.
- **Reviewer protection:** valid identity + QC + human-approved primary style are all necessary before a WAV/segment is copied into the library. Suspect foreign/game voices, poor audio, ambiguous style, errors and review backlog stay outside promoted folders.
- **Human usability:** new immutable `outputs/datasets/<profile>/<version>/audio/<Japanese-category>/` folders for **populated** categories, `README.txt`, `index.csv` and `manifest.jsonl`, playable in Explorer without Notion/Python or a custom browser. Manually pick at least one published WAV in official VoiceDesign to validate compatibility.
- **Data safety:** no writes to existing `data/`, no changes inside `Irodori-TTS/`, no overwriting previous output versions. Run manifests retain source hashes, offsets, QC, classification and approval provenance; no silent duplication or deletion.
- **Unverified candidate library first:** All 1,167 input WAVs receive best-effort AST candidate grouping or an explicit unknown/error record; browse copies are distinctly labeled unverified under `outputs/candidates/` and never treated as approved. They are not blocked by individual human reviews. The requirements above for identity/QC/human approval apply to the **separate, verified** `outputs/datasets/` area.
- **Out of scope for v1:** 48-sample **ASR** accuracy comparison, complete ASR transcription, new-stream archive E2E (LV-R07), automated VoiceDesign operations and LoRA training. These are **not** blockers for LV-R04/06/05. Test the actual archive-to-library pathway in LV-R07 after library v1.

Phase 3 Pilot must not auto-sort WAVs based on transcript text. LV-R04 owns acoustic candidates, LV-R06 owns identity/QC/reviewer evidence, and LV-R05 owns approved versioned library export. The ASR result may be included as optional metadata only.
