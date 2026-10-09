# LocalVoice — 新規話者のReference Bank自動ブートストラップ

Status: code + offline regression tests committed; **real Windows/pyannote/audio-separator end-to-end acceptance pending**.

## Problem and change

Previously, `scripts/localvoice.py` required `data/reference_bank/<profile>/self_reference_bank.npz` **before** even extracting from an archive. Obtaining manually approved training snippets upfront was inefficient and created a circular first-profile onboarding path.

The existing `localvoice.py` now supports **one entry point and two runtime branches**:

- Existing `self_reference_bank.npz` → ordinary acquisition / separation / diarization / speaker-embedding comparison / `SELF`, `REVIEW`, `OTHER` classifications; existing Bank and original WAVs are unchanged.
- Missing `self_reference_bank.npz` → acquisition and vocals separation first → run pyannote diarization **once** → generate a few representative speech previews per diarized label → **ask a human which anonymous label(s) belong to the target speaker** → quality-filter and automatically select reference clips across the recording → run existing `build_reference_bank.py` → reuse the previous diarization for ordinary speaker comparison.

This does **not** claim that pyannote identifies the person's real name automatically. A label can be wrong, and **the same person may be split across multiple labels**. The user can select multiple labels only after listening and confirming they are the same target. The other labels **must not automatically become negatives**.

## Initial operator workflow

~~~powershell
cd F:\AIProjects\LocalVoice
git pull --ff-only origin feature/phase4-acoustic-pilot

.\.venv\Scripts\python.exe .\scripts\localvoice.py --profile New_Speaker --url "<URL>"
~~~

Use `--wav <WAV>` for an existing source (which the script passes through audio-separator), or `--vocals <WAV>` when vocals are already separated. These start the same automatic bootstrap when no Bank exists. The `--url` path uses yt-dlp; other sites work only where yt-dlp's extractor supports the URL. Current cross-platform ID collisions are a separate LV-R07 issue.

The terminal lists candidate speaker labels and stored preview WAVs:

~~~text
Speaker SPEAKER_00: 43 qualifying speech clips
  preview: ...\bootstrap\previews\SPEAKER_00_01.wav
Speaker SPEAKER_01: 9 qualifying speech clips
...
Target speaker label(s) [p <label> / labels / q]:
~~~

At the prompt:

- `p SPEAKER_00`: play up to three previews using ffplay; if ffplay isn't installed, open the printed paths manually.
- `SPEAKER_00`: select one target label.
- `SPEAKER_00,SPEAKER_01`: select several labels *only if both are confirmed to be the same person*.
- `YES`: explicitly confirm the person selection and start reference creation.
- `q`: stop safely without making a Bank; use cleaner audio or manually prepared reference WAVs.

The review is an interactive terminal workflow. In a non-interactive session, the process **fails closed after creating preview WAVs**. Don't substitute “longest/loudest speaker” as auto-identity.

## Candidate quality gates and stored outputs

Bank candidates must be:

- Speech turns between **2–20 seconds**.
- Non-overlapping with another diarized speaker.
- Non-silent, with RMS >= `1e-4`.
- Not heavily clipped (fraction of absolute samples >= `0.999` must be <= `0.005`).
- Selected across the entire archive, **up to 24 clips**, rather than only its first minutes. At least 3 qualifying clips must exist across confirmed labels.

These are **mechanical quality filters, not a semantic or final identity/quality guarantee**. Other voice/BGM in an apparently clean turn is possible. The generated Bank is provisional and must be evaluated with actual normal/other voice samples.

Stored paths:

~~~text
data/source/<source-key>/...                     # acquisition source cache (existing behavior)
data/wav_master/<source-key>.wav                 # WAV master cache (existing behavior)
data/runs/<profile>/<run-name>/
  separated/                                    # vocals separation (unless --vocals)
  bootstrap/
    previews/                                   # small WAV clips for listening
    selected_reference_wavs/                    # up to 24 eligible WAV clips
  my_voice/                                     # classification output
  review/
  review_unscored/
  rejected/
  classification.tsv
data/reference_bank/<profile>/
  self_reference_bank.npz                       # built by existing script
  self_reference_bank.json                      # reference audit report
  thresholds.json                               # optional prior/manual calibration
~~~

The bootstrap **never deletes previous outputs** and refuses to overwrite an existing NPZ or JSON. It preserves the upstream Irodori checkout, AST classification histories and raw original audio.

The first classification uses existing `config.toml` threshold fallback. An optional `thresholds.json` can later be created using **verified negative voice** from another speaker:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\calibrate_threshold.py --profile New_Speaker --negative-source "<verified-other-voice-WAV-folder>"
~~~

Never use a raw other pyannote label without confirming it belongs to a different person.

## After bootstrapping

~~~powershell
.\.venv\Scripts\python.exe .\scripts\review_training_audio.py --profile New_Speaker
.\.venv\Scripts\python.exe .\scripts\collect_training_audio.py --profile New_Speaker --dry-run
# Inspect MY_VOICE and candidate files before adding
.\.venv\Scripts\python.exe .\scripts\collect_training_audio.py --profile New_Speaker
~~~

Important: `collect_training_audio.py` automatically includes `my_voice/` based on model SELF classifications. Bank bootstrap does **not** make these decisions infallible. Do not use the result as a “fully human approved” dataset without a separate QC and promotion gate.

AST classification and metadata versioning remain an independent phase: `style-batch run`, HTML reviewer, `metadata archive/revise`. Nothing silently edits Ui_Shigure v2 when a new archive is collected.

## Failure, restart and non-goals

- `--force` deletes an **existing run directory** and must not be used casually. Choose a **fresh** `--run-name` after reviewing an interrupted run; original source WAVs and caches remain. The CLI does not promise resume of incomplete speaker bootstrap yet.
- If the Bank subprocess crashes, inspect the report and staged WAVs and don't rely on an incomplete NPZ/JSON. The CLI refuses an inconsistent partial Bank rather than destroying it.
- If too few safe 2–20-second turns exist, use a cleaner/longer archive or manually collect at least 3 verified WAVs with the existing `build_reference_bank.py`. There is no silent downgrade to an unidentified speaker.
- Source cache IDs are still based on yt-dlp IDs and **can collide across platforms**. Do not claim safe multi-site collection E2E before LV-R07 addresses namespacing.
- This implementation does not add new CLAP/ASR requirements, a new independent downloader, automatic review of all clips, dataset approval, or an automatic linkage to v2 AST metadata.

## Acceptance (pending in the real user environment)

- [ ] A fresh `New_Speaker` with no Bank: `--url` yields Vocals → speaker previews → human label selection → `self_reference_bank.npz` + JSON → SELF/REVIEW/OTHER with a single diarization run.
- [ ] Confirm preview playback/label prompts, including multiple labels known to belong to one person.
- [ ] Existing Ui_Shigure Bank: ordinary classification is unchanged; it is never replaced and bootstrap preview files aren't emitted.
- [ ] No cross-profile or previous-run files are altered; retry with fresh run name after failure.
- [ ] Verify new output count and manually inspect selected references / SELF, REVIEW and OTHER.
- [ ] Run `tests/test_reference_bootstrap.py` plus the existing regressions with repo-local Python. The tests mock the model and must not download models.

