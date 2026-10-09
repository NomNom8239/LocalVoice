# LocalVoice — 新規話者のReference Bank自動ブートストラップ

Status: 2026-10-10 field issue reproduced conceptually from source: label-level approval promoted unseen clips. Source and offline tests updated to **explicit preview approval and resume**; **new Windows pytest and full end-to-end acceptance are pending**. The previous 43-pass checkpoint applies to the *old implementation*, not this change.

## Problem and change

Previously, `scripts/localvoice.py` required `data/reference_bank/<profile>/self_reference_bank.npz` **before** even extracting from an archive. Obtaining manually approved training snippets upfront was inefficient and created a circular first-profile onboarding path.

The existing `localvoice.py` now supports **one entry point and two runtime branches**:

- Existing `self_reference_bank.npz` → ordinary acquisition / separation / diarization / speaker-embedding comparison / `SELF`, `REVIEW`, `OTHER` classifications; existing Bank and original WAVs are unchanged.
- Missing `self_reference_bank.npz` → acquisition and vocals separation first → run pyannote diarization **once** → generate a few representative speech previews per diarized label → **ask a human to approve precisely which preview WAVs contain only the target voice** → use only those approved previews (not unlistened archive clips) for `build_reference_bank.py` → reuse the previous diarization for ordinary speaker comparison.

This does **not** claim that pyannote identifies the person's real name automatically. A label can be wrong, and **the same person may be split across multiple labels**. The user can select multiple labels after listening, or individual preview IDs to skip silence/other-person/mixed samples. Selecting a label includes only its displayed previews, never unseen turns. The other labels **must not automatically become negatives**.

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
Confirmed labels/clip IDs [p <ID> / IDs / q]:
~~~

At the prompt:

- `p SPEAKER_00`: play up to three previews using ffplay; if ffplay isn't installed, open the printed paths manually.
- `SPEAKER_00`: include only this label's up to three displayed previews, after confirming all are the target's voice.
- `SPEAKER_00,SPEAKER_01`: include only the displayed previews from both labels after confirming every clip is the target.
- `p SPEAKER_02_02`: play just that one specific preview.
- `SPEAKER_02_02,SPEAKER_02_03`: approve two specified preview clips and skip a silent or incorrect `SPEAKER_02_01`.
- `SPEAKER_03`: **do not select** when it contains other persons or mixed speech.
- The selection must contain at least **three unique approved preview files** (across labels).
- `YES`: explicitly confirm **every listed preview WAV** belongs to the target and start reference creation. No other WAV can be promoted.
- `q`: stop safely without making a Bank; use cleaner audio or manually prepared reference WAVs.


### Real acceptance finding and safe restart

In the first archive check the user's observations were:

- SPEAKER_00 and SPEAKER_01: target voice heard.
- SPEAKER_02: 1 silent preview and 2 target voice previews.
- SPEAKER_03: 2 other-person previews and 1 target + other mixed preview.

The previous implementation displayed **3 previews per label but silently used up to 24 previously unheard clips**. This is an unsafe expansion of the approval scope, particularly when pyannote labels are split or noisy. That behavior is removed. The only valid Bank inputs are **the exact preview WAVs that were approved**, and silent/other/mixed previews can be omitted by their individual IDs.

Stop the *old running process* at the prompt with `q` (before `YES`), then update to the fix and resume the original run without another download:

~~~powershell
git pull --ff-only origin feature/phase4-acoustic-pilot
.\.venv\Scripts\python.exe .\scripts\localvoice.py --profile LV_Bootstrap_Test --resume-bootstrap bootstrap_acceptance_001
~~~

The profile and run name must match the interrupted run. Example only if all clips in SPEAKER_00 and SPEAKER_01 were actually listened to and verified to be target-only:

~~~text
p SPEAKER_00
p SPEAKER_01
p SPEAKER_02_02
p SPEAKER_02_03
SPEAKER_00,SPEAKER_01,SPEAKER_02_02,SPEAKER_02_03
YES
~~~

Do **not** include SPEAKER_03 given the observed other-person and mixed voices. The silent SPEAKER_02_01 is omitted. If any other clip is uncertain, omit it too. The command will print every included filename and demand confirmation before creating the Bank.



**Important for the in-progress two-hour archive:** On the originally running process, type `q` before `YES`. After pulling the corrected code, use exactly:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\localvoice.py --profile LV_Bootstrap_Test --resume-bootstrap bootstrap_acceptance_001
~~~

Replace with the actual profile and run name. **Do not add `--classify-after-bootstrap`**, which would run pyannote over the full long archive again. The default command produces a Bank and stops; it **does not produce `classification.tsv` or a complete dataset from the original long archive**. Further full-archive classification, if later requested, needs another pass because the original turns were not persisted. This limitation is explicit rather than silently requiring hours of processing.

The review is an interactive terminal workflow. In a non-interactive session, the process **fails closed after creating preview WAVs**. Don't substitute “longest/loudest speaker” as auto-identity.

## Candidate quality gates and stored outputs

Bank candidates must be:

- Speech turns between **2–20 seconds**.
- Non-overlapping with another diarized speaker.
- Non-silent, with RMS >= `1e-4`.
- Not heavily clipped (fraction of absolute samples >= `0.999` must be <= `0.005`).
- Previewed clips are distributed across the archive. Up to **three preview clips per detected speaker label** are available, but only the specifically human-approved previews are copied into the Bank stage. **No previously unlistened clips** are promoted. At least 3 unique approved clips are required.

These are **mechanical quality filters, not a semantic or final identity/quality guarantee**. Other voice/BGM in an apparently clean turn is possible. The generated Bank is provisional and must be evaluated with actual normal/other voice samples.

Stored paths:

~~~text
data/source/<source-key>/...                     # acquisition source cache (existing behavior)
data/wav_master/<source-key>.wav                 # WAV master cache (existing behavior)
data/runs/<profile>/<run-name>/
  separated/                                    # vocals separation (unless --vocals)
  bootstrap/
    previews/                                   # small WAV clips for listening
    selected_reference_wavs/                    # ONLY explicitly approved preview clips
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

- `--force` deletes an **existing run directory** and must not be used casually. Use `--resume-bootstrap <existing-run-name>` after canceling at speaker selection to **reuse the preview WAVs and build ONLY the Reference Bank**, without downloading, separating or reading full Vocals, running pyannote, or classifying the full archive. This intentionally stops after Bank creation. A separate explicit `--classify-after-bootstrap` switch opts into the expensive full-archive diarization (which may take hours). For originally `--vocals` runs, that explicit opt-in also needs `--resume-vocals <same-Vocals.wav>`. Prior diarization turns from interrupted legacy runs were not saved, so the full classification cannot be resumed without recomputing them. Once Bank-only resume succeeds, the same resume command is blocked by existing-Bank protection; do not try to bypass that safeguard.
- If the Bank subprocess crashes, inspect the report and staged WAVs and don't rely on an incomplete NPZ/JSON. The CLI refuses an inconsistent partial Bank rather than destroying it.
- If too few safe 2–20-second turns exist, use a cleaner/longer archive or manually collect at least 3 verified WAVs with the existing `build_reference_bank.py`. There is no silent downgrade to an unidentified speaker.
- Source cache IDs are still based on yt-dlp IDs and **can collide across platforms**. Do not claim safe multi-site collection E2E before LV-R07 addresses namespacing.
- This implementation does not add new CLAP/ASR requirements, a new independent downloader, automatic review of all clips, dataset approval, or an automatic linkage to v2 AST metadata.

## Acceptance (pending in the real user environment)

- [ ] A fresh `New_Speaker` with no Bank: `--url` yields Vocals → speaker previews → human label selection → `self_reference_bank.npz` + JSON → SELF/REVIEW/OTHER with a single diarization run.
- [ ] Confirm preview playback/label prompts, including multiple labels and per-preview clip selection to exclude silence/mixed voices.
- [ ] Cancel the original run at the selection prompt, then resume using `--resume-bootstrap <existing-run-name>` and verify **no download, audio-separator, pyannote, full Vocals reading, or full classification** occurs.
- [ ] Prove the default Bank-only path does not call `classify` or `diarize_turns`, and that `--classify-after-bootstrap` is the only opt-in path that invokes full-archive classification.
- [ ] Existing Ui_Shigure Bank: ordinary classification is unchanged; it is never replaced and bootstrap preview files aren't emitted.
- [ ] No cross-profile or previous-run files are altered; retry with fresh run name after failure.
- [ ] Verify new output count and manually inspect selected references / SELF, REVIEW and OTHER.
- [ ] Run `tests/test_reference_bootstrap.py` plus the existing regressions with repo-local Python. The tests mock the model and must not download models.

