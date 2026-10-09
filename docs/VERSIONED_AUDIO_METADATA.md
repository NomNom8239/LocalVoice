# LocalVoice — WAVを共通化した分類・レビュー履歴管理

Status: implemented, with Windows browser acceptance pending. This document defines the metadata-first successor to
the copy-per-version `outputs/candidates/<profile>/v1,v2` releases.

## Purpose and ownership

`data/training_audio/<profile>/audio/` contains the canonical audio bytes and remains
**read-only**. AST batch evidence `work/<run-id>/` remains read-only after completion.
Versioning keeps classification/review decisions and source hashes, **not a new copy of
1,167 WAV files per revision**. A browsable WAV folder can be *materialized on demand*.

New owned root: `outputs/metadata/<profile>/` (ignored by Git; local-only):

```text
outputs/metadata/Ui_Shigure/
  inbox/                       # manually place downloaded CSV/JSON here
    README.txt
  versions/
    v1/
      manifest.jsonl            # every source ID, SHA, AST category, source location
      index.csv
      summary.json
      receipt.json              # manifest checksum, parent, input SHA, provenance
    v2/
      manifest.jsonl
      index.csv
      summary.json
      receipt.json
      review/
        review_decisions.csv    # immutable copy of HTML CSV
        review_state.json       # optional original HTML save
    v3/                         # later revisions: only metadata, no WAV
      ...
```

Version names are immutable. Each revision stores the *full resolved view* of every
source, plus the exact input CSV/JSON files (where provided). No implicit merge of
partial CSV exports. HTML review CSVs must be **cumulative**: when adding decisions,
load the previous JSON in the reviewer first, then export a full new CSV.

`outputs/candidates/<profile>/<name>/` is a materialized *view/cache*, not the
canonical version history. Previous `v1` and `v2` WAV copies are NEVER automatically
deleted. They may only be manually retired after successful migration,
version verification, restoration to a separate directory, and a backup check.

`outputs/datasets/` remains reserved for separately approved audio and is not
touched by any metadata command. Review state `keep+self+ok` is
`reviewed_usable`, not final dataset approval. Ambiguous clips go to
`96_要確認_使用不可`, hold to `97_保留`, reject to `98_除外`.
Unreviewed clips retain the AST category and are **not automatically approved**.

## Operating flow

1. Archive existing v1 and v2 to metadata-only versions **after full SHA checks**.
   For v2, provide the CSV used to generate it, and optionally saved HTML JSON.
2. Confirm metadata versions via `verify` and the source IDs/SHA.
3. Later human revisions use `revise` with a cumulative HTML review CSV; it
   creates v3/v4 under the metadata root, without copying WAVs.
4. Use `materialize --version vN --output-name <unused name>` when a
   category-separated Explorer folder is actually required. Source WAVs are
   copied *only on explicit request* into a new output name; no silent overwrite.
5. Keep `inbox/` as the human-facing location for downloaded `.csv` and `.json`.
   Importing a version makes its own immutable evidence copies in `review/`.

## Manual reclassification in the existing browser reviewer

The tracked repository-root `localvoice_ast_candidate_reviewer.html` opens locally
in Chrome/Edge. It is the same reviewer used for v1, now extended for v2 and
later metadata versions. No new HTML copy or WAV copy is required.

**Method A (when the v2 WAV candidate folder still exists):**

1. Open `localvoice_ast_candidate_reviewer.html`.
2. Select `outputs/candidates/Ui_Shigure/v2` with the first folder picker
   (select v2 itself, **not** its `audio` child).
3. The reviewer reconstructs existing manual decisions from v2's
   `manifest.jsonl`, preserving reviewed records on subsequent export.

**Method B (when the materialized v2 WAV folder was removed):**

1. Select `outputs/metadata/Ui_Shigure/versions/v2` using the metadata picker.
2. Select `data/training_audio/Ui_Shigure/audio` using the original-WAV picker.
3. Browse/reclassify directly against the original WAVs by `relative_path`;
   do not run `materialize` solely to use the reviewer.

For either method, a prior review CSV/JSON can also be imported with the
**CSV/JSON picker**. Existing decisions are never silently dropped: all
previously reviewed source IDs must appear in the imported cumulative file.
The reviewer checks each recorded `source_id`, `source_sha256`, and
**original** `ast_category` from v2 (not v2's category folder name). A review
that contains an unknown/mismatched source is rejected without partial import.

After reviewing, download **both** JSON (future resume) and cumulative CSV
(`metadata revise` input). Put them in
`outputs/metadata/Ui_Shigure/inbox/` and run:

```powershell
$csv = ".\outputs\metadata\Ui_Shigure\inbox\localvoice_ast_review_decisions_v2_cumulative.csv"
$json = ".\outputs\metadata\Ui_Shigure\inbox\localvoice_ast_candidate_review_v2.json"

.\.venv\Scripts\python.exe -m localvoice style-batch metadata revise `
  --profile Ui_Shigure --from-version v2 --version v3 `
  --review-csv $csv --review-json $json --dry-run

# After inspecting the dry-run, run the same command without --dry-run.
```

If the downloaded filenames differ, substitute the actual paths. The HTML
does **not** write to v2 or automatically publish v3. `metadata revise`
creates an immutable metadata-only v3 with zero copied WAVs.
`keep + self + ok` is needed for `reviewed_usable`; editing the manual
category alone records the classification but stays in a caution grouping.

The browser validates metadata references and exact selected file paths; it
does not recompute all 1,167 WAV byte SHA-256 hashes. Run
`metadata verify --profile Ui_Shigure --version v2` before reviewing and
`metadata verify --profile Ui_Shigure --version v3` after publishing.
Browser verification on the user's actual folder selection is still pending.

## Safety gates

- Fail closed on unknown/duplicate source IDs, CSV SHA/AST mismatches, unknown
  category/status values, missing or altered original WAVs, corrupted version manifests,
  changed input fingerprints, existing target versions, and linked source paths.
- No destructive writes outside newly owned `outputs/metadata/` and new
  `outputs/candidates/<profile>/<unused name>` materializations.
- SHA-256 each original before archiving/revising and when materializing.
- Record schema, run ID, parent version, source library and review input hashes.
- Source audio is not immutable *because it has a SHA recorded*; the WAV files and
  completed AST run must be retained or backed up to enable restoration.
- Do not infer that a CSV alone recreates audio; an external audio backup and
  historical version metadata are necessary.

## Windows migration and new revisions

All commands below use the repository-local `.venv` and the existing completed
AST run. They **never** remove the old materialized `v1/` and `v2/`.

```powershell
cd F:\AIProjects\LocalVoice
git fetch origin
git switch feature/phase4-acoustic-pilot
git pull --ff-only origin feature/phase4-acoustic-pilot
.\.venv\Scripts\python.exe -m pytest tests/test_style_batch.py tests/test_review_export.py -q

$runId = "ast_batch_20261009_163035451"
$csv = "$env:USERPROFILE\Downloads\localvoice_ast_review_decisions_v1.csv"
Test-Path $csv
```

Archive original AST v1 (dry-run then execute):

```powershell
.\.venv\Scripts\python.exe -m localvoice style-batch metadata archive `
  --run-id $runId --version v1 --dry-run
.\.venv\Scripts\python.exe -m localvoice style-batch metadata archive `
  --run-id $runId --version v1
```

Archive reviewed v2, **using the exact original CSV** (its SHA must match
`v2/summary.json`). Supply `--review-json <path>` if the saved HTML JSON
is available. It is optional; the v2 CSV + full manifest are sufficient to
preserve recorded decisions.

```powershell
.\.venv\Scripts\python.exe -m localvoice style-batch metadata archive `
  --run-id $runId --version v2 --review-csv $csv --dry-run
.\.venv\Scripts\python.exe -m localvoice style-batch metadata archive `
  --run-id $runId --version v2 --review-csv $csv
```

The CSV and optional JSON are copied under
`outputs/metadata/Ui_Shigure/versions/v2/review/`; a dedicated
`outputs/metadata/Ui_Shigure/inbox/` is also created so later downloaded
CSV/JSON files can be gathered in one place.

Verify the history and source WAVs:

```powershell
.\.venv\Scripts\python.exe -m localvoice style-batch metadata verify --profile Ui_Shigure --version v1
.\.venv\Scripts\python.exe -m localvoice style-batch metadata verify --profile Ui_Shigure --version v2

Get-ChildItem ".\outputs\metadata\Ui_Shigure" -Recurse -Filter *.wav
```

The final command should return **no WAV files** under `outputs/metadata/`.
Archived versions must each show `wav_referenced=1167` and zero WAV copies
created. v1/v2 original directories stay intact until separately retired.

For **new** HTML review results, first reload earlier saved HTML JSON to
keep previous decisions, export a *cumulative* CSV, and place the CSV/JSON in
`outputs/metadata/Ui_Shigure/inbox/`. Then:

```powershell
$newCsv = ".\outputs\metadata\Ui_Shigure\inbox\localvoice_ast_review_decisions_v2.csv"
.\.venv\Scripts\python.exe -m localvoice style-batch metadata revise `
  --profile Ui_Shigure --from-version v2 --version v3 `
  --review-csv $newCsv --dry-run
.\.venv\Scripts\python.exe -m localvoice style-batch metadata revise `
  --profile Ui_Shigure --from-version v2 --version v3 `
  --review-csv $newCsv
```

If an accompanying HTML JSON exists, add `--review-json <path>` to the
`archive` or `revise` commands. Each version records exact raw CSV/JSON bytes
and their SHA-256. The v3 command **does not copy audio**. If you want to
browse old or new categories in Explorer, regenerate a separate folder:

```powershell
.\.venv\Scripts\python.exe -m localvoice style-batch metadata materialize `
  --profile Ui_Shigure --version v1 --output-name restored-v1 --dry-run
.\.venv\Scripts\python.exe -m localvoice style-batch metadata materialize `
  --profile Ui_Shigure --version v1 --output-name restored-v1
```

The same process works for `v2` or `v3` with a **new unused output name**.
This copies WAVs into `outputs/candidates/Ui_Shigure/restored-v1/` only on
explicit request. Interrupted materialization can continue with `--resume`
using the same input parameters. Nothing overwrites another completed folder.

**Storage note:** Existing v1/v2 WAV folders consume disk until manually
retired. This migration does not reclaim disk immediately. Do not delete them
until both metadata archives verify, a restoration has succeeded, and original
audio plus metadata have appropriate backups. The original audio remains
necessary for future restoration; metadata/CSV alone cannot recreate bytes.
