# LocalVoice — WAVを共通化した分類・レビュー履歴管理

Status: implementing. This document defines the metadata-first successor to
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
