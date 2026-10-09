# AST音声レビューCSV → カテゴリ修正版ライブラリv2

**対象：** 既存の `localvoice_ast_candidate_reviewer.html` から出力した
`localvoice_ast_review_decisions_v1.csv` を、完成済みAST候補ライブラリv1に適用する。

**重要：** CSVの `keep` だけでは本人・品質確認済みとは判定しない。
今回のv2は `outputs/candidates/` に残す**未承認候補**であり、
`outputs/datasets/` には一切出力しない。v1および `data/` は読み取り専用。

## このCSVの実データ（2026-10-09）

- 対象1,167件、CSV記録105件
- `keep=105`、`hold=0`、`reject=0`
- `keep + identity=self + quality=ok`：100件（レビュー適合候補）
- 上記以外の`keep`：5件（要確認。誤って使用しないよう別フォルダ）
- CSVに含まれない1,062件：**未レビュー**（既存AST分類を保持）
- 手動カテゴリは元のASTと区別して原本カテゴリ・修正値の両方を記録する

5件の要確認ID：
`c00402`、`c00408`、`c00409`、`c00989`、`c00998`。

## Windows PowerShellで事前確認

```powershell
cd F:\AIProjects\LocalVoice
git pull --ff-only origin feature/phase4-acoustic-pilot

.\.venv\Scripts\python.exe -m pytest tests/test_style_batch.py tests/test_review_export.py -q

$runId = "ast_batch_20261009_163035451"
$reviewCsv = "$env:USERPROFILE\Downloads\localvoice_ast_review_decisions_v1.csv"
Test-Path $reviewCsv

.\.venv\Scripts\python.exe -m localvoice style-batch apply-review `
  --run-id $runId `
  --review-csv $reviewCsv `
  --source-version v1 `
  --version v2 `
  --dry-run
```

CSVがDownloads以外にある場合は、`$reviewCsv`を実際の保存先に直す。
`Test-Path` が `True` になるまで適用コマンドを進めない。

`--dry-run` は、CSVの列・重複・本人/品質値・カテゴリ名、runとv1の一致、
1,167件の既存WAVのSHAを確認し、出力物は作らない。

実データで期待する集計：

| 項目 | 期待値 |
| --- | ---: |
| `source_count` | 1167 |
| `review_records` | 105 |
| `reviewed_usable` | 100 |
| `needs_attention` | 5 |
| `held` / `rejected` | 0 / 0 |
| `unreviewed` | 1062 |

異なる場合は**v2作成を開始せず**入力・理由を確認する。

## v2の生成（dry-run成功後）

```powershell
.\.venv\Scripts\python.exe -m localvoice style-batch apply-review `
  --run-id $runId `
  --review-csv $reviewCsv `
  --source-version v1 `
  --version v2
```

独立した新規の `outputs/candidates/Ui_Shigure/v2/` を一時フォルダで作り、
1,167件すべてのWAVを**元v1からコピー**してから公開する。

```text
outputs/candidates/Ui_Shigure/
  v1/                          # 既存の不変版
  v2/
    README.txt
    summary.json
    index.csv                   # 全1,167件、元AST/手動分類/出力先/判定状態
    reviewed_usable.csv         # keep + self + ok の100件
    needs_attention.csv         # 要確認・保留・除外
    manifest.jsonl              # 1,167件の原本SHA・監査証跡
    review_source.json          # 反映元CSV SHA・AST run ID
    audio/
      01_通常会話/              # 未レビューを含む分類候補
      02_囁き/                  # 手動修正の候補も入る
      ...                      # カテゴリのあるものだけ
      90_その他/
      96_要確認_使用不可/       # keepでも本人性/音質不適合の5件
      97_保留/                  # 今後holdがあればここへ
      98_除外/                  # 今後rejectがあればここへ
```

**各WAVは1か所にだけコピーされる。** 要確認・保留・除外のWAVも証跡のためv2内に保持するが、
カテゴリから離しており使用対象ではない。判断・手動修正前のカテゴリは
`ast_category` / `effective_category` に残す。
`human_approved` は全件 `false` のまま。
`reviewed_usable` は**正式な学習データへの昇格を許可する最終判定ではない**。

## 出力確認

```powershell
$v2 = ".\outputs\candidates\Ui_Shigure\v2"
(Get-ChildItem "$v2\audio" -Recurse -File -Filter *.wav).Count
Import-Csv "$v2\reviewed_usable.csv" | Measure-Object
Import-Csv "$v2\needs_attention.csv" | Select-Object source_id,review_status,review_quality
Get-Content "$v2\summary.json" -Raw -Encoding UTF8
```

期待はWAV **1,167件**、`reviewed_usable.csv` **100件**、
`needs_attention.csv` **5件**。

### コピー途中で中断した場合だけ

```powershell
.\.venv\Scripts\python.exe -m localvoice style-batch apply-review `
  --run-id $runId `
  --review-csv $reviewCsv `
  --source-version v1 `
  --version v2 `
  --resume-export
```

保存途中の一時フォルダがある場合は、**同一内容のCSVでのみ再開**できる。
既存の完成済みv2には上書きしない。レビュー内容を追加・変更したらv3などの
新バージョンを指定する。元WAV・v1・runの分類証跡を削除しない。

## 実装の境界

- 実装：`src/localvoice/style/review_export.py`（既存`style-batch`内の`apply-review`サブコマンド）
- テスト：`tests/test_review_export.py`
- 書込先：`outputs/candidates/<profile>/<new version>/` のみ
- 範囲外：ASR/AST再推論、全件音声QCの保証、本人性の再判定、
  `outputs/datasets/`への自動昇格、元ファイル削除
