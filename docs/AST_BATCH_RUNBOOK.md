# AST 1,167 WAV 一括分類 → 暫定カテゴリ別ライブラリ

**実装状況（2026-10-09）：** コードとオフラインテストを実装済み。Windows / RTX 5060 Ti での実モデルバッチ実行、全1,167件の実データ処理はまだ未検証。

## 目的と対象

既存の data/training_audio/Ui_Shigure/audio/ にある1,167件のWAVを、AST単独で判別できる範囲から日本語カテゴリ別フォルダにコピーする。精密な囁き・喘ぎ・吐息の識別、CLAP比較、ASR48件比較、全件人手承認を先行条件にしない。

**注意：出力は未確認のAST分類候補であり、話者本人性・音声品質・カテゴリの承認を意味しない。** 使う前に必要に応じて試聴する。承認済み出力 outputs/datasets/ と混同しない。

成果物は以下。

~~~text
data/training_audio/Ui_Shigure/audio/             # 元データ：read-only
work/<run-id>/
  input_manifest.jsonl                           # 固定インベントリ＋SHA
  run_manifest.json                              # 状態と再開情報
  style_predictions.jsonl                        # 1WAVにつき分類/不明/失敗
  failures.jsonl                                 # 失敗詳細
outputs/candidates/Ui_Shigure/v1/
  README.txt / index.csv / manifest.jsonl / summary.json
  audio/
    01_通常会話/
    02_囁き/
    03_吐息/
    04_笑い/
    05_喘ぎ_うめき/
    06_息切れ_荒い呼吸/
    07_息をのむ/
    08_泣き声/
    09_悲鳴/
    11_鼻歌/
    99_未分類_要確認/
~~~

空フォルダは作らない。ASTが出せないカテゴリは無理に創作しない。混在した短いクリップも現段階ではWAV全体を単位とする。

## STEP 1：Windowsテスト

PowerShell で実行する。

~~~powershell
cd F:\AIProjects\LocalVoice
git fetch origin
git switch feature/phase4-acoustic-pilot
git pull --ff-only origin feature/phase4-acoustic-pilot
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_asr_pilot.py tests/test_style_pilot.py tests/test_style_batch.py -q
~~~

既存の LocalVoice/.venv を使用する。音声原本・Irodori-TTS/.venv を変更しない。pytestは人工WAVとFakeモデルのテスト。エラー時はログを確認し、依存ライブラリを一括アップグレードしない。

## STEP 2：入力の確認（推論・コピーなし）

~~~powershell
.\.venv\Scripts\python.exe -m localvoice style-batch inventory --profile Ui_Shigure --expected-count 1167
~~~

期待：INVENTORY: profile=Ui_Shigure count=1167 bytes=... sha=... (read-only)。件数・原本SHA・参照パスを確認。件数が異なる場合は処理せず原因を調べる。

## STEP 3：最初の12件でASTを実機動作確認

~~~powershell
$runId = "ast_batch_$(Get-Date -Format 'yyyyMMdd_HHmmssfff')"
.\.venv\Scripts\python.exe -m localvoice style-batch run --profile Ui_Shigure --expected-count 1167 --run-id $runId --device cuda --max-new 12

Get-Content ".\work\$runId\run_manifest.json" -Raw -Encoding UTF8 | ConvertFrom-Json | Select-Object status, processed, remaining, initialization_error | Format-List
Get-Content ".\work\$runId\style_predictions.jsonl" -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json } | Select-Object source_id, category_dir, category_score, status | Format-Table -AutoSize
~~~

**部分実行は終了コード3、partial: 12/1167 が正常。** 本処理は1WAVごとに永続化される。ASTの初回モデルダウンロードとCUDAの動作確認が完了したら、そのrunIdを使って残件を一括処理する。

~~~powershell
.\.venv\Scripts\python.exe -m localvoice style-batch run --profile Ui_Shigure --expected-count 1167 --run-id $runId --device cuda --resume
~~~

中断した場合は同じ --resume を再実行する。すでに結果のあるWAVは再処理しない。入力WAVや設定が途中で変化した場合は安全のため中断。ステータスが completed または completed_with_errors になれば、1,167件に対応する分類/保留/失敗証跡が揃っている。

## STEP 4：件数確認と暫定フォルダの出力

~~~powershell
.\.venv\Scripts\python.exe -m localvoice style-batch summary --run-id $runId
.\.venv\Scripts\python.exe -m localvoice style-batch export --run-id $runId --version v1

$library = ".\outputs\candidates\Ui_Shigure\v1"
Get-ChildItem "$library\audio" -Directory | Select-Object Name
(Get-ChildItem "$library\audio" -File -Filter *.wav -Recurse).Count
Get-Content "$library\README.txt" -Encoding UTF8
~~~

期待は total=1167、processed=1167、remaining=0、コピー後のWAV数1,167。元WAVは読み取るだけで、コピー先は独立した outputs/candidates/ になる。出力前に全入力SHA、ディスク容量、既存版衝突を検査する。出力完了後に既存のv1へ上書きしない。

コピー中に中断した場合だけ：

~~~powershell
.\.venv\Scripts\python.exe -m localvoice style-batch export --run-id $runId --version v1 --resume-export
~~~

未完成の一時フォルダの中でコピー済みハッシュを照合し、残りをコピーする。完成済みv1への再出力は禁止（変更するならv2など新規版）。

## AST候補カテゴリの考え方

- ASTのAudioSetスコアから、分類表にある音響カテゴリの最大スコアを利用する。
- 既定の暫定振分けは、カテゴリスコアが0.03以上かつAST全体のTopスコアの10%以上の場合に一番強い対応カテゴリを選ぶ。それ以外は未分類。
- この基準は**校正された確率・品質保証ではなく暫定の整理用ヒューリスティック**。必要なら次のrunで変更する。スコアと元のAudioSetラベルを証跡に残す。
- 完全無音・弱い非対応ラベル・デコード失敗・推論例外も入力件数から消さず、未分類/要確認へコピーする。
- 個別の曖昧な声を深追いせず、まず1,167件全体を整理する。

## 範囲外・今後の改善

このコマンドは本人性や音質QC、人手承認、非言語区間の高精度な切出し、正式な outputs/datasets/ へのプロモートは行わない。認定済みライブラリは後続LV-R06/R05で扱う。Irodoriへの参照WAV指定は従来どおり手動。