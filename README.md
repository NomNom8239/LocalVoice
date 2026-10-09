# LocalVoice

配信アーカイブ・既存WAVから対象話者の音声を抽出・本人候補として保存し、**AST音響分類 → 必要な音声だけ人手レビュー → 分類・レビュー履歴をバージョン管理**するローカルパイプラインです。既存の音源分離・話者照合・RVCデータセット生成処理も維持しています。

公式Irodori-TTSのZero-shot VoiceDesignは、独立した `Irodori-TTS/` クローンで動作させます。LocalVoice独自の機能は公式クローンへ追加せず、ユーザーが選んだ参照WAVを手動で指定します。

**初回話者のReference Bank自動セットアップを実装。** Windows全体テスト43件は一度PASSしたものの、実音声受入で無音・混話を検出し、採用方式を個別プレビュー承認へ変更したため、**修正版の回帰テスト・実音声E2Eは未完了**です。

**最初に読む資料：** [ディレクトリ責務・安全規則](docs/PROJECT_LAYOUT.md) ／ [AST一括分類手順](docs/AST_BATCH_RUNBOOK.md) ／ [分類・レビュー履歴の運用](docs/VERSIONED_AUDIO_METADATA.md)

## 現在の到達点（2026-10-09）

**`Ui_Shigure` の既存1,167 WAVについて、分類フェーズはv2を現行版として一旦完了**しています。v2はあくまで**暫定候補の分類ライブラリ**であり、全件が本人性・音質・学習適格性の最終承認済みという意味ではありません。

| 項目 | 確認済みの結果 |
| --- | --- |
| AST一括分類 | 1,167/1,167件、unknown 5、推論エラー0 |
| AST実行ID | `ast_batch_20261009_163035451` |
| 人手レビュー | 重点105件（レビュー適合候補100、要確認5）、未レビュー1,062 |
| 現在の分類版 | **v2**（`outputs/candidates/Ui_Shigure/v2/`） |
| 履歴保存 | `outputs/metadata/Ui_Shigure/versions/v1/`、`v2/` にメタデータのみ保存 |
| 履歴整合性 | v1・v2とも `metadata verify` が `VERIFIED_METADATA_AND_SOURCE_WAV`、参照1,167・新規WAVコピー0 |
| Windows回帰テスト | `tests/test_style_batch.py` + `tests/test_review_export.py`：**22 passed** |
| 共通レビューHTML | 実装・模擬JavaScript検証済み。**実Chrome/Edge受入は未確認** |

- **Done：** AST一括候補分類、v1暫定候補出力、重点105件レビュー反映v2、メタデータ履歴v1/v2の保存・SHA照合。
- **受入待ち：** 共通HTMLでの実フォルダ読み込み・音声再生・CSV/JSON往復・次版dry-run。別profileの初回v1・再分類v2以降は模擬データで確認済み。
- **後続・別工程：** 独立した全件QC/本人性と正式な `outputs/datasets/` 採用、ASR48件比較、新規配信アーカイブの継続取り込みE2E。現在の分類v2の完了条件にはしません。

## AST分類から手動レビューまで（全profile共通）

音響分類のPython CLIは `python -m localvoice style-batch` です。**HTMLは分類済み候補を試聴・修正するツールであり、AST自動推論自体は実行しません。**

1. **初回の自動分類：** 対象profileの元WAVを確認し、`style-batch inventory` → `run` → `summary`。初回のAST候補ライブラリが必要なら `export --version v1` で `outputs/candidates/<profile>/v1/` に出力（初回だけのコピー）。
2. **初回の手動レビュー：** リポジトリ直下の [`localvoice_ast_candidate_reviewer.html`](localvoice_ast_candidate_reviewer.html) をChrome/Edgeで開き、**v1候補フォルダ**を選ぶ。試聴・分類修正を行い、**累積レビューCSVと保存用JSON**をダウンロードする。
3. **履歴として初回版を保存：** `style-batch metadata archive --run-id <run> --version v1` → `metadata verify`。初回レビューからv2の**メタデータ版**を作成する場合は `metadata revise --profile <profile> --from-version v1 --version v2 --review-csv <CSV>` を使用。これにはv1履歴保存が必要です。
4. **2回目以降のレビュー：** HTMLに最新のv2、v3…を読み込み、以前のレビュー判断を引き継いで必要な音声だけ再分類する。累積CSV/JSONを `outputs/metadata/<profile>/inbox/` に保存し、`metadata revise --from-version <現行版> --version <新しい版> --review-csv <CSV>` で次の不変メタデータ版を作成する。
5. **必要な場合だけWAVフォルダを再生成：** `metadata materialize` は元WAVから実際にコピーを作成するコマンドです。HTMLでは**メタデータ版フォルダ＋原本WAVフォルダ**を読み込めるため、通常の試聴や再分類に追加コピーは不要です。

### ブラウザレビューのフォルダ選択

- **方法A（候補WAVフォルダがある）：** `outputs/candidates/<profile>/<version>/` を選択。直下の `manifest.jsonl` と `summary.json`、`audio/` を利用。v1初回レビュー、v2以降の既存レビュー復元に対応。
- **方法B（候補WAVのコピーがない）：** `outputs/metadata/<profile>/versions/<version>/` と `data/training_audio/<profile>/audio/` を別々に指定。WAVを複製せず試聴する。
- CSV/JSON読込みではsource ID、記録済みSHA、元AST分類を検証し、既存レビューが欠落する読み込みは拒否します。HTMLは音声の**全件バイトSHA照合を実行しない**ため、保全確認にはCLIの `metadata verify` を使用してください。

**重要：** `keep + identity=self + quality=ok` は「レビュー適合**候補**」であって正式な学習素材承認ではありません。曖昧な音声は `96_要確認_使用不可`、保留は `97_保留`、除外は `98_除外` に区分し、未レビューも勝手に承認しません。

### 分類・レビュー履歴の保存先

```text
data/training_audio/<profile>/audio/          # 原本WAV（保持・読み取り専用）
work/<AST-run-id>/                           # AST入力・予測・SHA等の証跡
outputs/candidates/<profile>/v1,v2.../       # 必要な場合だけ保持するWAV付き閲覧用候補
outputs/metadata/<profile>/
  inbox/                                    # ブラウザから保存したCSV・JSON
  versions/v1,v2,v3.../                     # メタデータだけの不変分類・レビュー履歴
    manifest.jsonl / index.csv / summary.json / receipt.json
    review/                                 # レビュー版のCSVと任意のJSON
outputs/datasets/<profile>/<version>/        # 別のQC/承認ゲートを経た正式採用専用
```

既存v1/v2候補WAVフォルダは**自動削除されません**。実際の復元コピーの実証は未完了です。整理前には元WAV・AST run証跡・メタデータのバックアップを確認してください。**CSV/JSONだけではWAV自体を復元できません**。版の上書きはせず、v3以降に更新を積み重ねます。

### 既存v1→v2コピー方式について

初回レビューCSVをv1に適用してWAV付きv2を作る `style-batch apply-review` は、**既存リリースの再現・互換用途**として残っています。手順は [旧v1→v2レビューCSV反映](docs/REVIEW_CSV_APPLY_RUNBOOK.md)。**今後のv3以降の履歴更新には使わず、WAVを追加コピーしない `metadata revise` を使ってください。**

詳細：[履歴・再分類のWindows手順](docs/VERSIONED_AUDIO_METADATA.md) ／ [AST分類バッチ手順](docs/AST_BATCH_RUNBOOK.md)

## ASR補助機能（後続保留）

[ASR設計](docs/PHASE3_JAPANESE_ASR.md) ／ [12件Pilot手順](docs/ASR_PILOT_RUNBOOK.md) ／ [音響Pilot手順](docs/STYLE_PILOT_RUNBOOK.md)

Kotoba-Whisper v2.0とWhisper large-v3の12件CUDA Pilotは両方処理完了。人工無音でのハルシネーションや非言語での反復があり、**48件の本比較・正式モデル採用は保留**。音響スタイル分類の必須先行ゲートではありません。

## 既存の配信取得・RVCパイプライン

```text
YouTube / WAV
    ↓
yt-dlp / FFmpeg
    ↓
audio-separator
    ↓
Vocals
    ↓
pyannote.audio diarization
    ↓
speaker embedding / reference bank
    ↓
SELF / REVIEW / OTHER classification
    ↓
my_voice + manually approved review
    ↓
training_audio accumulation / deduplication
    ↓
RVC dataset QC
    ↓
RVC-WebUI
```

## Repository layout

```text
LocalVoice/
├─ README.md
├─ localvoice_ast_candidate_reviewer.html   # 共通レビューUI（Git管理）
├─ config.toml
├─ scripts/                                 # 既存の取得・話者照合・RVC用コード
│  ├─ localvoice.py
│  ├─ diarize.py
│  ├─ build_reference_bank.py
│  ├─ calibrate_threshold.py
│  ├─ review_training_audio.py
│  ├─ collect_training_audio.py
│  └─ build_rvc_dataset.py
├─ src/localvoice/
│  ├─ style/                                # AST分類・レビューCSV反映・メタデータ版管理
│  └─ transcription/                        # ASR Pilot
├─ tests/                                   # 回帰テスト
├─ docs/                                    # 作業手順・責務の正本
├─ work/                                    # AST等の実行証跡（Git管理外）
├─ outputs/
│  ├─ candidates/                          # 未承認の候補WAV・閲覧用コピー
│  ├─ metadata/                            # 版ごとのCSV/JSON履歴（WAV複製なし）
│  └─ datasets/                            # 正式QC承認済み出力専用（後続）
├─ data/                                    # 原音声・legacy取得成果物（Git管理外）
│  ├─ source/
│  ├─ wav_master/
│  ├─ diarization/
│  ├─ reference_bank/
│  ├─ runs/
│  ├─ training_audio/
│  └─ rvc_dataset/
└─ Irodori-TTS/                             # 公式独立クローン（Git管理外）
```

`outputs/datasets/` など将来用の場所は、この構成を示すための**責務**であり、未作成でも問題ありません。元音声・モデル・`work/`・`outputs/`・仮想環境はGitへ登録しません。原本は新分類処理から読み取り専用で扱います。

## Environment

現在確認済みのローカル環境:

- Windows 11
- Python 3.12.10
- NVIDIA GPU / CUDA
- PyTorch 2.14.1+cu130
- pyannote.audio 4.0.7
- audio-separator 0.47.0
- FFmpeg 9.x
- yt-dlp

LocalVoice と RVC-WebUI の仮想環境は分離してください。

```text
F:\AIProjects\LocalVoice\.venv
F:\AIProjects\RVC-WebUI\.venv
```

### Base setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
python -m pip install "audio-separator[gpu]" audioread pyannote.audio soundfile scipy
```

CUDA 対応 PyTorch / ONNX Runtime は GPU と CUDA 環境に合わせて導入してください。現在の検証環境では CUDA 13.0 系を利用しています。

Hugging Face の gated model を使うため、利用条件を承認したうえでログインします。

```powershell
hf auth login
```

主に使用するモデル:

- `pyannote/speaker-diarization-community-1`
- community-1 内蔵 speaker embedding
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt`（audio-separator）

## Profile layout

```text
data/reference_bank/<profile>/
├─ audio/
├─ self_reference_bank.npz
├─ self_reference_bank.json
└─ thresholds.json
```

`thresholds.json` が存在する場合、`localvoice.py` は profile 固有値を優先します。

## Usage

### 話者分離

```powershell
python .\scripts\diarize.py `
  --input ".\path\to\Vocals.wav" `
  --output ".\data\diarization\archive_001"
```

### 新しい話者の初回収集（Reference Bankなしで開始）

**推奨経路：** 新規profileはReference Bankの手動準備不要。通常の `localvoice.py --url` から直接、配信取得 → Vocals抽出 → pyannote話者分離 → 代表WAV試聴 → **確認済みの個別WAVだけ**でBank作成 → 本人声分類まで進みます。

~~~powershell
cd F:\AIProjects\LocalVoice
.\.venv\Scripts\python.exe .\scripts\localvoice.py --profile New_Speaker --url "<配信アーカイブURL>"
~~~

取得後、`bootstrap/previews/` に話者ラベルごと最大3件ずつ短いWAVを出力。ターミナルで試聴できます。

~~~text
p SPEAKER_00                        # 話者00のプレビューすべてを再生
p SPEAKER_02_02                     # 話者02の2番目だけ再生
SPEAKER_00,SPEAKER_01              # 00と01の「表示済みプレビューだけ」を採用
SPEAKER_02_02,SPEAKER_02_03        # 02の1番目が無音なら、2・3番目だけ採用
SPEAKER_00,SPEAKER_01,SPEAKER_02_02,SPEAKER_02_03  # 組合せも可
q                                   # Bank生成せず安全に中断
~~~

ラベル単位で選択しても、**Bankに入れるのは表示・試聴できた最大3件だけ**で、同じラベルの未試聴の全区間は採用しません。個別クリップIDなら**その1件だけ**を採用します。最低3件の異なる確認済みWAVが必要。指定後に具体的な採用ファイルパスがすべて表示され、すべて本人だけの声であると確認できた場合のみ `YES` を入力します。無音、他人声、混話のWAVは選択しないでください。代表音声の無音・他人声混入が見つかった実機受入を受けて、未試聴クリップの自動採用を禁止しました。

**途中で中断した場合：** 生成済みプレビューを再利用するので、アーカイブの再取得やaudio-separatorの再実行は不要です。最初の実行時に指定した`--run-name`または既定の動画IDを確認し、次で再開します。

~~~powershell
.\.venv\Scripts\python.exe .\scripts\localvoice.py --profile New_Speaker --resume-bootstrap "<元のrun-name>"
~~~

この方法は既存 `data/runs/<profile>/<run-name>/bootstrap/previews/` を再利用し、**確認したクリップからReference Bankだけを生成して終了**します。**再ダウンロード・audio-separator・アーカイブ全体のpyannote再実行・SELF/REVIEW/OTHER分類は行いません。** 既存の話者分離結果は中断前に保存されていないため、後続の全編分類を実行するには別途pyannoteの再実行が必要になります。

**全編分類が必要な場合だけ**、`--resume-bootstrap` に `--classify-after-bootstrap` を追加できます。ただし**これを指定すると2時間以上かかる可能性のあるpyannoteを全編再実行**します。元が `--vocals` 入力の場合は `--resume-vocals "<元のVocals.wav>"` も必要です。Bankのみを先に作成した後は既存Bank保護により同じresumeコマンドで全編分類はできません。意図しない長時間処理を避けるため、今回の再開ではオプションを追加しないでください。

**安全上の制約：** 初回Bankの音声はまだ暫定基準であり、すべての分類結果の本人性や最終QCを保証しません。`self_reference_bank.npz`/`self_reference_bank.json`の不整合や既存Bankは上書きしません。`--force`は既存runを削除するため使わず、再開には`--resume-bootstrap`を使用します。別ラベル＝他人と自動でnegativeにはしません。本人以外の確実なWAVがある場合だけ、`calibrate_threshold.py`で別途閾値校正します。

プレビュー: `data/runs/<profile>/<run-name>/bootstrap/previews/`。実際にBankに渡したファイル: `bootstrap/selected_reference_wavs/`。Bank: `data/reference_bank/<profile>/`。この段階では学習用素材の正式承認やAST分類は行わず、従来どおり`review_training_audio.py`→`collect_training_audio.py --dry-run`で結果を確認します。

**代替ルート：** 音声が短すぎたり、試聴候補が十分に得られない場合に限り、`yt-dlp`/`ffmpeg`/`diarize.py`で本人WAVを手動収集し、次節の生成コマンドでBankを作成できます。

### Reference Bankを手動生成する場合（代替・既存Bankは上書き注意）

前節の音声準備が終わってから実行します。既存bankがある場合は上書きになるため、初期登録のために安易に再実行しないでください。

```powershell
python .\scripts\build_reference_bank.py `
  --profile Toto_Kogara `
  --source ".\data\reference_bank\Toto_Kogara\audio"
```

### Threshold calibration

```powershell
python .\scripts\calibrate_threshold.py `
  --profile Toto_Kogara `
  --negative-source ".\data\diarization\negative"
```

### 配信アーカイブURLから対象話者を抽出（yt-dlp）

```powershell
python .\scripts\localvoice.py `
  --profile Toto_Kogara `
  --url "https://www.youtube.com/watch?v=..."
```

**`--url` はYouTube専用ではありません。** コードは `yt-dlp` から媒体の `id` を取得し、`bestaudio/best` でダウンロードするため、TwitchやBilibiliなども **yt-dlpがそのURLを取得できる場合に限り**使用できる構造です。ただし**YouTube以外の実機E2Eは未検証**。認証・配信終了・地域制限・DRM・サイト仕様変更で取得に失敗することがあります。プレイリストの一括処理は `--no-playlist` により対象外です。

取得できるかの事前確認（音声をダウンロードしない）：

```powershell
yt-dlp --no-playlist --skip-download --print "%(extractor_key)s / %(id)s" "https://example.com/your-archive"
```

**現状のURL実装の制約：** 取得時の `id` はサイトを含まないため、異なるサイトで同じ動画IDを持つ場合 `data/source/<id>/` と `data/wav_master/<id>.wav` の既存ファイルを**誤再利用する可能性があります**。`--run-name` で実行フォルダだけを変えても、この問題は解消しません。別サイトの一括収集を本格運用する前に、保存キーへ媒体名/URL由来識別子を含める改修と実機E2E（LV-R07）が必要です。

同じrun名が既存の場合は停止します。**`--force` は既存の `data/runs/<profile>/<run-name>` を削除するので、通常は使用しないでください。** 別アーカイブとして実行する場合は、重複しない `--run-name` を指定できます（ただし上記のソースID衝突対策ではありません）。

ローカルの既存音声WAV（元動画は先にFFmpeg等で音声WAVへ変換）:

```powershell
python .\scripts\localvoice.py `
  --profile Toto_Kogara `
  --wav ".\data\wav_master\example.wav"
```

既に Vocals 抽出済みの場合:

```powershell
python .\scripts\localvoice.py `
  --profile Toto_Kogara `
  --vocals ".\path\to\Vocals.wav"
```

**前提：** Python環境・FFmpeg・audio-separator・pyannote等が必要です（URL取得時はyt-dlpも必要）。**Bankがないprofileでは初回セットアップが自動起動**し、試聴済みの本人クリップだけを採用する操作を求めます。Bankがあるprofileは通常の本人照合へ直行します。出力は話者ごとに分割した候補クリップであり、AST日本語カテゴリ分類は別工程です。

分類結果:

```text
data/runs/<profile>/<run-name>/
├─ separated/
├─ my_voice/
├─ review/
├─ review_unscored/
├─ review_approved/        # 手動採用した通常声
├─ review_emotion/         # 手動採用した感情・極端発声
├─ review_rejected/        # 手動却下
├─ rejected/
├─ review_decisions.tsv
└─ classification.tsv
```

reference bank は自動更新しません。誤判定の自己増殖を避けるため、`my_voice/` を reference bank へ自動投入しないでください。

### 手動レビュー

通常の speaker embedding で確信できない `review/` と、短すぎてスコアを出していない
`review_unscored/` は、対話式レビューで仕分けします。

```powershell
python .\scripts\review_training_audio.py `
  --profile Toto_Kogara
```

キー:

```text
y = 本人の通常声として採用     -> review_approved/
e = 本人の感情・極端発声として採用 -> review_emotion/
n = 却下                     -> review_rejected/
r = 再生し直す
s = 今回は保留
q = 終了
```

元 WAV は削除・移動せず、そのまま保持します。判断結果は各 run の
`review_decisions.tsv` に記録されるので、次回は判断済みクリップを自動で飛ばします。
`ffplay` を自動再生に使用します。自動再生しない場合は `--no-play` を指定できます。

`review_emotion/` は「感情声を機械判定できた」という意味ではありません。
通常話者照合から外れやすい喘ぎ・叫び・笑い・息声などを、本人だと人間が確認したうえで
通常声と分けて保持するためのカテゴリです。

大量の `review_unscored/` から感情声だけを拾う場合は、対象を絞ってレビューできます。

```powershell
python .\scripts\review_training_audio.py `
  --profile <profile> `
  --run <run-name> `
  --kind unscored `
  --min-duration 0.3 `
  --max-duration 2.0 `
  --limit 100 `
  --emotion-only
```

`--emotion-only` では `e` が `review_emotion/` への採用、
`n` が「感情声ではない」として判断済みの記録になります。
`n` のクリップは `review_rejected/` へコピーせず、通常学習素材にも自動追加しません。
`--limit` を使えば100件ずつなどの単位で進められます。

### 新しいアーカイブから素材を追加する時の実行順

**取得と素材の蓄積はコマンドから可能ですが、全工程を1コマンドで連結する実装ではありません。**

1. `localvoice.py --url`（または `--wav`, `--vocals`）で取得→必要ならVocals分離→話者分離→SELF/REVIEW/OTHER判定。**Reference Bankがなければ初回話者選択とBank生成へ自動移行**。各runの `classification.tsv` と `my_voice/` の音源を確認。
2. `review_training_audio.py --profile <profile>` で `review/`、`review_unscored/` を必要な範囲だけ再生し、本人声・極端な声を採用/保留/却下。
3. `collect_training_audio.py --profile <profile> --dry-run` で**追加候補と重複・失敗件数を検査**。問題がなければ `--dry-run` を外して `data/training_audio/<profile>/audio/` へ収集。
4. **必要に応じて別途** ASTの新run分類と、共通HTMLでのレビュー/メタデータ版更新を行う。既存の `Ui_Shigure` v2へ自動合流する仕組みは未実装。新規アーカイブを含むフルE2Eは後続のLV-R07で扱う。

```powershell
$profile = "Toto_Kogara"
python .\scripts\localvoice.py --profile $profile --url "https://www.youtube.com/watch?v=..."
python .\scripts\review_training_audio.py --profile $profile
python .\scripts\collect_training_audio.py --profile $profile --dry-run
# dry-runの追加候補・重複・失敗を確認してから
python .\scripts\collect_training_audio.py --profile $profile
```

**重要：** `collect_training_audio.py` は、話者判定で `SELF` となった `my_voice/` を人手承認なしで収集対象にします。 `review_approved/` と `review_emotion/` は手動採用分だけを収集し、`review/`・`review_unscored/`・`rejected/` は自動採用しません。音声内容で重複排除しますが、話者本人性と録音品質の最終保証はありません。素材蓄積と`outputs/datasets/`への正式承認を混同しないでください。

### 学習素材を蓄積

`collect_training_audio.py` には URL や run 名を渡しません。
`--profile` に対応する `data/runs/<profile>/` 配下の run を自動探索し、各 run の:

- `my_voice/`
- `review_approved/`
- `review_emotion/`

だけを `data/training_audio/<profile>/audio/` へ蓄積します。
`review/` と `review_unscored/` は自動採用しません。
先に `review_training_audio.py` で人間が判断します。

通常実行:

```powershell
python .\scripts\collect_training_audio.py `
  --profile Toto_Kogara
```

先に追加予定だけ確認:

```powershell
python .\scripts\collect_training_audio.py `
  --profile Toto_Kogara `
  --dry-run
```

保存先は既定で profile 名です。必要な場合だけ別名を指定できます。

```powershell
python .\scripts\collect_training_audio.py `
  --profile Toto_Kogara `
  --destination Toto_Kogara_main
```

出力:

```text
data/training_audio/Toto_Kogara/
├─ audio/
├─ manifest.tsv
└─ summary.json
```

同じ音声はデコード後の音声内容から作る fingerprint で重複排除します。
元の `runs/` 内の WAV は移動・削除せず、training_audio へコピーします。

### RVC dataset 作成

`--source` と `--profile` は必須です。

```powershell
python .\scripts\build_rvc_dataset.py `
  --profile Toto_Kogara `
  --source ".\data\training_audio\Toto_Kogara\audio"
```

出力:

```text
data/rvc_dataset/Toto_Kogara/
├─ self/
├─ review/
├─ rejected/
├─ duplicates/
├─ manifest.tsv
└─ summary.json
```

RVC-WebUI には `self/` を dataset path として渡します。

## Safety boundaries

- reference bank には対象話者であることを確認済みの音声だけを入れる。
- SELF 判定結果を reference bank へ無確認で追加しない。
- `collect_training_audio.py` は profile 配下の run を自動探索し、`my_voice/`、`review_approved/`、`review_emotion/` のみを採用する。
- training_audio への追加は元ファイルを削除・移動せずコピーで行い、provenance を `manifest.tsv` に残す。
- RVC dataset の ACCEPT は機械的 QC 通過を意味し、話者・分離品質の最終確認を代替しない。
- 他人の声を学習・変換する場合は、利用許諾のある音声だけを使用する。

## Git policy

Git にはコードと設定だけを保存します。音声・モデル・キャッシュ・学習成果物はローカル管理です。

GitHub Actions は使用せず、検証はローカル環境で行います。
