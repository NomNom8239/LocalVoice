# LocalVoice

配信アーカイブや既存音声から、対象話者の音声を抽出・確認し、データセットを作成するローカルパイプラインです。既存のRVCデータセット生成処理は維持しています。

**Irodori-TTSのZero-shot VoiceDesignは、公式リポジトリを `Irodori-TTS/` に独立してクローンして使用します。** 旧LoRA/旧ASR/旧スタイル分類コードを戻さず、Phase 3以降の文字起こしと分類は新規設計します。

**必読： [ディレクトリ責務・追加ルール](docs/PROJECT_LAYOUT.md)** — `data/` 保全、公式Irodoriへの独自ファイル追加禁止、新規コード・一時出力・成果物の保存先を定義しています。下記のパイプラインとパス説明は、既存の取得・RVC処理についての説明です。

**Phase 3日本語ASRの設計：** [PHASE3_JAPANESE_ASR.md](docs/PHASE3_JAPANESE_ASR.md)。Irodori公式の Text / Caption / Reference の境界に準拠し、ASR・Emoji分類・最終フォルダ書き出しを別Phaseとして扱います。**12音声Pilotの実装は `feature/phase3-asr-pilot` にあり、実機GPU未検証です。** [Windows Pilot実行手順](docs/ASR_PILOT_RUNBOOK.md) を参照。全件ASRの実装やモデル採用はまだ行っていません。

## Pipeline

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
├─ config.toml
├─ scripts/
│  ├─ common.py
│  ├─ localvoice.py
│  ├─ diarize.py
│  ├─ build_reference_bank.py
│  ├─ calibrate_threshold.py
│  ├─ review_training_audio.py
│  ├─ collect_training_audio.py
│  └─ build_rvc_dataset.py
└─ data/                  # generated/local data; Git 管理外
   ├─ source/
   ├─ wav_master/
   ├─ diarization/
   ├─ reference_bank/
   ├─ runs/
   ├─ training_audio/
   └─ rvc_dataset/
```

`data/`、音声ファイル、学習済みモデル、Python 仮想環境は Git に含めません。

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

### Reference bank 作成

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

### YouTube から対象話者を自動抽出

```powershell
python .\scripts\localvoice.py `
  --profile Toto_Kogara `
  --url "https://www.youtube.com/watch?v=..."
```

既存 WAV:

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
