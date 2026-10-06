# LocalVoice

配信アーカイブや既存音声から、対象話者の音声を抽出して **RVC 学習用データセット**まで整形するローカルパイプラインです。

このリポジトリの責務は、音声取得・人声分離・話者分離・話者照合・reference bank・RVC dataset 生成までです。RVC の学習・推論本体は別リポジトリ `NomNom8239/RVC-WebUI` で管理します。

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
confirmed target-speaker audio
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
│  └─ build_rvc_dataset.py
└─ data/                  # generated/local data; Git 管理外
   ├─ source/
   ├─ wav_master/
   ├─ separated/
   ├─ diarization/
   ├─ reference_bank/
   ├─ rvc_dataset/
   └─ runs/
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

このリポジトリ用の仮想環境と RVC-WebUI 用の仮想環境は分離してください。

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

CUDA 対応 PyTorch / ONNX Runtime は GPU と CUDA 環境に合わせて別途導入してください。現在の検証環境では CUDA 13.0 系を利用しています。

Hugging Face の gated model を使うため、利用条件を承認したうえでログインします。

```powershell
hf auth login
```

主に使用するモデル:

- `pyannote/speaker-diarization-community-1`
- community-1 内蔵 speaker embedding
- `model_bs_roformer_ep_317_sdr_12.9755.ckpt`（audio-separator）

## Configuration

共通設定は `config.toml` にまとめています。

話者固有の reference bank は次の形で管理します。

```text
data/reference_bank/<profile>/
├─ audio/
├─ self_reference_bank.npz
├─ self_reference_bank.json
└─ thresholds.json
```

`thresholds.json` が存在する場合、`localvoice.py` はその profile 固有の threshold を優先して使用します。

## Usage

### 1. 話者分離

```powershell
python .\scripts\diarize.py `
  --input ".\path\to\Vocals.wav" `
  --output ".\data\diarization\archive_001"
```

### 2. Reference bank 作成

対象話者であることを確認済みの WAV を用意して実行します。

```powershell
python .\scripts\build_reference_bank.py `
  --profile Toto_Kogara `
  --source ".\data\reference_bank\Toto_Kogara\audio"
```

### 3. Threshold calibration

対象話者ではないことを確認済みの音声を negative sample として使用します。

```powershell
python .\scripts\calibrate_threshold.py `
  --profile Toto_Kogara `
  --negative-source ".\data\diarization\negative"
```

結果は以下へ保存されます。

```text
data/reference_bank/Toto_Kogara/thresholds.json
```

### 4. YouTube から対象話者を自動抽出

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

分類結果は profile ごとの run directory に保存されます。

```text
data/runs/<profile>/<run-name>/
├─ my_voice/
├─ review/
├─ review_unscored/
├─ rejected/
└─ classification.tsv
```

reference bank は自動更新しません。誤判定の自己増殖を避けるため、`my_voice/` を確認してから reference audio に追加します。

### 5. RVC dataset 作成

`--source` と `--profile` は必須です。

```powershell
python .\scripts\build_rvc_dataset.py `
  --profile Toto_Kogara `
  --source ".\data\reference_bank\Toto_Kogara\audio"
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
- SELF 自動判定結果を reference bank へ無確認で追加しない。
- RVC dataset の `ACCEPT` は機械的 QC を通過したという意味であり、話者・分離品質の最終確認を代替しない。
- 他人の声を学習・変換する場合は、利用許諾のある音声だけを使用する。

## Git policy

Git にはコードと設定だけを保存します。音声・モデル・キャッシュ・学習成果物はローカル管理です。

GitHub Actions は使用せず、検証はローカル環境で行います。
