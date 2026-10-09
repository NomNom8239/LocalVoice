# LocalVoice

配信アーカイブや既存音声から、対象話者の音声を抽出して **RVC 学習用データセット**まで整形するローカルパイプラインです。

このリポジトリは音声取得・人声分離・話者分離・話者照合・reference bank・RVC dataset 生成を担当します。RVC の学習・推論本体は別リポジトリ `NomNom8239/RVC-WebUI` で管理します。

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


## Irodori-TTS LoRA dataset preparation

Use the repository-owned, read-only dataset workflow instead of unpacking separate ZIP tools. See [docs/irodori_dataset.md](docs/irodori_dataset.md). Existing review data is reused; ASR runs in a separate environment.

## Scripts and LV-03 diagnostic (no Codex)

Executable scripts are centralized under [scripts/](scripts/README.md).
Do not create a parallel top-level scripts location or move existing
Irodori/RVC entrypoints without updating the dispatcher and tests.

To run the standalone LV-03 recovery after pulling the work branch:

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv03_no_codex.py
~~~

This uses existing local Irodori dependencies and writes only fresh attempt folders.
See [scripts/README.md](scripts/README.md) for grouping, diagnostics and outputs.
## LV-04 official LoRA baseline (Codex-free)

After LV-03's DACVAE result is confirmed, see the [script catalog](scripts/README.md)
for a one-command preflight and an explicit official training run.

~~~powershell
git pull --ff-only origin feature/irodori-lora-dataset-cli
.\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py
# After preflight PASS, opt in to local training:
.\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py --run
~~~

This only uses the eight LV-02 approved training sources (7 training / 1
internal validation); the three independent evaluations remain held out.
The official trainer runs in Irodori-TTS/.venv and saves outputs into a fresh
lv04_lora_NNN directory. A non-explicit inference smoke test follows a
successful adapter build. This is *not* a claim of voice quality or suitability
for any particular vocal style, which requires later evaluation.

## Git policy

Git にはコードと設定だけを保存します。音声・モデル・キャッシュ・学習成果物はローカル管理です。

GitHub Actions は使用せず、検証はローカル環境で行います。

## Single-command expressive LoRA from existing WAV files

Use only voice data you own or have permission to process. Input is the existing
LocalVoice/data/training_audio/Ui_Shigure/audio directory and existing LV-02 tables.

Run from the LocalVoice root after git pull:

~~~powershell
.\.venv\Scripts\python.exe .\scripts\irodori_train_direct.py --run
~~~

The run automatically reuses curated training voice data (up to 320 additional
normal clips, distributed by source video), frozen original 8 clips, and existing
human-labelled tokenizer-checked nonverbal breath/groan pilot data. The independent
3 evaluation examples and their source video stay excluded from training.
Auto-generated ASR text is provisional, not independently verified.

One fresh direct_lora_NNN output directory holds the checked source CSV,
official DACVAE manifest, weighted experimental nonverbal manifest,
LoRA adapter, training logs, result.json, and matched Base vs LoRA WAVs
for normal Japanese speech plus the present breath/groan classes.

The official Irodori training is configured with duration_predictor frozen;
when nonverbal latents are repeated, potentially leaking internal random
validation is disabled. This is a technical trial, not proof of similarity
or NSFW quality. It does not support absent classes without real labeled data.
No Codex, new downloads, full ASR rerun, or 48-item review loop is required.

LocalVoice .venv drives the workflow; Irodori-TTS/.venv runs official
prepare_manifest.py, train.py, and infer.py. Existing artifacts are not overwritten.
