# Phase 3 — 12音声ASR Pilot実行手順（Windows）

状態：実装済み／**実GPU未検証**。Phase 3完了ではない。公式IrodoriはTTSであり、これはLocalVoiceの独立したASR比較。旧LoRAやIrodoriクローンの変更は不要。

## 公式リファレンス

- [Irodori v4-Small：Text/Caption/Reference](https://github.com/Aratako/Irodori-TTS/blob/main/README.md)
- [Irodori Emoji Palette](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/gradio_emoji_palette.py)
- [Kotoba-Whisper v2.0の公式モデルカード](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)
- [Whisper large-v3の公式モデルカード](https://huggingface.co/openai/whisper-large-v3)

ASRは両モデルともTransformersの同一pipeline API、language=ja・task=transcribe・区間timestamp=trueで実施。Emojiによる音声分類はしない。

## 0. GitとPython環境

PowerShell:

~~~powershell
cd F:\AIProjects\LocalVoice
git fetch origin
git switch feature/phase3-asr-pilot

# LocalVoice既存.venvで検証。Irodori-TTS/.venvには触らない。
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip show torch transformers accelerate soundfile scipy numpy
.\.venv\Scripts\python.exe -m pytest tests/test_asr_pilot.py -q
~~~

`transformers` (4.39以上、5未満)、`accelerate`が既存環境になければ、依存整合を確認してから追加する。既存pyannoteを壊す依存変更が予想されればSTOPし、専用`.venv-asr`の採用を別途判断する。

~~~powershell
.\.venv\Scripts\python.exe -m pip install "transformers>=4.39,<5" "accelerate>=0.27"
.\.venv\Scripts\python.exe -m pip check
~~~

`seal`/`validate`にはsoundfileが必要。モデル重みは`run`で初めてダウンロードする。

## 1. 人間が選んだ12件のWAVだけを使用

250ms〜30秒、1〜2ch、8kHz以上のWAVを実際に聴いて選ぶ。`data/`の原本を変更・移動・切断しない。長過ぎる音声は別の短いクリップを選択。内訳は通常会話3、囁き2、発話＋笑い等2、非言語（笑い/喘ぎ/息切れ等）3、無音/ノイズ1、他人声/複数話者1。

`category`は**Pilot選定のための人手区分**であり、ASRのスタイル判定ではない。`identity_review`は既に人手承認された場合だけ`approved_self`、未確認は`candidate`/`unknown`、他人声の負例は`rejected_other`。

~~~powershell
New-Item -ItemType Directory .\work -Force | Out-Null
Copy-Item .\docs\asr_pilot_selection.example.csv .\work\asr_pilot_selection.csv
notepad .\work\asr_pilot_selection.csv
~~~

`REPLACE_WITH_WAV_*.wav`を各音声の**フルパス**へ置換。CSVのsample_idとcategoryを維持し、必要ならprofileを変更。パスにコンマがあれば二重引用符で囲む。

## 2. 入力のSHA固定（まだモデルは起動しない）

~~~powershell
.\.venv\Scripts\python.exe -m localvoice asr-pilot seal `
  --csv .\work\asr_pilot_selection.csv `
  --manifest .\work\asr_pilot_selection.sealed.jsonl

.\.venv\Scripts\python.exe -m localvoice asr-pilot validate `
  --manifest .\work\asr_pilot_selection.sealed.jsonl
~~~

PowerShellでは行末の`はバッククォート（継続記号）。`VALID: 12 source WAVs, all SHA-256 verified`が期待結果。誤パス、分類の数、空/長いWAV、重複、SHA不一致ではSTOP。seal済みManifestは編集しない。変更時は別の新規manifest名でsealし直す。

## 3. 同じ入力を1モデルずつ推論

**seal / validate成功後のみ実行。** 大きなモデルの初回ダウンロードが必要。RTX 5060 Ti上のCUDA動作はこの段階で初めて確認する。

~~~powershell
.\.venv\Scripts\python.exe -m localvoice asr-pilot run `
  --manifest .\work\asr_pilot_selection.sealed.jsonl `
  --model kotoba --run-id asr_kotoba_pilot_001

.\.venv\Scripts\python.exe -m localvoice asr-pilot run `
  --manifest .\work\asr_pilot_selection.sealed.jsonl `
  --model whisper --run-id asr_whisper_pilot_001
~~~

別々のrunに保存。再実行は`...002`など**新しいrun-id**。CUDA未使用時は黙ってCPUへ切り替えず初期化失敗証跡を残す。CPUを診断するときのみ`--device cpu`を明示する。

## 4. 出力と受入確認

~~~text
work/
├─ asr_pilot_selection.csv
├─ asr_pilot_selection.sealed.jsonl
├─ asr_kotoba_pilot_001/
│  ├─ transcription.jsonl
│  ├─ failures.jsonl
│  └─ run_manifest.json
└─ asr_whisper_pilot_001/
   ├─ transcription.jsonl
   ├─ failures.jsonl
   └─ run_manifest.json
~~~

`run_manifest.json`の`status=completed`、`completed=12`、`failed=0`で技術的なPilot処理成功（音声認識品質は未承認）。`transcription.jsonl`は各WAV全体の証跡行を残す。元WAVの0〜末尾は、発話が本当に存在した区間とは限らない。ASRが喘ぎ/笑いに誤テキストを出しても`speech_candidate`扱いに留め、正解ラベルにしない。

Pilot12件の結果だけでモデル採用や一括ASR実行を決定しない。次の固定48件比較は別ゲート。音声そのものや生のログはGitへcommitしない。

## スコープ

- 既存`data/`に書込み禁止。公式`Irodori-TTS/`にも独自コードを置かない。
- 旧ASR・LoRA・Emoji分類を復活させない。
- テストはモデルをMock化したオフラインの安全性検証。実モデル、Windows、GPU・精度は**未検証**。
- 最終的な通常会話/笑い/喘ぎ/息切れの日本語カテゴリフォルダはPhase 5。
