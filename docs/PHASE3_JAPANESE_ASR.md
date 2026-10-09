# Phase 3 — 日本語文字起こし（ASR）新規設計

**Status:** DRAFT — 仕様・候補選定段階。実装未着手。設計の受入ゲートを満たすまで実装完了としない。2026-10-09確認。

## 1. 目的・範囲

既存の配信アーカイブ処理が生成した音声候補について、「聞き取れる日本語の発話」と「ASRで文字にできないが有用かもしれない非言語音声候補」を**区間単位で保存**する。最終目的は、通常会話・囁き・吐息・笑い・喘ぎ・息切れ等を人間が音声フォルダから探し、Irodori VoiceDesignの参照音声として利用できること。

本Phaseは文字起こしと後工程に必要な証跡の生成のみ。**Emoji Palette分類はPhase 4、承認済みWAVのフォルダ出力はPhase 5。** ASR結果から声のスタイルを決めつけない。旧Irodori ASR、スタイル分類、LoRA実験のコード・出力を実装の前提にしない。

### 公式リファレンス（2026-10-09確認）

1. [Irodori-TTS README](https://github.com/Aratako/Irodori-TTS/blob/main/README.md)（参照したREADME blob SHA: `03818b912f4e689ac194e5c1fd618d5a73b27b4d`）。v4.1-Smallは**text、caption、reference audio**を統合し、Emojiによる生成表現の指定をサポート。複数のクリーンな短い参照WAVが推奨される。
2. [VoiceDesign WebUI](https://github.com/Aratako/Irodori-TTS/blob/main/gradio_app_voicedesign.py)（blob SHA: `45e674f31e54b3348290d2e889dbf09f847f734b`）。Emoji Paletteは**Text欄**に絵文字を挿入する。Captionは別欄、参照音声は複数アップロード可。
3. [Emoji Palette実装](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/gradio_emoji_palette.py)（blob SHA: `33fce37be1e042b05fefe8e2b9ab5ac110b91fda`）。🥵喘ぎ＝うめき/唸り、🌬️息切れ＝荒い息遣い/呼吸音等。**Paletteは出力を操作するUIであり、音声分類モデルでも正解データでもない。**
4. [Irodoriの音声学習Manifest生成](https://github.com/Aratako/Irodori-TTS/blob/main/prepare_manifest.py)（blob SHA: `3d9d2ac4ac5b4b42d20fd64262efe37bd686a296`）。学習入力`text`が空ならskip（`empty_text`）。学習ManifestはDACVAE latentへのパスを含み、`caption`と`speaker_id`はoptional。**LocalVoiceのASRログをこの学習Manifestと混同しない。**
5. [公式推論パラメータ](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md)（blob SHA: `d458b2550419a6fb2a2503b26f878259cfce0a9d`）。v4.1-Small参照WAVはidentity conditioning。通常の参照品質を優先し、複数の短いクリーンなクリップの組合せで約30秒までに類似性向上の大半が観測されたと説明（最大120秒）。

**結論:** IrodoriはTTSでありASRではない。ASRモデルは別に選び、結果とPaletteの分類・Captionを別フィールドで扱う。VoiceDesign参照用WAVには学習用`text`非空条件を適用しない。

## 2. 候補ASRと選定ゲート（未決定）

- **評価第一候補:** [Kotoba-Whisper v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)。日本語向けdistilled Whisper。公開したReazonSpeech/JSUT/CommonVoice評価と音声例を確認。音声区間のタイムスタンプはモデル・推論設定の実機結果で検証する。
- **比較対象:** [OpenAI Whisper large-v3](https://github.com/openai/whisper)。別ドメインで精度が逆転するため、Kotobaを無検証採用しない。
- **代替候補:** [Kotoba-Whisper v2.1](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.1)（主な差分は句読点付与等のpost-processing）。可読性の追加機能として必要な場合のみ評価する。モデル依存が増えることに注意。
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)はOpenAI Whisperの高速な推論実装であってIrodoriの機能ではない。Windows NVIDIAで用いる場合、**CUDA 12とcuDNN 9等の実行条件を実機検証してから**バックエンドを決定。GPUで動くと推定して確定しない。

公式ベンチマークは**配信音声や吐息・喘ぎの分類精度を保証しない**。候補は実データで同条件比較し、`CER`、誤挿入・幻覚、非言語音声の保持率、推論失敗、速度・VRAMを記録した後に1つを決定。仮想環境は既存`LocalVoice/.venv`と依存衝突がないか検証し、必要な場合のみ`.venv-asr/`を許可（理由を記録）。

### 評価セット（Git管理外）

`data/` の既存素材**そのものを変更せず**、由来の異なる配信から例を抽出。最低限の構成：
- 通常会話・短い相槌
- 囁き・小声
- 発話と笑い／吐息などの混在
- 笑い・喘ぎ/うめき・息切れ・息をのむ等の非言語主体
- 無音/小音量/音楽・ノイズ
- 他人の声・複数話者／重なり・不明な素材

まず小規模pilotでモデル動作・出力の違いを確認し、人間の手動ラベルを固定した本比較へ進む。採用判定基準・評価サンプル数・許容誤差は**実機比較前に固定**し、都合のよいモデルに合わせて事後変更しない。判定不能の音声は手動レビューに回す。

## 3. 入出力境界

**入力**：承認済み/候補の音声ファイルをユーザー指定manifestに列挙。フォルダ全件の自動探索・無差別全件文字起こしは既定でしない。

入力各行の必須:
- `source_path`: 読み取り専用で存在する元WAV。manifestから基準パスを安全に解決。意図しないパス・リンク・ディレクトリを拒否。
- `source_sha256`: 開始前と終了後に検証し、変更時は処理失敗。
- `source_id`: 入力ファイルを識別する安定ID（衝突しないもの）。
- `profile`: スピーカー対象プロファイル。本人承認状態はあくまで上流の情報でありASRが決めない。
- `identity_review`: `approved_self | candidate | rejected_other | unknown`。 `rejected_other` は本番処理/出力対象から除外（理由は残す）。

**出力**：実行ごとに新規 `work/<run-id>/` のみ書込み。`work/<run-id>/transcription.jsonl` と `run_manifest.json`、`failures.jsonl`を所有する。既存の `data/` は一切変更しない。

出力1行/区間の契約（設計上の必須キー）:
- `schema_version`, `run_id`, `source_id`, `source_path`, `source_sha256`, `profile`, `identity_review`
- `segment_id`, `start_ms`, `end_ms`：元入力WAVに対する0起点の時間境界。ミリ秒単位だが**実際の推定精度まで保証しない**。
- `asr_status`: `speech_candidate | non_speech_candidate | uncertain | error`（このPhaseで笑い/喘ぎの最終分類はしない）。
- `transcript_raw`: 文字起こしモデルの原出力または`null`。`transcript_checked`は人手確認した場合のみ保存、確認者情報付き。
- `asr_diagnostics`: モデルが公開する情報だけを保存。確信度が未対応なら`null`。no-speech probabilityも**声の表現ラベルではない**。
- `model_id`, `model_revision`, `backend_version`, `config_fingerprint`, `audio_decode_info`, `failure_reason`。

区間が未検出/空文字でも、**元音声全体を後段が参照できる証跡行を必ず出す**。`asr_status=non_speech_candidate`は「非言語音声だと確定」ではなく、**文字起こしが可能な発話を認識できなかった候補**として扱う。明確な無音か有用な息/笑い声かはPhase 4・人間が判定。

## 4. 実行・フォルダ・再実行ルール

```text
LocalVoice/
  data/                                     # 既存・常に保全
  src/localvoice/
    transcription/                          # Phase 3 本体（設計後に作成）
    __main__.py                             # 単一CLI
  tests/                                    # 単体/境界/回帰
  work/<run-id>/
    input_manifest.jsonl                    # 入力の固定コピー
    run_manifest.json
    transcription.jsonl
    failures.jsonl
  outputs/datasets/<profile>/<version>/    # Phase 5のみ。Phase 3は書き込まない
  Irodori-TTS/                              # 公式クローン。触らない
```

予定の呼出し口：`python -m localvoice transcribe --input-manifest <path> --run-id <unique-id>`。**現時点では未実装**。1つのCLI/1つの実装を維持し、実験用の`*_v2.py`, `*_final.py`等を増やさない。

- 同一run IDの出力を上書きしない。成功・失敗・保留を追跡する。入力SHA不一致・デコード失敗・区間矛盾はfail-closed。
- VADや言語モデルが除外した音声も、入力に対する記録を残す。speech-only VADの結果を「非言語音声不要」の根拠にしない。
- 長時間録音はセグメント別の推論結果を元WAVへの時刻位置に還元。自動区間切断で息遣い・笑いなどの重要な音を失わない設計とする。
- 生のASRテキストと表示用の正規化テキストを区別する。原文に存在しない句読点・絵文字・非言語擬音を無条件で追加しない。
- 正式フォルダ/タグの仕様は[EMOJI_AUDIO_DATASET.md](EMOJI_AUDIO_DATASET.md)。ASRによる`text`、スタイルによる`tags`、人間による`approved`を混同しない。
- 本PhaseでIrodori用DACVAE`latent_path`や学習Manifestを生成しない。ユーザーからの別途明示要求がある場合だけ、公式`prepare_manifest.py`を新しい工程として設計する。

## 5. Phase 3実装ゲートと受入判定

### 設計レビューで先に確定するもの
- [ ] ASR候補モデルの小規模pilot結果と使用する実行バックエンド
- [ ] 人手確認済みの固定評価音声と測定指標/許容条件
- [ ] `input_manifest`, `transcription.jsonl` のスキーマと時間境界の解釈
- [ ] 非言語/無音/失敗を**捨てない**出力契約
- [ ] ASR用依存環境の判断（原則既存`.venv`）

### 実装後に必須の受入
- [ ] 日本語通常会話が実際の音声に沿って文字起こしされ、元音声と時刻で照合できる
- [ ] 喘ぎ・笑い・吐息・息切れの候補が文字起こし無しでも**失われず**Phase 4へ渡る
- [ ] 無音/音楽/複数話者/他人音声で勝手な文字列や本人承認が確定しない
- [ ] パス衝突、再実行、入力不整合、decode不良で既存WAVを破壊しない
- [ ] `data/`の保全と`Irodori-TTS/`が未変更であることを検証する
- [ ] 失敗・保留の証跡が残る。実機のモデルとGPUで検証を実施する

**STOP:** 設計ゲート未達なら、ASRの自動大量処理やPhase 4分類の実装に進まない。
