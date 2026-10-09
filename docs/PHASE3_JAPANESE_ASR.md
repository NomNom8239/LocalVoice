# Phase 3 — 日本語文字起こし（ASR）新規設計

**Status (2026-10-09):** 12件ASR Pilot実装・CUDA実機検証済み。全件ASR・48件本比較・モデル正式採用は未実施で後続保留。**カテゴリ別参照WAVライブラリv1の完了・LV-R04/06/05着手に必須の先行ゲートではない。**

## 1. 目的・範囲

本書はASRによる**補助的な日本語文字起こし**、実施済み12件Pilot、将来の本格ASR採用条件を扱う。発話の推定テキスト・区間を記録し、認識できない非言語音声も元WAV・由来・時刻・SHAを捨てない。

**当面の最優先成果物は、既存 `data/training_audio/Ui_Shigure/audio/` の1,167件の本人音声候補を、音響分類・本人性/録音品質QC・必要な人手レビューを経て、日本語カテゴリ別フォルダに非破壊で版付きコピーする参照WAVライブラリv1。** エクスプローラーからユーザーが参照WAVを選び、公式Irodori VoiceDesignに手動指定する。

**ASRはライブラリv1の必須入力ではない。** LV-R04の音響スタイル分類・LV-R06のQC/レビュー・LV-R05のフォルダ出力は、ASRテキストや48件モデル比較なしで実施可能とする。 `speech_candidate`やASR出力は本人承認・スタイル正解・公開可否を意味しない。Emoji Paletteの分類はASRから推測しない。新規配信取り込みの一括E2EはLV-R07、Irodori操作の完全自動化とLoRAはv1スコープ外。

旧Irodori ASR/スタイル分類/LoRA実験のコード・出力を実装の前提にしない。既存`data/`と公式`Irodori-TTS/`を破壊・変更しない。

### 公式リファレンス（2026-10-09確認）

1. [Irodori-TTS README](https://github.com/Aratako/Irodori-TTS/blob/main/README.md)（参照したREADME blob SHA: `03818b912f4e689ac194e5c1fd618d5a73b27b4d`）。v4.1-Smallは**text、caption、reference audio**を統合し、Emojiによる生成表現の指定をサポート。複数のクリーンな短い参照WAVが推奨される。
2. [VoiceDesign WebUI](https://github.com/Aratako/Irodori-TTS/blob/main/gradio_app_voicedesign.py)（blob SHA: `45e674f31e54b3348290d2e889dbf09f847f734b`）。Emoji Paletteは**Text欄**に絵文字を挿入する。Captionは別欄、参照音声は複数アップロード可。
3. [Emoji Palette実装](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/gradio_emoji_palette.py)（blob SHA: `33fce37be1e042b05fefe8e2b9ab5ac110b91fda`）。🥵喘ぎ＝うめき/唸り、🌬️息切れ＝荒い息遣い/呼吸音等。**Paletteは出力を操作するUIであり、音声分類モデルでも正解データでもない。**
4. [Irodoriの音声学習Manifest生成](https://github.com/Aratako/Irodori-TTS/blob/main/prepare_manifest.py)（blob SHA: `3d9d2ac4ac5b4b42d20fd64262efe37bd686a296`）。学習入力`text`が空ならskip（`empty_text`）。学習ManifestはDACVAE latentへのパスを含み、`caption`と`speaker_id`はoptional。**LocalVoiceのASRログをこの学習Manifestと混同しない。**
5. [公式推論パラメータ](https://github.com/Aratako/Irodori-TTS/blob/main/docs/parameters.md)（blob SHA: `d458b2550419a6fb2a2503b26f878259cfce0a9d`）。v4.1-Small参照WAVはidentity conditioning。通常の参照品質を優先し、複数の短いクリーンなクリップの組合せで約30秒までに類似性向上の大半が観測されたと説明（最大120秒）。

**結論:** IrodoriはTTSでありASRではない。ASRモデルは別に選び、結果とPaletteの分類・Captionを別フィールドで扱う。VoiceDesign参照用WAVには学習用`text`非空条件を適用しない。

## 2. ASRの実績と将来のモデル選定条件

### 2.1 12件の実機Pilot（2026-10-09完了）

- 固定入力は`work/asr_pilot_selection_v2_20261009_135606446.sealed.jsonl`。実機seal成功、`VALID: 12 source WAVs, all SHA-256 verified`。無音負例`sample_11`は人工3秒完全無音で、実録音ノイズ評価を代替しない。
- [Kotoba-Whisper v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)：CUDA実行`completed=12, failed=0`。無音に「ごめん」を誤生成。
- [Whisper large-v3](https://github.com/openai/whisper)：CUDA実行`completed=12, failed=0`。無音に「ご視聴ありがとうございました」を誤生成。短い非言語`sample_09/10`で「あ」「う」等の長大な反復生成が発生。
- **技術的な実機Pilotは両モデルPASS。** ただし文字起こし正解との精度比較・非言語の認識適否・正式なASRモデル採用は未完了。Pilotの`pilot_category`はASR評価用途であり音響分類の正解ラベルではない。
- 実装済み：`src/localvoice/transcription/pilot.py` の`asr-pilot seal/validate/run`と単一CLI。詳細は[ASR_PILOT_RUNBOOK.md](ASR_PILOT_RUNBOOK.md)。元WAVや既存runを上書きしない。

### 2.2 将来のASR正式選定（**ライブラリv1とは別ゲート**）

発話検索や大規模文字起こしが必要になった段階で再開する既存の評価計画。Pilot12件だけでモデルを採用せず、1,167件に無差別ASRをかけない。

- 比較候補はKotoba-Whisper v2.0とWhisper large-v3。必要ならKotoba v2.1を句読点後処理目的で検討。`faster-whisper`は推論バックエンド候補であり、Windows/CUDA/cuDNN条件を別途実機確認する。既存`.venv`を基本とする。
- 固定48件（Pilotとは重複なし、できれば3配信以上）：通常会話12、囁き6、発話＋表現6、非言語16（笑い4、喘ぎ/うめき4、息切れ4、吐息/息をのむ4）、無音/ノイズ4、他人声/複数話者4。実録音の不足を人工無音で埋めて完了扱いしない。
- 2モデルに同じ48件、同等前処理・`language=ja`・`task=transcribe`・同一GPU条件を与える。発話24件は人手正解に対する日本語正規化CER（NFKC、句読点/空白差を主指標から除外）、非言語/負例24件は幻覚・誤挿入数と根拠を残せない確定数を比較。単純平均せず、推論失敗・速度・VRAMも別に記録。
- 全48件の出力または理由付き失敗証跡、SHA・時間境界・`data/`非破壊を必須にする。特に非言語・負例の誤挿入が少ない候補を優先。差が明確でなければ`requires_review`にして、無理に全自動採用しない。
- 48件が未準備なら**ASR正式採用だけを保留**。LV-R04/06/05の音響分類・品質レビュー・ライブラリ構築は停止させない。

## 3. 入出力境界

**入力（ASRを実行する場合のみ）**：承認済み/候補の音声ファイルをユーザー指定manifestに列挙。フォルダ全件の自動探索・無差別全件文字起こしは既定でしない。LV-R04/06/05は`transcript_raw`が無くてもWAVから処理できること。

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

実装済みの呼出し口は`python -m localvoice asr-pilot seal|validate|run`。`python -m localvoice transcribe --input-manifest <path> --run-id <unique-id>` は将来の正式ASR用**未実装案**。機能追加時も単一CLI・共通実装を維持し、`*_v2.py`/`*_final.py`等を増やさない。

- 同一run IDの出力を上書きしない。成功・失敗・保留を追跡する。入力SHA不一致・デコード失敗・区間矛盾はfail-closed。
- VADや言語モデルが除外した音声も、入力に対する記録を残す。speech-only VADの結果を「非言語音声不要」の根拠にしない。
- 長時間録音はセグメント別の推論結果を元WAVへの時刻位置に還元。自動区間切断で息遣い・笑いなどの重要な音を失わない設計とする。
- 生のASRテキストと表示用の正規化テキストを区別する。原文に存在しない句読点・絵文字・非言語擬音を無条件で追加しない。
- 正式フォルダ/タグの仕様は[EMOJI_AUDIO_DATASET.md](EMOJI_AUDIO_DATASET.md)。ASRによる`text`、スタイルによる`tags`、人間による`approved`を混同しない。
- 本PhaseでIrodori用DACVAE`latent_path`や学習Manifestを生成しない。ユーザーからの別途明示要求がある場合だけ、公式`prepare_manifest.py`を新しい工程として設計する。

## 5. 独立した受入ゲートと保留範囲

### 5.1 12件Pilot（実施済み）

- [x] 12件WAV選定、SHA付きseal/validate
- [x] Kotoba-Whisper v2.0・Whisper large-v3をWindows CUDAで各12件実行
- [x] 12件分の出力/失敗証跡、無音の幻覚と非言語の反復を確認
- [ ] 2モデルの正解テキストに対する本比較・正式採用は**未着手（v1スコープ外）**

### 5.2 本格ASRを追加する場合のゲート（後続）

- [ ] 48件の実録音評価セットと人手正解・測定指標を固定
- [ ] `input_manifest`と`transcription.jsonl`の実装契約・モデル/バックエンド・時間境界の意味を確定
- [ ] 非言語/無音を破棄しないこと、ASR幻覚・長大反復を自動的な正解や本人承認にしないことを検証
- [ ] 実機でデコード不良、衝突・再実行・不整合、GPU・失敗証跡、`data/`/公式Irodori非変更を確認

**STOPの適用先は正式ASRの採用・一括文字起こしのみ。** LV-R04の音響分類、LV-R06の本人性・音質QC/レビュー、LV-R05のカテゴリ別ライブラリv1出力をブロックしない。

## 6. Phase 4/5へのデータ契約

LV-R04/06/05はASR結果がなくても、入力WAVの安定ID・source SHA・出典・元音声/区間、音響分類候補・本人性/QC・人手承認/保留記録を使って実施する。文字起こしが存在する場合だけ任意の補助メタデータとして参照する。`transcript_raw`、`speech_candidate`およびASRの誤挿入はスタイル/本人性の根拠にしない。正式ライブラリには人手承認済みかつQC合格の音声のみコピーする。詳細は[EMOJI_AUDIO_DATASET.md](EMOJI_AUDIO_DATASET.md)と[PROJECT_LAYOUT.md](PROJECT_LAYOUT.md)。
