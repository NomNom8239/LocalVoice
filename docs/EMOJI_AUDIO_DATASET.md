# Human-browsable audio library aligned with Irodori Emoji Palette

Status: Phase 3–5 **design requirement**; not implemented. This is a new design, not a reuse of legacy Irodori ASR/style/LoRA experiments.

## User-facing goal

After acquisition, identity verification, transcription, independent acoustic-style classification and human review, export a **folder-browsable reference-audio library**. A person should be able to open a category such as `通常会話`, `笑い`, `喘ぎ`, or `息切れ`, listen to a clip, and select its WAV directly in the official Irodori VoiceDesign WebUI. This export is useful without opening a CSV or running Python.

The labels align with **the actual upstream palette** as implemented in [`Aratako/Irodori-TTS/irodori_tts/gradio_emoji_palette.py`](https://github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/gradio_emoji_palette.py). **Palette buttons are generation controls, not validated audio-classification labels or guarantees of how a recording should be transcribed.** Track the upstream palette revision when freezing the label schema.

## Initial category taxonomy

| Stable code | Export directory | Corresponding Palette entry | Label evidence |
| --- | --- | --- | --- |
| `normal_speech` | `01_通常会話` | None (LocalVoice baseline) | Predominantly ordinary, intelligible speech |
| `whisper` | `02_囁き` | 👂 囁き | Audible whisper-like delivery |
| `exhale` | `03_吐息` | 😮‍💨 吐息 | Exhalation, sigh, sleep breath |
| `laugh` | `04_笑い` | 🤭 笑い | Giggle or laughter |
| `moan` | `05_喘ぎ_うめき` | 🥵 喘ぎ | Moaning, groaning, vocal panting-like moans |
| `pant` | `06_息切れ_荒い呼吸` | 🌬️ 息切れ | Rapid/heavy breathing and breathlessness |
| `gasp` | `07_息をのむ` | 😮 息をのむ | Short audible sharp inhale/gasp |
| `cry` | `08_泣き声` | 😭 泣き声 | Sobbing / crying |
| `scream` | `09_悲鳴` | 😱 悲鳴 | Screams or yells |
| `yawn` | `10_あくび` | 🥱 あくび | Yawning |
| `hum` | `11_鼻歌` | 🎵 鼻歌 | Humming |
| `other_approved` | `90_その他_承認済み` | None (fallback) | Usable clip manually approved but not a listed category |

Only create populated categories; do not create 40+ empty folders to mirror every palette button. Labels such as `😏 からかう`, `🫶 優しく`, `😠 怒り`, `😪 眠そう`, etc. may be proposed as **secondary tags** until their audible boundaries and human-labeled evaluation data support a separate folder. `⏸️ 間` is not a standalone reference voice class; `📢 エコー` and `📞 電話越し` describe recording/effect conditions and are not separate clean reference voice classes. Their presence should be tracked and may be a QC reason to exclude the clip.

## Ownership by phase

- **Phase 3 / transcription:** Identify whether a segment has intelligible Japanese words; output timestamps, text when supported, `speech | non_speech | uncertain` evidence and failure reasons. ASR must not call a clip `喘ぎ`, `笑い`, etc. based on text alone. Its non-speech marker is not a final style label.
- **Phase 4 / style:** Assess acoustic events and delivery independently of transcript text, using the defined categories and optional secondary tags. Require human review for uncertainty and cases where multiple categories plausibly apply. Distinguish `exhale`, `pant`, `gasp` and `moan`; do not collapse all breath sounds.
- **Phase 5 / export:** After identity approval, audio QC and human approval, **copy** clean, relevant audio segments to a versioned directory named for the selected primary class. One canonical copy per segment. Other confirmed categories are recorded in metadata; do not duplicate audio across folders by default.

The output folder is a **manual lookup tool**, not evidence that emoji-only conditioning or LoRA training works. The upstream VoiceDesign UI remains untouched.

## Intended on-disk output

```text
LocalVoice/
├─ data/                              # Existing assets: NEVER migrate or modify
├─ work/<run-id>/
│  ├─ transcription.jsonl             # Phase 3 evidence and failures
│  ├─ style_predictions.jsonl         # Phase 4 evidence
│  └─ review_queue.csv                # Pending decisions, not final exports
└─ outputs/datasets/<profile>/<version>/
   ├─ README.txt                       # Japanese guide, label/emoji mapping, selection tips
   ├─ index.csv                        # Human-readable search/index (UTF-8 with BOM for Excel)
   ├─ manifest.jsonl                   # Machine-readable provenance and labels
   └─ audio/
      ├─ 01_通常会話/
      ├─ 02_囁き/
      ├─ 03_吐息/
      ├─ 04_笑い/
      ├─ 05_喘ぎ_うめき/
      ├─ 06_息切れ_荒い呼吸/
      ├─ 07_息をのむ/
      ├─ 08_泣き声/
      ├─ 09_悲鳴/
      ├─ 10_あくび/
      ├─ 11_鼻歌/
      └─ 90_その他_承認済み/
```

These directories are examples of the *final schema*, not directories to create yet; omit empty classes.

## Clip and review rules

1. Classify **segments**, not entire multi-minute streams or mixed-event WAVs as a single expression. Preserve source audio and segment start/end offsets. If clipping would remove important phonetic/respiratory context, leave an intact clip with secondary tags or queue for review rather than cutting blindly.
2. One clip has one **human-approved primary category** for directory placement and zero or more secondary labels. When `通常会話` and `笑い` coexist, pick a dominant class **only if audibly justified**. Otherwise queue for review; never copy the same WAV into several categories silently.
3. Candidate classes and `confidence` are not approvals. Missing/unsupported confidence is `null`, not a fabricated score. Keep `unknown`, `requires_review`, rejected and non-voice/foreign-speaker clips **out of the promoted folders**, in the corresponding run's review/evidence area only.
4. Each exported audio file has a stable neutral filename (not a generated descriptive claim), and `index.csv` provides original source, Japanese transcript when intelligible, primary Japanese label/emoji, secondary tags, length and listening notes. Avoid raw emoji in the actual filenames to keep cross-tool compatibility.
5. `manifest.jsonl` records source WAV path and SHA-256, original acquisition/run identity, segment offsets, output SHA-256/path, speech/transcription status, transcript or null, primary and secondary label codes, mapped emoji, model/config/taxonomy revisions, reviewer decision, and approval provenance.
6. Do not change files under `data/`; published versions are immutable. On collision, refuse to overwrite and require a new version. Do not put LocalVoice scripts or exported library under `Irodori-TTS/`.
7. Allow manual playback and category browsing **without dependency on Notion, model downloads, or a custom browser UI**. `README.txt` explains that Irodori Palette controls inference, not category selection.
8. An acceptance test must walk a **real archive → speaker check → ASR → style review → export → pick WAV in VoiceDesign** flow. Check category correctness and ease of human browsing, not merely JSON validity.

## Scope gate

Phase 3 must define compatible speech/non-speech output so Phase 4 can classify events. **Phase 3 does not auto-sort WAVs into final Emoji Palette folders.** Phase 4 owns acoustic classification; Phase 5 owns the human-browsable folder export. Do not build shortcut scripts or premature classifier scaffolding while designing ASR.
