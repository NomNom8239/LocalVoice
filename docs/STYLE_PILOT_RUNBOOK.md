# LV-R04 — Acoustic-expression classification Pilot (Windows)

**Status:** Pilot implementation committed; **real RTX 5060 Ti inference and real recording classification quality NOT YET VERIFIED**. This is NOT a complete style classifier and NOT the versioned WAV library. Existing `data/` sources and upstream `Irodori-TTS/` stay untouched.

## Goal and comparison boundary

The priority deliverable is a browsable Japanese-category reference-WAV library from the **existing 1,167** candidates, not full ASR transcription or automatic Irodori operation. This Pilot checks whether an acoustic model can help triage actual recordings before a larger style validation.

Compare:

| Candidate | HF model | Strength | Known limitation |
| --- | --- | --- | --- |
| AST | [MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | Pretrained AudioSet sound labels such as Whispering, Laughter, Groan, Pant and Gasp | English/general AudioSet labels, not calibrated to this speaker or all desired categories. It has no assumed Yawn/exhale exact coverage. |
| CLAP | [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) | Compare acoustic evidence to custom English sound descriptions including gasp, sigh, moan and silence | Relative score depends on the full candidate-text set. Not a calibrated probability or label approval. |

**Never** use ASR `transcript_raw`, ASR's `speech_candidate`, or the 12-item ASR `category` values as acoustically verified class labels. Pilot category is retained as `selection_category_hint` for provenance ONLY.

The initial Pilot may reuse the existing, SHA-sealed **12-WAV selection**. This avoids reselecting/relabeling WAVs just to test model inference; it does NOT replace independently reviewed acoustic ground truth. More representative recordings (separate breaths, laughs, moans, whispering, game speech and actual noise) will be needed to evaluate model quality. The 3-second artificial-silence item is a negative integration check, not a replacement for real recording noise.

## 1. Branch / environment / read-only preflight

PowerShell:

```powershell
cd F:\AIProjects\LocalVoice
git fetch origin
git switch feature/phase4-acoustic-pilot
git pull --ff-only origin feature/phase4-acoustic-pilot

# Reuse LocalVoice/.venv; DO NOT activate Irodori-TTS/.venv
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_asr_pilot.py tests/test_style_pilot.py -q
```

If an import/dependency issue occurs, inspect the error before any package upgrade. The already functioning ASR environment has `torch`, `transformers`, `soundfile`, `numpy`, `scipy`. No new environment or broad dependency update is required by the design. AST/CLAP model downloads are only performed when running a model for the first time; their compatibility with this exact GPU setup is not yet confirmed.

## 2. Reuse existing sealed selection safely

```powershell
$manifest = ".\work\asr_pilot_selection_v2_20261009_135606446.sealed.jsonl"

.\.venv\Scripts\python.exe -m localvoice style-pilot validate --manifest $manifest
```

Expected: `VALID: 12 acoustic Pilot WAVs; SHA-256 unchanged`. Reads all source files and confirms SHA/duration; does not touch any source. If the manifest is missing, STOP and locate the prior sealed manifest; never fabricate paths or replace existing files.

Other SHA-sealed JSONL manifests with `sample_id` or `source_id`, `source_path`, `source_sha256` and optional duration/profile/identity may be used (1–32 explicitly selected WAVs, 0.25–30 s, mono/stereo, sample rate >=8kHz). No implicit folder-wide traversal.

## 3. AST FIRST, standalone CUDA

```powershell
$astRunId = "style_ast_pilot_$(Get-Date -Format 'yyyyMMdd_HHmmssfff')"
.\.venv\Scripts\python.exe -m localvoice style-pilot run `
  --manifest $manifest `
  --model ast `
  --run-id $astRunId `
  --device cuda
```

Inspect (do not edit the JSONL):

```powershell
Get-Content ".\work\$astRunId\run_manifest.json" -Encoding UTF8 -Raw | ConvertFrom-Json |
  Select-Object status, completed, failed, initialization_error | Format-List

Get-Content ".\work\$astRunId\style_predictions.jsonl" -Encoding UTF8 |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Select-Object source_id, selection_category_hint, digital_silence, candidate_ranking, review_status |
  Format-List
```

`status=completed, completed=12, failed=0` means **only technical inference success**. Every record intentionally has `review_status=requires_review` and `human_approved=false`. Predicted scores/ranks do not imply acceptance.

## 4. CLAP SECOND, new run and separate model process

After AST exits fully and its run is inspected:

```powershell
$clapRunId = "style_clap_pilot_$(Get-Date -Format 'yyyyMMdd_HHmmssfff')"
.\.venv\Scripts\python.exe -m localvoice style-pilot run `
  --manifest $manifest `
  --model clap `
  --run-id $clapRunId `
  --device cuda
```

Then inspect `work/$clapRunId/run_manifest.json` and `style_predictions.jsonl` as above. Compare by **same source_id**, looking at `category_scores`, ranks, confusions, and `digital_silence`. CLAP score is softmax over the current English caption set; changing captions changes the scale/rank. Do not compare AST numeric scores directly against CLAP numeric scores as if both were calibrated probabilities.

## 5. Outputs and minimum acceptance

```text
work/<unique-run-id>/
├─ run_manifest.json            # model/config hash, requested device, status
├─ style_predictions.jsonl      # one full-source result/error row per WAV
└─ failures.jsonl               # explicit failure evidence
```

- Run IDs create new directories and **never overwrite** old results.
- Source WAVs are SHA-checked before and after inference, resampled in memory, never modified or copied to `data/`.
- Real-world categories can overlap. The model returns a **ranked hypothesis**, NOT a final style or identity approval. Silence is flagged as a risk even if a model returns a class.
- A technically successful Pilot is not a quality PASS. Next evaluation must listen to real sound and label it separately; uncalibrated scores may not be used to auto-export clips. The full 1,167 input batch, uncertainty thresholds, QC, reviewer UX and Japanese folder export are **not implemented here**.
- This work does not build, use, or alter an Irodori training manifest or latent.

**Next decision after inspecting both model outputs:** design a small independently human-annotated acoustic reference set, select labels that are actually distinguishable, and define review/abstention behavior before implementing a full batch candidate classifier.
