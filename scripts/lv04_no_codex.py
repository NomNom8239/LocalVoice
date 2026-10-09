#!/usr/bin/env python3
r"""LV-04 Irodori LoRA baseline: offline preflight by default; --run trains.

From the LocalVoice repository root:
    .\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py
    .\.venv\Scripts\python.exe .\scripts\lv04_no_codex.py --run

Codex, new model downloads, and new Python environments are not required.
The eight LV-02 approved sources stay separate from three independent eval
clips. The official train.py internally reserves one of eight as validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from lv03_no_codex import WORKSPACE_PARTS, verify_sources
from irodori_nonverbal_lora import (
    CHECKPOINT_SHA256,
    checkpoint_path,
    file_hash,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA_EXPECTED = "89f9d8fbd4d51ea019867ee1197725ede1df13c5"
MANIFEST_DIR = "lv03_dacvae_005"
BASELINE_STEPS = 120
VALID_RATIO = 0.125
SPEAKER_ID = "LocalVoice-approved:Ui_Shigure"


def hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def parse_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        values = [json.loads(line) for line in f if line.strip()]
    if len(values) < 2:
        raise ValueError(f"Expected at least 2 rows in official manifest: got {len(values)}")
    visited = set()
    for row in values:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Empty text in official manifest")
        if row.get("speaker_id") != SPEAKER_ID:
            raise ValueError("Unexpected speaker_id in manifest")
        if not isinstance(row.get("num_frames"), int) or row["num_frames"] < 1:
            raise ValueError("Invalid num_frames in manifest")
        latent = Path(str(row["latent_path"]))
        if not latent.is_absolute():
            latent = path.parent / latent
        latent = latent.resolve()
        if not latent.is_file() or latent in visited:
            raise ValueError(f"Missing or duplicate latent: {latent}")
        visited.add(latent)
    return values


def read_lv03_result(workspace: Path, source_sha: dict[str, str],
                     attempt: str = MANIFEST_DIR) -> Path:
    if not re.fullmatch(r"lv03_dacvae_\d{3}", attempt):
        raise ValueError("Expected a versioned lv03_dacvae_NNN attempt name")
    codec = workspace / attempt
    result = codec / "lv03_final_result.json"
    if not result.is_file():
        raise FileNotFoundError(f"LV-03 completion evidence missing: {result}")
    document = json.loads(result.read_text(encoding="utf-8"))
    if (document.get("status") != "PASS_LV03_DACVAE_AND_DATASET"
            or document.get("train_rows") != len(source_sha) - 3
            or document.get("evaluation_rows") != 3
            or document.get("training_started") is not False):
        raise ValueError("LV-03 completion result is incomplete or invalid")
    if document.get("input_sha256_by_clip") != source_sha:
        raise ValueError("LV-03 SHA evidence differs from current WAVs")
    manifest = Path(str(document.get("manifest", ""))).resolve()
    expected = (codec / "train_manifest.jsonl").resolve()
    if manifest != expected:
        raise ValueError(f"Unexpected LV-03 manifest path: {manifest}")
    return manifest


def official_version(upstream: Path) -> str:
    git = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if git.returncode:
        raise RuntimeError("Irodori git revision cannot be verified")
    revision = git.stdout.strip()
    if revision != UPSTREAM_SHA_EXPECTED:
        raise RuntimeError(
            f"Irodori official revision changed: {revision}; "
            "review upstream and configs before training"
        )
    return revision


def resolve_checkpoint(upstream: Path, selected: Path | None) -> tuple[Path, str]:
    if selected is None:
        path, kind = checkpoint_path(upstream, None, fetch=False)
    else:
        path, kind = selected.expanduser().resolve(), "user-provided"
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Base checkpoint missing: {path}. Restore previously verified "
            "unquantized model.safetensors; no automatic download."
        )
    if path.name != "model.safetensors":
        raise ValueError("Only official unquantized model.safetensors is accepted")
    if file_hash(path) != CHECKPOINT_SHA256:
        raise ValueError("Official base model SHA256 mismatch; training blocked")
    return path, kind


def build_command(python: Path, upstream: Path, config: Path, manifest: Path,
                  checkpoint: Path, output: Path, steps: int,
                  resume: Path | None = None, *,
                  valid_ratio: float = VALID_RATIO) -> list[str]:
    if not 20 <= steps <= 10000:
        raise ValueError("LoRA steps must be 20..10000")
    if not 0 < valid_ratio < 1:
        raise ValueError("Validation ratio must be between 0 and 1")
    cmd = [
        str(python), "-u", str(upstream / "train.py"),
        "--config", str(config),
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--device", "cuda", "--precision", "bf16",
        "--lora", "--train-mode", "rf",
        "--max-steps", str(steps),
        "--batch-size", "1", "--gradient-accumulation-steps", "2",
        "--num-workers", "0", "--optimizer", "adamw",
        "--lr", "0.00005", "--lr-scheduler", "cosine",
        "--warmup-steps", "8",
        "--save-every", "30", "--log-every", "5",
        "--checkpoint-best-n", "2",
        "--valid-ratio", str(valid_ratio), "--valid-every", "20",
        "--ref-min-seconds", "1", "--ref-max-seconds", "12",
        "--gradient-checkpointing", "--no-compile-model", "--no-wandb",
        "--seed", "0",
    ]
    if resume is not None:
        if not resume.is_dir() or not (resume / "adapter_config.json").is_file():
            raise ValueError("--resume expects an existing LoRA checkpoint directory")
        cmd += ["--resume", str(resume), "--init-checkpoint", str(checkpoint)]
    else:
        cmd += ["--init-checkpoint", str(checkpoint)]
    return cmd


def fresh_attempt(workspace: Path) -> Path:
    for n in range(1, 1000):
        candidate = workspace / f"lv04_lora_{n:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
    raise RuntimeError("No unused LV-04 attempt number")


def save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def stream(command: list[str], *, cwd: Path, log: Path, env: dict[str, str],
           timeout: int) -> tuple[int, str]:
    with log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=str(cwd), env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        start = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - start >= timeout:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return process.returncode or 124, "timeout"
            time.sleep(3)
    return process.returncode, "exited"


def validate_adapter(directory: Path) -> dict:
    config = directory / "adapter_config.json"
    model = next(
        (directory / x for x in ("adapter_model.safetensors", "adapter_model.bin")
         if (directory / x).is_file() and (directory / x).stat().st_size > 0),
        None,
    )
    if not config.is_file() or model is None:
        raise ValueError(f"PEFT adapter files not found in {directory}")
    meta = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not meta.get("peft_type"):
        raise ValueError("Invalid PEFT adapter config")
    return {"directory": str(directory),
            "adapter_sha256": hash_file(model),
            "adapter_config_sha256": hash_file(config),
            "weight_file": model.name}


def infer_command(root: Path, upstream: Path, python: Path, checkpoint: Path,
                  adapter: Path, manifest: Path, output: Path) -> list[str]:
    first = parse_manifest(manifest)[0]
    ref = Path(first["latent_path"])
    if not ref.is_absolute():
        ref = (manifest.parent / ref).resolve()
    shared = (upstream / ".localvoice-toolchain" /
              "ffmpeg-7.0.2-full-shared" / "bin")
    launcher = root / "scripts" / "irodori_codec_entry.py"
    if os.name == "nt":
        if not shared.is_dir() or not launcher.is_file():
            raise FileNotFoundError("Known-good FFmpeg shared DLL launcher missing")
        cmd = [str(python), "-u", str(launcher), str(shared), str(upstream / "infer.py")]
    else:
        cmd = [str(python), "-u", str(upstream / "infer.py")]
    return cmd + [
        "--checkpoint", str(checkpoint),
        "--lora-adapter", str(adapter),
        "--text", "おはようございます。今日はいい天気ですね。",
        "--output-wav", str(output),
        "--ref-latent", str(ref),
        "--num-steps", "12", "--seconds", "3",
        "--model-device", "cuda", "--codec-device", "cuda",
        "--model-precision", "bf16", "--codec-precision", "fp32",
        "--seed", "0",
    ]


def prepare(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    workspace = root.joinpath(*WORKSPACE_PARTS)
    upstream = root / "Irodori-TTS"
    python = upstream / ".venv" / "Scripts" / "python.exe"
    config = upstream / "configs" / "train_v4_small_lora.yaml"
    if not python.is_file() or not config.is_file() or not (upstream / "train.py").is_file():
        raise FileNotFoundError("Existing Irodori train/config/venv missing")
    lv03_attempt = getattr(args, "lv03_attempt", MANIFEST_DIR)
    if not re.fullmatch(r"lv03_dacvae_\d{3}", lv03_attempt):
        raise ValueError("Invalid LV-03 attempt name")
    lv03_evidence = workspace / lv03_attempt / "lv03_final_result.json"
    if not lv03_evidence.is_file():
        raise FileNotFoundError(lv03_evidence)
    lv03_report = json.loads(lv03_evidence.read_text(encoding="utf-8"))
    selected_csv = Path(lv03_report.get("training_csv") or
                        workspace / "dataset_for_prepare_manifest_approved_train.csv").resolve()
    train_source, source_sha = verify_sources(workspace, selected_csv)
    if lv03_report.get("training_csv_sha256") and (
            hash_file(train_source) != lv03_report["training_csv_sha256"]):
        raise ValueError("LV-03 training CSV content changed")
    manifest = read_lv03_result(workspace, source_sha, lv03_attempt)
    rows = parse_manifest(manifest)
    if len(rows) != len(source_sha) - 3:
        raise ValueError("Manifest row count differs from frozen source evidence")
    revision = official_version(upstream)
    checkpoint, kind = resolve_checkpoint(upstream, args.checkpoint)
    text = config.read_text(encoding="utf-8")
    if ("lora_enabled: true" not in text or
            "text_tokenizer_repo: sbintuitions/modernbert-ja-310m" not in text or
            "latent_dim: 32" not in text):
        raise ValueError("Unexpected Irodori LoRA base configuration")
    return {
        "root": root, "workspace": workspace, "upstream": upstream,
        "python": python, "manifest": manifest, "config": config,
        "checkpoint": checkpoint, "checkpoint_source": kind,
        "upstream_revision": revision,
        "source_sha": source_sha,
        "training_input_csv": str(train_source),
        "train_rows": len(rows),
        "lv03_attempt": lv03_attempt,
        "train_manifest_sha256": hash_file(manifest),
        "config_sha256": hash_file(config),
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=BASELINE_STEPS)
    parser.add_argument("--lv03-attempt", default=MANIFEST_DIR,
                        help="Use a passed newer LV-03 attempt, not an unverified manifest")
    parser.add_argument("--valid-ratio", type=float, default=VALID_RATIO)
    parser.add_argument("--timeout", type=int, default=10800,
                        help="Training timeout in seconds")
    parser.add_argument("--infer-timeout", type=int, default=1800)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run", action="store_true",
                        help="Explicitly start one LoRA baseline training experiment")
    args = parser.parse_args()
    if args.timeout < 1 or args.infer_timeout < 1:
        raise ValueError("timeouts must be positive")
    if not 0 < args.valid_ratio < 1:
        raise ValueError("Invalid validation ratio")
    data = prepare(args)
    workspace = data["workspace"]
    # Construct preview only; no output directory is created in dry run.
    candidate = next((workspace / f"lv04_lora_{i:03d}" for i in range(1, 1000)
                      if not (workspace / f"lv04_lora_{i:03d}").exists()), None)
    if candidate is None:
        raise RuntimeError("No available output directory")
    resume = args.resume.expanduser().resolve() if args.resume else None
    cmd = build_command(data["python"], data["upstream"], data["config"],
                        data["manifest"], data["checkpoint"],
                        candidate / "adapter", args.steps, resume,
                        valid_ratio=args.valid_ratio)
    plan = {
        "status": "PREFLIGHT_PASS_PLAN_ONLY" if not args.run else "TRAIN_STARTING",
        "upstream_commit": data["upstream_revision"],
        "manifest": str(data["manifest"]),
        "manifest_sha256": data["train_manifest_sha256"],
        "train_source_csv": data["training_input_csv"],
        "approved_train": data["train_rows"],
        "internal_train_expected": data["train_rows"] - max(1, min(data["train_rows"] - 1,
                                                                  int(data["train_rows"] * args.valid_ratio))),
        "internal_validation_expected": max(1, min(data["train_rows"] - 1,
                                                    int(data["train_rows"] * args.valid_ratio))),
        "independent_eval_untouched": 3,
        "validation_ratio": args.valid_ratio,
        "steps": args.steps,
        "base_checkpoint_sha256": data["checkpoint_sha256"],
        "base_checkpoint": str(data["checkpoint"]),
        "upstream_config_sha256": data["config_sha256"],
        "source_sha256_by_clip": data["source_sha"],
        "command": cmd,
        "lora_quality_validated": False,
    }
    if not args.run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    attempt = fresh_attempt(workspace)
    cmd = build_command(data["python"], data["upstream"], data["config"],
                        data["manifest"], data["checkpoint"],
                        attempt / "adapter", args.steps, resume,
                        valid_ratio=args.valid_ratio)
    plan["command"] = cmd
    plan["output"] = str(attempt)
    save_json(attempt / "plan.json", plan)
    (attempt / "upstream_config.yaml").write_bytes(data["config"].read_bytes())
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
    print(f"Starting LV-04 LoRA baseline; output: {attempt}", flush=True)
    code, reason = stream(cmd, cwd=data["upstream"],
                          log=attempt / "train.log", env=env, timeout=args.timeout)
    result = dict(plan, status="TRAIN_FAILED", training_exit_code=code,
                  training_exit_reason=reason)
    if code:
        save_json(attempt / "result.json", result)
        print(f"Training failed ({reason}, exit={code}); inspect {attempt / 'train.log'}")
        return 2
    try:
        result["adapter"] = validate_adapter(attempt / "adapter" / "checkpoint_final")
    except ValueError as err:
        result["status"] = "TRAIN_EXITED_ZERO_ADAPTER_MISSING"
        result["reason"] = str(err)
        save_json(attempt / "result.json", result)
        print(str(err))
        return 2
    # Separate external three evaluation clips remain untouched.
    _, observed_sha = verify_sources(workspace, Path(data["training_input_csv"]))
    if observed_sha != data["source_sha"]:
        raise RuntimeError("Source WAV SHA set changed during training")
    if hash_file(data["manifest"]) != data["train_manifest_sha256"]:
        raise RuntimeError("Official training manifest changed during training")
    result["status"] = "TRAIN_ADAPTER_VALID_INFERENCE_PENDING"
    save_json(attempt / "result.json", result)
    infer = infer_command(data["root"], data["upstream"], data["python"],
                          data["checkpoint"], attempt / "adapter" / "checkpoint_final",
                          data["manifest"], attempt / "inference_smoke.wav")
    result["infer_command"] = infer
    print("Adapter verified. Running non-explicit inference smoke test.", flush=True)
    inf_code, inf_reason = stream(infer, cwd=data["upstream"],
                                  log=attempt / "infer.log", env=env,
                                  timeout=args.infer_timeout)
    wav = attempt / "inference_smoke.wav"
    result["inference_exit_code"] = inf_code
    result["inference_exit_reason"] = inf_reason
    result["inference_wav"] = str(wav)
    if inf_code == 0 and wav.is_file() and wav.stat().st_size > 44:
        result["status"] = "PASS_LV04_ADAPTER_AND_INFER_SMOKE_NOT_QUALITY_VALIDATED"
        result["inference_sha256"] = hash_file(wav)
    else:
        result["status"] = "TRAIN_ADAPTER_VALID_INFERENCE_FAILED"
    save_json(attempt / "result.json", result)
    print(f"{result['status']}\nResult: {attempt / 'result.json'}", flush=True)
    return 0 if result["status"].startswith("PASS_LV04") else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"LV-04 BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
