#!/usr/bin/env python3
r"""LV-05 paired voice comparison, Codex-free, without retraining.

Run from LocalVoice root:
    .\.venv\Scripts\python.exe .\scripts\lv05_voice_ab.py
    .\.venv\Scripts\python.exe .\scripts\lv05_voice_ab.py --run

Four WAVs: base/LoRA with ONE reference latent and with FOUR reference
latents. Each matched pair uses identical seed, prompt, and sampler.
Never labels synthesized voice as matching the source automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lv04_no_codex import (
    hash_file,
    parse_manifest,
    prepare,
    stream,
    validate_adapter,
)

ROOT = Path(__file__).resolve().parents[1]
LV04_DIR = "lv04_lora_001"
REFERENCE_COUNT = 4
DEFAULT_TEXT = "おはようございます。今日はよろしくお願いします。"


def verified_adapter(workspace: Path, expected_manifest: Path) -> tuple[Path, dict]:
    attempt = workspace / LV04_DIR
    result_path = attempt / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("status")
            != "PASS_LV04_ADAPTER_AND_INFER_SMOKE_NOT_QUALITY_VALIDATED"
            or result.get("training_exit_code") != 0
            or result.get("inference_exit_code") != 0):
        raise ValueError("LV-04 completed adapter/inference evidence missing")
    recorded_manifest = Path(result.get("manifest", "")).resolve()
    if recorded_manifest != expected_manifest.resolve():
        raise ValueError("LV-04 training manifest differs from LV-05 input")
    adapter = attempt / "adapter" / "checkpoint_final"
    evidence = validate_adapter(adapter)
    expected = result.get("adapter", {})
    if (evidence["adapter_sha256"] != expected.get("adapter_sha256")
            or evidence["adapter_config_sha256"]
            != expected.get("adapter_config_sha256")):
        raise ValueError("LV-04 adapter SHA differs from completion evidence")
    return adapter, evidence


def latent_references(manifest: Path) -> tuple[Path, list[Path]]:
    rows = parse_manifest(manifest)
    paths = []
    for row in rows[:REFERENCE_COUNT]:
        name = Path(str(row["latent_path"]))
        paths.append((name if name.is_absolute() else
                      manifest.parent / name).resolve())
    if len(set(paths)) != REFERENCE_COUNT or not all(p.is_file() for p in paths):
        raise ValueError("Expected four distinct approved reference latents")
    return paths[0], paths


def fresh_attempt(workspace: Path, *, create: bool) -> Path:
    for index in range(1, 1000):
        target = workspace / f"lv05_voice_ab_{index:03d}"
        if not target.exists():
            if create:
                target.mkdir(exist_ok=False)
            return target
    raise RuntimeError("No unused LV-05 AB output directory")


def command(python: Path, root: Path, upstream: Path, checkpoint: Path,
            adapter: Path | None, ref_latents: list[Path], text: str,
            output_wav: Path, *, seed: int, steps: int,
            seconds: float | None = None) -> list[str]:
    if not ref_latents or steps < 1:
        raise ValueError("Invalid A/B reference or sampling step count")
    if seconds is not None and not 1.0 <= seconds <= 20.0:
        raise ValueError("Fixed duration must be within 1..20 seconds")
    if os.name == "nt":
        shared = (upstream / ".localvoice-toolchain" /
                  "ffmpeg-7.0.2-full-shared" / "bin")
        launcher = root / "scripts" / "irodori_codec_entry.py"
        if not shared.is_dir() or not launcher.is_file():
            raise FileNotFoundError("Official Windows FFmpeg7 launcher unavailable")
        cmd = [str(python), "-u", str(launcher), str(shared),
               str(upstream / "infer.py")]
    else:
        cmd = [str(python), "-u", str(upstream / "infer.py")]
    cmd += [
        "--checkpoint", str(checkpoint),
        "--text", text,
        "--output-wav", str(output_wav),
        "--num-steps", str(steps),
        "--seed", str(seed),
        "--model-device", "cuda",
        "--codec-device", "cuda",
        "--model-precision", "bf16",
        "--codec-precision", "fp32",
    ]
    if len(ref_latents) == 1:
        cmd += ["--ref-latent", str(ref_latents[0])]
    else:
        cmd += ["--ref-latents", *(str(p) for p in ref_latents)]
    if adapter is not None:
        cmd += ["--lora-adapter", str(adapter)]
    # Default: official duration prediction. Optional fixed seconds isolates
    # voice/timbre from the LoRA-trained duration predictor.
    if seconds is not None:
        cmd += ["--seconds", str(seconds)]
    return cmd


def evaluate(args: argparse.Namespace) -> int:
    if not args.text.strip() or not 1 <= args.steps <= 100:
        raise ValueError("Provide nonempty text and 1..100 sampling steps")
    if args.fixed_seconds is not None and not 1.0 <= args.fixed_seconds <= 20.0:
        raise ValueError("Fixed duration must be within 1..20 seconds")
    data = prepare(args)  # Rechecks LV-03 split, original WAV hashes, model SHA, upstream commit
    workspace = data["workspace"]
    adapter, adapter_info = verified_adapter(workspace, data["manifest"])
    ref_one, ref_four = latent_references(data["manifest"])
    target = fresh_attempt(workspace, create=args.run)
    jobs = []
    references = (("single", [ref_one]), ("multi4", ref_four))
    for reference, latent_paths in references:
        if args.ref_mode not in ("both", reference):
            continue
        for variant, selected_adapter in (("base", None), ("lora", adapter)):
            output = target / f"{reference}_{variant}.wav"
            jobs.append({
                "reference": reference,
                "variant": variant,
                "output": output,
                "cmd": command(
                    data["python"], data["root"], data["upstream"],
                    data["checkpoint"], selected_adapter, latent_paths,
                    args.text, output, seed=args.seed, steps=args.steps,
                    seconds=args.fixed_seconds,
                ),
                "reference_latent_sha256": [hash_file(p) for p in latent_paths],
            })
    report = {
        "status": "PLAN_ONLY" if not args.run else "INFERENCE_RUNNING",
        "experiment": "LV-05 reference-length vs LoRA matched-pairs",
        "output": str(target),
        "text": args.text,
        "seed": args.seed,
        "sampling_steps": args.steps,
        "duration_mode": ("official_predicted" if args.fixed_seconds is None
                          else "fixed_seconds"),
        "fixed_seconds": args.fixed_seconds,
        "reference_mode": args.ref_mode,
        "training_changed": False,
        "trained_approved_samples": 8,
        "independent_evaluation_samples_used_for_training": 0,
        "upstream_commit": data["upstream_revision"],
        "base_checkpoint_sha256": data["checkpoint_sha256"],
        "source_manifest_sha256": data["train_manifest_sha256"],
        "adapter": adapter_info,
        "reference_policy": "approved-training latents only",
        "quality_assessed": False,
        "jobs": [{
            "variant": job["variant"], "reference": job["reference"],
            "output": str(job["output"]),
            "reference_latent_sha256": job["reference_latent_sha256"],
        } for job in jobs],
    }
    if not args.run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    (target / "plan.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    results = []
    try:
        for job in jobs:
            label = f"{job['reference']}_{job['variant']}"
            print(f"[LV-05] generating {label}", flush=True)
            code, reason = stream(
                job["cmd"], cwd=data["upstream"],
                log=target / f"{label}.log",
                env=dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1"),
                timeout=args.timeout,
            )
            wav = job["output"]
            if code != 0 or not wav.is_file() or wav.stat().st_size <= 44:
                raise RuntimeError(f"{label} failed: exit={code}, reason={reason}; "
                                   f"see {target / (label + '.log')}")
            with wav.open("rb") as audio_file:
                header = audio_file.read(12)
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise RuntimeError(f"Invalid WAV header in {wav}")
            results.append({
                "reference": job["reference"],
                "variant": job["variant"],
                "wav": str(wav),
                "sha256": hash_file(wav),
                "bytes": wav.stat().st_size,
            })
        report["status"] = "PAIRED_WAVS_READY_NOT_QUALITY_VALIDATED"
    except Exception as exc:
        report["status"] = "BLOCK"
        report["error"] = str(exc)
        raise
    finally:
        report["generated"] = results
        (target / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"{report['status']}\n{target / 'result.json'}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--fixed-seconds", type=float, default=None,
                        help="Fix output duration in matched base/LoRA (1..20 s)")
    parser.add_argument("--ref-mode", choices=("both", "single", "multi4"),
                        default="both",
                        help="Which reference condition(s) to generate")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    return evaluate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"LV-05 BLOCK: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
