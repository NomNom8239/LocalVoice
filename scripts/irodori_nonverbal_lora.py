"""Stage a controlled Irodori nonverbal LoRA feasibility run, without training by default.

The six-item DACVAE smoke dataset is stratified 2 train + 1 holdout per
nonverbal cue. It is NOT a meaningful assessment of model quality by itself.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from .irodori_dataset import ROOT, paths
    from .irodori_nonverbal_manifest import validate_sources, validate_result, environment
else:
    from irodori_dataset import ROOT, paths
    from irodori_nonverbal_manifest import validate_sources, validate_result, environment


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def group_holdout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        text = row.get("text", "")
        if not text or not row.get("caption") or not row.get("speaker_id"):
            raise ValueError("Incomplete latent manifest conditioning")
        groups[text].append(row)
    if len(groups) != 2 or any(len(v) != 3 for v in groups.values()):
        raise ValueError(
            "Feasibility pilot requires exactly two cues with three clips each; "
            "do not silently mix styles or change the experiment."
        )
    train, holdout = [], []
    for cue in sorted(groups):
        train.extend(groups[cue][0:2])
        holdout.extend(groups[cue][2:])
    return train, holdout


def fresh_attempt(pilot: Path) -> Path:
    for n in range(1, 100):
        trial = pilot / f"lora_attempt_{n:03d}"
        if not trial.exists():
            return trial
    raise RuntimeError("No free LoRA attempt folder")


def command(python: Path, script: Path, config: Path, manifest: Path,
            output: Path, checkpoint: Path, steps: int) -> list[str]:
    return [
        str(python), str(script),
        "--config", str(config),
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--init-checkpoint", str(checkpoint),
        "--device", "cuda",
        "--lora", "--max-steps", str(steps),
        "--batch-size", "1",
        "--gradient-accumulation-steps", "1",
        "--num-workers", "0",
        "--optimizer", "adamw",
        "--lr-scheduler", "none",
        "--warmup-steps", "0",
        "--save-every", str(steps),
        "--log-every", "1",
        "--checkpoint-best-n", "1",
        "--valid-ratio", "0",
        "--valid-every", "0",
        "--gradient-checkpointing",
        "--no-compile-model",
        "--no-wandb",
    ]


def plan(args: argparse.Namespace, source: Path, out: Path, *,
         root: Path = ROOT, runner=subprocess.run) -> dict:
    if not 1 <= args.steps <= 100:
        raise ValueError("--steps must be 1..100 for an experimental run")
    validated = validate_sources(source, out, args.profile)
    pilot = out / "nonverbal_pilot"
    codec = pilot / args.codec_attempt
    result_path = codec / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError("A successful DACVAE codec_attempt is required")
    evidence = json.loads(result_path.read_text(encoding="utf-8"))
    if evidence.get("status") != "PASS_DACVAE_MANIFEST_ONLY":
        raise ValueError("DACVAE manifest experiment has not passed")
    manifest_path = (codec / "train_manifest.jsonl").resolve()
    latent_dir = (codec / "latents").resolve()
    validate_result(manifest_path, latent_dir, len(validated))
    manifest = load_jsonl(manifest_path)
    if [x["text"] for x in manifest] != [x["text"] for x in validated]:
        raise ValueError("Latent manifest text differs from source pilot")
    train, holdout = group_holdout(manifest)
    upstream, python = environment(root)
    train_py = upstream / "train.py"
    config = upstream / "configs" / "train_v4_small_lora.yaml"
    if not train_py.is_file() or not config.is_file():
        raise FileNotFoundError("Irodori v4-Small LoRA train.py/config missing")
    config_text = config.read_text(encoding="utf-8")
    if ("lora_enabled: true" not in config_text
            or "text_tokenizer_repo: sbintuitions/modernbert-ja-310m" not in config_text):
        raise ValueError("Unexpected Irodori LoRA model configuration")
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint else (upstream / "model.safetensors").resolve()
    )
    checkpoint_exists = checkpoint.is_file()
    if args.run and not checkpoint_exists:
        raise FileNotFoundError(
            f"Full-precision v4.1-Small checkpoint not found: {checkpoint}. "
            "Supply --checkpoint pointing at an unquantized model.safetensors."
        )
    attempt = fresh_attempt(pilot)
    train_manifest = attempt / "train.jsonl"
    output = attempt / "adapter"
    cmd = command(python, train_py, config, train_manifest,
                  output, checkpoint, args.steps)
    report = {
        "status": "PLAN_ONLY" if not args.run else "RUNNING_NOT_VALIDATED",
        "train_clips": len(train), "holdout_clips": len(holdout),
        "by_style_train": sorted([x["text"] for x in train]),
        "by_style_holdout": sorted([x["text"] for x in holdout]),
        "checkpoint": str(checkpoint), "checkpoint_exists": checkpoint_exists,
        "max_steps": args.steps, "output": str(attempt),
        "lora_training": "NOT_RUN" if not args.run else "STARTED",
        "important": "Tiny feasibility run only; no generalization or voice-quality claim.",
        "command": cmd,
    }
    if not args.run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    attempt.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("train.jsonl", train), ("holdout.jsonl", holdout)):
        with (attempt / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copy2(config, attempt / "upstream_config.yaml")
    (attempt / "plan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    process = runner(cmd, cwd=str(upstream), text=True, check=False)
    if process.returncode != 0:
        report["status"] = "TRAIN_FAILED"
        report["exit_code"] = process.returncode
    else:
        report["status"] = "TRAIN_COMMAND_EXITED_ZERO_NOT_QUALITY_VALIDATED"
    (attempt / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if process.returncode:
        raise RuntimeError(f"Irodori LoRA pilot training failed (exit {process.returncode})")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--codec-attempt", default="codec_attempt_001")
    parser.add_argument("--checkpoint", help="Unquantized Irodori v4.1-Small model.safetensors")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--run", action="store_true",
                        help="Explicit opt-in: actually train; otherwise PLAN_ONLY")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        plan(args, source, out)
        return 0
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
