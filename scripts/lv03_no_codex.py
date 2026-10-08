#!/usr/bin/env python3
"""LV-03 DACVAE recovery without Codex. Run from the LocalVoice repository root:

    .\.venv\Scripts\python.exe .\scripts\lv03_no_codex.py

Uses the existing Irodori-TTS/.venv, and only writes fresh lv03_dacvae_NNN
attempt directories. No audio, existing attempts or learned models are modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PARTS = ("Irodori-TTS", "outputs", "localvoice_lora_dataset")

# This runs inside the *existing* Irodori-TTS .venv; no new dependencies.
CHILD = r'''
import faulthandler
import os
import runpy
import sys
import time
from pathlib import Path

def stage(name):
    print("[LV03] " + time.strftime("%Y-%m-%d %H:%M:%S") + " " + name, flush=True)

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
upstream = Path(sys.argv[1]).resolve()
shared = Path(sys.argv[2]).resolve()
prepare = upstream / "prepare_manifest.py"
if not prepare.is_file():
    raise FileNotFoundError(prepare)
if os.name == "nt":
    if not shared.is_dir():
        raise FileNotFoundError("Missing FFmpeg 7 shared DLL directory: " + str(shared))
    dll_guard = os.add_dll_directory(str(shared))
    os.environ["PATH"] = str(shared) + os.pathsep + os.environ.get("PATH", "")
stage("importing irodori_tts.codec")
sys.path.insert(0, str(upstream))
from irodori_tts.codec import DACVAECodec
original_load = DACVAECodec.load
@classmethod
def observed_load(cls, *args, **kwargs):
    stage("DACVAECodec.load START (cache/model/CUDA/dummy encode)")
    loaded = original_load(*args, **kwargs)
    stage("DACVAECodec.load SUCCESS")
    return loaded
DACVAECodec.load = observed_load
stage("starting official prepare_manifest.py")
sys.argv = [str(prepare), *sys.argv[3:]]
runpy.run_path(str(prepare), run_name="__main__")
stage("official prepare_manifest.py completed")
'''

CHECK = r'''
import json
import sys
from pathlib import Path
import torch
from irodori_tts.dataset import LatentTextDataset

manifest = Path(sys.argv[1]).resolve()
count = int(sys.argv[2])
items = [json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
if len(items) != count:
    raise RuntimeError("manifest row count: " + str(len(items)) + " expected: " + str(count))
seen = set()
for item in items:
    if not item.get("text") or not item.get("speaker_id"):
        raise RuntimeError("Missing text or speaker_id")
    if not isinstance(item.get("num_frames"), int) or item["num_frames"] < 1:
        raise RuntimeError("Invalid num_frames")
    if item.get("caption") is not None and not isinstance(item["caption"], str):
        raise RuntimeError("Invalid caption")
    path = Path(item["latent_path"])
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file() or path in seen:
        raise RuntimeError("Missing or duplicate latent: " + str(path))
    seen.add(path)
    latent = torch.load(path, map_location="cpu", weights_only=True)
    if (latent.ndim != 2 or latent.shape[1] != 32 or
            latent.shape[0] != item["num_frames"] or
            not bool(torch.isfinite(latent).all())):
        raise RuntimeError("Invalid latent tensor: " + str(path))
dataset = LatentTextDataset(manifest, latent_dim=32, enable_caption_condition=True)
if len(dataset) != count:
    raise RuntimeError("Official dataset length mismatch")
for i in range(count):
    item = dataset[i]
    if item["num_frames"] < 1 or not bool(torch.isfinite(item["latent"]).all()):
        raise RuntimeError("Official dataset read failed at index " + str(i))
print(json.dumps({"status": "PASS", "latents": len(seen), "dataset_reads": len(dataset)}), flush=True)
'''


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    obj = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            obj.update(block)
    return obj.hexdigest()


def source_path(row: dict[str, str]) -> Path:
    name = (row.get("audio") or row.get("source_path") or "").strip()
    if not name:
        raise ValueError("Missing source path in approved CSV")
    return Path(name).resolve()


def verify_sources(workspace: Path) -> tuple[Path, dict[str, str]]:
    train_csv = workspace / "dataset_for_prepare_manifest_approved_train.csv"
    train = rows(train_csv)
    eval_rows = rows(workspace / "lv02_approved_evaluation.csv")
    inventory = rows(workspace / "inventory.csv")
    if (len(train), len(eval_rows), len(inventory)) != (8, 3, 1167):
        raise ValueError("Expected train=8, evaluation=3, inventory=1167")
    by_path = {str(Path(r["source_path"]).resolve()): r for r in inventory}
    if len(by_path) != len(inventory):
        raise ValueError("Duplicate inventory paths")
    ids, hashes, videos = {}, {}, {}
    all_sources = {}
    for label, group in (("train", train), ("eval", eval_rows)):
        ids[label], hashes[label], videos[label] = set(), set(), set()
        for row in group:
            path = source_path(row)
            if not path.is_file():
                raise FileNotFoundError(path)
            inv = by_path.get(str(path))
            if inv is None:
                raise ValueError("WAV is not in approved inventory: " + str(path))
            calculated = digest(path)
            if calculated != inv["sha256"] or (row.get("sha256") and calculated != row["sha256"]):
                raise ValueError("SHA-256 mismatch: " + str(path))
            if row.get("clip_id") and row["clip_id"] != inv["clip_id"]:
                raise ValueError("clip_id mismatch: " + str(path))
            if not (row.get("text") or "").strip():
                raise ValueError("Verified text missing: " + str(path))
            if label == "train" and not (row.get("speaker") or "").strip():
                raise ValueError("Training speaker missing: " + str(path))
            clip_id = inv["clip_id"]
            if clip_id in ids[label] or calculated in hashes[label]:
                raise ValueError("Duplicate within split: " + clip_id)
            ids[label].add(clip_id)
            hashes[label].add(calculated)
            videos[label].add(path.name.split("__", 1)[0])
            all_sources[clip_id] = calculated
    for key, partition in (("clip_id", ids), ("SHA-256", hashes), ("video", videos)):
        if partition["train"] & partition["eval"]:
            raise ValueError("Train/evaluation leakage: " + key)
    if len(all_sources) != 11:
        raise ValueError("Expected 11 distinct approved clip identities")
    return train_csv, all_sources


def new_attempt(workspace: Path) -> Path:
    for number in range(1, 1000):
        path = workspace / f"lv03_dacvae_{number:03d}"
        if not path.exists():
            path.mkdir(exist_ok=False)
            return path
    raise RuntimeError("No free attempt numbers")


def execute(command: list[str], *, cwd: Path, log: Path,
            env: dict[str, str], seconds: int) -> tuple[bool, str]:
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=str(cwd), env=env,
                                   stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - started > seconds:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return False, "timeout after " + str(seconds) + "s"
            time.sleep(2)
    return process.returncode == 0, "exit code " + str(process.returncode)


def tail(path: Path, n: int = 45) -> str:
    if not path.is_file():
        return "[log unavailable]"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])


def encode(root: Path, workspace: Path, csv_file: Path, py: Path,
           shared: Path, env: dict[str, str], count: int, timeout: int) -> tuple[bool, Path]:
    attempt = new_attempt(workspace)
    print(f"Encoding {count} approved clips in new attempt: {attempt}", flush=True)
    arguments = [
        "--dataset", "csv", "--split", "train",
        "--data-files", "train=" + str(csv_file),
        "--audio-column", "audio", "--text-column", "text",
        "--caption-column", "caption", "--speaker-column", "speaker",
        "--speaker-id-prefix", "LocalVoice-approved",
        "--output-manifest", str(attempt / "train_manifest.jsonl"),
        "--latent-dir", str(attempt / "latents"),
        "--device", "cuda", "--codec-deterministic-encode",
        "--prefetch", "0", "--flush-every", "1", "--log-every", "1",
        "--max-samples", str(count), "--no-progress",
    ]
    (attempt / "invocation.json").write_text(
        json.dumps({"parameters": arguments, "expected": count,
                    "normalization_db": -16, "ffmpeg_shared": str(shared)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    command = [str(py), "-u", "-c", CHILD, str(root / "Irodori-TTS"),
               str(shared), *arguments]
    okay, reason = execute(command, cwd=root / "Irodori-TTS",
                           log=attempt / "process.log", env=env, seconds=timeout)
    (attempt / "process_result.json").write_text(
        json.dumps({"success": okay, "reason": reason}, indent=2),
        encoding="utf-8")
    if not okay:
        print("DACVAE failed: " + reason + "\n" + tail(attempt / "process.log"), flush=True)
        return False, attempt
    manifest = attempt / "train_manifest.jsonl"
    if not manifest.is_file():
        print("DACVAE returned without manifest", flush=True)
        return False, attempt
    try:
        validation = subprocess.run(
            [str(py), "-u", "-c", CHECK, str(manifest), str(count)],
            cwd=str(root / "Irodori-TTS"), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180,
            check=False)
    except subprocess.TimeoutExpired as exc:
        (attempt / "validation.log").write_text("Validation timeout\n", encoding="utf-8")
        print("Validation timed out: " + str(exc), flush=True)
        return False, attempt
    (attempt / "validation.log").write_text(validation.stdout or "", encoding="utf-8")
    if validation.returncode != 0 or '"status": "PASS"' not in (validation.stdout or ""):
        print("Manifest/latent validation failed:\n" + tail(attempt / "validation.log"), flush=True)
        return False, attempt
    return True, attempt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ffmpeg-shared-bin", type=Path)
    parser.add_argument("--probe-timeout", type=int, default=300)
    parser.add_argument("--full-timeout", type=int, default=1200)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    if args.probe_timeout < 1 or args.full_timeout < 1:
        raise ValueError("Timeouts must be positive")
    root = args.root.resolve()
    workspace = root.joinpath(*WORKSPACE_PARTS)
    py = root / "Irodori-TTS" / ".venv" / "Scripts" / "python.exe"
    upstream = root / "Irodori-TTS" / "prepare_manifest.py"
    shared = (args.ffmpeg_shared_bin or root / "Irodori-TTS" / ".localvoice-toolchain"
              / "ffmpeg-7.0.2-full-shared" / "bin").resolve()
    if not py.is_file() or not upstream.is_file() or (os.name == "nt" and not shared.is_dir()):
        raise FileNotFoundError("Existing Irodori .venv/prepare_manifest/FFmpeg7 shared DLL missing")
    train_csv, initial = verify_sources(workspace)
    print("PASS: approved train8/eval3; identity, WAV SHA, source video disjoint", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"  # Reuse cached codec weights; never stall on a download.
    good, diagnostic = encode(root, workspace, train_csv, py, shared, env, 1, args.probe_timeout)
    if not good:
        print("LV-03 NOT DONE. Diagnostic: " + str(diagnostic), flush=True)
        return 2
    print("PASS: one-clip codec and official dataset read", flush=True)
    if args.diagnose_only:
        return 0
    good, full = encode(root, workspace, train_csv, py, shared, env, 8, args.full_timeout)
    if not good:
        print("LV-03 NOT DONE. Full attempt: " + str(full), flush=True)
        return 2
    _, after = verify_sources(workspace)
    if initial != after:
        raise RuntimeError("Source WAVs changed during run")
    result = {"status": "PASS_LV03_DACVAE_AND_DATASET",
              "train_rows": 8, "evaluation_rows": 3,
              "manifest": str(full / "train_manifest.jsonl"),
              "input_sha256_by_clip": initial, "training_started": False}
    report_path = full / "lv03_final_result.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("PASS: LV-03 manifest8 / verified latent8 / dataset reads8", flush=True)
    print("Result: " + str(report_path), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print("LV-03 BLOCKED: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        raise SystemExit(2)
