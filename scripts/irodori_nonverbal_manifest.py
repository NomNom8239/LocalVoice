"""Isolated, six-file Irodori DACVAE manifest smoke test; never start LoRA.

Uses the existing Irodori-TTS/.venv and pinned experimental JSONL.
Any failed/partial attempt remains separate; original data are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .irodori_dataset import ROOT, digest, paths, read_csv
    from .irodori_ffmpeg_runtime import resolve as resolve_ffmpeg
else:
    from irodori_dataset import ROOT, digest, paths, read_csv
    from irodori_ffmpeg_runtime import resolve as resolve_ffmpeg


def validate_sources(source: Path, out: Path, profile: str) -> list[dict]:
    pilot = out / "nonverbal_pilot"
    data_path = pilot / "hf_audio_dataset_hypothesis.jsonl"
    audit_path = pilot / "audit.csv"
    token_path = pilot / "tokenizer_audit.csv"
    if not all(x.is_file() for x in (data_path, audit_path, token_path)):
        raise FileNotFoundError("Run 'nonverbal-pilot' then 'nonverbal-check' first")
    originals = read_csv(out / "inventory.csv")
    originals_by_path = {r["source_path"]: r for r in originals}
    if len(originals_by_path) != len(originals):
        raise ValueError("Inventory duplicate WAV paths")
    chosen = [r for r in read_csv(audit_path)
              if r.get("status") == "pilot_hypothesis"]
    chosen_by_path = {r["source_path"]: r for r in chosen}
    tokens = read_csv(token_path)
    tokens_by_id = {r["clip_id"]: r for r in tokens}
    if (not chosen or len(chosen_by_path) != len(chosen)
            or len(tokens_by_id) != len(tokens)):
        raise ValueError("Pilot audit/tokenizer identity mismatch")
    with data_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if len(rows) != len(chosen) or len(rows) != len(tokens):
        raise ValueError("Pilot JSONL, audit and tokenizer counts do not match")
    seen = set()
    for row in rows:
        raw = row.get("audio", "")
        matched = chosen_by_path.get(raw)
        original = originals_by_path.get(raw)
        if (not matched or not original or raw in seen
                or row.get("text") != matched.get("experimental_text")
                or row.get("caption") != matched.get("caption")
                or row.get("speaker") != profile):
            raise ValueError("Stale or mismatched pilot data")
        seen.add(raw)
        token = tokens_by_id.get(matched["clip_id"])
        if (not token or token.get("status") != "pass"
                or token.get("emoji") != row["text"]
                or matched["sha256"] != original["sha256"]):
            raise ValueError("Tokenizer gate not satisfied for pilot row")
        wav = Path(raw).resolve()
        if (not wav.is_file() or source not in wav.parents
                or wav != Path(original["source_path"]).resolve()
                or digest(wav) != matched["sha256"]):
            raise ValueError("Pilot WAV was removed or changed")
    return rows


def environment(root: Path) -> tuple[Path, Path]:
    upstream = (root / "Irodori-TTS").resolve()
    script = upstream / "prepare_manifest.py"
    python = upstream / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not script.is_file():
        raise FileNotFoundError(f"Missing upstream prepare_manifest.py: {script}")
    if not python.is_file():
        raise FileNotFoundError(
            f"Irodori-TTS .venv not found: {python}. "
            "Set up Irodori's own environment; do not mix it with LocalVoice .venv."
        )
    return upstream, python


def command(python: Path, upstream: Path, data_file: Path,
            manifest: Path, latents: Path, device: str) -> list[str]:
    return [
        str(python), str(upstream / "prepare_manifest.py"),
        "--dataset", "json",
        "--split", "train",
        "--data-files", "train=" + str(data_file),
        "--audio-column", "audio",
        "--text-column", "text",
        "--caption-column", "caption",
        "--speaker-column", "speaker",
        "--speaker-id-prefix", "LocalVoice-nonverbal-pilot",
        "--output-manifest", str(manifest),
        "--latent-dir", str(latents),
        "--device", device,
        "--normalize-db", "none",
        "--codec-deterministic-encode",
        "--no-progress",
    ]


def validate_result(manifest: Path, latents: Path, expected: int) -> dict:
    if not manifest.is_file():
        raise RuntimeError("Irodori did not create the latent manifest")
    with manifest.open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if len(entries) != expected:
        raise RuntimeError(
            f"Irodori encoded {len(entries)}/{expected}; do not claim success"
        )
    targets = set()
    for row in entries:
        target = (manifest.parent / row.get("latent_path", "")).resolve()
        if (not row.get("text") or not isinstance(row.get("num_frames"), int)
                or row["num_frames"] < 1 or not row.get("speaker_id")
                or target in targets or not target.is_file()
                or (target != latents and latents not in target.parents)):
            raise RuntimeError("Invalid Irodori latent manifest row")
        targets.add(target)
    return {"latent_entries": len(entries), "verified_latents": len(targets)}


def version_info(python: Path, upstream: Path, runner) -> dict:
    """Inspect wheel metadata without importing the failing native extension."""
    probe = (
        "import json; from importlib.metadata import version; "
        "print(json.dumps({x:version(x) for x in ('torch','torchcodec')}))"
    )
    result = runner([str(python), "-c", probe], cwd=str(upstream),
                    text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Cannot inspect Irodori Torch/TorchCodec wheel versions: "
                           + (result.stderr or "")[-600:])
    try:
        versions = json.loads(result.stdout.strip())
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Irodori wheel version probe did not return JSON") from exc
    torch_v = versions.get("torch", "").split("+")[0]
    codec_v = versions.get("torchcodec", "").split("+")[0]
    if not (torch_v.startswith("2.10.") and codec_v.startswith("0.10.")):
        raise RuntimeError(
            f"Incompatible Torch/TorchCodec pair: {versions}. "
            "Upstream Irodori expects torch 2.10.x and torchcodec 0.10.x. "
            "Do not automatically replace the CUDA toolchain."
        )
    return versions


def native_probe(python: Path, upstream: Path, runner, shared: Path | None = None):
    imports = ("import torch, torchaudio, datasets, dacvae, torchcodec; "
               "print('cuda_available=' + str(torch.cuda.is_available()))")
    if shared is not None:
        # Python 3.8+ Windows requires adding DLL directories explicitly.
        preamble = (
            "import os; _ffmpeg_dll=os.add_dll_directory(" + repr(str(shared)) + "); "
            "os.environ['PATH']=" + repr(str(shared) + os.pathsep)
            + "+os.environ.get('PATH',''); "
        )
        imports = preamble + imports
    return runner([str(python), "-c", imports], cwd=str(upstream),
                  text=True, capture_output=True, check=False)


def smoke(args: argparse.Namespace, source: Path, out: Path, *,
          runner=subprocess.run, root: Path = ROOT,
          shared_resolver=resolve_ffmpeg,
          windows: bool | None = None) -> dict:
    rows = validate_sources(source, out, args.profile)
    upstream, python = environment(root)
    dataset = out / "nonverbal_pilot" / "hf_audio_dataset_hypothesis.jsonl"
    is_windows = os.name == "nt" if windows is None else windows
    shared = None

    # First try the existing Irodori environment untouched. Only bootstrap
    # a private FFmpeg shared runtime if the native TorchCodec DLL is missing.
    preflight = native_probe(python, upstream, runner)
    if preflight.returncode != 0:
        err = preflight.stderr or preflight.stdout
        if is_windows and ("torchcodec" in err.lower()
                           or "libtorchcodec" in err.lower()):
            versions = version_info(python, upstream, runner)
            shared = shared_resolver(upstream, getattr(args, "ffmpeg_shared_bin", None))
            preflight = native_probe(python, upstream, runner, shared)
            if preflight.returncode:
                raise RuntimeError(
                    "TorchCodec still cannot load using the isolated FFmpeg "
                    f"shared runtime {shared}. Torch wheels={versions}. "
                    "Check VC++ runtime and native DLL dependencies. "
                    + (preflight.stderr or preflight.stdout)[-1000:]
                )
        else:
            raise RuntimeError(
                "Irodori-TTS .venv native dependency preflight failed; "
                "no automatic PyTorch reinstall was attempted: "
                + err[-1200:]
            )
    if args.device == "cuda" and "cuda_available=True" not in preflight.stdout:
        raise RuntimeError("Irodori-TTS .venv cannot use CUDA")
    pilot = out / "nonverbal_pilot"
    # Each attempt is isolated, so an interrupted run cannot silently reuse
    # partial latents or clobber a previously successful manifest.
    attempt = None
    for n in range(1, 100):
        candidate = pilot / f"codec_attempt_{n:03d}"
        if not candidate.exists():
            attempt = candidate
            break
    if attempt is None:
        raise RuntimeError("No unused codec attempt directory")
    manifest = attempt / "train_manifest.jsonl"
    latents = attempt / "latents"
    args_list = command(python, upstream, dataset, manifest, latents, args.device)
    if shared is not None:
        # Add DLL search directory before importing torchcodec in the real
        # upstream process. This changes only the child interpreter.
        entry = Path(__file__).with_name("irodori_codec_entry.py").resolve()
        args_list = [str(python), str(entry), str(shared),
                     str(upstream / "prepare_manifest.py"), *args_list[2:]]
    if args.dry_run:
        result = {
            "status": "PREFLIGHT_ONLY", "samples": len(rows),
            "device": args.device, "proposed_output": str(attempt),
            "command": args_list, "lora_training": "NOT_RUN",
            "ffmpeg_shared_bin": str(shared) if shared else "existing_environment",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result

    attempt.mkdir(parents=True, exist_ok=False)
    try:
        process = runner(args_list, cwd=str(upstream), text=True, check=False)
        if process.returncode:
            raise RuntimeError(f"Irodori DACVAE prepare_manifest failed (exit {process.returncode})")
        verified = validate_result(manifest, latents, len(rows))
        report = {
            "status": "PASS_DACVAE_MANIFEST_ONLY",
            "source_rows": len(rows), **verified,
            "manifest": str(manifest), "latent_dir": str(latents),
            "lora_training": "NOT_RUN", "training_ready": False,
            "ffmpeg_shared_bin": str(shared) if shared else "existing_environment",
            "next_gate": "controlled LoRA pilot and inference comparison",
        }
    except Exception as exc:
        report = {
            "status": "BLOCK", "source_rows": len(rows),
            "reason": f"{type(exc).__name__}: {exc}",
            "partial_attempt": str(attempt),
            "lora_training": "NOT_RUN",
        }
        (attempt / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    (attempt / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ffmpeg-shared-bin", help="Existing FFmpeg 7 full-shared bin directory; skips portable bootstrap")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        smoke(args, source, out)
        return 0
    except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
