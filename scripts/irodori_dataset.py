"""Prepare a reviewed Irodori TTS dataset from existing LocalVoice audio.

Only reads training_audio source WAVs. Generated tables live outside the source.
Human approval is mandatory; ASR output never becomes training text automatically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = (
    "clip_id", "source_path", "relative_path", "sha256", "duration_sec",
    "sample_rate", "channels", "size_bytes", "scan_flag", "source_kind",
    "asr_suggestion", "asr_status", "text", "caption", "style",
    "speaker_ok", "quality", "decision", "notes"
)
ASR_COLUMNS = ("clip_id", "sha256", "asr_status", "asr_suggestion",
               "language", "error_detail")
VALID_STYLES = {"normal", "whisper", "breath", "panting", "groan", "emotion", "other"}
SPECIAL_STYLES = VALID_STYLES - {"normal", "other"}
ERROR_STATUSES = {"asr_failed", "error_review_audio"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def audio_info(path: Path) -> tuple[float, int, int]:
    # ffprobe supports the float PCM produced by the LocalVoice pipeline.
    process = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=duration,sample_rate,channels",
         "-of", "json", str(path)], text=True, capture_output=True,
        encoding="utf-8", check=False,
    )
    if process.returncode == 0:
        streams = json.loads(process.stdout).get("streams") or []
        if streams:
            s = streams[0]
            duration = s.get("duration")
            if duration not in (None, "N/A"):
                return float(duration), int(s["sample_rate"]), int(s["channels"])
    # Fallback for ordinary PCM files.
    with wave.open(str(path), "rb") as wav:
        return (wav.getnframes() / wav.getframerate(),
                wav.getframerate(), wav.getnchannels())


def paths(args) -> tuple[Path, Path]:
    if not args.profile or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in args.profile):
        raise ValueError("Invalid --profile")
    source = (ROOT / "data" / "training_audio" / args.profile / "audio").resolve()
    legacy = ROOT / "Irodori-TTS" / "outputs" / "localvoice_lora_dataset"
    if args.workspace:
        out = Path(args.workspace).expanduser().resolve()
    elif (legacy / "review.csv").is_file():
        out = legacy.resolve()  # Reuse already reviewed data without migration.
    else:
        out = (ROOT / "data" / "irodori_lora" / args.profile).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Dataset workspace must be separate from the WAV source")
    return source, out


def provenance(profile: str) -> dict[str, str]:
    result: dict[str, str] = {}
    path = ROOT / "data" / "training_audio" / profile / "manifest.tsv"
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("output"):
                    result[str(Path(row["output"]).resolve()).casefold()] = row.get("source_kind", "")
    return result


def scan(args, source: Path, out: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Training audio not found: {source}")
    if (out / "review.csv").exists():
        raise FileExistsError(f"Review already exists; use status: {out / 'review.csv'}")
    wavs = sorted(source.rglob("*.wav"))
    if not wavs:
        raise ValueError("No WAV files found")
    origins = provenance(args.profile)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for wav in wavs:
        wav = wav.resolve()
        if source not in wav.parents:
            raise ValueError(f"Source path escaped training directory: {wav}")
        file_hash = digest(wav)
        if file_hash in seen:
            continue
        seen.add(file_hash)
        try:
            duration, rate, channels = audio_info(wav)
            flag = ("short" if duration < 1.0 else
                    "long" if duration > 25.0 else
                    "low_sample_rate" if rate < 16000 else "ok")
        except (OSError, ValueError, KeyError, json.JSONDecodeError, wave.Error):
            duration, rate, channels, flag = 0.0, 0, 0, "unreadable"
        rows.append({
            "clip_id": file_hash[:20], "source_path": str(wav),
            "relative_path": str(wav.relative_to(source)),
            "sha256": file_hash, "duration_sec": f"{duration:.4f}",
            "sample_rate": rate, "channels": channels,
            "size_bytes": wav.stat().st_size, "scan_flag": flag,
            "source_kind": origins.get(str(wav).casefold(), ""),
            "decision": "pending",
        })
    write_csv(out / "inventory.csv", COLUMNS[:10] + ("source_kind",), rows)
    write_csv(out / "review.csv", COLUMNS, rows)
    print(f"Scanned {len(rows)} WAVs; flags={dict(Counter(r['scan_flag'] for r in rows))}")
    print(f"Review sheet: {out / 'review.csv'}")


def asr(args, source: Path, out: Path) -> None:
    del source
    inventory = read_csv(out / "inventory.csv")
    saved = out / "asr_suggestions.csv"
    old = read_csv(saved) if saved.is_file() else []
    by_id = {r["clip_id"]: r for r in old}
    if len(by_id) != len(old):
        raise ValueError("Duplicate ASR clip IDs")
    selected = []
    for clip in inventory:
        previous = by_id.get(clip["clip_id"])
        if clip["scan_flag"] != "ok":
            continue
        if previous is None or (args.retry_errors and previous.get("asr_status") in ERROR_STATUSES):
            selected.append(clip)
    selected = selected[:args.limit] if args.limit else selected
    if not selected:
        print("No ASR candidates left")
        return
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is missing in the active ASR environment") from exc
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    failures = 0
    for item in selected:
        try:
            wav = Path(item["source_path"]).resolve()
            if not wav.is_file() or digest(wav) != item["sha256"]:
                raise ValueError("Source WAV missing or changed since scan")
            segments, info = model.transcribe(
                str(wav), language="ja", beam_size=5, vad_filter=True,
                condition_on_previous_text=False,
            )
            text = "".join(s.text.strip() for s in segments).strip()
            row = {"clip_id": item["clip_id"], "sha256": item["sha256"],
                   "asr_status": "suggested" if text else "no_detected_text_review_audio",
                   "asr_suggestion": text, "language": info.language, "error_detail": ""}
            failures = 0
        except Exception as exc:
            row = {"clip_id": item["clip_id"], "sha256": item["sha256"],
                   "asr_status": "asr_failed", "asr_suggestion": "",
                   "language": "", "error_detail": f"{type(exc).__name__}: {exc}"[:1200]}
            failures += 1
            print(f"ASR error {row['clip_id']}: {row['error_detail']}", file=sys.stderr)
        by_id[item["clip_id"]] = row
        # Checkpoint each file; do not lose previously successful transcripts.
        write_csv(saved, ASR_COLUMNS, list(by_id.values()))
        if failures >= 3:
            raise RuntimeError("Three consecutive ASR errors; inspect error_detail")
    print(f"ASR suggestion file: {saved} ({len(selected)} attempted)")


def merge(args, source: Path, out: Path) -> None:
    del args, source
    lookup = {r["clip_id"]: r for r in read_csv(out / "asr_suggestions.csv")}
    rows = read_csv(out / "review.csv")
    updated = 0
    for row in rows:
        candidate = lookup.get(row["clip_id"])
        if (candidate and candidate["sha256"] == row["sha256"]
                and candidate["asr_status"] in {"suggested", "no_detected_text_review_audio"}
                and not row.get("asr_suggestion")):
            row["asr_suggestion"] = candidate["asr_suggestion"]
            row["asr_status"] = candidate["asr_status"]
            updated += 1
    write_csv(out / "review.csv", COLUMNS, rows)
    print(f"Merged {updated} suggestions; manual text and approval were NOT altered")


def status(args, source: Path, out: Path) -> None:
    del args, source
    rows = read_csv(out / "review.csv")
    report = {
        "workspace": str(out), "total": len(rows),
        "scan_flags": dict(Counter(r.get("scan_flag", "") for r in rows)),
        "decisions": dict(Counter(r.get("decision", "") for r in rows)),
        "asr": dict(Counter(r.get("asr_status", "") for r in rows)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    suggestion_path = out / "asr_suggestions.csv"
    if suggestion_path.exists():
        errors = [r for r in read_csv(suggestion_path) if r.get("asr_status") in ERROR_STATUSES]
        for row in errors[:3]:
            print(f"ASR ERROR {row['clip_id']}: {row.get('error_detail') or '(old record lacks details)'}")


def export(args, source: Path, out: Path) -> None:
    rows = read_csv(out / "review.csv")
    inv = {r["clip_id"]: r for r in read_csv(out / "inventory.csv")}
    chosen: list[dict[str, str]] = []
    reject: Counter[str] = Counter()
    for r in rows:
        if r.get("decision", "").strip().lower() != "approved":
            reject["not_approved"] += 1
            continue
        cid = r["clip_id"]
        ref = inv.get(cid)
        if ref is None or any(r.get(k) != ref.get(k) for k in
                              ("source_path", "sha256", "relative_path", "duration_sec")):
            raise ValueError(f"Inventory mismatch: {cid}")
        wav = Path(r["source_path"]).resolve()
        if source not in wav.parents or not wav.is_file() or digest(wav) != r["sha256"]:
            raise ValueError(f"Source mismatch/changed: {wav}")
        if r.get("speaker_ok", "").lower() != "yes" or r.get("quality", "").lower() != "good":
            reject["not_human_verified"] += 1
            continue
        style = r.get("style", "").lower().strip()
        text = r.get("text", "").strip()
        caption = r.get("caption", "").strip()
        if style not in VALID_STYLES or not text:
            reject["style_or_text_missing"] += 1
            continue
        if style in SPECIAL_STYLES and not caption:
            reject["style_caption_missing"] += 1
            continue
        if r["scan_flag"] in {"unreadable", "low_sample_rate"}:
            reject["invalid_audio"] += 1
            continue
        chosen.append({"audio": str(wav), "text": text,
                       "caption": caption, "speaker": args.profile})
    if not chosen:
        raise ValueError(f"No human-approved training items: {dict(reject)}")
    target = out / "dataset_for_prepare_manifest.csv"
    if target.exists() and not args.replace:
        raise FileExistsError("Export already exists; use export --replace")
    write_csv(target, ("audio", "text", "caption", "speaker"), chosen)
    print(f"Exported {len(chosen)} reviewed clips to {target}")
    print("This is an intermediate CSV, not the Irodori latent manifest.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace", help="Existing dataset workspace (optional)")
    subs = parser.add_subparsers(dest="action", required=True)
    for command in ("scan", "status", "merge"):
        subs.add_parser(command)
    p = subs.add_parser("asr")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--retry-errors", action="store_true")
    p = subs.add_parser("export")
    p.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        actions = {"scan": scan, "asr": asr, "merge": merge,
                   "status": status, "export": export}
        actions[args.action](args, source, out)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
