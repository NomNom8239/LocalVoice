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
import re
import subprocess
import shutil
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
VALID_STYLES = {"normal", "whisper", "breath", "panting", "groan", "laugh", "emotion", "other"}
SPECIAL_STYLES = VALID_STYLES - {"normal", "other"}
ERROR_STATUSES = {"asr_failed", "error_review_audio"}
STYLE_ALIASES = {
    "laughter": "laugh",
    "laughing": "laugh",
    "笑い": "laugh",
    "笑い声": "laugh",
}


def normalize_style(value: str) -> str:
    style = value.strip().lower()
    return STYLE_ALIASES.get(style, style)



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
    try:
        process = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "format=duration:stream=sample_rate,channels",
         "-of", "json", str(path)], text=True, capture_output=True,
        encoding="utf-8", check=False,
        )
    except FileNotFoundError:
        process = subprocess.CompletedProcess(args=[], returncode=127, stdout="", stderr="")
    if process.returncode == 0:
        streams = json.loads(process.stdout).get("streams") or []
        if streams:
            s = streams[0]
            duration = (json.loads(process.stdout).get("format") or {}).get("duration")
            if duration not in (None, "N/A"):
                return float(duration), int(s["sample_rate"]), int(s["channels"])
    # Fallback for ordinary PCM files.
    with wave.open(str(path), "rb") as wav:
        return (wav.getnframes() / wav.getframerate(),
                wav.getframerate(), wav.getnchannels())


def paths(args) -> tuple[Path, Path]:
    if not args.profile or args.profile in {".", ".."} or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in args.profile):
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
        if clip["scan_flag"] != "ok" and not (args.include_short and clip["scan_flag"] == "short"):
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
    # ASR never approves or edits human-verified text. Merge hints automatically
    # so 'status' reflects new results without a separate command.
    merge(None, None, out)
    print(f"ASR suggestion file: {saved} ({len(selected)} attempted)")


def merge(args, source: Path, out: Path) -> None:
    del args, source
    lookup = {r["clip_id"]: r for r in read_csv(out / "asr_suggestions.csv")}
    rows = read_csv(out / "review.csv")
    updated = 0
    for row in rows:
        candidate = lookup.get(row["clip_id"])
        if not candidate or candidate.get("sha256") != row.get("sha256"):
            continue
        if candidate.get("asr_status") not in {
            "suggested", "no_detected_text_review_audio"
        }:
            continue

        new_text = candidate.get("asr_suggestion", "")
        new_status = candidate["asr_status"]
        # Preserve nonempty review-side suggestions. Human-confirmed 'text',
        # speaker/quality, and decisions are never changed by ASR.
        if row.get("asr_suggestion") and row["asr_suggestion"] != new_text:
            continue
        if (row.get("asr_suggestion", "") == new_text
                and row.get("asr_status", "") == new_status):
            continue

        row["asr_suggestion"] = new_text
        row["asr_status"] = new_status
        updated += 1

    if updated:
        write_csv(out / "review.csv", COLUMNS, rows)
    print(
        f"Merged {updated} new/changed suggestions; "
        "manual text and approval were NOT altered"
    )


def status(args, source: Path, out: Path) -> None:
    del args, source
    rows = read_csv(out / "review.csv")
    suggestion_path = out / "asr_suggestions.csv"
    suggestions = read_csv(suggestion_path) if suggestion_path.is_file() else []
    asr_counts = Counter(r.get("asr_status", "") for r in suggestions)
    actual_errors = [
        r for r in suggestions
        if r.get("asr_status") in ERROR_STATUSES and r.get("error_detail", "").strip()
    ]
    old_errors = sum(
        1 for r in suggestions
        if r.get("asr_status") in ERROR_STATUSES and not r.get("error_detail", "").strip()
    )
    report = {
        "workspace": str(out),
        "total": len(rows),
        "scan_flags": dict(Counter(r.get("scan_flag", "") for r in rows)),
        "decisions": dict(Counter(r.get("decision", "") for r in rows)),
        "review_asr_status": dict(Counter(r.get("asr_status", "") for r in rows)),
        "asr_suggestions": {
            "file": str(suggestion_path) if suggestion_path.is_file() else None,
            "total_attempted": len(suggestions),
            "statuses": dict(asr_counts),
            "old_errors_without_details": old_errors,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for row in actual_errors[:3]:
        print(f"ASR ERROR {row['clip_id']}: {row['error_detail']}")
    if old_errors:
        print(f"Old ASR failures without error details: {old_errors} (use -RetryErrors)")


def triage(args, source: Path, out: Path) -> None:
    """Write a prioritized, non-destructive listening queue.

    Labels are heuristic review priorities, not verified speech-style labels.
    All decisions and transcriptions stay in review.csv.
    """
    del args, source
    reviews = read_csv(out / "review.csv")
    suggestion_path = out / "asr_suggestions.csv"
    suggestions = {
        r["clip_id"]: r for r in read_csv(suggestion_path)
    } if suggestion_path.is_file() else {}
    output: list[dict[str, str | int]] = []
    counts: Counter[str] = Counter()
    for review in reviews:
        if review.get("decision", "").strip().lower() not in {"pending", "tagged", "needs_text"}:
            continue
        suggestion = suggestions.get(review["clip_id"], {})
        if suggestion and suggestion.get("sha256") != review.get("sha256"):
            raise ValueError(f"ASR hash mismatch: {review['clip_id']}")
        status = suggestion.get("asr_status") or review.get("asr_status", "")
        text = review.get("asr_suggestion") or suggestion.get("asr_suggestion", "")
        text = text.strip()
        flag = review.get("scan_flag", "")
        if review.get("decision", "").strip().lower() == "needs_text":
            rank, group, why = (6, "needs_transcript", "Voice/style verified; transcription still needs confirmation")
        elif review.get("decision", "").strip().lower() == "tagged":
            rank, group, why = (7, "tagged_nonverbal", "Tagged for later transcription/manifest review")
        elif flag in {"unreadable", "low_sample_rate"}:
            rank, group, why = (0, "invalid_audio", "Inspect or reject corrupted/unsupported audio")
        elif flag == "short":
            rank, group, why = (1, "short_audio", "Listen: short clip, ASR not run")
        elif status == "no_detected_text_review_audio":
            rank, group, why = (2, "no_detected_text", "Listen: could be silence, noise or nonverbal voice")
        elif len(text) < 5 or re.search(r"(.)\1{3,}", text):
            rank, group, why = (3, "expressive_or_unclear", "Listen: very short or repeated text; label manually")
        elif len(text) < 10:
            rank, group, why = (4, "short_transcript", "Check brief utterance and exact wording")
        else:
            rank, group, why = (5, "ordinary_candidate", "Verify wording, speaker and quality")
        counts[group] += 1
        output.append({
            "review_priority": rank,
            "review_group": group,
            "reason": why,
            "clip_id": review["clip_id"],
            "source_path": review.get("source_path", ""),
            "duration_sec": review.get("duration_sec", ""),
            "scan_flag": flag,
            "asr_status": status,
            "asr_suggestion": text,
            "decision": review.get("decision", ""),
        })
    output.sort(key=lambda r: (int(r["review_priority"]), str(r["clip_id"])))
    target = out / "triage.csv"
    write_csv(target,
              ("review_priority", "review_group", "reason", "clip_id",
               "source_path", "duration_sec", "scan_flag", "asr_status",
               "asr_suggestion", "decision"),
              output)
    print(f"Created read-only review queue: {target}")
    print(json.dumps({
        "pending": sum(r["decision"] == "pending" for r in output),
        "tagged_for_later": sum(r["decision"] == "tagged" for r in output),
        "needs_transcript": sum(r["decision"] == "needs_text" for r in output),
        "queue_total": len(output),
        "groups": dict(counts)
    }, ensure_ascii=False, indent=2))
    print("Edit only review.csv to approve/reject; triage.csv is regenerated.")


def review_origin(row: dict[str, str]) -> str:
    declared = (row.get("source_kind") or "").strip().lower()
    if declared in {"my_voice", "review_emotion", "review_approved"}:
        return declared
    name = Path(row.get("source_path", "")).name.lower()
    for kind in ("review_emotion", "review_approved", "my_voice"):
        if f"__{kind}__" in name:
            return kind
    return "unknown"


def review(args, source: Path, out: Path) -> None:
    """Listen, verify and persist training decisions one clip at a time."""
    review_path = out / "review.csv"
    triage_path = out / "triage.csv"
    if not triage_path.is_file():
        raise FileNotFoundError(f"Run 'triage' first: {triage_path}")
    if not args.no_play and shutil.which(args.player) is None:
        raise RuntimeError(f"Audio player not found: {args.player}")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")

    reviews = read_csv(review_path)
    indices = {r["clip_id"]: index for index, r in enumerate(reviews)}
    if len(indices) != len(reviews):
        raise ValueError("Duplicate clip IDs in review.csv")
    queue = read_csv(triage_path)
    pending: list[tuple[int, dict[str, str]]] = []
    for item in queue:
        index = indices.get(item["clip_id"])
        if index is None:
            raise ValueError(f"Unknown clip in triage: {item['clip_id']}")
        entry = reviews[index]
        if item.get("source_path") != entry.get("source_path"):
            raise ValueError(f"Stale triage path: {item['clip_id']}")
        decision = entry.get("decision", "").strip().lower()
        if decision not in {"pending", "tagged", "needs_text"}:
            continue
        if decision == "tagged" and not args.include_tagged:
            continue
        if decision == "needs_text" and args.group != "needs_transcript":
            continue
        origin = review_origin(entry)
        if args.kind == "emotion" and origin != "review_emotion":
            continue
        if args.kind == "other" and origin == "review_emotion":
            continue
        if args.group and item.get("review_group") != args.group:
            continue
        pending.append((index, item))
    pending = pending[:args.limit]
    if not pending:
        print("No matching pending clips. Refresh triage if needed.")
        return

    # Preserve manually added review columns rather than dropping them.
    columns = list(dict.fromkeys(
        list(COLUMNS) + [key for row in reviews for key in row.keys()]
    ))
    done = 0
    for number, (index, item) in enumerate(pending, start=1):
        entry = reviews[index]
        wav = Path(entry["source_path"]).resolve()
        if source not in wav.parents or not wav.is_file() or digest(wav) != entry["sha256"]:
            raise ValueError(f"WAV missing/changed/outside input directory: {wav}")
        origin = review_origin(entry)
        print("\n" + "-" * 72)
        print(f"[{number}/{len(pending)}] {item['review_group']} | {origin} | {item['duration_sec']}s")
        print(wav)
        hint = (entry.get("asr_suggestion") or item.get("asr_suggestion") or "").strip()
        print(f"ASR hint: {hint or '(none)'}")
        while True:
            if not args.no_play:
                subprocess.run([args.player, "-nodisp", "-autoexit", "-loglevel",
                                "error", str(wav)], check=False)
            command = input("[a] approve / [t] tag-only / [n] reject / [r] replay / [s] skip / [q] quit > ").strip().lower()
            if command == "q":
                print(f"Stopped; {done} changes saved.")
                return
            if command == "s":
                break
            if command == "r":
                continue
            if command not in {"a", "t", "n"}:
                print("Unknown command")
                continue

            if command == "n":
                entry["decision"] = "rejected"
                entry["notes"] = input("Reason (optional): ").strip()
            else:
                options = ", ".join(sorted(VALID_STYLES))
                style = normalize_style(input(f"Style ({options}): "))
                if style not in VALID_STYLES:
                    print("Unrecognized style; no changes saved for this clip.")
                    continue
                if command == "t":
                    entry["style"] = style
                    entry["notes"] = input("Notes (optional): ").strip()
                    entry["decision"] = "tagged"
                    print("Tagged for later review; not approved for training.")
                else:
                    if hint:
                        print("ASR text is unverified. Type corrected text or '=' to explicitly confirm the suggestion.")
                    print("Press Enter to save the speaker/style check and defer transcription.")
                    prompt = f"Verified transcription [{hint}]: " if hint else "Verified transcription (optional now): "
                    typed = input(prompt).strip()
                    verified = hint if typed == "=" and hint else (typed if typed != "=" else "")
                    caption = ""
                    if verified:
                        caption = input("Verified caption (required for special styles; Enter for normal): ").strip()
                        if style in SPECIAL_STYLES and not caption:
                            print("Caption is required for special delivery; nothing saved.")
                            continue
                    if input("Confirm speaker and quality good? [y/N]: ").strip().lower() != "y":
                        print("Not confirmed; no approval saved.")
                        continue
                    entry.update({
                        "text": verified,
                        "caption": caption if verified else "",
                        "style": style,
                        "speaker_ok": "yes",
                        "quality": "good",
                        "decision": "approved" if verified else "needs_text",
                    })
                    if not verified:
                        print("Saved voice/style confirmation as needs_text; NOT export-ready.")
            write_csv(review_path, columns, reviews)
            done += 1
            print(f"Saved: {entry['decision']} / {entry.get('style', '')} ({done} in this session)")
            break
    print(f"Review session complete: {done} changes saved.")
    print("Run 'triage' again to refresh the pending listening queue.")


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
        style = normalize_style(r.get("style", ""))
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
    for command in ("scan", "status", "merge", "triage"):
        subs.add_parser(command)
    p = subs.add_parser("asr")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--include-short", action="store_true", help="Also attempt ASR on short clips; results still require listening")
    p = subs.add_parser("review")
    p.add_argument("--kind", choices=("emotion", "other", "all"), default="emotion")
    p.add_argument("--group", choices=(
        "short_audio", "no_detected_text", "expressive_or_unclear",
        "short_transcript", "ordinary_candidate", "invalid_audio", "tagged_nonverbal", "needs_transcript"
    ))
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--include-tagged", action="store_true")
    p.add_argument("--player", default="ffplay")
    p = subs.add_parser("export")
    p.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        actions = {"scan": scan, "asr": asr, "merge": merge,
                   "status": status, "triage": triage, "review": review, "export": export}
        actions[args.action](args, source, out)
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
