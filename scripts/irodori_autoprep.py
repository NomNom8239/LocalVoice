"""Automatically stage training candidates; human input only for ambiguous vocal style."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from .irodori_dataset import digest, normalize_style, paths, read_csv, review_origin, write_csv
else:
    from irodori_dataset import digest, normalize_style, paths, read_csv, review_origin, write_csv

# These are delivery descriptions, NOT transcriptions.
CAPTIONS = {
    "normal": "", "whisper": "囁くような小さな声で話している",
    "laugh": "笑いながら発声している",
    "breath": "息を吐く、吐息を伴う発声",
    "panting": "荒く息をしながら発声している",
    "groan": "うめくような声を出している",
    "emotion": "感情を強く込めて発声している", "other": "",
}
NONVERBAL = {"laugh", "breath", "panting", "groan"}
# README demonstrates these emojis with spoken words, not standalone.
# This is an experiment hint only, never a validated transcript.
EMOJI_HINTS = {"laugh": "🤭", "breath": "😮‍💨"}
DETAIL_COLUMNS = (
    "clip_id", "sha256", "source_path", "source_kind", "decision",
    "route", "reason", "candidate_style", "style_source", "text_source",
    "text", "caption", "training_ready",
)
REVIEW_COLUMNS = (
    "clip_id", "source_path", "duration_sec", "source_kind",
    "asr_status", "candidate_style", "top_events", "reason",
)
EXPERIMENT_COLUMNS = (
    "clip_id", "source_path", "sha256", "style", "style_source",
    "possible_emoji_text", "auto_caption", "reason",
)


def index(rows: list[dict[str, str]], name: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        cid = row.get("clip_id", "")
        if not cid or cid in result:
            raise ValueError(f"Missing or duplicate {name} clip_id: {cid}")
        result[cid] = row
    return result


def sound_hint(hint: dict[str, str]) -> tuple[str, str]:
    if not hint:
        return "", "not_classified"
    if hint.get("status") != "suggested":
        return "", "classification_failed"
    value = hint.get("candidate_style", "")
    if value not in CAPTIONS and value != "unknown":
        return "", "invalid_style_prediction"
    if value == "unknown":
        return "", "model_abstained"
    if hint.get("uncertainty") != "unvalidated":
        return value, "style_prediction_uncertain"
    return value, "style_prediction_candidate"


def strong_style(hint: dict[str, str], state: str) -> bool:
    """Conservative routing heuristic, never a calibrated confidence."""
    if state != "style_prediction_candidate":
        return False
    try:
        top = float(hint.get("candidate_score", ""))
        second = float(hint.get("runner_up_score", ""))
    except ValueError:
        return False
    return top >= 0.70 and top - second >= 0.20


def build(args: argparse.Namespace, source: Path, out: Path) -> dict[str, object]:
    """Non-interactive assembly; WAV and human judgments remain untouched."""
    reviews = read_csv(out / "review.csv")
    inv = index(read_csv(out / "inventory.csv"), "inventory")
    if len(reviews) != len(inv) or {r.get("clip_id") for r in reviews} != set(inv):
        raise ValueError("Inventory/review identity mismatch")
    apath, spath = out / "asr_suggestions.csv", out / "style_suggestions.csv"
    asr = index(read_csv(apath), "ASR") if apath.is_file() else {}
    hints = index(read_csv(spath), "AST") if spath.is_file() else {}
    details, ambiguous, experiments, ready = [], [], [], []
    summary = Counter()
    for row in reviews:
        cid = row["clip_id"]
        orig = inv[cid]
        if any(orig.get(k) != row.get(k) for k in (
            "sha256", "source_path", "relative_path", "duration_sec",
        )):
            raise ValueError(f"Inventory mismatch: {cid}")
        wav = Path(row["source_path"]).resolve()
        if source not in wav.parents or not wav.is_file() or digest(wav) != row["sha256"]:
            raise ValueError(f"Source WAV missing or changed: {wav}")
        ar, st = asr.get(cid, {}), hints.get(cid, {})
        for name, item in (("ASR", ar), ("AST", st)):
            if item and item.get("sha256") != row["sha256"]:
                raise ValueError(f"Stale {name} result: {cid}")

        decision = row.get("decision", "").strip().lower()
        origin = review_origin(row)
        style, state = sound_hint(st)
        human_style = normalize_style(row.get("style", ""))
        confirmed = decision in {"approved", "needs_text", "needs_caption", "style_confirmed"}
        tentative = decision == "tagged"
        if (confirmed or tentative) and human_style in CAPTIONS:
            style = human_style
            style_source = "human_confirmed" if confirmed else "human_tentative"
        else:
            style_source = "ast_experimental" if st else "not_available"
        verified_text = row.get("text", "").strip() if decision in {"approved", "needs_text", "needs_caption"} else ""
        asr_text = ar.get("asr_suggestion", "").strip() if ar.get("asr_status") == "suggested" else ""
        content = verified_text or asr_text
        text_source = ("human_verified" if verified_text else
                       "asr_unverified" if asr_text else "none")
        caption = (row.get("caption", "").strip() if confirmed else "") or CAPTIONS.get(style, "")
        route, reason = "", ""

        if decision == "rejected":
            route, reason = "excluded", "human_rejected"
        elif row.get("scan_flag") in {"unreadable", "low_sample_rate", "long"}:
            route, reason = "excluded", "invalid_or_out_of_range_audio"
        elif decision not in {"pending", "tagged", "approved", "needs_text", "needs_caption", "style_confirmed"}:
            route, reason = "excluded", "unknown_decision"
        elif not content:
            if style in NONVERBAL and (confirmed or tentative or state == "style_prediction_candidate"):
                route, reason = "nonverbal_experiment", "nonverbal_text_protocol_unvalidated"
            elif state in {"not_classified", "classification_failed"}:
                route, reason = "automatic_processing", "missing_style_inference"
            else:
                route, reason = "ambiguous_vocal_style", "speech_vs_nonverbal_uncertain"
        elif confirmed:
            route, reason = "training_candidate", "human_confirmed_style_with_automatic_caption"
        elif tentative:
            if st and state == "style_prediction_uncertain":
                route, reason = "ambiguous_vocal_style", "style_evidence_conflict"
            else:
                route, reason = "training_candidate", "human_tentative_style_with_asr_text"
        elif origin in {"my_voice", "review_approved"}:
            if not st or (style == "normal" and state == "style_prediction_candidate"):
                style, style_source = "normal", "curated_source_normal_default"
                caption = CAPTIONS["normal"]
                route, reason = "training_candidate", "curated_source_with_asr"
            elif strong_style(st, state):
                style_source = "ast_strong_heuristic_unverified"
                caption = CAPTIONS.get(style, "")
                route, reason = "training_candidate", "strong_non_normal_event_candidate"
            else:
                route, reason = "ambiguous_vocal_style", "speech_vs_audio_style_disagreement"
        elif origin == "review_emotion":
            if not st:
                route, reason = "automatic_processing", "missing_style_inference"
            elif strong_style(st, state):
                if style == "normal":
                    style, style_source = "emotion", "emotional_source_plus_ast_speech"
                else:
                    style_source = "ast_strong_heuristic_unverified"
                caption = CAPTIONS[style]
                route, reason = "training_candidate", "strong_event_plus_emotional_source"
            else:
                route, reason = "ambiguous_vocal_style", "expressive_delivery_needs_identification"
        else:
            route, reason = "ambiguous_vocal_style", "unknown_source_style"

        if route == "training_candidate":
            if style not in CAPTIONS or not content:
                route, reason = "ambiguous_vocal_style", "incomplete_candidate"
            else:
                ready.append({"audio": str(wav), "text": content,
                              "caption": caption, "speaker": args.profile})
        if route == "ambiguous_vocal_style":
            ambiguous.append({
                "clip_id": cid, "source_path": str(wav),
                "duration_sec": row.get("duration_sec", ""),
                "source_kind": origin, "asr_status": ar.get("asr_status", ""),
                "candidate_style": style or "unknown",
                "top_events": st.get("top_events_json", ""), "reason": reason,
            })
        if route == "nonverbal_experiment":
            experiments.append({
                "clip_id": cid, "source_path": str(wav), "sha256": row["sha256"],
                "style": style, "style_source": style_source,
                "possible_emoji_text": EMOJI_HINTS.get(style, ""),
                "auto_caption": caption, "reason": reason + ";not_training_ready",
            })
        summary[route] += 1
        details.append({
            "clip_id": cid, "sha256": row["sha256"], "source_path": str(wav),
            "source_kind": origin, "decision": decision,
            "route": route, "reason": reason, "candidate_style": style,
            "style_source": style_source, "text_source": text_source,
            "text": content, "caption": caption,
            "training_ready": str(route == "training_candidate").lower(),
        })

    # No mutation until every row and source file passes preflight.
    write_csv(out / "auto_preparation_report.csv", DETAIL_COLUMNS, details)
    write_csv(out / "ambiguous_vocal_review.csv", REVIEW_COLUMNS, ambiguous)
    write_csv(out / "nonverbal_experiments.csv", EXPERIMENT_COLUMNS, experiments)
    write_csv(out / "dataset_for_prepare_manifest_auto.csv",
              ("audio", "text", "caption", "speaker"), ready)
    report = {
        "total": len(details), "routes": dict(summary),
        "training_candidates": len(ready),
        "asr_text_is_machine_generated": True,
        "human_verified_only": False,
        "nonverbal_experiment_is_training_ready": False,
        "manual_action": "Only classify unclear vocal styles in ambiguous_vocal_review.csv",
        "outputs": {
            "training": str(out / "dataset_for_prepare_manifest_auto.csv"),
            "vocal_exceptions": str(out / "ambiguous_vocal_review.csv"),
            "nonverbal_experiments": str(out / "nonverbal_experiments.csv"),
            "source_trace": str(out / "auto_preparation_report.csv"),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

def resolve(args: argparse.Namespace, source: Path, out: Path) -> None:
    """Ask for ONE style choice per ambiguous clip, not text or captions."""
    path = out / "ambiguous_vocal_review.csv"
    if not path.is_file():
        raise FileNotFoundError("Run 'prepare' to build the vocal-style queue")
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.no_play and shutil.which("ffplay") is None:
        raise RuntimeError("ffplay is missing from PATH")
    reviews = read_csv(out / "review.csv")
    by_id = index(reviews, "review")
    columns = tuple(dict.fromkeys(
        [*read_csv_columns(out / "review.csv"),
         *[key for item in reviews for key in item]]
    ))
    pending = read_csv(path)[:]
    saved = 0
    shown = 0
    for queued in pending:
        item = by_id.get(queued.get("clip_id", ""))
        if not item or item.get("source_path") != queued.get("source_path"):
            raise ValueError("Stale vocal queue; run prepare again")
        if item.get("decision") in {"approved", "rejected", "style_confirmed"}:
            continue
        wav = Path(item["source_path"]).resolve()
        if source not in wav.parents or not wav.is_file() or digest(wav) != item["sha256"]:
            raise ValueError(f"Source WAV changed: {wav}")
        if shown >= args.limit:
            break
        shown += 1
        print(f"\n[{shown}] {wav.name} | AI: {queued.get('candidate_style')} | {queued.get('reason')}")
        while True:
            if not args.no_play:
                subprocess.run(["ffplay", "-nodisp", "-autoexit",
                                "-loglevel", "error", str(wav)], check=False)
            answer = input(
                "Style [normal/whisper/laugh/breath/panting/groan/emotion/other], "
                "[=] AI suggestion, [r] replay, [s] skip, [x] reject, [q] quit > "
            ).strip().lower()
            if answer == "q":
                print(f"Saved {saved} vocal-style classifications.")
                return
            if answer == "s":
                break
            if answer == "r":
                continue
            if answer == "x":
                item["decision"] = "rejected"
                item["notes"] = "rejected_during_vocal_style_classification"
            else:
                if answer == "=":
                    answer = queued.get("candidate_style", "")
                answer = normalize_style(answer)
                if answer not in CAPTIONS:
                    print("Unrecognized style; use s to defer.")
                    continue
                item["style"] = answer
                item["decision"] = "style_confirmed"
                # Do not mark text, speaker identity or audio quality verified.
            write_csv(out / "review.csv", columns, reviews)
            saved += 1
            print(f"Saved style: {item['decision']} / {item.get('style', '')}")
            break
    print(f"Saved {saved} vocal-style classifications. Run prepare again.")


def read_csv_columns(path: Path) -> list[str]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream).fieldnames or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--no-play", action="store_true")
    args = parser.parse_args()
    try:
        source, out = paths(args)
        if args.resolve:
            resolve(args, source, out)
        else:
            build(args, source, out)
        return 0
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
