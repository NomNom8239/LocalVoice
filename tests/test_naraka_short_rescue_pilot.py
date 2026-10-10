"""Regression tests for the naraka short rescue pilot (no GPU/audio files)."""
from scripts import naraka_short_rescue_pilot as pilot


def clip(start, end, classification="UNSCORED", speaker="SPEAKER_00", human="UNREVIEWED", v2="REVIEW"):
    result = {
        "start": start,
        "end": end,
        "speaker": speaker,
        "classification": classification,
        "source_turns": "1",
        "file": f"synthetic_{start:.3f}_{end:.3f}.wav",
        "audit": None,
    }
    if classification != "UNSCORED":
        result["audit"] = {
            "human_truth": human,
            "v2_class": v2,
            "qc_status": "PASS",
        }
    return result


def test_short_only_grouping_keeps_label_boundaries_and_scored_anchor():
    rows = [
        clip(0.0, 1.1),
        clip(1.5, 2.6),
        clip(2.7, 5.0, "SELF"),
        clip(5.1, 6.1, speaker="SPEAKER_01"),
        clip(6.2, 7.5),
    ]
    groups, eligible = pilot.build_groups(rows)
    assert len(groups) == 3
    assert len(eligible) == 1
    assert eligible[0] == rows[:2]


def test_human_negative_is_never_used_as_positive_anchor():
    neg = clip(2.0, 4.2, "SELF", human="OTHER", v2="SELF")
    verified = clip(5.0, 7.1, "OTHER", human="SELF", v2="OTHER")
    assert pilot.self_source(neg) == ""
    assert pilot.self_source(verified) == "HUMAN_SELF"


def test_context_and_join_scores_do_not_auto_approve():
    target = clip(0.0, 1.1)
    neighbor = clip(1.2, 3.5, "SELF", human="UNREVIEWED", v2="SELF")
    rows = [target, neighbor]
    scored = pilot.build_candidates(rows, [], [])
    assert len(scored) == 1
    assert scored[0]["formal_approval"] == "NO"
    assert scored[0]["context_tier"] == "CONTEXT_OR_JOIN_REVIEW"
    assert scored[0]["near_auto_self_count"] == 1
    assert scored[0]["human_truth_for_target"] == "UNREVIEWED"


def test_negative_proximity_is_a_warning_not_target_truth():
    target = clip(0.0, 0.7)
    negative = clip(0.8, 2.9, "OTHER", human="MIXED", v2="OTHER")
    candidates = pilot.build_candidates([target, negative], [], [])
    assert len(candidates) == 1
    result = candidates[0]
    assert result["context_tier"] == "KNOWN_NEGATIVE_CONTEXT_REVIEW"
    assert result["human_truth_for_target"] == "UNREVIEWED"
    assert "HUMAN_NEGATIVE_WITHIN_5S_NOT_TARGET_TRUTH" in result["flags"]
    assert result["formal_approval"] == "NO"


def test_high_scoring_join_remains_a_candidate_and_does_not_write_wav():
    target = clip(0.0, 1.1)
    buddy = clip(1.5, 2.6)
    anchor = clip(2.7, 4.9, "SELF", human="SELF", v2="SELF")
    rows = [target, buddy, anchor]
    groups, eligible = pilot.build_groups(rows)
    assert len(eligible) == 1
    infos = pilot.group_info(eligible, [])
    infos[0]["bank_v2_score"] = "0.6800"
    infos[0]["score_ge_049"] = True
    candidates = pilot.build_candidates(rows, eligible, infos)
    assert len(candidates) == 2
    assert all(c["context_tier"] == "MULTI_EVIDENCE_SELF_CANDIDATE" for c in candidates)
    assert all(c["formal_approval"] == "NO" for c in candidates)
    assert all(c["joined_bank_v2_score"] == "0.6800" for c in candidates)
