from __future__ import annotations

from testing_agent_harness.harness import _normalize_bug_localization
from testing_agent_harness.schemas import BugLocalization


def test_score_is_mapped_to_confidence_and_clamped() -> None:
    raw = {
        "summary": "",
        "candidates": [
            {"path": "src/buggy_calc/core.py", "score": 1.1},
            {"path": "src/buggy_calc/util.py", "probability": 0.42, "reason": "fails test X"},
        ],
        "should_repair": True,
    }
    norm = _normalize_bug_localization(raw)
    decision = BugLocalization.model_validate(norm)
    assert decision.candidates[0].path == "src/buggy_calc/core.py"
    assert decision.candidates[0].confidence == 1.0
    assert decision.candidates[1].confidence == 0.42
    assert decision.candidates[1].reasons == ["fails test X"]
    assert decision.should_repair is True
    assert decision.confidence == 1.0


def test_missing_fields_get_defaults() -> None:
    decision = BugLocalization.model_validate(_normalize_bug_localization({}))
    assert decision.summary == "No summary provided."
    assert decision.candidates == []
    assert decision.should_repair is False
    assert decision.confidence == 0.0


def test_invalid_candidates_are_dropped() -> None:
    raw = {
        "candidates": [
            {"score": 0.9},  # no path
            "not a dict",
            {"path": "", "confidence": 0.5},  # empty path
            {"path": "a.py", "confidence": "not a number"},  # non-numeric fallback to default
        ]
    }
    norm = _normalize_bug_localization(raw)
    decision = BugLocalization.model_validate(norm)
    assert [c.path for c in decision.candidates] == ["a.py"]
    assert decision.candidates[0].confidence == 0.5


def test_percent_scale_is_squashed_into_unit_range() -> None:
    raw = {"candidates": [{"path": "x.py", "score": 80}]}
    norm = _normalize_bug_localization(raw)
    decision = BugLocalization.model_validate(norm)
    assert decision.candidates[0].confidence == 0.8
