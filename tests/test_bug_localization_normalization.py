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


def test_nested_localization_shape_is_mapped_to_candidate() -> None:
    raw = {
        "localization": {
            "file": "src/terminal_snake/game.py",
            "reason": "Snake growth and wall collision logic are wrong.",
        },
        "repair_needed": True,
        "repair_details": [
            {"issue": "Snake not growing when eating food", "fix": "Keep the tail when food is eaten."},
            {"issue": "Wall collision failure", "code_change": "Use >= for boundary checks."},
        ],
    }
    norm = _normalize_bug_localization(raw)
    decision = BugLocalization.model_validate(norm)
    assert decision.summary == "Snake growth and wall collision logic are wrong."
    assert decision.should_repair is True
    assert decision.candidates[0].path == "src/terminal_snake/game.py"
    assert decision.candidates[0].confidence == 0.5
    assert "Snake not growing when eating food" in decision.candidates[0].reasons
    assert "Use >= for boundary checks." in decision.candidates[0].reasons


def test_buggy_code_locations_shape_is_mapped_to_candidate() -> None:
    raw = {
        "buggy_code_locations": [
            {
                "file": "src/terminal_snake/game.py",
                "function": "turn",
                "reason": "Opposite-direction handling is inconsistent with the requested behavior.",
            }
        ],
        "repairs_justified": True,
        "repair_description": "turn() should preserve the current state for opposite-direction commands.",
    }
    norm = _normalize_bug_localization(raw)
    decision = BugLocalization.model_validate(norm)
    assert decision.summary == "turn() should preserve the current state for opposite-direction commands."
    assert decision.should_repair is True
    assert decision.candidates[0].path == "src/terminal_snake/game.py"
    assert decision.candidates[0].confidence == 0.5
    assert "Opposite-direction handling is inconsistent with the requested behavior." in decision.candidates[0].reasons


def test_buggy_files_and_lines_shape_is_mapped_to_candidates() -> None:
    raw = {
        "buggy_files": ["game.py"],
        "buggy_lines": [
            {"file": "game.py", "line": "new_snake = new_snake[:-1]"},
            {"file": "game.py", "line": "return point.col > state.width"},
        ],
        "repairs_needed": True,
    }
    norm = _normalize_bug_localization(raw)
    decision = BugLocalization.model_validate(norm)
    assert decision.should_repair is True
    assert decision.candidates[0].path == "game.py"
    assert decision.candidates[0].confidence == 0.5
    assert "new_snake = new_snake[:-1]" in decision.candidates[0].reasons
    assert "return point.col > state.width" in decision.candidates[0].reasons
    assert "game.py: new_snake = new_snake[:-1]" in decision.summary
