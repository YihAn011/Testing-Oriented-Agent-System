from __future__ import annotations

from testing_agent_harness.harness import (
    _coerce_bool,
    _normalize_file_changes,
    _normalize_final_judgement,
    _normalize_report_payload,
)
from testing_agent_harness.schemas import FileChange, FinalJudgement


def test_final_judgement_aliases_are_mapped() -> None:
    raw = {
        "achieved": "true",
        "verdict_summary": "Coverage target met; all tests green.",
        "gaps": "Edge cases for divide by zero",
        "risks": ["coverage may be fragile"],
        "next_steps": ["add fuzz test"],
    }
    fj = FinalJudgement.model_validate(_normalize_final_judgement(raw))
    assert fj.goal_achieved is True
    assert "Coverage target met" in fj.summary
    assert fj.unmet_goals == ["Edge cases for divide by zero"]
    assert fj.remaining_risks == ["coverage may be fragile"]
    assert fj.recommended_next_actions == ["add fuzz test"]


def test_final_judgement_defaults_when_empty() -> None:
    fj = FinalJudgement.model_validate(_normalize_final_judgement({}))
    assert fj.goal_achieved is False
    assert fj.summary == "No summary provided."
    assert fj.unmet_goals == []


def test_coerce_bool_handles_strings_and_numbers() -> None:
    assert _coerce_bool("true") is True
    assert _coerce_bool("YES") is True
    assert _coerce_bool("no") is False
    assert _coerce_bool(1) is True
    assert _coerce_bool(0) is False
    assert _coerce_bool(None, default=True) is True


def test_normalize_file_changes_drops_metadata_only_items() -> None:
    raw = [
        {"path": "src/foo.py", "test_coverage": {"functions": ["x"]}},  # no content
        {"path": "", "content": "x"},  # empty path
        {"file": "tests/test_a.py", "code": "import pytest\ndef test_a(): assert True\n"},  # aliases
        {"path": "tests/test_b.py", "content": "   "},  # whitespace content
        "not a dict",
    ]
    files = _normalize_file_changes(raw)
    assert len(files) == 1
    assert files[0]["path"] == "tests/test_a.py"
    assert "import pytest" in files[0]["content"]
    # Pydantic should accept the normalized items.
    FileChange.model_validate(files[0])


def test_normalize_report_payload_fills_missing_markdown() -> None:
    out = _normalize_report_payload({"report": "# hi\nbody"})
    assert out["markdown"].startswith("# hi")
    assert out["json_payload"] == {}

    out2 = _normalize_report_payload(None)
    assert "No markdown body" in out2["markdown"]
    assert out2["json_payload"] == {}


def test_normalize_report_wraps_non_dict_json_payload() -> None:
    out = _normalize_report_payload({"markdown": "# hi", "json_payload": "not a dict"})
    assert out["json_payload"] == {"raw": "not a dict"}
