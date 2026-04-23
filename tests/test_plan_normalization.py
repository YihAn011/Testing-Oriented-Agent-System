from __future__ import annotations

from testing_agent_harness.harness import _normalize_plan_payload
from testing_agent_harness.schemas import TestPlan


def test_normalizes_phase_to_stage_and_missing_fields() -> None:
    raw = {
        "goal": "test harness",
        "scope": ["unit"],
        "constraints": None,
        "target_coverage": "0.9",
        "workflow": [
            {
                "phase": "Setup",
                "objective": "prepare sandbox",
                "success_criteria": "sandbox ready",
            },
            {
                "phase": "Initial Test Execution",
                "objective": "run baseline",
                "success_criteria": ["coverage collected"],
            },
        ],
        "done_definition": "final report written",
        "risk_flags": [],
    }
    normalized = _normalize_plan_payload(raw)
    plan = TestPlan.model_validate(normalized)
    assert plan.target_coverage == 0.9
    assert [s.stage for s in plan.workflow] == ["setup", "initial_test_execution"]
    assert plan.workflow[0].success_criteria == ["sandbox ready"]
    assert plan.workflow[0].stop_conditions == []
    assert plan.done_definition == ["final report written"]


def test_normalizes_list_wrappers_and_missing_workflow() -> None:
    raw = {"goal": "g", "workflow": {"phase": "x", "objective": "y"}}
    normalized = _normalize_plan_payload(raw)
    plan = TestPlan.model_validate(normalized)
    assert len(plan.workflow) == 1
    assert plan.workflow[0].stage == "x"
    assert plan.workflow[0].objective == "y"
