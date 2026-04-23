from __future__ import annotations

from testing_agent_harness.harness import _normalize_route_payload
from testing_agent_harness.schemas import RouteDecision


def test_test_generation_is_mapped_to_iterative_improvement() -> None:
    raw = {"next_stage": "test_generation", "reason": "grow coverage", "stop": False}
    normalized = _normalize_route_payload(raw)
    decision = RouteDecision.model_validate(normalized)
    assert decision.next_stage == "iterative_improvement"
    assert decision.stop is False
    assert decision.stop_reason is None


def test_bug_fix_and_stop_string_true_are_coerced() -> None:
    raw = {"next_stage": "bug_fix", "reason": "apply patch", "stop": "true"}
    decision = RouteDecision.model_validate(_normalize_route_payload(raw))
    assert decision.next_stage == "repair"
    assert decision.stop is True


def test_exact_canonical_values_pass_through() -> None:
    for canonical in [
        "baseline_execution",
        "iterative_improvement",
        "failure_localization",
        "bug_localization",
        "repair",
        "final_judgement",
    ]:
        decision = RouteDecision.model_validate(
            _normalize_route_payload({"next_stage": canonical, "reason": "", "stop": False})
        )
        assert decision.next_stage == canonical


def test_unknown_label_falls_back_to_iterative_improvement() -> None:
    decision = RouteDecision.model_validate(
        _normalize_route_payload({"next_stage": "wut", "reason": "???", "stop": False})
    )
    assert decision.next_stage == "iterative_improvement"
