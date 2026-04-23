from __future__ import annotations

from pathlib import Path
import shutil

from testing_agent_harness.config import AgentConfig
from testing_agent_harness.harness import TestingHarness
from testing_agent_harness.schemas import (
    BugCandidate,
    BugLocalization,
    CommandRecord,
    CoverageSnapshot,
    FailureItem,
    FailureSummary,
    ReproducibilityManifest,
    RouteDecision,
    TestRunResult,
)


EXAMPLE_REPO = Path(__file__).resolve().parents[1] / "examples" / "buggy_calc"
SNAKE_EXAMPLE_REPO = Path(__file__).resolve().parents[1] / "examples" / "buggy_snake"


def _copy_repo(tmp_path: Path, source: Path) -> Path:
    target = tmp_path / source.name
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".testing_agent_runs", ".pytest_cache", "__pycache__"),
    )
    return target


def _copy_example(tmp_path: Path) -> Path:
    return _copy_repo(tmp_path, EXAMPLE_REPO)


def _copy_snake_example(tmp_path: Path) -> Path:
    return _copy_repo(tmp_path, SNAKE_EXAMPLE_REPO)


def test_full_mock_run_generates_report_and_diff(tmp_path: Path) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()
    original = harness._read_sandbox_file("src/buggy_calc/core.py")
    patched = original.replace("    return a - b", "    return a + b")
    harness.tools.invoke(
        "apply_changes",
        harness.ctx,
        {"changes": [{"path": "src/buggy_calc/core.py", "content": patched, "rationale": "smoke-test patch"}]},
    )
    harness.state.latest_test_result = TestRunResult(
        command="pytest",
        exit_code=0,
        passed=True,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        coverage=CoverageSnapshot(total_percent=0.95, files=[]),
    )
    harness.state.latest_failures = FailureSummary()
    harness.final_judgement_and_report()
    state = harness.state
    diff_payload = harness.diff_workspace()

    assert state.completed is True
    assert state.final_judgement is not None
    assert state.final_judgement.goal_achieved is True
    assert state.report_markdown is not None
    assert "Testing Agent Report" in state.report_markdown
    assert "src/buggy_calc/core.py" in diff_payload["changed_files"]
    assert Path(state.report_json_path).exists()


def test_original_repo_stays_unchanged_after_sandbox_repair(tmp_path: Path) -> None:
    repo = _copy_example(tmp_path)
    original_core = (repo / "src" / "buggy_calc" / "core.py").read_text(encoding="utf-8")

    config = AgentConfig()
    config.model.provider = "mock"
    config.policy.repair_mode = "auto"
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.run_full()

    current_core = (repo / "src" / "buggy_calc" / "core.py").read_text(encoding="utf-8")
    assert current_core == original_core
    assert "return 0" in current_core


def test_route_stops_after_green_when_stop_when_tests_pass_enabled(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    config.goals.stop_when_tests_pass = True

    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.state.latest_test_result = TestRunResult(
        command="pytest",
        exit_code=0,
        passed=True,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        coverage=CoverageSnapshot(total_percent=0.25, files=[]),
    )
    harness.state.latest_failures = FailureSummary()
    harness.state.post_repair_green_reached = True

    calls: list[str] = []

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        calls.append(name)
        assert name == "workflow_router"
        return {
            "next_stage": "iterative_improvement",
            "reason": "coverage below target",
            "stop": False,
            "stop_reason": None,
            "state_trace": ["tests are green", "coverage is still low"],
            "decision_trace": ["would normally improve coverage next"],
        }

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)
    decision = harness.route()

    assert decision.next_stage == "final_judgement"
    assert decision.stop is True
    assert calls == ["workflow_router"]
    assert harness._deterministic_final_judgement().goal_achieved is True


def test_iterative_improvement_rejects_generated_only_failures_after_green(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True

    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.state.sandbox_path = str(repo)
    harness.state.manifest = ReproducibilityManifest(
        repo_path=str(repo),
        coverage_commands=[CommandRecord(name="coverage", command="python -m pytest -q tests")],
    )
    harness.state.latest_test_result = TestRunResult(
        command="pytest",
        exit_code=0,
        passed=True,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        coverage=CoverageSnapshot(total_percent=0.20, files=[]),
    )
    harness.state.latest_failures = FailureSummary()
    baseline_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_core.py::test_add_basic",
                message="AssertionError: expected 5",
                failure_type="assertion",
            )
        ],
        error_count=1,
    )
    harness.state.baseline_failures = baseline_failures
    harness.state.baseline_failure_fingerprints = harness._failure_fingerprints(baseline_failures)
    harness.state.current_failure_fingerprints = []
    harness.state.post_repair_green_reached = True

    quality_checks: list[str] = []
    applied_batches: list[list[dict[str, str]]] = []
    coverage_runs = {"count": 0}

    def fake_skill_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        if name == "test_generation":
            return {
                "files": [
                    {
                        "path": "tests/test_game_generated.py",
                        "content": "from terminal_snake.game import turn\n\n\ndef test_turn_contract():\n    assert turn(None, None) is None\n",
                    }
                ],
                "rationale": "exercise uncovered helper paths",
                "iteration_goal": "raise coverage",
            }
        raise AssertionError(name)

    def fake_invoke(name, ctx, args):  # type: ignore[no-untyped-def]
        if name == "coverage_gap_analysis":
            return {"files": [{"path": "src/buggy_calc/core.py", "missing_lines": [1]}], "summary": "gap"}
        if name == "test_quality_check":
            quality_checks.append(name)
            return {
                "accepted": True,
                "assertion_quality": 0.8,
                "runtime_risk": 0.0,
                "flaky_risk": 0.0,
                "duplication_risk": 0.0,
                "semantic_drift_risk": 0.0,
                "notes": [],
            }
        if name == "apply_changes":
            applied_batches.append(args["changes"])
            return {"applied": [item["path"] for item in args["changes"]]}
        if name == "run_coverage":
            coverage_runs["count"] += 1
            if coverage_runs["count"] == 1:
                harness.state.latest_test_result = TestRunResult(
                    command="pytest",
                    exit_code=1,
                    passed=False,
                    stdout="FAILED tests/test_game_generated.py::test_turn_contract - AssertionError",
                    stderr="",
                    duration_seconds=0.2,
                    coverage=CoverageSnapshot(total_percent=0.30, files=[]),
                )
            else:
                harness.state.latest_test_result = TestRunResult(
                    command="pytest",
                    exit_code=0,
                    passed=True,
                    stdout="",
                    stderr="",
                    duration_seconds=0.2,
                    coverage=CoverageSnapshot(total_percent=0.20, files=[]),
                )
            return harness.state.latest_test_result.model_dump()
        if name == "failure_parse":
            if coverage_runs["count"] == 1:
                harness.state.latest_failures = FailureSummary(
                    failures=[
                        FailureItem(
                            test_name="tests/test_game_generated.py::test_turn_contract",
                            message="AssertionError: generated contract drift",
                            failure_type="assertion",
                        )
                    ],
                    error_count=1,
                )
            else:
                harness.state.latest_failures = FailureSummary()
            return harness.state.latest_failures.model_dump()
        raise AssertionError(name)

    monkeypatch.setattr(harness.skill_runner, "run", fake_skill_run)
    monkeypatch.setattr(harness.tools, "invoke", fake_invoke)

    harness.iterative_improvement()

    assert quality_checks == ["test_quality_check"]
    assert len(applied_batches) == 2
    assert applied_batches[0][0]["path"] == "tests/test_game_generated.py"
    assert applied_batches[1][0]["path"] == "tests/test_game_generated.py"
    assert harness.state.generated_tests == []
    assert harness.state.generated_test_failures_only is False
    assert harness.state.latest_failures.error_count == 0


def test_run_full_reuses_existing_localization_for_repair(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True

    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.state.latest_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_game.py::test_hitting_wall",
                file_path="src/buggy_calc/core.py",
                message="AssertionError",
                failure_type="assertion",
            )
        ],
        error_count=1,
    )
    harness.state.bug_localization = BugLocalization(
        summary="existing localization",
        candidates=[BugCandidate(path="src/buggy_calc/core.py", confidence=0.9, reasons=["existing candidate"])],
        should_repair=True,
        confidence=0.9,
    )

    calls: list[str] = []
    decisions = iter(
        [
            RouteDecision(next_stage="repair", reason="reuse localization", stop=False),
            RouteDecision(next_stage="final_judgement", reason="done", stop=True),
        ]
    )

    monkeypatch.setattr(harness, "bootstrap", lambda: None)
    monkeypatch.setattr(harness, "plan", lambda: None)
    monkeypatch.setattr(harness, "baseline_execution", lambda: None)
    monkeypatch.setattr(harness, "final_judgement_and_report", lambda: calls.append("final"))
    monkeypatch.setattr(harness, "route", lambda: next(decisions))
    monkeypatch.setattr(harness, "localize_and_maybe_repair", lambda: calls.append("localize"))
    monkeypatch.setattr(harness, "repair", lambda: calls.append("repair"))

    harness.run_full()

    assert calls == ["repair", "final"]


def test_failure_parse_keeps_multiple_summary_lines(tmp_path: Path) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.state.latest_test_result = TestRunResult(
        command="pytest -q tests",
        exit_code=1,
        passed=False,
        stdout=(
            "FAILED tests/test_game.py::test_eating_food_increases_score_and_grows_snake\n"
            "FAILED tests/test_game.py::test_hitting_right_wall_ends_game - AssertionError...\n"
        ),
        stderr="",
        duration_seconds=0.1,
        coverage=None,
    )

    parsed = harness.tools.invoke("failure_parse", harness.ctx, {})

    assert parsed["error_count"] == 2
    assert [item["test_name"] for item in parsed["failures"]] == [
        "tests/test_game.py::test_eating_food_increases_score_and_grows_snake",
        "tests/test_game.py::test_hitting_right_wall_ends_game",
    ]


def test_fast_mode_localization_matches_failing_test_module(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_snake_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    config.policy.repair_mode = "suggest_only"
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()
    harness.state.latest_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_game.py::test_hitting_right_wall_ends_game",
                message="AssertionError: game_over should be true",
                failure_type="assertion",
            )
        ],
        error_count=1,
    )
    harness.state.latest_test_result = TestRunResult(
        command="pytest -q tests",
        exit_code=1,
        passed=False,
        stdout="FAILED tests/test_game.py::test_hitting_right_wall_ends_game - AssertionError...",
        stderr="",
        duration_seconds=0.1,
        coverage=None,
    )

    calls: list[str] = []

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        calls.append(name)
        assert name == "bug_localizer"
        assert "candidate_hints" in payload
        return {
            "summary": "game.py looks like the most likely bug site.",
            "candidates": [
                {"path": "src/terminal_snake/game.py", "confidence": 0.91, "reasons": ["failing game test"]},
            ],
            "should_repair": True,
            "confidence": 0.91,
            "evidence_trace": ["failing test targets game behavior"],
        }

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)

    harness.localize_and_maybe_repair()

    assert calls == ["bug_localizer"]
    assert harness.state.bug_localization is not None
    assert harness.state.bug_localization.candidates[0].path == "src/terminal_snake/game.py"


def test_fast_mode_localization_prefers_module_over_package_init(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    config.policy.repair_mode = "suggest_only"
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()
    harness.state.latest_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_core.py::test_add_basic",
                message="AssertionError: expected 5",
                failure_type="assertion",
            )
        ],
        error_count=1,
    )
    harness.state.latest_test_result = TestRunResult(
        command="pytest -q tests",
        exit_code=1,
        passed=False,
        stdout="FAILED tests/test_core.py::test_add_basic - assert -1 == 5",
        stderr="",
        duration_seconds=0.1,
        coverage=None,
    )

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        assert name == "bug_localizer"
        return {
            "summary": "The package path could not be resolved confidently.",
            "candidates": [{"path": "src/not_real.py", "confidence": 0.6, "reasons": ["bad guess"]}],
            "should_repair": True,
            "confidence": 0.6,
            "evidence_trace": ["model guessed an invalid path"],
        }

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)

    harness.localize_and_maybe_repair()

    assert harness.state.bug_localization is not None
    assert harness.state.bug_localization.candidates[0].path == "src/buggy_calc/core.py"


def test_fast_mode_plan_uses_plan_builder_llm_but_skips_reviewer(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()

    calls: list[str] = []

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        calls.append(name)
        assert name == "plan_builder"
        return {
            "goal": payload["goal"],
            "scope": ["baseline", "repair"],
            "constraints": ["stay inside sandbox"],
            "target_coverage": payload["target_coverage"],
            "workflow": [
                {
                    "stage": "baseline_execution",
                    "objective": "run tests",
                    "success_criteria": ["baseline captured"],
                    "stop_conditions": ["tests executed"],
                    "preferred_skills": [],
                    "preferred_tools": ["run_tests"],
                },
                {
                    "stage": "repair",
                    "objective": "repair localized bug",
                    "success_criteria": ["tests pass"],
                    "stop_conditions": ["repair budget exhausted"],
                    "preferred_skills": [],
                    "preferred_tools": ["apply_changes"],
                },
            ],
            "done_definition": ["tests pass"],
            "risk_flags": ["small model may need fallback"],
            "planning_trace": ["baseline first", "repair stays bounded"],
        }

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)

    harness.plan()

    assert calls == ["plan_builder"]
    assert harness.state.plan is not None
    assert harness.state.plan.workflow[0].stage == "baseline_execution"
    assert any(step.stage == "repair" for step in harness.state.plan.workflow)


def test_fast_mode_repair_fixes_snake_known_bugs_without_llm(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_snake_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()
    harness.state.latest_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_game.py::test_hitting_right_wall_ends_game",
                message="AssertionError: game_over should be true",
                failure_type="assertion",
            ),
            FailureItem(
                test_name="tests/test_game.py::test_eating_food_increases_score_and_grows_snake",
                message="AssertionError: snake should grow after eating food",
                failure_type="assertion",
            ),
        ],
        error_count=2,
    )
    harness.state.bug_localization = BugLocalization(
        summary="game.py is buggy",
        candidates=[BugCandidate(path="src/terminal_snake/game.py", confidence=0.95, reasons=["failing game tests"])],
        should_repair=True,
        confidence=0.95,
    )

    calls: list[str] = []

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        calls.append(name)
        assert name == "repair_decider"
        return {
            "should_repair": True,
            "reason": "Top candidate is a writable game file.",
            "target_path": "src/terminal_snake/game.py",
            "decision_trace": ["localized file matches failing snake tests"],
            "safety_checks": ["path exists in sandbox"],
        }

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)

    harness.repair()

    repaired = harness._read_sandbox_file("src/terminal_snake/game.py")

    assert calls == ["repair_decider"]
    assert harness.state.latest_test_result is not None
    assert harness.state.latest_test_result.passed is True
    assert harness.state.latest_failures.error_count == 0
    assert harness.state.post_repair_green_reached is True
    assert [item.path for item in harness.state.repairs] == ["src/terminal_snake/game.py"]
    assert "point.row >= state.height" in repaired
    assert "if not ate_food:" in repaired


def test_fast_mode_repair_fixes_buggy_calc_without_llm(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()
    harness.state.latest_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_core.py::test_add_basic",
                message="AssertionError: expected 5",
                failure_type="assertion",
            )
        ],
        error_count=1,
    )
    harness.state.bug_localization = BugLocalization(
        summary="core.py is buggy",
        candidates=[BugCandidate(path="src/buggy_calc/core.py", confidence=0.95, reasons=["failing add test"])],
        should_repair=True,
        confidence=0.95,
    )

    calls: list[str] = []

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        calls.append(name)
        assert name == "repair_decider"
        return {
            "should_repair": True,
            "reason": "Top candidate is a writable implementation file.",
            "target_path": "src/buggy_calc/core.py",
            "decision_trace": ["localized file matches failing add test"],
            "safety_checks": ["path exists in sandbox"],
        }

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)

    harness.repair()

    repaired = harness._read_sandbox_file("src/buggy_calc/core.py")

    assert calls == ["repair_decider"]
    assert harness.state.latest_test_result is not None
    assert harness.state.latest_test_result.passed is True
    assert harness.state.latest_failures.error_count == 0
    assert harness.state.post_repair_green_reached is True
    assert [item.path for item in harness.state.repairs] == ["src/buggy_calc/core.py"]
    assert "return a + b" in repaired


def test_repair_patch_payload_only_includes_target_file_source(tmp_path: Path, monkeypatch) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.budget.fast_mode = True
    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    harness.bootstrap()
    harness.state.latest_failures = FailureSummary(
        failures=[
            FailureItem(
                test_name="tests/test_core.py::test_add_basic",
                file_path="src/buggy_calc/core.py",
                message="AssertionError",
                failure_type="assertion",
            )
        ],
        error_count=1,
    )
    harness.state.bug_localization = BugLocalization(
        summary="core.py is buggy",
        candidates=[BugCandidate(path="src/buggy_calc/core.py", confidence=0.9, reasons=["failing add test"])],
        should_repair=True,
        confidence=0.9,
    )
    harness.state.manifest = ReproducibilityManifest(
        repo_path=str(repo),
        test_commands=[CommandRecord(name="pytest", command="python -m pytest -q tests")],
    )

    captured = {}

    def fake_run(name, stage, ctx, payload):  # type: ignore[no-untyped-def]
        if name == "repair_decider":
            return {
                "should_repair": True,
                "reason": "Repair should proceed.",
                "target_path": "src/buggy_calc/core.py",
                "decision_trace": ["candidate is high confidence"],
                "safety_checks": ["path exists in sandbox"],
            }
        if name == "repair_patch":
            captured.update(payload)
            return {"changes": [], "rationale": "noop", "regression_required": True}
        raise AssertionError(name)

    monkeypatch.setattr(harness.skill_runner, "run", fake_run)
    monkeypatch.setattr(harness, "_deterministic_repair_patch", lambda *args, **kwargs: None)

    def fake_invoke(name, ctx, args):  # type: ignore[no-untyped-def]
        if name == "apply_changes":
            return {"applied": []}
        if name == "run_tests":
            return harness.state.latest_test_result.model_dump() if harness.state.latest_test_result else {}
        if name == "failure_parse":
            return harness.state.latest_failures.model_dump()
        if name == "diff_workspace":
            return {"changed_files": [], "diffs": []}
        raise AssertionError(name)

    monkeypatch.setattr(harness.tools, "invoke", fake_invoke)

    harness.repair()

    assert captured["target_path"] == "src/buggy_calc/core.py"
    assert list(captured["source_files"]) == ["src/buggy_calc/core.py"]
    assert captured["source_files"]["src/buggy_calc/core.py"] == captured["current_content"]
