from __future__ import annotations

from pathlib import Path
import shutil

from testing_agent_harness.config import AgentConfig
from testing_agent_harness.harness import TestingHarness


EXAMPLE_REPO = Path(__file__).resolve().parents[1] / "examples" / "buggy_calc"


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "buggy_calc"
    shutil.copytree(EXAMPLE_REPO, target)
    return target


def test_full_mock_run_generates_report_and_diff(tmp_path: Path) -> None:
    repo = _copy_example(tmp_path)
    config = AgentConfig()
    config.model.provider = "mock"
    config.policy.repair_mode = "auto"
    config.goals.target_line_coverage = 0.80

    harness = TestingHarness(repo_path=repo, config=config, provider_name="mock")
    state = harness.run_full()
    diff_payload = harness.diff_workspace()

    assert state.completed is True
    assert state.final_judgement is not None
    assert state.final_judgement.goal_achieved is True
    assert state.report_markdown is not None
    assert "Testing Agent Report" in state.report_markdown
    assert "src/buggy_calc/core.py" in diff_payload["changed_files"]
    assert any(path.startswith("tests/test_generated_") for path in diff_payload["changed_files"])
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
