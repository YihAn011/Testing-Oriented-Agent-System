from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field


class DependencyRecord(BaseModel):
    name: str
    version: str | None = None
    source: str | None = None


class CommandRecord(BaseModel):
    name: str
    command: str
    status: Literal["candidate", "validated", "failed", "skipped"] = "candidate"
    notes: str = ""


class ReproducibilityManifest(BaseModel):
    repo_path: str
    sandbox_path: str | None = None
    python_version: str | None = None
    platform: str | None = None
    dependencies: list[DependencyRecord] = Field(default_factory=list)
    install_commands: list[CommandRecord] = Field(default_factory=list)
    test_commands: list[CommandRecord] = Field(default_factory=list)
    coverage_commands: list[CommandRecord] = Field(default_factory=list)
    random_seed: int = 42
    environment_variables: list[str] = Field(default_factory=list)
    test_entry_points: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ProjectScan(BaseModel):
    project_type: str = "unknown"
    package_manager: str | None = None
    root_files: list[str] = Field(default_factory=list)
    test_files: list[str] = Field(default_factory=list)
    source_files: list[str] = Field(default_factory=list)
    config_files: list[str] = Field(default_factory=list)
    python_modules: list[str] = Field(default_factory=list)
    discovered_commands: list[str] = Field(default_factory=list)
    summary: str = ""


class PlanStep(BaseModel):
    stage: str
    objective: str
    success_criteria: list[str]
    stop_conditions: list[str]
    preferred_skills: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)


class TestPlan(BaseModel):
    goal: str
    scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_coverage: float = 0.85
    workflow: list[PlanStep] = Field(default_factory=list)
    done_definition: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)


class CoverageFileGap(BaseModel):
    path: str
    missing_lines: list[int] = Field(default_factory=list)
    missing_functions: list[str] = Field(default_factory=list)
    covered_percent: float | None = None


class CoverageSnapshot(BaseModel):
    total_percent: float = 0.0
    files: list[CoverageFileGap] = Field(default_factory=list)
    raw_json_path: str | None = None


class TestRunResult(BaseModel):
    command: str
    exit_code: int
    passed: bool
    stdout: str
    stderr: str
    duration_seconds: float
    junit_xml: str | None = None
    coverage: CoverageSnapshot | None = None


class FailureItem(BaseModel):
    test_name: str
    file_path: str | None = None
    message: str
    stack_excerpt: list[str] = Field(default_factory=list)
    failure_type: str = "unknown"


class FailureSummary(BaseModel):
    failures: list[FailureItem] = Field(default_factory=list)
    error_count: int = 0
    suspected_environment_issue: bool = False
    notes: list[str] = Field(default_factory=list)


class BugCandidate(BaseModel):
    path: str
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class BugLocalization(BaseModel):
    summary: str
    candidates: list[BugCandidate] = Field(default_factory=list)
    should_repair: bool = False
    confidence: float = 0.0


class FileChange(BaseModel):
    path: str
    content: str
    rationale: str = ""


class GeneratedTestBundle(BaseModel):
    files: list[FileChange] = Field(default_factory=list)
    rationale: str = ""
    iteration_goal: str = ""


class QualityScore(BaseModel):
    assertion_quality: float = 0.0
    runtime_risk: float = 0.0
    flaky_risk: float = 0.0
    duplication_risk: float = 0.0
    accepted: bool = False
    notes: list[str] = Field(default_factory=list)


class RepairProposal(BaseModel):
    changes: list[FileChange] = Field(default_factory=list)
    rationale: str = ""
    regression_required: bool = True


class RouteDecision(BaseModel):
    next_stage: Literal[
        "baseline_execution",
        "iterative_improvement",
        "failure_localization",
        "bug_localization",
        "repair",
        "final_judgement",
    ]
    reason: str
    stop: bool = False
    stop_reason: str | None = None


class FinalJudgement(BaseModel):
    goal_achieved: bool
    summary: str
    unmet_goals: list[str] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)


class EventRecord(BaseModel):
    run_id: str
    stage: str
    kind: str
    name: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str
    repo_path: str
    sandbox_path: str
    created_at: str
    config_path: str | None = None
    project_scan: ProjectScan | None = None
    manifest: ReproducibilityManifest | None = None
    plan: TestPlan | None = None
    baseline_result: TestRunResult | None = None
    baseline_failures: FailureSummary | None = None
    latest_test_result: TestRunResult | None = None
    latest_failures: FailureSummary | None = None
    bug_localization: BugLocalization | None = None
    final_judgement: FinalJudgement | None = None
    generated_tests: list[FileChange] = Field(default_factory=list)
    repairs: list[FileChange] = Field(default_factory=list)
    route_history: list[RouteDecision] = Field(default_factory=list)
    baseline_failure_fingerprints: list[str] = Field(default_factory=list)
    current_failure_fingerprints: list[str] = Field(default_factory=list)
    post_repair_green_reached: bool = False
    generated_test_failures_only: bool = False
    events_path: str | None = None
    report_markdown: str | None = None
    report_json_path: str | None = None
    iteration_count: int = 0
    failed_repair_count: int = 0
    tool_call_count: int = 0
    completed: bool = False

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "RunState":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
