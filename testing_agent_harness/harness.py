from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import uuid

from .config import AgentConfig
from .models import BaseProvider, GeminiProvider, MockProvider, OpenAICompatibleProvider
from .registry import EventLogger, SkillRegistry, SkillRunner
from .schemas import BugLocalization, FinalJudgement, FailureSummary, RouteDecision, RunState, TestPlan
from .tools import ToolContext, ToolRegistry, build_default_tool_registry
from .utils import ensure_dir, utc_now_iso, write_json


class HarnessError(RuntimeError):
    pass


_MAX_STREAM_CHARS = 4000  # Keep stdout/stderr blobs small before sending to the LLM.
_MAX_STACK_LINES = 40


def _tail(text: Any, limit: int = _MAX_STREAM_CHARS) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    return "...<truncated>...\n" + text[-limit:]


def _trim_test_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    out = dict(result)
    out["stdout"] = _tail(out.get("stdout"))
    out["stderr"] = _tail(out.get("stderr"))
    return out


def _trim_failure_summary(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    out = dict(summary)
    trimmed: list[dict[str, Any]] = []
    for f in out.get("failures", []) or []:
        if not isinstance(f, dict):
            continue
        item = dict(f)
        stack = item.get("stack_excerpt") or []
        if isinstance(stack, list) and len(stack) > _MAX_STACK_LINES:
            item["stack_excerpt"] = stack[-_MAX_STACK_LINES:]
        item["message"] = _tail(item.get("message"), limit=1500)
        trimmed.append(item)
    out["failures"] = trimmed
    return out


class TestingHarness:
    def __init__(self, repo_path: str | Path, config: AgentConfig, provider_name: str | None = None) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config
        self.provider = self._build_provider(provider_name or config.model.provider)
        self.tools: ToolRegistry = build_default_tool_registry()
        self.skills = SkillRegistry(Path(__file__).parent / "prompts")
        self.run_id = uuid.uuid4().hex[:10]
        self.run_dir = ensure_dir(self.repo_path / ".testing_agent_runs" / self.run_id)
        self.state = RunState(
            run_id=self.run_id,
            repo_path=str(self.repo_path),
            sandbox_path="",
            created_at=utc_now_iso(),
            config_path=str(self.repo_path / ".testing_agent.yaml") if (self.repo_path / ".testing_agent.yaml").exists() else None,
            events_path=str(self.run_dir / "events.jsonl"),
        )
        self.logger = EventLogger(self.run_dir / "events.jsonl", self.run_id)
        self.skill_runner = SkillRunner(self.provider, self.tools, self.skills, self.logger)

    def _build_provider(self, provider_name: str) -> BaseProvider:
        if provider_name == "mock":
            return MockProvider(self.config)
        if provider_name == "gemini":
            return GeminiProvider(self.config)
        if provider_name == "openai_compatible":
            return OpenAICompatibleProvider(self.config)
        raise HarnessError(f"Unsupported provider: {provider_name}")

    @property
    def ctx(self) -> ToolContext:
        return ToolContext(config=self.config, state=self.state, run_dir=self.run_dir)

    def _trimmed_latest_result(self) -> dict[str, Any]:
        return _trim_test_result(self.state.latest_test_result.model_dump() if self.state.latest_test_result else None)

    def _trimmed_latest_failures(self) -> dict[str, Any]:
        return _trim_failure_summary(self.state.latest_failures.model_dump() if self.state.latest_failures else None)

    _MAX_SOURCE_BYTES = 8000  # cap a single file's content sent to the LLM

    def _read_sandbox_file(self, rel_path: str) -> str:
        # Best-effort read of a file from the sandbox; returns "" on any error.
        if not rel_path:
            return ""
        try:
            sandbox = Path(self.state.sandbox_path)
            if not sandbox:
                return ""
            p = (sandbox / rel_path).resolve()
            if sandbox.resolve() not in p.parents and p != sandbox.resolve():
                return ""
            if not p.exists() or not p.is_file():
                return ""
            text = p.read_text(encoding="utf-8", errors="replace")
            if len(text) > self._MAX_SOURCE_BYTES:
                text = text[: self._MAX_SOURCE_BYTES] + "\n# ...<truncated>...\n"
            return text
        except Exception:  # noqa: BLE001
            return ""

    def _existing_source_paths(self) -> list[str]:
        # Real source files as discovered by project_scan; used to validate
        # paths that the LLM hallucinates (e.g. drops a package directory).
        scan = self.state.project_scan
        return list(scan.source_files) if scan and scan.source_files else []

    def _resolve_llm_path(self, raw: str | None) -> str:
        # Map an LLM-supplied path to a real file in the sandbox.
        # Strategy:
        #   1) exact match in project_scan.source_files -> keep
        #   2) file exists on disk relative to sandbox -> keep
        #   3) same basename as exactly one source file -> remap to that
        #   4) otherwise -> return "" (caller drops the candidate)
        if not raw or not isinstance(raw, str):
            return ""
        raw = raw.strip().lstrip("/")
        sources = self._existing_source_paths()
        if raw in sources:
            return raw
        sandbox = Path(self.state.sandbox_path) if self.state.sandbox_path else None
        if sandbox and (sandbox / raw).is_file():
            return raw
        base = Path(raw).name
        matches = [s for s in sources if Path(s).name == base]
        if len(matches) == 1:
            return matches[0]
        return ""

    def _source_snapshot(self) -> dict[str, str]:
        # Snapshot of current source files (from project_scan) so test/repair
        # skills can see the ACTUAL code instead of hallucinating it.
        out: dict[str, str] = {}
        scan = self.state.project_scan
        if not scan:
            return out
        budget_bytes = 24000  # total cap across all files
        used = 0
        for rel in scan.source_files:
            if used >= budget_bytes:
                break
            content = self._read_sandbox_file(rel)
            if not content:
                continue
            remaining = budget_bytes - used
            if len(content) > remaining:
                content = content[: max(0, remaining)] + "\n# ...<truncated>...\n"
            out[rel] = content
            used += len(content)
        return out

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def save_state(self) -> Path:
        return self.state.save(self.state_path)

    def emit_stage(self, stage: str, status: str, payload: dict[str, Any] | None = None) -> None:
        self.logger.emit(stage, "stage", stage, status, payload or {})
        self.save_state()

    def bootstrap(self) -> None:
        stage = "bootstrap"
        self.emit_stage(stage, "started", {"repo_path": str(self.repo_path)})
        self.tools.invoke("snapshot_sandbox", self.ctx, {})
        if self.config.budget.fast_mode:
            # Deterministic bootstrap: the LLM previously just forwarded the
            # project_scan and build_manifest tool outputs. Calling the tools
            # directly skips one slow LLM round-trip entirely.
            scan = self.tools.invoke("project_scan", self.ctx, {})
            manifest = self.tools.invoke("build_manifest", self.ctx, {})
            self.state.project_scan = self.state.project_scan or self._coerce_project_scan(scan)
            self.state.manifest = self.state.manifest or self._coerce_manifest(manifest)
            self.emit_stage(stage, "completed", {"project_scan": scan, "manifest": manifest, "fast_mode": True})
            return
        result = self.skill_runner.run(
            "environment_bootstrap",
            stage,
            self.ctx,
            {"goal": self.config.goals.user_goal},
        )
        self.state.project_scan = self.state.project_scan or self._coerce_project_scan(result.get("project_scan"))
        self.state.manifest = self.state.manifest or self._coerce_manifest(result.get("manifest"))
        self.emit_stage(stage, "completed", result)

    def plan(self) -> None:
        stage = "planning"
        self.emit_stage(stage, "started")
        plan_payload = self.skill_runner.run(
            "plan_builder",
            stage,
            self.ctx,
            {
                "goal": self.config.goals.user_goal,
                "target_coverage": self.config.goals.target_line_coverage,
                "project_scan": self.state.project_scan.model_dump() if self.state.project_scan else {},
                "manifest": self.state.manifest.model_dump() if self.state.manifest else {},
                "repair_mode": self.config.policy.repair_mode,
            },
        )
        if self.config.budget.fast_mode:
            # Skip the plan reviewer LLM call entirely — for small models it
            # rarely produces useful feedback and costs a full round-trip.
            review_payload = {"approved": True, "issues": [], "revised_plan": None}
        else:
            review_payload = self.skill_runner.run("plan_reviewer", stage, self.ctx, {"plan": plan_payload})
        approved = bool(review_payload.get("approved", False))
        issues = review_payload.get("issues", []) or []
        revised = review_payload.get("revised_plan") or {}
        raw_plan: Any
        if approved:
            raw_plan = revised or plan_payload
        elif isinstance(revised, dict) and revised:
            self.emit_stage(stage, "review_warning", {"issues": issues, "action": "using_revised_plan"})
            raw_plan = revised
        elif isinstance(plan_payload, dict) and plan_payload:
            self.emit_stage(stage, "review_warning", {"issues": issues, "action": "using_original_plan"})
            raw_plan = plan_payload
        else:
            raise HarnessError(f"Plan review failed with no usable plan: {issues}")
        self.state.plan = TestPlan.model_validate(_normalize_plan_payload(raw_plan))
        write_json(self.run_dir / "plan.json", self.state.plan.model_dump())
        self.emit_stage(stage, "completed", {"plan": self.state.plan.model_dump()})

    def baseline_execution(self) -> None:
        stage = "baseline_execution"
        self.emit_stage(stage, "started")
        manifest = self.state.manifest
        if not manifest:
            raise HarnessError("Manifest not available before baseline execution.")
        test_command = manifest.test_commands[0].command
        test_result = self.tools.invoke("run_tests", self.ctx, {"command": test_command})
        self.state.baseline_result = self.state.latest_test_result
        failure_summary = self.tools.invoke("failure_parse", self.ctx, {})
        # Always try a coverage run after baseline unless baseline command itself is a coverage command.
        coverage_command = manifest.coverage_commands[0].command
        coverage_result = self.tools.invoke("run_coverage", self.ctx, {"command": coverage_command})
        self.tools.invoke("failure_parse", self.ctx, {})
        self.emit_stage(
            stage,
            "completed",
            {"test_result": test_result, "coverage_result": coverage_result, "failures": failure_summary},
        )

    def route(self) -> RouteDecision:
        stage = "routing"
        self.emit_stage(stage, "started")
        if self.config.budget.fast_mode:
            decision = self._deterministic_route()
            self.state.route_history.append(decision)
            self.emit_stage(stage, "completed", {**decision.model_dump(), "source": "deterministic"})
            return decision
        payload = self.skill_runner.run(
            "workflow_router",
            stage,
            self.ctx,
            {
                "plan": self.state.plan.model_dump() if self.state.plan else {},
                "latest_result": self._trimmed_latest_result(),
                "latest_failures": self._trimmed_latest_failures(),
                "iteration_count": self.state.iteration_count,
                "failed_repair_count": self.state.failed_repair_count,
            },
        )
        decision = RouteDecision.model_validate(_normalize_route_payload(payload))
        self.state.route_history.append(decision)
        self.emit_stage(stage, "completed", decision.model_dump())
        return decision

    def _deterministic_route(self) -> RouteDecision:
        # Simple, boring, and fast. Mirrors what workflow_router is supposed
        # to decide on well-behaved small projects.
        failures = self.state.latest_failures
        result = self.state.latest_test_result
        coverage = (result.coverage.total_percent if result and result.coverage else 0.0) if result else 0.0
        target = self.config.goals.target_line_coverage

        if failures and failures.error_count > 0 and not self.state.bug_localization:
            return RouteDecision(
                next_stage="bug_localization",
                reason="Failures present and no localization yet.",
                stop=False,
            )
        if failures and failures.error_count > 0 and self.state.failed_repair_count < self.config.budget.max_failed_repairs:
            return RouteDecision(
                next_stage="repair",
                reason="Failures still present; attempting repair.",
                stop=False,
            )
        if coverage < target and self.state.iteration_count < self.config.budget.max_iterations:
            return RouteDecision(
                next_stage="iterative_improvement",
                reason=f"Coverage {coverage:.2f} < target {target:.2f}.",
                stop=False,
            )
        return RouteDecision(
            next_stage="final_judgement",
            reason="Tests pass and/or coverage/budget reached.",
            stop=True,
        )

    def iterative_improvement(self) -> None:
        stage = "iterative_improvement"
        self.emit_stage(stage, "started")
        previous_coverage = self.state.latest_test_result.coverage.total_percent if self.state.latest_test_result and self.state.latest_test_result.coverage else 0.0
        stagnant_rounds = 0
        while self.state.iteration_count < self.config.budget.max_iterations:
            self.state.iteration_count += 1
            gaps = self.tools.invoke("coverage_gap_analysis", self.ctx, {})
            latest_cov = self.state.latest_test_result.coverage.total_percent if self.state.latest_test_result and self.state.latest_test_result.coverage else 0.0
            if latest_cov >= self.config.goals.target_line_coverage:
                self.logger.emit(stage, "loop", stage, "completed", {"reason": "coverage target reached", "coverage": latest_cov})
                break
            try:
                generated = self.skill_runner.run(
                    "test_generation",
                    stage,
                    self.ctx,
                    {
                        "coverage_gaps": gaps,
                        "latest_failures": self._trimmed_latest_failures(),
                        "goal": self.config.goals.user_goal,
                        # Give the model the actual source so generated tests
                        # import real symbols with real signatures.
                        "source_files": self._source_snapshot(),
                    },
                )
            except Exception as exc:  # noqa: BLE001 — let the loop fail gracefully
                self.logger.emit(stage, "loop", stage, "stopped", {"reason": f"test_generation failed: {exc}"})
                break
            raw_files = generated.get("files") or []
            valid_files = [f for f in raw_files if isinstance(f, dict) and isinstance(f.get("content"), str) and f["content"].strip() and isinstance(f.get("path"), str)]
            if len(valid_files) < len(raw_files):
                self.logger.emit(
                    stage,
                    "loop",
                    stage,
                    "warning",
                    {"reason": "dropped test files without content", "dropped": len(raw_files) - len(valid_files)},
                )
            if not valid_files:
                self.logger.emit(stage, "loop", stage, "stopped", {"reason": "no usable test files (missing 'content')"})
                break
            if self.config.budget.fast_mode:
                # The deterministic test_quality_check tool already filtered
                # out unusable files above; skip the LLM critic entirely.
                quality = {"accepted": True, "notes": ["fast_mode: deterministic quality only"]}
            else:
                try:
                    quality = self.skill_runner.run("test_quality_critic", stage, self.ctx, {"files": valid_files})
                except Exception as exc:  # noqa: BLE001
                    self.logger.emit(stage, "loop", stage, "warning", {"reason": f"quality check failed: {exc}"})
                    quality = {"accepted": True, "notes": ["quality check skipped due to error"]}
            self.logger.emit(stage, "quality", "test_quality_critic", "completed", quality)
            if not quality.get("accepted", False):
                self.logger.emit(stage, "loop", stage, "stopped", {"reason": "generated tests rejected", "quality": quality})
                break
            self.tools.invoke("apply_changes", self.ctx, {"changes": valid_files})
            generated["files"] = valid_files
            self.state.generated_tests.extend([] if not generated.get("files") else [
                self._coerce_file_change(item) for item in generated["files"]
            ])
            manifest = self.state.manifest
            coverage_command = manifest.coverage_commands[0].command
            self.tools.invoke("run_coverage", self.ctx, {"command": coverage_command})
            self.tools.invoke("failure_parse", self.ctx, {})
            new_coverage = self.state.latest_test_result.coverage.total_percent if self.state.latest_test_result and self.state.latest_test_result.coverage else 0.0
            if new_coverage <= previous_coverage + 0.001:
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            previous_coverage = new_coverage
            if stagnant_rounds >= 2:
                self.logger.emit(stage, "loop", stage, "stopped", {"reason": "stagnation detected", "coverage": new_coverage})
                break
            if self.state.latest_test_result and self.state.latest_test_result.passed and new_coverage >= self.config.goals.target_line_coverage:
                break
        self.emit_stage(stage, "completed", {"iterations": self.state.iteration_count, "latest_result": self.state.latest_test_result.model_dump() if self.state.latest_test_result else {}})

    def localize_and_maybe_repair(self) -> None:
        if not self.state.latest_failures or self.state.latest_failures.error_count == 0:
            return
        stage = "localization"
        self.emit_stage(stage, "started")
        try:
            if not self.config.budget.fast_mode:
                # The narrative failure_localizer skill is purely informational
                # — bug_localizer already reads the raw failures directly.
                self.skill_runner.run("failure_localizer", stage, self.ctx, {"failures": self._trimmed_latest_failures()})
            bug_payload = self.skill_runner.run(
                "bug_localizer",
                stage,
                self.ctx,
                {
                    "failures": self._trimmed_latest_failures(),
                    "latest_result": self._trimmed_latest_result(),
                    # Anchor the LLM to real paths so it stops dropping package
                    # directories (e.g. "src/core.py" instead of
                    # "src/buggy_calc/core.py"). Also include source contents
                    # so the summary/reasons stay grounded.
                    "candidate_paths": self._existing_source_paths(),
                    "source_files": self._source_snapshot(),
                },
            )
            normalized = _normalize_bug_localization(bug_payload)
            cleaned: list[dict[str, Any]] = []
            dropped: list[dict[str, Any]] = []
            for cand in normalized.get("candidates", []) or []:
                if not isinstance(cand, dict):
                    continue
                resolved = self._resolve_llm_path(cand.get("path"))
                if resolved:
                    cand = dict(cand)
                    cand["path"] = resolved
                    cleaned.append(cand)
                else:
                    dropped.append(cand)
            if dropped:
                self.logger.emit(stage, "normalize", "bug_localizer", "path_fixups", {"dropped": dropped, "kept": cleaned})
            normalized["candidates"] = cleaned
            self.state.bug_localization = BugLocalization.model_validate(normalized)
            self.emit_stage(stage, "completed", self.state.bug_localization.model_dump())
        except Exception as exc:  # noqa: BLE001 — don't let one skill kill the run
            self.emit_stage(stage, "failed", {"error": str(exc)})
            return
        if self.config.policy.repair_mode == "suggest_only":
            return
        if self.config.policy.repair_mode in {"auto", "ask"} and self.config.policy.allow_code_repair:
            self.repair()

    def repair(self) -> None:
        stage = "repair"
        self.emit_stage(stage, "started")
        if not self.state.bug_localization:
            self.emit_stage(stage, "skipped", {"reason": "No bug localization available."})
            return
        try:
            if self.config.budget.fast_mode:
                # Deterministic decision: pick the highest-confidence candidate
                # whose path actually exists in the sandbox. This stops us from
                # "repairing" a hallucinated file like ``src/core.py`` while
                # the real buggy file ``src/buggy_calc/core.py`` sits untouched.
                bl = self.state.bug_localization
                candidates = sorted(bl.candidates, key=lambda c: c.confidence, reverse=True)
                target_path = ""
                for c in candidates:
                    resolved = self._resolve_llm_path(c.path)
                    if resolved:
                        target_path = resolved
                        break
                decision = {
                    "should_repair": bool(target_path),
                    "target_path": target_path,
                    "source": "deterministic",
                }
            else:
                decision = self.skill_runner.run(
                    "repair_decider",
                    stage,
                    self.ctx,
                    {"bug_localization": self.state.bug_localization.model_dump()},
                )
            if not _coerce_bool(decision.get("should_repair")):
                self.emit_stage(stage, "skipped", decision)
                return
            # Record pre-repair baseline (error count + test content) so we can
            # detect and roll back a regression if the LLM wrecks the file.
            pre_errors = self.state.latest_failures.error_count if self.state.latest_failures else 0
            target_path = decision.get("target_path") or ""
            current_content = self._read_sandbox_file(target_path)
            pre_snapshot: dict[str, str] = {}
            if target_path and current_content:
                pre_snapshot[target_path] = current_content

            patch = self.skill_runner.run(
                "repair_patch",
                stage,
                self.ctx,
                {
                    "target_path": target_path,
                    # Pass the actual current content so qwen3:4b doesn't have
                    # to hallucinate the rest of the file. This is the single
                    # biggest correctness fix for small local models.
                    "current_content": current_content,
                    "source_files": self._source_snapshot(),
                    "bug_localization": self.state.bug_localization.model_dump(),
                    "failures": self._trimmed_latest_failures(),
                },
            )
            raw_changes = _normalize_file_changes(patch.get("changes"))
            if not raw_changes:
                self.emit_stage(stage, "skipped", {"reason": "no usable changes (missing path/content)", "raw": patch})
                return
            # Only allow patches to files that actually exist in the sandbox.
            # If the LLM produced a bogus path, either remap via basename or
            # drop the change. We do NOT let repair create new source files.
            valid_changes: list[dict[str, Any]] = []
            rejected: list[dict[str, Any]] = []
            for ch in raw_changes:
                raw_path = ch.get("path") or ""
                resolved = self._resolve_llm_path(raw_path)
                if resolved:
                    if resolved != raw_path:
                        ch = dict(ch)
                        ch["path"] = resolved
                    valid_changes.append(ch)
                else:
                    rejected.append(ch)
            if rejected:
                self.logger.emit(stage, "normalize", "repair_patch", "rejected_paths", {"rejected": [r.get("path") for r in rejected]})
            if not valid_changes:
                self.emit_stage(stage, "skipped", {"reason": "all patch paths unknown in sandbox", "rejected": [r.get("path") for r in rejected]})
                return
            # Snapshot every file we're about to overwrite so we can revert.
            for ch in valid_changes:
                p = ch.get("path") or ""
                if p and p not in pre_snapshot:
                    prev = self._read_sandbox_file(p)
                    if prev:
                        pre_snapshot[p] = prev
            self.tools.invoke("apply_changes", self.ctx, {"changes": valid_changes})
            manifest = self.state.manifest
            if manifest and manifest.test_commands:
                self.tools.invoke("run_tests", self.ctx, {"command": manifest.test_commands[0].command})
                self.tools.invoke("failure_parse", self.ctx, {})
                if self.config.policy.require_regression_after_change and manifest.coverage_commands:
                    self.tools.invoke("run_coverage", self.ctx, {"command": manifest.coverage_commands[0].command})
                    self.tools.invoke("failure_parse", self.ctx, {})
            post_errors = self.state.latest_failures.error_count if self.state.latest_failures else 0
            # Revert when:
            #   * started green, ended red   -> regression
            #   * started red, got strictly worse or no better -> wasted effort
            regressed = (pre_errors == 0 and post_errors > 0) or (pre_errors > 0 and post_errors >= pre_errors)
            if regressed and pre_snapshot:
                revert_changes = [{"path": p, "content": c, "rationale": "auto-revert: repair did not reduce failures"} for p, c in pre_snapshot.items()]
                self.tools.invoke("apply_changes", self.ctx, {"changes": revert_changes})
                if manifest and manifest.test_commands:
                    self.tools.invoke("run_tests", self.ctx, {"command": manifest.test_commands[0].command})
                    self.tools.invoke("failure_parse", self.ctx, {})
                self.state.failed_repair_count += 1
                self.emit_stage(stage, "reverted", {"pre_errors": pre_errors, "post_errors": post_errors, "reverted_files": list(pre_snapshot)})
                return
            # Repair accepted: record it.
            self.state.repairs.extend([self._coerce_file_change(item) for item in valid_changes])
            if self.state.latest_failures and self.state.latest_failures.error_count > 0:
                self.state.failed_repair_count += 1
            self.emit_stage(stage, "completed", {"patch": {"changes": valid_changes}, "diff": self.tools.invoke("diff_workspace", self.ctx, {})})
        except Exception as exc:  # noqa: BLE001 — keep the run going
            self.emit_stage(stage, "failed", {"error": str(exc)})

    def _deterministic_final_judgement(self) -> FinalJudgement:
        result = self.state.latest_test_result
        failures = self.state.latest_failures
        coverage = (result.coverage.total_percent if result and result.coverage else 0.0) if result else 0.0
        target = self.config.goals.target_line_coverage
        tests_pass = bool(result and result.passed and (not failures or failures.error_count == 0))
        coverage_ok = coverage >= target
        unmet: list[str] = []
        if not tests_pass:
            unmet.append(f"tests still failing ({failures.error_count if failures else 'unknown'} failures)")
        if not coverage_ok:
            unmet.append(f"coverage {coverage:.2f} below target {target:.2f}")
        goal_achieved = tests_pass and coverage_ok
        summary = (
            f"Tests {'passed' if tests_pass else 'FAILING'}; "
            f"coverage {coverage:.2%} (target {target:.2%}); "
            f"{len(self.state.repairs)} repair(s), {len(self.state.generated_tests)} generated test file(s)."
        )
        return FinalJudgement(
            goal_achieved=goal_achieved,
            summary=summary,
            unmet_goals=unmet,
            remaining_risks=[] if goal_achieved else ["run did not fully satisfy target goals"],
            recommended_next_actions=[] if goal_achieved else ["inspect failing tests and rerun"],
        )

    def _deterministic_report_payload(self) -> dict[str, Any]:
        fj = self.state.final_judgement.model_dump() if self.state.final_judgement else {}
        result = self.state.latest_test_result
        coverage = (result.coverage.total_percent if result and result.coverage else 0.0) if result else 0.0
        lines = [
            "# Testing Agent Report",
            "",
            f"**Goal:** {self.config.goals.user_goal}",
            f"**Goal achieved:** {fj.get('goal_achieved', False)}",
            f"**Summary:** {fj.get('summary', '')}",
            "",
            "## Test Result",
            f"- Passed: {bool(result and result.passed)}",
            f"- Coverage: {coverage:.2%}",
            f"- Iterations: {self.state.iteration_count}",
            f"- Failed repair rounds: {self.state.failed_repair_count}",
            "",
            "## Generated Tests",
        ]
        if self.state.generated_tests:
            for ft in self.state.generated_tests:
                lines.append(f"- `{ft.path}`")
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("## Repairs")
        if self.state.repairs:
            for ft in self.state.repairs:
                lines.append(f"- `{ft.path}`: {ft.rationale or '(no rationale)'}")
        else:
            lines.append("- (none)")
        markdown = "\n".join(lines) + "\n"
        json_payload = {
            "goal": self.config.goals.user_goal,
            "final_judgement": fj,
            "coverage": coverage,
            "iterations": self.state.iteration_count,
            "failed_repairs": self.state.failed_repair_count,
            "generated_tests": [t.model_dump() for t in self.state.generated_tests],
            "repairs": [r.model_dump() for r in self.state.repairs],
        }
        return {"markdown": markdown, "json_payload": json_payload}

    def final_judgement_and_report(self) -> None:
        stage = "finalization"
        self.emit_stage(stage, "started")
        if self.config.budget.fast_mode:
            self.state.final_judgement = self._deterministic_final_judgement()
            report_payload = self._deterministic_report_payload()
        else:
            try:
                final_payload = self.skill_runner.run(
                    "final_judge",
                    stage,
                    self.ctx,
                    {
                        "plan": self.state.plan.model_dump() if self.state.plan else {},
                        "latest_result": self._trimmed_latest_result(),
                        "latest_failures": self._trimmed_latest_failures(),
                        "bug_localization": self.state.bug_localization.model_dump() if self.state.bug_localization else {},
                        "repairs": [item.model_dump() for item in self.state.repairs],
                    },
                )
                self.state.final_judgement = FinalJudgement.model_validate(_normalize_final_judgement(final_payload))
            except Exception as exc:  # noqa: BLE001
                self.emit_stage(stage, "judgement_failed", {"error": str(exc)})
                self.state.final_judgement = FinalJudgement(
                    goal_achieved=False,
                    summary=f"final_judge skill failed: {exc}",
                    unmet_goals=["final judgement could not be produced"],
                )
            try:
                report_payload = self.skill_runner.run(
                    "report_writer",
                    stage,
                    self.ctx,
                    {
                        "goal": self.config.goals.user_goal,
                        "plan": self.state.plan.model_dump() if self.state.plan else {},
                        "final_judgement": self.state.final_judgement.model_dump(),
                        "latest_result": self._trimmed_latest_result(),
                        "latest_failures": self._trimmed_latest_failures(),
                        "repairs": [item.model_dump() for item in self.state.repairs],
                        "generated_tests": [item.model_dump() for item in self.state.generated_tests],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                self.emit_stage(stage, "report_failed", {"error": str(exc)})
                report_payload = {}
            report_payload = _normalize_report_payload(report_payload)
        try:
            report_artifacts = self.tools.invoke("write_report", self.ctx, report_payload)
        except Exception as exc:  # noqa: BLE001
            self.emit_stage(stage, "write_report_failed", {"error": str(exc)})
            report_artifacts = {"json_path": ""}
        self.state.report_markdown = report_payload["markdown"]
        self.state.report_json_path = report_artifacts.get("json_path", "")
        self.state.completed = True
        self.emit_stage(stage, "completed", {"final_judgement": self.state.final_judgement.model_dump(), "report_artifacts": report_artifacts})

    def run_full(self) -> RunState:
        self.bootstrap()
        self.plan()
        self.baseline_execution()
        while True:
            decision = self.route()
            if decision.stop:
                break
            if decision.next_stage == "iterative_improvement":
                self.iterative_improvement()
                continue
            if decision.next_stage in {"failure_localization", "bug_localization", "repair"}:
                self.localize_and_maybe_repair()
                # after repair or localization, route again
                if self.state.failed_repair_count >= self.config.budget.max_failed_repairs:
                    break
                # if failures resolved and coverage adequate, routing will send us to final_judgement
                continue
            if decision.next_stage == "baseline_execution":
                self.baseline_execution()
                continue
            if decision.next_stage == "final_judgement":
                break
        # If failures still exist after improvement, localize before final judgement.
        if self.state.latest_failures and self.state.latest_failures.error_count > 0 and not self.state.bug_localization:
            self.localize_and_maybe_repair()
        self.final_judgement_and_report()
        self.save_state()
        return self.state

    def report_only(self) -> str:
        if self.state.report_markdown:
            return self.state.report_markdown
        if self.state_path.exists():
            self.state = RunState.load(self.state_path)
            return self.state.report_markdown or ""
        raise HarnessError("No report available for this run.")

    def diff_workspace(self) -> dict[str, Any]:
        return self.tools.invoke("diff_workspace", self.ctx, {})

    def discard_run(self) -> None:
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)

    def _coerce_manifest(self, payload: dict[str, Any] | None):
        if payload is None:
            return None
        from .schemas import ReproducibilityManifest
        try:
            return ReproducibilityManifest.model_validate(payload)
        except Exception:  # noqa: BLE001 — fall back to the deterministic tool output
            try:
                tool_payload = self.tools.invoke("build_manifest", self.ctx, {})
                return ReproducibilityManifest.model_validate(tool_payload)
            except Exception:  # noqa: BLE001
                return None

    def _coerce_project_scan(self, payload: dict[str, Any] | None):
        if payload is None:
            return None
        from .schemas import ProjectScan
        try:
            return ProjectScan.model_validate(payload)
        except Exception:  # noqa: BLE001 — fall back to the deterministic tool output
            try:
                tool_payload = self.tools.invoke("project_scan", self.ctx, {})
                return ProjectScan.model_validate(tool_payload)
            except Exception:  # noqa: BLE001
                return None

    def _coerce_file_change(self, payload: dict[str, Any]):
        from .schemas import FileChange
        return FileChange.model_validate(payload)


_PLAN_STEP_ALIASES = {
    "stage": ("stage", "phase", "name", "id", "title", "step"),
    "objective": ("objective", "goal", "description", "summary", "purpose"),
    "success_criteria": ("success_criteria", "criteria", "success", "acceptance_criteria", "done"),
    "stop_conditions": ("stop_conditions", "conditions", "stop", "exit_conditions", "terminators"),
    "preferred_skills": ("preferred_skills", "skills"),
    "preferred_tools": ("preferred_tools", "tools"),
}


def _as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if isinstance(value, (tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def _normalize_plan_step(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {"objective": str(raw)}
    out: dict[str, Any] = {}
    for target, aliases in _PLAN_STEP_ALIASES.items():
        for key in aliases:
            if key in raw:
                out[target] = raw[key]
                break
    out.setdefault("stage", f"step_{index}")
    out.setdefault("objective", "")
    out["stage"] = str(out["stage"]).strip().lower().replace(" ", "_") or f"step_{index}"
    out["objective"] = str(out["objective"])
    out["success_criteria"] = _as_list_of_str(out.get("success_criteria"))
    out["stop_conditions"] = _as_list_of_str(out.get("stop_conditions"))
    out["preferred_skills"] = _as_list_of_str(out.get("preferred_skills"))
    out["preferred_tools"] = _as_list_of_str(out.get("preferred_tools"))
    return out


_ALLOWED_ROUTE_STAGES = {
    "baseline_execution",
    "iterative_improvement",
    "failure_localization",
    "bug_localization",
    "repair",
    "final_judgement",
}


_ROUTE_STAGE_ALIASES = {
    # "do more tests" family → iterative_improvement
    "test_generation": "iterative_improvement",
    "test_creation": "iterative_improvement",
    "generate_tests": "iterative_improvement",
    "write_tests": "iterative_improvement",
    "coverage_improvement": "iterative_improvement",
    "improve_coverage": "iterative_improvement",
    "tests": "iterative_improvement",
    "coverage": "iterative_improvement",
    # baseline family
    "baseline": "baseline_execution",
    "run_tests": "baseline_execution",
    "initial_run": "baseline_execution",
    "execute": "baseline_execution",
    # failure / bug family
    "failure": "failure_localization",
    "failures": "failure_localization",
    "triage": "failure_localization",
    "bug": "bug_localization",
    "debug": "bug_localization",
    "localize": "bug_localization",
    # repair family
    "fix": "repair",
    "bug_fix": "repair",
    "patch": "repair",
    "apply_fix": "repair",
    # finish family
    "final": "final_judgement",
    "finalize": "final_judgement",
    "done": "final_judgement",
    "report": "final_judgement",
    "end": "final_judgement",
    "stop": "final_judgement",
}


def _canonical_route_stage(raw: Any) -> str:
    value = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if value in _ALLOWED_ROUTE_STAGES:
        return value
    if value in _ROUTE_STAGE_ALIASES:
        return _ROUTE_STAGE_ALIASES[value]
    # Partial match fallback: if the model added prefixes/suffixes (e.g. "stage_repair")
    for canonical in _ALLOWED_ROUTE_STAGES:
        if canonical in value:
            return canonical
    for alias, canonical in _ROUTE_STAGE_ALIASES.items():
        if alias in value:
            return canonical
    return "iterative_improvement"


def _normalize_route_payload(payload: Any) -> dict[str, Any]:
    """Coerce a workflow_router LLM reply into the RouteDecision schema.

    The enum-restricted ``next_stage`` field is the most common place where a
    real LLM hallucinates a fresh label such as ``"test_generation"``. We map
    those synonyms to a valid canonical stage rather than failing validation.
    """
    if not isinstance(payload, dict):
        payload = {"next_stage": str(payload)}
    out = dict(payload)
    out["next_stage"] = _canonical_route_stage(out.get("next_stage"))
    out["reason"] = str(out.get("reason") or "")
    stop = out.get("stop")
    if isinstance(stop, str):
        stop = stop.strip().lower() in {"1", "true", "yes", "y"}
    out["stop"] = bool(stop)
    stop_reason = out.get("stop_reason")
    out["stop_reason"] = None if stop_reason in (None, "", "null") else str(stop_reason)
    return out


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "achieved", "pass", "passed"}
    return default


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [str(value)]


_FINAL_JUDGE_ALIASES = {
    "goal_achieved": ("goal_achieved", "achieved", "success", "passed", "verdict", "goal_met", "met"),
    "summary": ("summary", "verdict_summary", "conclusion", "rationale", "description"),
    "unmet_goals": ("unmet_goals", "unmet", "gaps", "failed_goals", "missing"),
    "remaining_risks": ("remaining_risks", "risks", "risk_flags", "concerns"),
    "recommended_next_actions": ("recommended_next_actions", "next_actions", "next_steps", "recommendations", "todo", "actions"),
}


def _normalize_final_judgement(payload: Any) -> dict[str, Any]:
    """Coerce a ``final_judge`` LLM reply into the FinalJudgement schema."""
    if not isinstance(payload, dict):
        payload = {"summary": str(payload) if payload else ""}
    out: dict[str, Any] = {}
    for canonical, aliases in _FINAL_JUDGE_ALIASES.items():
        for key in aliases:
            if key in payload:
                out[canonical] = payload[key]
                break
    return {
        "goal_achieved": _coerce_bool(out.get("goal_achieved"), default=False),
        "summary": str(out.get("summary") or "No summary provided."),
        "unmet_goals": _as_str_list(out.get("unmet_goals")),
        "remaining_risks": _as_str_list(out.get("remaining_risks")),
        "recommended_next_actions": _as_str_list(out.get("recommended_next_actions")),
    }


def _normalize_file_changes(raw: Any) -> list[dict[str, Any]]:
    """Filter & coerce a list of file-change dicts to the FileChange schema.

    Drops items that are missing ``path`` or ``content``, or that accidentally
    swapped ``content`` with structural metadata (like ``test_coverage``).
    Common aliases like ``file``/``filepath`` and ``code``/``source`` are mapped.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("file") or item.get("filepath") or item.get("filename")
        content = item.get("content") or item.get("code") or item.get("source") or item.get("body")
        if not isinstance(path, str) or not path.strip():
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        rationale = item.get("rationale") or item.get("reason") or item.get("explanation") or ""
        out.append({"path": path.strip(), "content": content, "rationale": str(rationale)})
    return out


def _normalize_report_payload(payload: Any) -> dict[str, Any]:
    """Ensure a ``report_writer`` reply has ``markdown`` (str) and ``json_payload`` (dict)."""
    if not isinstance(payload, dict):
        payload = {}
    markdown = (
        payload.get("markdown")
        or payload.get("report")
        or payload.get("body")
        or payload.get("text")
        or payload.get("content")
    )
    if not isinstance(markdown, str) or not markdown.strip():
        markdown = "# Run report\n\n(No markdown body was produced by the report_writer skill.)\n"
    json_payload = payload.get("json_payload") or payload.get("json") or payload.get("data") or {}
    if not isinstance(json_payload, dict):
        json_payload = {"raw": json_payload}
    return {"markdown": markdown, "json_payload": json_payload}


_BUG_CANDIDATE_CONFIDENCE_ALIASES = ("confidence", "score", "probability", "weight", "suspicion", "likelihood")


def _clamp_unit(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    if f > 1.0:
        # If the model emitted a 0-100 scale or something like 1.1, squash it.
        f = f / 100.0 if f > 10.0 else min(f, 1.0)
    if f < 0.0:
        f = 0.0
    return f


def _normalize_bug_candidate(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    path = item.get("path") or item.get("file") or item.get("filepath")
    if not isinstance(path, str) or not path.strip():
        return None
    conf_raw: Any = None
    for key in _BUG_CANDIDATE_CONFIDENCE_ALIASES:
        if key in item:
            conf_raw = item[key]
            break
    reasons = item.get("reasons") or item.get("reason") or item.get("evidence") or []
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = []
    return {
        "path": path.strip(),
        "confidence": _clamp_unit(conf_raw, default=0.5),
        "reasons": [str(r) for r in reasons if r],
    }


def _normalize_bug_localization(payload: Any) -> dict[str, Any]:
    """Coerce a ``bug_localizer`` skill reply into the BugLocalization schema.

    Common deviations we fix:
      * Candidates use ``score`` / ``probability`` / ``weight`` instead of ``confidence``.
      * Confidence values above 1.0 (e.g. 1.1 or a 0-100 scale).
      * Missing ``summary`` / ``should_repair`` / top-level ``confidence``.
    """
    if not isinstance(payload, dict):
        payload = {}
    raw_candidates = payload.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    candidates = [c for c in (_normalize_bug_candidate(x) for x in raw_candidates) if c is not None]
    top_conf = payload.get("confidence")
    if top_conf is None and candidates:
        top_conf = max((c["confidence"] for c in candidates), default=0.0)
    return {
        "summary": str(payload.get("summary") or "No summary provided."),
        "candidates": candidates,
        "should_repair": bool(payload.get("should_repair", False)),
        "confidence": _clamp_unit(top_conf, default=0.0),
    }


def _normalize_plan_payload(payload: Any) -> dict[str, Any]:
    """Best-effort coercion of a free-form LLM plan into the TestPlan schema.

    Handles common mistakes such as using ``phase`` instead of ``stage``,
    missing ``stop_conditions``, or returning scalars instead of lists.
    """
    if not isinstance(payload, dict):
        return {"goal": str(payload), "workflow": []}
    out = dict(payload)
    out.setdefault("goal", "")
    out["scope"] = _as_list_of_str(out.get("scope"))
    out["constraints"] = _as_list_of_str(out.get("constraints"))
    out["done_definition"] = _as_list_of_str(out.get("done_definition"))
    out["risk_flags"] = _as_list_of_str(out.get("risk_flags"))
    try:
        out["target_coverage"] = float(out.get("target_coverage", 0.85))
    except (TypeError, ValueError):
        out["target_coverage"] = 0.85
    workflow_raw = out.get("workflow") or []
    if not isinstance(workflow_raw, list):
        workflow_raw = [workflow_raw]
    out["workflow"] = [_normalize_plan_step(item, idx) for idx, item in enumerate(workflow_raw)]
    return out
