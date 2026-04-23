from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request

from .config import AgentConfig
from .registry import EventLogger, SkillSpec
from .tools import ToolContext, ToolRegistry
from .utils import truncate


class ProviderError(RuntimeError):
    pass


RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})


def _default_retry_reporter(msg: str) -> None:
    try:
        sys.stderr.write(f"\x1b[33m[{msg}]\x1b[0m\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _urlopen_with_retry(
    request: urllib.request.Request,
    timeout: float,
    label: str,
    *,
    max_retries: int = 4,
    base_delay: float = 1.5,
    on_status: Optional[Callable[[str], None]] = None,
) -> bytes:
    """Open a urllib request with exponential backoff on 5xx / 429 / transient network errors.

    ``on_status`` lets the caller surface progress (e.g. to the chat spinner):
    it is invoked with short strings like ``"retry 2/4 in 3.1s"``. When not
    provided, retry notices are echoed dimly to stderr.
    """
    if on_status is None:
        on_status = _default_retry_reporter
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            if exc.code in RETRYABLE_STATUS and attempt < max_retries:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 0.4)
                msg = f"retry {attempt + 1}/{max_retries} in {wait:.1f}s ({label} {exc.code})"
                if on_status:
                    on_status(msg)
                time.sleep(wait)
                continue
            raise ProviderError(f"{label} API error {exc.code}: {detail or exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 0.4)
                msg = f"retry {attempt + 1}/{max_retries} in {wait:.1f}s ({label} network)"
                if on_status:
                    on_status(msg)
                time.sleep(wait)
                continue
            raise ProviderError(f"{label} connection error: {exc}") from exc
    raise ProviderError(f"{label} failed after {max_retries} retries: {last_error}")


def parse_llm_json_response(text: str) -> Any:
    """Parse JSON from an LLM reply, tolerating markdown fences or extra prose."""
    text = text.strip()
    if not text:
        raise ProviderError("Model returned an empty JSON response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if match:
            return json.loads(match.group(1))
        raise ProviderError(f"Could not parse JSON from model output: {truncate(text, 1000)}")


class BaseProvider:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Optional callback used when long-running HTTP calls retry. Callers
        # (e.g. the chat REPL) can attach this to a spinner to surface
        # "retry 2/4 in 3.1s" progress without spamming stderr.
        self.status_callback: Optional[Callable[[str], None]] = None
        # Optional callback that receives every streamed LLM token
        # (``delta.content``) as it arrives during a skill call. Lets the
        # chat UI show the model's raw thinking live.
        self.token_callback: Optional[Callable[[str], None]] = None

    def run_skill(
        self,
        spec: SkillSpec,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        ctx: ToolContext,
        logger: EventLogger,
        stage: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        """Free-form conversational reply, no tools, plain text.

        ``messages`` follows the OpenAI chat shape: list of
        ``{"role": "system"|"user"|"assistant", "content": str}``.
        """
        raise NotImplementedError

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        on_delta: "Optional[Callable[[str], None]]" = None,
        on_status: "Optional[Callable[[str], None]]" = None,
    ) -> str:
        """Stream a free-form reply, invoking ``on_delta(chunk)`` as tokens arrive.

        Default implementation falls back to the blocking ``chat_text`` and
        emits the full text once at the end. Providers that support real
        streaming should override this.
        """
        text = self.chat_text(messages)
        if on_delta:
            on_delta(text)
        return text


class MockProvider(BaseProvider):
    """Offline deterministic provider used for local testing and CI.

    It follows the same tool-and-skill contracts as the real provider, but answers
    with rules instead of an external LLM call. This keeps the harness testable even
    when network access is unavailable.
    """

    def run_skill(
        self,
        spec: SkillSpec,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        ctx: ToolContext,
        logger: EventLogger,
        stage: str,
    ) -> dict[str, Any]:
        handler_name = f"_skill_{spec.name}"
        if hasattr(self, handler_name):
            return getattr(self, handler_name)(payload, tool_registry, ctx, logger, stage)
        raise ProviderError(f"Mock skill not implemented: {spec.name}")

    def _skill_environment_bootstrap(self, payload, tool_registry, ctx, logger, stage):
        scan = tool_registry.invoke("project_scan", ctx, {})
        manifest = tool_registry.invoke("build_manifest", ctx, {})
        return {
            "project_scan": scan,
            "manifest": manifest,
            "notes": ["Mock bootstrap completed.", "Sandbox snapshot and reproducibility manifest are ready."],
        }

    def _skill_plan_builder(self, payload, tool_registry, ctx, logger, stage):
        goal = payload["goal"]
        target_coverage = payload.get("target_coverage", 0.85)
        return {
            "goal": goal,
            "scope": ["environment setup", "baseline execution", "coverage-guided test improvement", "failure localization", "optional repair", "final report"],
            "constraints": ["stay inside sandbox", "log every tool and skill action", "respect repair preference"],
            "target_coverage": target_coverage,
            "workflow": [
                {
                    "stage": "bootstrap",
                    "objective": "Create isolated sandbox and reproducibility manifest.",
                    "success_criteria": ["Sandbox exists", "Manifest exists"],
                    "stop_conditions": ["project scan complete"],
                    "preferred_skills": ["environment_bootstrap"],
                    "preferred_tools": ["snapshot_sandbox", "project_scan", "build_manifest"],
                },
                {
                    "stage": "baseline_execution",
                    "objective": "Run existing test suite and collect baseline failures and coverage.",
                    "success_criteria": ["One test command executed", "Coverage or failure info collected"],
                    "stop_conditions": ["baseline test run complete"],
                    "preferred_skills": ["workflow_router"],
                    "preferred_tools": ["run_tests", "run_coverage", "failure_parse"],
                },
                {
                    "stage": "iterative_improvement",
                    "objective": "Increase line coverage and test quality in bounded iterations.",
                    "success_criteria": ["Coverage target met or budget exhausted"],
                    "stop_conditions": ["coverage target met", "iteration budget exhausted", "stagnation detected"],
                    "preferred_skills": ["test_generation", "test_quality_critic"],
                    "preferred_tools": ["coverage_gap_analysis", "apply_changes", "run_coverage"],
                },
                {
                    "stage": "repair",
                    "objective": "If allowed, apply a focused fix in sandbox and validate with regression testing.",
                    "success_criteria": ["Patch validated or rejected safely"],
                    "stop_conditions": ["user preference forbids repair", "regression passes", "failed repair budget exhausted"],
                    "preferred_skills": ["repair_decider", "repair_patch"],
                    "preferred_tools": ["suspect_files", "read_file", "apply_changes", "diff_workspace", "run_tests"],
                },
                {
                    "stage": "final_judgement",
                    "objective": "Judge goal completion and generate final report.",
                    "success_criteria": ["Final judgement recorded", "Report written"],
                    "stop_conditions": ["report generated"],
                    "preferred_skills": ["final_judge", "report_writer"],
                    "preferred_tools": ["write_report", "diff_workspace"],
                },
            ],
            "done_definition": [
                "reproducibility manifest saved",
                "all executed actions logged",
                "final report generated",
            ],
            "risk_flags": ["coverage quality tradeoff", "potential false positive repairs"],
        }

    def _skill_plan_reviewer(self, payload, tool_registry, ctx, logger, stage):
        plan = payload["plan"]
        issues = []
        if not plan.get("workflow"):
            issues.append("Plan is missing workflow stages.")
        if not plan.get("done_definition"):
            issues.append("Plan is missing a done definition.")
        return {"approved": not issues, "issues": issues, "revised_plan": plan}

    def _skill_workflow_router(self, payload, tool_registry, ctx, logger, stage):
        latest = ctx.state.latest_test_result
        plan = ctx.state.plan
        target = plan.target_coverage if plan else ctx.config.goals.target_line_coverage
        if latest is None:
            return {"next_stage": "baseline_execution", "reason": "No test execution yet.", "stop": False}
        if ctx.state.latest_failures and ctx.state.latest_failures.error_count > 0:
            return {"next_stage": "bug_localization", "reason": "Failures remain after execution.", "stop": False}
        if latest.coverage and latest.coverage.total_percent < target and ctx.state.iteration_count < ctx.config.budget.max_iterations:
            return {"next_stage": "iterative_improvement", "reason": "Coverage below target.", "stop": False}
        return {"next_stage": "final_judgement", "reason": "Coverage target reached or no actionable failures remain.", "stop": False}

    def _skill_test_generation(self, payload, tool_registry, ctx, logger, stage):
        gap_files = payload.get("coverage_gaps", {}).get("files", [])
        created_files = []
        rationale_parts = []
        for file_gap in gap_files[:2]:
            path = file_gap["path"]
            if not path.endswith(".py") or path.startswith("tests/"):
                continue
            module = tool_registry.invoke("ast_summary", ctx, {"path": path})
            functions = module.get("functions", [])
            missing_functions = file_gap.get("missing_functions", []) or [fn["name"] for fn in functions[:1]]
            if not missing_functions:
                continue
            module_name = path.replace("src/", "").replace(".py", "").replace("/", ".")
            test_path = f"tests/test_generated_{PathLike(path).stem}.py"
            body_lines = [
                "import pytest",
                f"from {module_name} import {', '.join(sorted(set(missing_functions)))}",
                "",
            ]
            for fn in missing_functions:
                if fn == "divide":
                    body_lines.extend([
                        "",
                        "def test_generated_divide_normal_case():",
                        "    assert divide(8, 2) == 4",
                        "",
                        "def test_generated_divide_zero_division():",
                        "    with pytest.raises(ZeroDivisionError):",
                        "        divide(3, 0)",
                    ])
                elif fn == "normalize_score":
                    body_lines.extend([
                        "",
                        "def test_generated_normalize_score_bounds():",
                        "    assert normalize_score(0, 10) == 0.0",
                        "    assert normalize_score(10, 10) == 1.0",
                    ])
                elif fn == "multiply":
                    body_lines.extend([
                        "",
                        "def test_generated_multiply_handles_signs():",
                        "    assert multiply(-3, 4) == -12",
                        "    assert multiply(0, 100) == 0",
                    ])
                else:
                    body_lines.extend([
                        "",
                        f"def test_generated_{fn}_smoke():",
                        f"    result = {fn}()",
                        "    assert result is not None",
                    ])
            created_files.append({
                "path": test_path,
                "content": "\n".join(body_lines).strip() + "\n",
                "rationale": f"Target uncovered function(s) {', '.join(missing_functions)} from {path}.",
            })
            rationale_parts.append(f"Created {test_path} for {path}.")
        return {
            "files": created_files,
            "rationale": " ".join(rationale_parts) or "No targeted gaps were available.",
            "iteration_goal": "Improve coverage for uncovered Python functions.",
        }

    def _skill_test_quality_critic(self, payload, tool_registry, ctx, logger, stage):
        return tool_registry.invoke("test_quality_check", ctx, {"files": payload.get("files", [])})

    def _skill_failure_localizer(self, payload, tool_registry, ctx, logger, stage):
        failures = ctx.state.latest_failures.failures if ctx.state.latest_failures else []
        return {
            "summary": f"Parsed {len(failures)} failure(s).",
            "failures": [item.model_dump() for item in failures],
            "likely_environment_issue": bool(ctx.state.latest_failures and ctx.state.latest_failures.suspected_environment_issue),
        }

    def _skill_bug_localizer(self, payload, tool_registry, ctx, logger, stage):
        suspects = tool_registry.invoke("suspect_files", ctx, {})["candidates"]
        failures = ctx.state.latest_failures.failures if ctx.state.latest_failures else []
        should_repair = bool(failures and ctx.config.policy.allow_code_repair)
        summary = "No clear bug localization signal found."
        confidence = 0.2
        if suspects:
            top = suspects[0]
            summary = f"Most likely buggy file is {top['path']} based on failure traces and coverage gaps."
            confidence = min(0.95, 0.4 + top["score"] / 10.0)
        return {
            "summary": summary,
            "candidates": [
                {
                    "path": item["path"],
                    "confidence": min(0.99, 0.3 + item["score"] / 10.0),
                    "reasons": ["Referenced by failure trace or uncovered by generated tests."],
                }
                for item in suspects[:5]
            ],
            "should_repair": should_repair,
            "confidence": confidence,
        }

    def _skill_repair_decider(self, payload, tool_registry, ctx, logger, stage):
        if not ctx.config.policy.allow_code_repair:
            return {"should_repair": False, "reason": "Policy forbids code repair."}
        candidates = payload.get("bug_localization", {}).get("candidates", [])
        if not candidates:
            return {"should_repair": False, "reason": "No localized bug candidate."}
        return {"should_repair": True, "reason": f"Repair allowed for top candidate {candidates[0]['path']}.", "target_path": candidates[0]["path"]}

    def _skill_repair_patch(self, payload, tool_registry, ctx, logger, stage):
        target_path = payload.get("target_path")
        if not target_path:
            return {"changes": [], "rationale": "No repair target path was selected.", "regression_required": True}
        source = tool_registry.invoke("read_file", ctx, {"path": target_path})["content"]
        updated = source
        rationale = "No patch applied."
        if "def divide" in source and "return 0" in source:
            updated = source.replace("        return 0", "        raise ZeroDivisionError('division by zero')")
            rationale = "Replace silent divide-by-zero handling with explicit ZeroDivisionError."
        return {
            "changes": [{"path": target_path, "content": updated, "rationale": rationale}],
            "rationale": rationale,
            "regression_required": True,
        }

    def _skill_final_judge(self, payload, tool_registry, ctx, logger, stage):
        latest = ctx.state.latest_test_result
        target = ctx.state.plan.target_coverage if ctx.state.plan else ctx.config.goals.target_line_coverage
        goal_achieved = bool(latest and latest.passed and (latest.coverage is None or latest.coverage.total_percent >= target or ctx.config.goals.stop_when_tests_pass))
        remaining_risks = []
        if latest and latest.coverage and latest.coverage.total_percent < target:
            remaining_risks.append("Coverage target not fully met.")
        if ctx.state.latest_failures and ctx.state.latest_failures.error_count:
            remaining_risks.append("Some failures remain unresolved.")
        summary = "Goal achieved with reproducible artifacts and final report ready." if goal_achieved else "Run completed, but one or more goals remain unmet."
        unmet_goals = [] if goal_achieved else ["Reach configured success threshold."]
        next_actions = ["Review diff and report.", "Rerun with a higher iteration budget if needed."]
        return {
            "goal_achieved": goal_achieved,
            "summary": summary,
            "unmet_goals": unmet_goals,
            "remaining_risks": remaining_risks,
            "recommended_next_actions": next_actions,
        }

    def _skill_report_writer(self, payload, tool_registry, ctx, logger, stage):
        state = ctx.state
        coverage = state.latest_test_result.coverage.total_percent if state.latest_test_result and state.latest_test_result.coverage else None
        lines = [
            f"# Testing Agent Report for `{PathLike(state.repo_path).name}`",
            "",
            "## Goal",
            payload.get("goal", ctx.config.goals.user_goal),
            "",
            "## Reproducibility",
            f"- Original repo: `{state.repo_path}`",
            f"- Sandbox: `{state.sandbox_path}`",
            f"- Python: `{state.manifest.python_version if state.manifest else 'unknown'}`",
            "",
            "## Plan Summary",
            f"- Target coverage: `{state.plan.target_coverage if state.plan else ctx.config.goals.target_line_coverage}`",
            f"- Iterations used: `{state.iteration_count}`",
            f"- Tool calls: `{state.tool_call_count}`",
            "",
            "## Execution Outcome",
            f"- Latest command: `{state.latest_test_result.command if state.latest_test_result else 'n/a'}`",
            f"- Passed: `{state.latest_test_result.passed if state.latest_test_result else False}`",
            f"- Coverage: `{coverage if coverage is not None else 'n/a'}`",
            "",
            "## Failure and Localization",
            f"- Failure count: `{state.latest_failures.error_count if state.latest_failures else 0}`",
            f"- Bug localization: `{state.bug_localization.summary if state.bug_localization else 'n/a'}`",
            "",
            "## Changes",
        ]
        diff_payload = tool_registry.invoke("diff_workspace", ctx, {})
        if diff_payload["changed_files"]:
            for changed in diff_payload["changed_files"]:
                lines.append(f"- `{changed}`")
        else:
            lines.append("- No file changes")
        lines.extend([
            "",
            "## Final Judgement",
            f"- Goal achieved: `{state.final_judgement.goal_achieved if state.final_judgement else False}`",
            f"- Summary: {state.final_judgement.summary if state.final_judgement else 'n/a'}",
            "",
            "## Suggested Next Steps",
        ])
        for action in (state.final_judgement.recommended_next_actions if state.final_judgement else ["Review the report."]):
            lines.append(f"- {action}")
        return {
            "markdown": "\n".join(lines).strip() + "\n",
            "json_payload": payload,
        }

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return (
            "[mock provider] I cannot free-form chat without a real LLM. "
            "Use slash commands: /help, /repo <path>, /provider <mock|gemini|openai_compatible>, "
            "/plan, /run, /report, /diff, /status, /quit.\n"
            f"Heard: {truncate(str(last_user), 200)}"
        )


class GeminiProvider(BaseProvider):
    API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

    def run_skill(
        self,
        spec: SkillSpec,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        ctx: ToolContext,
        logger: EventLogger,
        stage: str,
    ) -> dict[str, Any]:
        if not self.config.model.api_key:
            raise ProviderError(f"Environment variable {self.config.model.api_key_env} is required for Gemini provider.")

        system_text = (
            f"You are the internal skill '{spec.name}'.\n"
            f"Skill description: {spec.description}\n\n"
            f"Instructions:\n{spec.prompt}\n\n"
            "Return concise, grounded outputs. Use tools when needed."
        )
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": f"{system_text}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"}]}
        ]
        allowed_tools = spec.allowed_tools or []
        tool_declarations = []
        if allowed_tools:
            tool_declarations = [{"functionDeclarations": tool_registry.list_schemas(allowed_tools)}]
        final_text = ""
        for _ in range(self.config.model.max_tool_turns):
            response = self._generate_content(
                contents=contents,
                tools=tool_declarations or None,
                generation_config=None,
            )
            candidate_content = response["candidates"][0]["content"]
            parts = candidate_content.get("parts", [])
            function_calls = [part["functionCall"] for part in parts if "functionCall" in part]
            if function_calls:
                contents.append(candidate_content)
                for call in function_calls:
                    name = call["name"]
                    args = call.get("args", {})
                    logger.emit(stage, "tool", name, "started", {"args": args})
                    result = tool_registry.invoke(name, ctx, args)
                    ctx.state.tool_call_count += 1
                    logger.emit(stage, "tool", name, "completed", {"result": result})
                    function_response = {
                        "name": name,
                        "response": result,
                    }
                    if call.get("id"):
                        function_response["id"] = call["id"]
                    contents.append({"role": "user", "parts": [{"functionResponse": function_response}]})
                continue
            final_text = self._extract_text(parts)
            break

        structured_contents = list(contents)
        if final_text:
            structured_contents.append({"role": "model", "parts": [{"text": final_text}]})
        structured_contents.append({
            "role": "user",
            "parts": [{"text": "Return the final answer strictly as JSON matching the requested schema. Do not call more tools."}],
        })
        structured_response = self._generate_content(
            contents=structured_contents,
            tools=None,
            generation_config={
                "responseMimeType": "application/json",
                "responseJsonSchema": spec.output_schema,
            },
        )
        text = self._extract_text(structured_response["candidates"][0]["content"].get("parts", []))
        parsed = parse_llm_json_response(text)
        if not isinstance(parsed, dict):
            raise ProviderError(f"Expected JSON object from skill {spec.name}, got: {type(parsed)}")
        return parsed

    def _generate_content(
        self,
        contents: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        generation_config: dict[str, Any] | None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"contents": contents}
        if tools:
            body["tools"] = tools
        if generation_config:
            body["generationConfig"] = generation_config
        request = urllib.request.Request(
            url=f"{self.API_ROOT}/{self.config.model.model_name}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.config.model.api_key or "",
            },
            method="POST",
        )
        raw = _urlopen_with_retry(
            request,
            timeout=self.config.model.timeout_seconds,
            label="Gemini",
            on_status=on_status or self.status_callback,
        )
        return json.loads(raw.decode("utf-8"))

    def _extract_text(self, parts: list[dict[str, Any]]) -> str:
        texts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(texts).strip()

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        return self._chat_text_impl(messages, on_status=None)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        on_delta: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        # Gemini REST stream path is not wired up yet; we fall back to the blocking
        # call but still surface retry status to the caller.
        text = self._chat_text_impl(messages, on_status=on_status)
        if on_delta and text:
            on_delta(text)
        return text

    def _chat_text_impl(
        self,
        messages: list[dict[str, Any]],
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        if not self.config.model.api_key:
            raise ProviderError(
                f"Environment variable {self.config.model.api_key_env} is required for Gemini provider."
            )
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            mapped = "user" if role == "user" else "model"
            contents.append({"role": mapped, "parts": [{"text": m.get("content", "")}]})
        if system_parts and contents and contents[0]["role"] == "user":
            prefix = "\n".join(system_parts) + "\n\n"
            contents[0]["parts"][0]["text"] = prefix + contents[0]["parts"][0]["text"]
        resp = self._generate_content(
            contents=contents, tools=None, generation_config=None, on_status=on_status
        )
        return self._extract_text(resp["candidates"][0]["content"].get("parts", []))


class OpenAICompatibleProvider(BaseProvider):
    """LLM backend using the OpenAI Chat Completions protocol (POST ``.../v1/chat/completions``).

    Works with local Qwen3 and other models served by Ollama (OpenAI compatibility), vLLM,
    LM Studio, llama.cpp server, etc.
    """

    def run_skill(
        self,
        spec: SkillSpec,
        payload: dict[str, Any],
        tool_registry: ToolRegistry,
        ctx: ToolContext,
        logger: EventLogger,
        stage: str,
    ) -> dict[str, Any]:
        schema_hint = json.dumps(spec.output_schema, ensure_ascii=False, indent=2)
        # /no_think disables Qwen3's <think> chain-of-thought, which otherwise
        # doubles every skill call. It is a no-op on other OpenAI-compatible
        # models. The instruction block below also forces JSON on the first
        # shot so we can skip the usual "repair JSON" second round-trip.
        system_text = (
            f"You are the internal skill '{spec.name}'.\n"
            f"Skill description: {spec.description}\n\n"
            f"Instructions:\n{spec.prompt}\n\n"
            "Return ONE JSON object that conforms EXACTLY to the schema below.\n"
            "Output ONLY the JSON object: no markdown fences, no prose, no preamble.\n"
            f"Schema:\n{schema_hint}\n\n"
            "/no_think"
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"{system_text}\n\nInput JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
            }
        ]
        allowed_tools = spec.allowed_tools or []
        openai_tools: list[dict[str, Any]] | None = None
        # Fast mode already pre-feeds skills with source_files / current_content
        # / failures, so the optional tool calls (ast_summary, read_file) would
        # just slow the run down — and they also force the request off the
        # streaming path, which kills the live token view. Skip them.
        fast_mode = getattr(self.config.budget, "fast_mode", False)
        if allowed_tools and not fast_mode:
            openai_tools = [
                {"type": "function", "function": decl}
                for decl in tool_registry.list_schemas(allowed_tools)
            ]
        final_text = ""
        final_finish_reason = ""
        # When there are no tools, request strict JSON mode on the first shot
        # so Ollama / vLLM constrain the decoder and we skip the second call.
        request_json_mode = not openai_tools
        # Prefer streaming whenever we don't need server-side tool calls — it
        # lets the caller watch the model think live via ``token_callback``.
        prefer_stream = not openai_tools and self.token_callback is not None
        for _ in range(self.config.model.max_tool_turns):
            if prefer_stream:
                data = self._chat_completions_streamed(
                    messages=messages,
                    json_mode=request_json_mode,
                    on_delta=self.token_callback,
                )
            else:
                data = self._chat_completions(
                    messages=messages,
                    tools=openai_tools,
                    json_mode=request_json_mode,
                )
            choice0 = data["choices"][0]
            msg = choice0["message"]
            final_finish_reason = str(choice0.get("finish_reason") or "")
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                request_json_mode = False  # tool round-trips cannot enforce JSON
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.get("content"),
                        "tool_calls": tool_calls,
                    }
                )
                for tc in tool_calls:
                    if tc.get("type") != "function":
                        continue
                    fn = tc["function"]
                    name = fn["name"]
                    raw_args = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Invalid tool arguments JSON for {name}: {truncate(raw_args, 500)}") from exc
                    logger.emit(stage, "tool", name, "started", {"args": args})
                    result = tool_registry.invoke(name, ctx, args)
                    ctx.state.tool_call_count += 1
                    logger.emit(stage, "tool", name, "completed", {"result": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        }
                    )
                continue
            final_text = (msg.get("content") or "").strip()
            break

        # Fast path: the first reply was already valid JSON matching the schema shape.
        parsed: Any = None
        if final_text:
            try:
                parsed = parse_llm_json_response(final_text)
            except Exception:  # noqa: BLE001 — fall through to the repair round
                parsed = None
        if isinstance(parsed, dict):
            return parsed

        # Targeted retry: only when the model hit the output-token cap (so the
        # JSON is valid up to the truncation point). We double the budget and
        # try once more. We do NOT retry for other parse failures — that was
        # the slow path that dominated latency on small models.
        if final_finish_reason == "length" and not openai_tools:
            doubled = int(getattr(self.config.model, "max_output_tokens", 1536) or 1536) * 2
            retry_body_extras = {"max_tokens": doubled}
            if logger is not None:
                logger.emit(stage, "provider", spec.name, "retry_length", {"max_tokens": doubled})
            data2 = self._chat_completions(
                messages=messages,
                tools=None,
                json_mode=True,
                body_overrides=retry_body_extras,
            )
            text2 = (data2["choices"][0]["message"].get("content") or "").strip()
            try:
                parsed2 = parse_llm_json_response(text2)
            except Exception:  # noqa: BLE001
                parsed2 = None
            if isinstance(parsed2, dict):
                return parsed2

        raise ProviderError(
            f"Skill {spec.name} did not return parseable JSON on first shot "
            f"(finish_reason={final_finish_reason!r}). "
            f"Raw head: {truncate(final_text, 300)}"
        )

    def _chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_status: Optional[Callable[[str], None]] = None,
        json_mode: bool = False,
        body_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.config.model.base_url.rstrip('/')}/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model.model_name,
            "temperature": self.config.model.temperature,
            "messages": messages,
            # Cap output length so small models stop as soon as they finish
            # the JSON answer instead of rambling.
            "max_tokens": int(getattr(self.config.model, "max_output_tokens", 1536) or 1536),
        }
        # Ollama-specific: shrink context window to speed up generation.
        # OpenAI / vLLM ignore unknown "options".
        num_ctx = int(getattr(self.config.model, "num_ctx", 0) or 0)
        if num_ctx > 0:
            body["options"] = {"num_ctx": num_ctx}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        elif json_mode:
            # OpenAI-compatible JSON mode. Ollama >= 0.1.30 and vLLM honor this.
            body["response_format"] = {"type": "json_object"}
        if body_overrides:
            body.update(body_overrides)
        req_headers = {"Content-Type": "application/json"}
        key = self.config.model.api_key
        if key:
            req_headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        raw = _urlopen_with_retry(
            request,
            timeout=self.config.model.timeout_seconds,
            label="Chat Completions",
            on_status=on_status or self.status_callback,
        )
        return json.loads(raw.decode("utf-8"))

    def _chat_completions_streamed(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool,
        on_delta: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        body_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Streaming equivalent of ``_chat_completions`` for tool-less calls.

        Emits ``delta.content`` chunks through ``on_delta`` as they arrive,
        then returns a response dict shaped identically to the non-streaming
        endpoint so ``run_skill`` can consume it unchanged.
        """
        url = f"{self.config.model.base_url.rstrip('/')}/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model.model_name,
            "temperature": self.config.model.temperature,
            "messages": messages,
            "stream": True,
            "max_tokens": int(getattr(self.config.model, "max_output_tokens", 1536) or 1536),
        }
        num_ctx = int(getattr(self.config.model, "num_ctx", 0) or 0)
        if num_ctx > 0:
            body["options"] = {"num_ctx": num_ctx}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if body_overrides:
            body.update(body_overrides)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        key = self.config.model.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        status_cb = on_status or self.status_callback
        # NB: we deliberately do NOT emit "connecting"/"thinking" status
        # updates here. The live reporter already prints a ``▸ skill_name
        # (thinking…)`` header, so those redundant hints only confused the
        # layout. Retries from ``_open_stream_with_retry`` still flow through.
        response = self._open_stream_with_retry(request, on_status=status_cb)
        chunks: list[str] = []
        finish_reason = ""
        try:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    chunks.append(piece)
                    if on_delta:
                        on_delta(piece)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = str(fr)
        except urllib.error.URLError as exc:
            raise ProviderError(f"Chat Completions stream error: {exc}") from exc
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass
        full = "".join(chunks)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": full},
                    "finish_reason": finish_reason,
                }
            ]
        }

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        data = self._chat_completions(messages=messages, tools=None)
        return (data["choices"][0]["message"].get("content") or "").strip()

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        on_delta: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Real SSE streaming against ``POST {base_url}/chat/completions`` with stream=true.

        Parses ``data: {...}`` SSE frames and invokes ``on_delta`` with each
        content chunk as the server emits it. Returns the full concatenated
        text once ``data: [DONE]`` is received.
        """
        url = f"{self.config.model.base_url.rstrip('/')}/chat/completions"
        body: dict[str, Any] = {
            "model": self.config.model.model_name,
            "temperature": self.config.model.temperature,
            "messages": messages,
            "stream": True,
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        key = self.config.model.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        if on_status:
            on_status("connecting")
        chunks: list[str] = []
        # Retry only for connection establishment; once bytes start streaming we
        # let transport errors propagate.
        response = self._open_stream_with_retry(request, on_status=on_status)
        try:
            if on_status:
                on_status("thinking")
            for raw in response:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                if line.startswith(":"):  # SSE comment / keepalive
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    chunks.append(piece)
                    if on_delta:
                        on_delta(piece)
        except urllib.error.URLError as exc:
            raise ProviderError(f"Chat Completions stream error: {exc}") from exc
        finally:
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass
        return "".join(chunks).strip()

    def _open_stream_with_retry(
        self,
        request: urllib.request.Request,
        on_status: Optional[Callable[[str], None]] = None,
        max_retries: int = 4,
        base_delay: float = 1.5,
    ):
        on_status = on_status or self.status_callback
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=self.config.model.timeout_seconds)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:  # noqa: BLE001
                    pass
                if exc.code in RETRYABLE_STATUS and attempt < max_retries:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 0.4)
                    if on_status:
                        on_status(f"retry {attempt + 1}/{max_retries} in {wait:.1f}s (Chat Completions {exc.code})")
                    time.sleep(wait)
                    continue
                raise ProviderError(f"Chat Completions API error {exc.code}: {detail or exc.reason}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = base_delay * (2 ** attempt) + random.uniform(0, 0.4)
                    if on_status:
                        on_status(f"retry {attempt + 1}/{max_retries} in {wait:.1f}s (Chat Completions network)")
                    time.sleep(wait)
                    continue
                raise ProviderError(f"Chat Completions connection error: {exc}") from exc
        raise ProviderError(f"Chat Completions failed after {max_retries} retries: {last_exc}")


class PathLike:
    def __init__(self, value: str) -> None:
        self.value = value

    @property
    def name(self) -> str:
        return self.value.rstrip("/").split("/")[-1]

    @property
    def stem(self) -> str:
        name = self.name
        if "." in name:
            return name.rsplit(".", 1)[0]
        return name
