from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
import os
import yaml

RepairMode = Literal["ask", "suggest_only", "auto"]
ProviderName = Literal["gemini", "mock", "openai_compatible"]


@dataclass(slots=True)
class ModelConfig:
    provider: ProviderName = "gemini"
    model_name: str = "gemini-2.5-flash"
    # Env var for API key. For openai_compatible (Ollama/vLLM/LM Studio) many servers need no key.
    api_key_env: str = "GEMINI_API_KEY"
    # OpenAI-compatible root URL ending in /v1 (Ollama: 11434, vLLM: often 8000).
    base_url: str = "http://127.0.0.1:11434/v1"
    temperature: float = 0.0
    max_tool_turns: int = 8
    timeout_seconds: int = 120
    # Output token cap per LLM call. Small models are dramatically faster when
    # they cannot ramble, but plan_builder / repair_patch can legitimately emit
    # 1-2k tokens of JSON. The provider will retry once with 2x this budget if
    # it sees finish_reason=length, so the common case stays cheap.
    max_output_tokens: int = 1536
    # Ollama-only hint: smaller context windows generate much faster.
    # Ignored by non-Ollama OpenAI-compatible servers.
    num_ctx: int = 4096

    @property
    def api_key(self) -> str | None:
        val = os.environ.get(self.api_key_env)
        if val is None or val == "":
            return None
        return val


@dataclass(slots=True)
class BudgetConfig:
    max_iterations: int = 6
    max_runtime_minutes: int = 30
    max_tool_calls: int = 100
    max_replans: int = 2
    max_failed_repairs: int = 2
    # Fast mode keeps the key orchestration skills LLM-driven, but shrinks
    # their payloads, skips non-essential review passes, and reserves
    # deterministic logic for parsing, validation, and safe fallbacks.
    fast_mode: bool = True


@dataclass(slots=True)
class PolicyConfig:
    repair_mode: RepairMode = "ask"
    apply_accepted_patch_to_original: bool = False
    allow_test_modification: bool = True
    allow_code_repair: bool = True
    allowed_code_paths: list[str] = field(default_factory=lambda: ["src", "app", "package", "."])
    blocked_globs: list[str] = field(default_factory=lambda: [".git/*", ".venv/*", "venv/*", "dist/*", "build/*", "__pycache__/*"])
    require_regression_after_change: bool = True
    require_diff_review: bool = True


@dataclass(slots=True)
class GoalConfig:
    user_goal: str = "Run the project's tests, improve targeted Python tests, localize failures, and optionally repair bugs safely."
    target_line_coverage: float = 0.85
    stop_when_tests_pass: bool = False
    require_reproducibility_manifest: bool = True
    generate_final_report: bool = True


@dataclass(slots=True)
class AgentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    goals: GoalConfig = field(default_factory=GoalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentConfig":
        return cls(
            model=ModelConfig(**payload.get("model", {})),
            budget=BudgetConfig(**payload.get("budget", {})),
            policy=PolicyConfig(**payload.get("policy", {})),
            goals=GoalConfig(**payload.get("goals", {})),
        )


def default_config_path(repo_path: str | Path) -> Path:
    return Path(repo_path).resolve() / ".testing_agent.yaml"


def save_config(config: AgentConfig, path: str | Path) -> Path:
    target = Path(path)
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return target


def load_config(path: str | Path | None = None) -> AgentConfig:
    if path is None:
        return AgentConfig()
    p = Path(path)
    if not p.exists():
        return AgentConfig()
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AgentConfig.from_dict(payload)
