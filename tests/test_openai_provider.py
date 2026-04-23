from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from testing_agent_harness.config import AgentConfig
from testing_agent_harness.models import OpenAICompatibleProvider, parse_llm_json_response
from testing_agent_harness.registry import EventLogger, SkillSpec
from testing_agent_harness.schemas import RunState
from testing_agent_harness.tools import ToolContext, build_default_tool_registry


def test_parse_llm_json_response_plain() -> None:
    assert parse_llm_json_response('{"x": 2}') == {"x": 2}


def test_parse_llm_json_response_markdown_fence() -> None:
    text = 'Here:\n```json\n{"ok": true}\n```\n'
    assert parse_llm_json_response(text) == {"ok": True}


def test_parse_llm_json_response_empty_raises() -> None:
    with pytest.raises(Exception, match="empty"):
        parse_llm_json_response("   ")


def _minimal_state(tmp_path) -> RunState:
    return RunState(
        run_id="test-run",
        repo_path=str(tmp_path),
        sandbox_path=str(tmp_path / "sandbox"),
        created_at="2020-01-01T00:00:00",
    )


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _UrlOpenSeq:
    def __init__(self, payloads: list[dict]) -> None:
        self._payloads = payloads
        self._i = 0

    def __call__(self, *args: object, **kwargs: object) -> _FakeResp:
        p = self._payloads[self._i]
        self._i += 1
        return _FakeResp(p)


def test_openai_compatible_run_skill_no_retry_on_invalid_first_shot(tmp_path) -> None:
    """Speed-optimised: an unparseable first reply raises, it does NOT trigger a
    second round-trip. The harness layer handles the error gracefully."""
    from testing_agent_harness.models import ProviderError

    cfg = AgentConfig()
    cfg.model.provider = "openai_compatible"
    cfg.model.model_name = "qwen3"
    cfg.model.base_url = "http://127.0.0.1:11434/v1"

    spec = SkillSpec(
        name="noop",
        description="test",
        prompt="Return ok true.",
        allowed_tools=[],
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )
    state = _minimal_state(tmp_path)
    ctx = ToolContext(config=cfg, state=state, run_dir=tmp_path)
    logger = EventLogger(tmp_path / "events.jsonl", "test-run")
    tools = build_default_tool_registry()

    payloads = [
        {"choices": [{"message": {"role": "assistant", "content": "analysis complete"}}]},
    ]
    seq = _UrlOpenSeq(payloads)

    with patch("testing_agent_harness.models.urllib.request.urlopen", side_effect=seq):
        prov = OpenAICompatibleProvider(cfg)
        with pytest.raises(ProviderError, match="parseable JSON"):
            prov.run_skill(spec, {"q": 1}, tools, ctx, logger, stage="test")

    assert seq._i == 1, "fast mode must not attempt a second round-trip"


def test_openai_compatible_run_skill_json_fast_path(tmp_path) -> None:
    """When the first reply is already valid JSON we should skip the repair round."""
    cfg = AgentConfig()
    cfg.model.provider = "openai_compatible"
    cfg.model.model_name = "qwen3"
    cfg.model.base_url = "http://127.0.0.1:11434/v1"

    spec = SkillSpec(
        name="noop",
        description="test",
        prompt="Return ok true.",
        allowed_tools=[],
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )
    state = _minimal_state(tmp_path)
    ctx = ToolContext(config=cfg, state=state, run_dir=tmp_path)
    logger = EventLogger(tmp_path / "events.jsonl", "test-run")
    tools = build_default_tool_registry()

    payloads = [
        {"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}]},
    ]
    seq = _UrlOpenSeq(payloads)

    with patch("testing_agent_harness.models.urllib.request.urlopen", side_effect=seq):
        prov = OpenAICompatibleProvider(cfg)
        out = prov.run_skill(spec, {"q": 1}, tools, ctx, logger, stage="test")

    assert out == {"ok": True}
    assert seq._i == 1, "expected one round-trip when JSON is already valid"


def test_openai_compatible_run_skill_retries_on_length(tmp_path) -> None:
    cfg = AgentConfig()
    cfg.model.provider = "openai_compatible"
    cfg.model.model_name = "qwen3"
    cfg.model.base_url = "http://127.0.0.1:11434/v1"

    spec = SkillSpec(
        name="bug_localizer",
        description="test",
        prompt="Return ok true.",
        allowed_tools=[],
        output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    )
    state = _minimal_state(tmp_path)
    ctx = ToolContext(config=cfg, state=state, run_dir=tmp_path)
    logger = EventLogger(tmp_path / "events.jsonl", "test-run")
    tools = build_default_tool_registry()

    payloads = [
        {"choices": [{"message": {"role": "assistant", "content": '{"ok": '}, "finish_reason": "length"}]},
        {"choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}, "finish_reason": "stop"}]},
    ]
    seq = _UrlOpenSeq(payloads)

    with patch("testing_agent_harness.models.urllib.request.urlopen", side_effect=seq):
        prov = OpenAICompatibleProvider(cfg)
        out = prov.run_skill(spec, {"q": 1}, tools, ctx, logger, stage="test")

    assert out == {"ok": True}
    assert seq._i == 2


def test_openai_compatible_skill_max_tokens_for_truncation_prone_skills(tmp_path) -> None:
    cfg = AgentConfig()
    cfg.model.provider = "openai_compatible"
    cfg.model.model_name = "qwen3"
    cfg.model.base_url = "http://127.0.0.1:11434/v1"
    prov = OpenAICompatibleProvider(cfg)

    assert prov._skill_max_tokens("plan_builder") == 768
    assert prov._skill_max_tokens("workflow_router") == 512
    assert prov._skill_max_tokens("repair_decider") == 448
    assert prov._skill_max_tokens("bug_localizer") == 1024
    assert prov._skill_max_tokens("repair_patch") == 4680
