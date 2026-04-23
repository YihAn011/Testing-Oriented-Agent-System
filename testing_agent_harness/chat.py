"""Interactive chat-style REPL for the testing agent harness.

Slash commands do deterministic work (switch repo/provider, run plan/full harness,
show reports and diffs). Any other input is sent to the LLM provider as free-form
chat, with a system prompt telling it about the slash commands so it can suggest
next steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import os
import shlex
import sys
import urllib.error
import urllib.request

import typer

from .config import AgentConfig, default_config_path, load_config, save_config
from .harness import TestingHarness
from .models import ProviderError
from .schemas import RunState
from .spinner import WaveSpinner
from .tools import ToolContext, build_default_tool_registry


PROVIDERS: list[tuple[str, str]] = [
    ("openai_compatible", "Local OpenAI-compatible server (Ollama / vLLM / LM Studio) — e.g. Qwen3"),
    ("gemini", "Google Gemini REST API (GEMINI_API_KEY)"),
    ("mock", "Offline deterministic stub for tests / CI"),
]


KNOWN_MODELS: dict[str, list[str]] = {
    "gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    "openai_compatible": [
        "qwen3",
        "qwen3:8b",
        "qwen3:14b",
        "qwen3:32b",
        "qwen2.5:7b",
        "qwen2.5-coder:7b",
        "llama3.1:8b",
        "llama3.2",
        "mistral",
        "mixtral",
        "deepseek-r1",
        "deepseek-coder-v2",
        "gemma2",
        "phi3",
    ],
    "mock": [],
}


SYSTEM_PROMPT = """You are an assistant embedded in a CLI testing harness called "testing-agent-harness".
You do NOT execute code yourself inside chat messages. Instead, tell the user which slash command to run.

Available slash commands (typed by the user in this chat):
  /help                       show help
  /repo <path>                set the repository under test
  /pwd                        print the current repo path
  /provider [name]            list providers, or set provider
  /providers                  list providers
  /model [name]               list available models, or set model
  /models                     list available models
  /base-url <url>             set OpenAI-compatible base URL (ends with /v1)
  /goal <text>                set the testing goal
  /coverage <float>           target line coverage (e.g. 0.85)
  /repair <ask|suggest_only|auto>   repair behavior
  /budget <max_iterations>    override iteration budget
  /plan                       run the plan stage only
  /run                        run the full harness on the current repo
  /report                     print the latest markdown report
  /diff                       print the latest sandbox diff
  /runs                       list past runs for the current repo
  /log [n]                    tail the latest run's events.jsonl (default 20 lines)
  /status                     show current config and latest run summary
  /config                     print current in-memory config as YAML
  /init                       write default config into the current repo
  /reset                      reset in-memory config to defaults
  /ls [subdir]                list files in the repo
  /open <path>                print a file from the repo
  /history                    print chat history
  /save [path]                save chat history as JSON
  /clear                      clear chat history
  /version                    print package version
  /quit                       exit

Be concise. When the user describes a goal, suggest the concrete /commands to run.

/no_think"""


@dataclass
class ChatState:
    repo_path: Optional[Path] = None
    config: AgentConfig = field(default_factory=AgentConfig)
    history: list[dict[str, str]] = field(default_factory=list)
    last_run_id: Optional[str] = None

    def ensure_repo(self) -> Path:
        if self.repo_path is None:
            raise typer.BadParameter("No repo selected. Use: /repo <path>")
        return self.repo_path


def _print(msg: str, color: str | None = None) -> None:
    if color:
        typer.secho(msg, fg=color)
    else:
        typer.echo(msg)


_PROVIDER_LABELS = {
    "openai_compatible": "local LLM (OpenAI Chat Completions protocol; e.g. Ollama/vLLM)",
    "gemini": "Google Gemini REST API",
    "mock": "offline mock",
}


def _banner(state: ChatState) -> None:
    _print("testing-agent-harness chat. Type /help for commands, /quit to exit.", color=typer.colors.CYAN)
    _print(f"  repo:     {state.repo_path or '(not set — /repo <path>)'}")
    prov = state.config.model.provider
    _print(f"  provider: {prov}  ({_PROVIDER_LABELS.get(prov, '?')})")
    _print(f"  model:    {state.config.model.model_name}")
    if prov == "openai_compatible":
        _print(f"  base_url: {state.config.model.base_url}  (your local server; no OpenAI cloud is called)")


def _latest_run_dir(repo_path: Path) -> Optional[Path]:
    root = repo_path / ".testing_agent_runs"
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _load_latest_state(repo_path: Path) -> Optional[RunState]:
    rd = _latest_run_dir(repo_path)
    if rd is None:
        return None
    sp = rd / "state.json"
    return RunState.load(sp) if sp.exists() else None


def _fetch_openai_models(base_url: str, timeout: float = 2.0) -> list[str]:
    """Query an OpenAI-compatible server's /models. Empty list if unreachable."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return []
    items = data.get("data") or []
    ids = [str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")]
    return sorted({i for i in ids if i})


HELP_TEXT = SYSTEM_PROMPT.split("Available slash commands", 1)[1]
HELP_TEXT = HELP_TEXT.split(":", 1)[1].strip()


def _cmd_help(_state: ChatState, _args: list[str]) -> None:
    _print("Commands:\n" + HELP_TEXT)


def _apply_env_overrides(state: ChatState) -> None:
    """Re-apply environment variable overrides (set by start.sh / .env).

    Loading a repo's ``.testing_agent.yaml`` must not silently downgrade the
    provider the user selected at session start. If ``TEST_AGENT_PROVIDER`` is
    set we honor it; the model name and base_url follow the same rule.
    """
    env_provider = os.environ.get("TEST_AGENT_PROVIDER")
    if env_provider in {"mock", "gemini", "openai_compatible"}:
        state.config.model.provider = env_provider  # type: ignore[assignment]
        if env_provider == "openai_compatible":
            if state.config.model.model_name in ("", "gemini-2.5-flash"):
                state.config.model.model_name = os.environ.get("QWEN_MODEL", "qwen3")
            base = os.environ.get("OLLAMA_BASE_URL")
            if base:
                state.config.model.base_url = base


def _cmd_repo(state: ChatState, args: list[str]) -> None:
    if not args:
        raise typer.BadParameter("Usage: /repo <path>")
    p = Path(args[0]).expanduser().resolve()
    if not p.is_dir():
        raise typer.BadParameter(f"Not a directory: {p}")
    state.repo_path = p
    cfg_path = default_config_path(p)
    previous_provider = state.config.model.provider
    previous_model = state.config.model.model_name
    previous_base_url = state.config.model.base_url
    if cfg_path.exists():
        loaded = load_config(cfg_path)
        if previous_provider != AgentConfig().model.provider or loaded.model.provider == previous_provider:
            loaded.model.provider = previous_provider  # type: ignore[assignment]
            if previous_provider == "openai_compatible":
                loaded.model.model_name = previous_model
                loaded.model.base_url = previous_base_url
        state.config = loaded
    _apply_env_overrides(state)
    _print(f"repo set to {p}", color=typer.colors.GREEN)
    _print(
        f"  (provider={state.config.model.provider} model={state.config.model.model_name})",
        color=typer.colors.BRIGHT_BLACK,
    )


def _cmd_pwd(state: ChatState, _args: list[str]) -> None:
    _print(str(state.repo_path or "(repo not set)"))


def _list_providers(state: ChatState) -> None:
    _print("Available providers:")
    for name, desc in PROVIDERS:
        marker = "*" if name == state.config.model.provider else " "
        _print(f"  {marker} {name:<20} {desc}")
    _print("\nUse: /provider <name>")


def _cmd_provider(state: ChatState, args: list[str]) -> None:
    if not args:
        _list_providers(state)
        return
    if args[0] not in {"mock", "gemini", "openai_compatible"}:
        raise typer.BadParameter("Usage: /provider mock|gemini|openai_compatible")
    state.config.model.provider = args[0]  # type: ignore[assignment]
    if args[0] == "openai_compatible" and state.config.model.model_name == "gemini-2.5-flash":
        state.config.model.model_name = "qwen3"
    _print(f"provider set to {args[0]} (model={state.config.model.model_name})", color=typer.colors.GREEN)


def _list_models(state: ChatState) -> None:
    provider = state.config.model.provider
    current = state.config.model.model_name
    _print(f"Models for provider '{provider}':")

    live: list[str] = []
    if provider == "openai_compatible":
        live = _fetch_openai_models(state.config.model.base_url)
        if live:
            _print(f"  (live from {state.config.model.base_url}/models)")
            for name in live:
                marker = "*" if name == current else " "
                _print(f"  {marker} {name}")

    curated = KNOWN_MODELS.get(provider, [])
    remaining = [m for m in curated if m not in live]
    if remaining:
        header = "  (curated presets)" if live else "  (presets)"
        _print(header)
        for name in remaining:
            marker = "*" if name == current else " "
            _print(f"  {marker} {name}")

    if not live and not curated:
        _print("  (no models to list for this provider)")

    _print("\nUse: /model <name>")


def _cmd_model(state: ChatState, args: list[str]) -> None:
    if not args:
        _list_models(state)
        return
    state.config.model.model_name = args[0]
    _print(f"model set to {args[0]}", color=typer.colors.GREEN)


def _cmd_base_url(state: ChatState, args: list[str]) -> None:
    if not args:
        raise typer.BadParameter("Usage: /base-url <url>")
    state.config.model.base_url = args[0]
    _print(f"base_url set to {args[0]}", color=typer.colors.GREEN)


def _cmd_goal(state: ChatState, args: list[str]) -> None:
    if not args:
        raise typer.BadParameter("Usage: /goal <text>")
    state.config.goals.user_goal = " ".join(args)
    _print("goal updated.", color=typer.colors.GREEN)


def _cmd_coverage(state: ChatState, args: list[str]) -> None:
    if not args:
        raise typer.BadParameter("Usage: /coverage <float 0-1>")
    state.config.goals.target_line_coverage = float(args[0])
    _print(f"target coverage = {state.config.goals.target_line_coverage}", color=typer.colors.GREEN)


def _cmd_repair(state: ChatState, args: list[str]) -> None:
    if not args or args[0] not in {"ask", "suggest_only", "auto"}:
        raise typer.BadParameter("Usage: /repair ask|suggest_only|auto")
    state.config.policy.repair_mode = args[0]  # type: ignore[assignment]
    _print(f"repair_mode = {args[0]}", color=typer.colors.GREEN)


def _cmd_budget(state: ChatState, args: list[str]) -> None:
    if not args:
        raise typer.BadParameter("Usage: /budget <max_iterations>")
    try:
        n = int(args[0])
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    state.config.budget.max_iterations = n
    _print(f"budget.max_iterations = {n}", color=typer.colors.GREEN)


def _cmd_init(state: ChatState, _args: list[str]) -> None:
    repo = state.ensure_repo()
    save_config(state.config, default_config_path(repo))
    _print(f"wrote config to {default_config_path(repo)}", color=typer.colors.GREEN)


def _cmd_reset(state: ChatState, _args: list[str]) -> None:
    state.config = AgentConfig()
    _print("in-memory config reset to defaults.", color=typer.colors.GREEN)


class _LiveStageReporter:
    """Prints stage headers and streams LLM tokens live under each skill call.

    Replaces the old opaque spinner. Every time the harness:
      * enters a stage  -> print a cyan header line
      * invokes a skill -> print a dim indented header + stream the LLM's
                           raw tokens in dim style as they arrive
      * leaves a stage  -> print a green ``done`` marker

    Network retries and connection status are printed inline as well so the
    user always knows what the model is doing.
    """

    def __init__(self, use_color: bool = True) -> None:
        self._use_color = use_color and sys.stdout.isatty()
        self._current_stage = ""
        self._skill_open = False  # are we currently inside a streamed skill call?
        self._skill_emitted_token = False  # has the current skill already printed text?
        self._in_think = False    # are we inside a <think>...</think> block?
        self._think_buf = ""      # detect think tags that arrive token-split

    def _c(self, text: str, color: Optional[str] = None, dim: bool = False) -> str:
        if not self._use_color:
            return text
        if not color and not dim:
            return text
        return typer.style(text, fg=color, dim=dim)

    def _write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_stage(self, stage: str, status: str, payload) -> None:  # type: ignore[no-untyped-def]
        if status == "started":
            self._close_skill_line()
            self._current_stage = stage
            self._write("\n" + self._c(f"● {stage}", typer.colors.CYAN) + "\n")
        elif status == "completed":
            self._close_skill_line()
            self._write(self._c(f"  ✓ {stage} done", typer.colors.GREEN) + "\n")
        elif status in {"skipped", "reverted", "review_warning"}:
            self._close_skill_line()
            tag = self._c(f"  • {stage} {status}", typer.colors.YELLOW)
            self._write(tag + "\n")
        elif status == "failed":
            self._close_skill_line()
            self._write(self._c(f"  × {stage} failed (continuing)", typer.colors.RED) + "\n")

    def on_skill_started(self, skill_name: str) -> None:
        self._close_skill_line()
        header = self._c(f"  ▸ {skill_name}", typer.colors.MAGENTA)
        self._write(header + "\n    ")
        self._skill_open = True
        self._skill_emitted_token = False
        self._in_think = False
        self._think_buf = ""

    def on_skill_completed(self, skill_name: str) -> None:  # noqa: ARG002
        # By the time we get here, the wrapper has already printed a fallback
        # text dump if streaming produced nothing, so just close the block.
        self._close_skill_line()

    def saw_any_tokens(self) -> bool:
        return self._skill_emitted_token

    def on_fallback_text(self, text: str) -> None:
        # Used when streaming silently fell back to a unary HTTP request
        # (e.g. length-truncation retry). We still want the user to see the
        # skill's parsed result inside the skill block.
        if not self._skill_open or not text:
            return
        rendered = text.replace("\n", "\n    ")
        self._write(self._c(rendered, typer.colors.WHITE, dim=True))
        self._skill_emitted_token = True

    def _close_skill_line(self) -> None:
        if self._skill_open:
            self._write("\n")
            self._skill_open = False
            self._skill_emitted_token = False
            self._in_think = False
            self._think_buf = ""

    def on_token(self, piece: str) -> None:
        # Rewrite newlines with indentation so streamed JSON keeps the "inside
        # a skill block" visual. Fold Qwen3's <think>...</think> chain-of-thought
        # into a muted grey so users can see it without drowning in it.
        if not self._skill_open:
            return
        piece = piece.replace("\r", "")
        out_parts: list[str] = []
        i = 0
        while i < len(piece):
            if not self._in_think and piece[i : i + 7] == "<think>":
                self._in_think = True
                i += 7
                continue
            if self._in_think and piece[i : i + 8] == "</think>":
                self._in_think = False
                i += 8
                continue
            out_parts.append(piece[i])
            i += 1
        rendered = "".join(out_parts)
        if not rendered:
            return
        self._skill_emitted_token = True
        rendered = rendered.replace("\n", "\n    ")
        if self._in_think:
            self._write(self._c(rendered, typer.colors.BRIGHT_BLACK, dim=True))
        else:
            self._write(self._c(rendered, typer.colors.WHITE, dim=True))

    def on_status(self, msg: str) -> None:
        # Only real events (HTTP retries) reach us now — "connecting" /
        # "thinking" hints were removed upstream. Print the event on a fresh
        # indented line WITHOUT closing the surrounding skill block, so the
        # next streamed tokens continue to render correctly.
        if self._skill_open:
            self._write("\n    " + self._c(f"· {msg}", typer.colors.YELLOW) + "\n    ")
        else:
            self._write(self._c(f"    · {msg}", typer.colors.YELLOW) + "\n")


def _attach_live_reporter(harness: TestingHarness) -> _LiveStageReporter:
    """Hook a ``_LiveStageReporter`` onto the harness: stage events, skill
    events, streamed LLM tokens, and HTTP retry messages all flow through it."""
    reporter = _LiveStageReporter()

    original_emit = harness.emit_stage

    def emit_with_live(stage: str, status: str, payload=None):  # type: ignore[no-untyped-def]
        reporter.on_stage(stage, status, payload)
        return original_emit(stage, status, payload)

    harness.emit_stage = emit_with_live  # type: ignore[method-assign]

    # Wrap skill_runner.run so we print ``▸ skill_name`` markers around each
    # LLM call and enable token streaming on the provider for just that call.
    original_skill_run = harness.skill_runner.run

    def run_with_live(skill_name: str, stage: str, ctx, payload):  # type: ignore[no-untyped-def]
        reporter.on_skill_started(skill_name)
        harness.provider.token_callback = reporter.on_token
        result = None
        try:
            result = original_skill_run(skill_name, stage, ctx, payload)
            return result
        finally:
            harness.provider.token_callback = None
            # If the provider took a non-streaming path (e.g. an internal
            # length-truncation retry) the reporter never saw any tokens.
            # Dump the parsed JSON so the user still sees the skill output.
            if result is not None and not reporter.saw_any_tokens():
                try:
                    dump = json.dumps(result, ensure_ascii=False, indent=2)
                    reporter.on_fallback_text(dump)
                except Exception:  # noqa: BLE001
                    pass
            reporter.on_skill_completed(skill_name)

    harness.skill_runner.run = run_with_live  # type: ignore[method-assign]

    try:
        harness.provider.status_callback = reporter.on_status
    except Exception:  # noqa: BLE001
        pass
    return reporter


# Backwards-compat shim: older code paths still import this name.
def _attach_stage_spinner(harness: TestingHarness, spinner: WaveSpinner) -> None:  # noqa: ARG001
    _attach_live_reporter(harness)


def _cmd_plan(state: ChatState, _args: list[str]) -> None:
    repo = state.ensure_repo()
    harness = TestingHarness(repo_path=repo, config=state.config)
    _attach_live_reporter(harness)
    try:
        harness.bootstrap()
        harness.plan()
    except Exception as exc:
        _print(f"\nplan error: {exc}", color=typer.colors.RED)
        return
    state.last_run_id = harness.run_id
    _print(f"\nplan generated. Run ID: {harness.run_id}", color=typer.colors.GREEN)
    _print((harness.run_dir / "plan.json").read_text(encoding="utf-8"))


def _cmd_run(state: ChatState, _args: list[str]) -> None:
    repo = state.ensure_repo()
    if state.config.policy.repair_mode == "ask":
        state.config.policy.repair_mode = "auto"
    harness = TestingHarness(repo_path=repo, config=state.config)
    _attach_live_reporter(harness)
    rs = None
    try:
        rs = harness.run_full()
    except Exception as exc:
        _print(f"\nrun error: {exc}", color=typer.colors.RED)
        _print(f"partial artifacts: {harness.run_dir}", color=typer.colors.BRIGHT_BLACK)
        return
    state.last_run_id = rs.run_id
    _print(f"\nrun complete. id={rs.run_id}  artifacts={harness.run_dir}", color=typer.colors.GREEN)
    if rs.report_markdown:
        _print(rs.report_markdown)
    diff = harness.diff_workspace()
    if diff["changed_files"]:
        _print("\nSandbox diff:")
        for item in diff["diffs"]:
            _print(item["diff"])
    else:
        _print("(no sandbox diff)")


def _cmd_report(state: ChatState, _args: list[str]) -> None:
    repo = state.ensure_repo()
    rs = _load_latest_state(repo)
    if rs is None or not rs.report_markdown:
        _print("no report yet for this repo.", color=typer.colors.YELLOW)
        return
    _print(rs.report_markdown)


def _cmd_diff(state: ChatState, _args: list[str]) -> None:
    repo = state.ensure_repo()
    rs = _load_latest_state(repo)
    if rs is None:
        _print("no runs yet for this repo.", color=typer.colors.YELLOW)
        return
    run_dir = repo / ".testing_agent_runs" / rs.run_id
    tools = build_default_tool_registry()
    ctx = ToolContext(config=state.config, state=rs, run_dir=run_dir)
    payload = tools.invoke("diff_workspace", ctx, {})
    if not payload["changed_files"]:
        _print("no workspace diff for the latest run.")
        return
    for item in payload["diffs"]:
        _print(item["diff"])


def _cmd_runs(state: ChatState, _args: list[str]) -> None:
    repo = state.ensure_repo()
    root = repo / ".testing_agent_runs"
    if not root.exists():
        _print("no runs yet for this repo.", color=typer.colors.YELLOW)
        return
    rows: list[tuple[float, str, bool]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        sp = p / "state.json"
        completed = False
        if sp.exists():
            try:
                completed = bool(RunState.load(sp).completed)
            except Exception:  # noqa: BLE001
                pass
        rows.append((p.stat().st_mtime, p.name, completed))
    if not rows:
        _print("no runs yet for this repo.", color=typer.colors.YELLOW)
        return
    rows.sort(reverse=True)
    _print("Runs (newest first):")
    for mtime, name, completed in rows:
        mark = "ok " if completed else "... "
        _print(f"  {mark} {name}  (mtime={int(mtime)})")


def _cmd_log(state: ChatState, args: list[str]) -> None:
    repo = state.ensure_repo()
    rd = _latest_run_dir(repo)
    if rd is None:
        _print("no runs yet for this repo.", color=typer.colors.YELLOW)
        return
    events = rd / "events.jsonl"
    if not events.exists():
        _print("no events.jsonl for the latest run.", color=typer.colors.YELLOW)
        return
    n = 20
    if args:
        try:
            n = max(1, int(args[0]))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    lines = events.read_text(encoding="utf-8").splitlines()
    for line in lines[-n:]:
        _print(line)


def _cmd_status(state: ChatState, _args: list[str]) -> None:
    _print(f"repo:     {state.repo_path or '(not set)'}")
    _print(f"provider: {state.config.model.provider}")
    _print(f"model:    {state.config.model.model_name}")
    _print(f"base_url: {state.config.model.base_url}")
    _print(f"goal:     {state.config.goals.user_goal}")
    _print(f"coverage: {state.config.goals.target_line_coverage}")
    _print(f"repair:   {state.config.policy.repair_mode}")
    _print(f"budget:   max_iterations={state.config.budget.max_iterations}")
    if state.repo_path:
        rs = _load_latest_state(state.repo_path)
        if rs:
            _print(f"last run: {rs.run_id}  completed={rs.completed}")


def _cmd_config(state: ChatState, _args: list[str]) -> None:
    import yaml

    _print(yaml.safe_dump(state.config.to_dict(), sort_keys=False))


def _cmd_ls(state: ChatState, args: list[str]) -> None:
    repo = state.ensure_repo()
    sub = args[0] if args else "."
    base = (repo / sub).resolve()
    if not str(base).startswith(str(repo)):
        raise typer.BadParameter("Path escapes repo.")
    if not base.exists():
        raise typer.BadParameter(f"Not found: {base}")
    if base.is_file():
        _print(str(base.relative_to(repo)))
        return
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build", ".testing_agent_runs"}
    count = 0
    for p in sorted(base.rglob("*")):
        if any(part in ignored for part in p.relative_to(repo).parts):
            continue
        if p.is_file():
            _print(str(p.relative_to(repo)))
            count += 1
            if count >= 300:
                _print(f"... (truncated at {count} entries)")
                return


def _cmd_open(state: ChatState, args: list[str]) -> None:
    repo = state.ensure_repo()
    if not args:
        raise typer.BadParameter("Usage: /open <path>")
    target = (repo / args[0]).resolve()
    if not str(target).startswith(str(repo)):
        raise typer.BadParameter("Path escapes repo.")
    if not target.is_file():
        raise typer.BadParameter(f"Not a file: {target}")
    max_chars = 12000
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated at {max_chars} chars)"
    _print(f"--- {target.relative_to(repo)} ---")
    _print(text)


def _cmd_history(state: ChatState, _args: list[str]) -> None:
    if not state.history:
        _print("(chat history is empty)")
        return
    for turn in state.history:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        tag = "you" if role == "user" else "assistant"
        _print(f"[{tag}] {content}")


def _cmd_save(state: ChatState, args: list[str]) -> None:
    target = Path(args[0]) if args else Path.cwd() / "chat_history.json"
    target.write_text(json.dumps(state.history, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(f"saved {len(state.history)} turns to {target}", color=typer.colors.GREEN)


def _cmd_clear(state: ChatState, _args: list[str]) -> None:
    state.history.clear()
    _print("chat history cleared.", color=typer.colors.GREEN)


def _cmd_version(_state: ChatState, _args: list[str]) -> None:
    try:
        from importlib.metadata import version

        _print(f"testing-agent-harness {version('testing-agent-harness')}")
    except Exception:  # noqa: BLE001
        _print("testing-agent-harness (version unknown)")


COMMANDS = {
    "/help": _cmd_help,
    "/repo": _cmd_repo,
    "/pwd": _cmd_pwd,
    "/provider": _cmd_provider,
    "/providers": lambda s, a: _list_providers(s),
    "/model": _cmd_model,
    "/models": lambda s, a: _list_models(s),
    "/base-url": _cmd_base_url,
    "/goal": _cmd_goal,
    "/coverage": _cmd_coverage,
    "/repair": _cmd_repair,
    "/budget": _cmd_budget,
    "/init": _cmd_init,
    "/reset": _cmd_reset,
    "/plan": _cmd_plan,
    "/run": _cmd_run,
    "/report": _cmd_report,
    "/diff": _cmd_diff,
    "/runs": _cmd_runs,
    "/log": _cmd_log,
    "/status": _cmd_status,
    "/config": _cmd_config,
    "/ls": _cmd_ls,
    "/open": _cmd_open,
    "/history": _cmd_history,
    "/save": _cmd_save,
    "/clear": _cmd_clear,
    "/version": _cmd_version,
}


def _handle_slash(state: ChatState, line: str) -> bool:
    """Return True if the loop should continue, False to quit."""
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        _print(f"parse error: {exc}", color=typer.colors.RED)
        return True
    if not parts:
        return True
    cmd = parts[0].lower()
    args = parts[1:]
    if cmd in {"/quit", "/exit"}:
        return False
    handler = COMMANDS.get(cmd)
    if handler is None:
        _print(f"unknown command: {cmd}. Type /help.", color=typer.colors.YELLOW)
        return True
    try:
        handler(state, args)
    except typer.BadParameter as exc:
        _print(str(exc), color=typer.colors.RED)
    except Exception as exc:  # noqa: BLE001
        _print(f"error: {exc}", color=typer.colors.RED)
    return True


class _StreamPrinter:
    """Prints streamed tokens to stdout, hiding ``<think>...</think>`` under a dim style.

    Holds back the last few characters of the buffer to avoid splitting a tag
    across chunk boundaries (e.g. ``<thi`` + ``nk>`` arrives in two chunks).
    """

    OPEN = "<think>"
    CLOSE = "</think>"
    HOLDBACK = max(len(OPEN), len(CLOSE)) - 1  # keep 7 chars to detect any tag prefix
    DIM = "\x1b[2m"
    RESET = "\x1b[0m"

    def __init__(self, enable_color: bool) -> None:
        self.enable_color = enable_color
        self.buffer = ""
        self.any_output = False
        self._in_think = False

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self.any_output = True
        self.buffer += chunk
        self._drain(final=False)

    def flush_rest(self) -> None:
        self._drain(final=True)
        if self._in_think and self.enable_color:
            sys.stdout.write(self.RESET)
        sys.stdout.flush()

    def _drain(self, final: bool) -> None:
        while self.buffer:
            if self._in_think:
                end = self.buffer.find(self.CLOSE)
                if end == -1:
                    safe = len(self.buffer) if final else max(0, len(self.buffer) - self.HOLDBACK)
                    if safe:
                        self._emit(self.buffer[:safe], dim=True)
                        self.buffer = self.buffer[safe:]
                    return
                self._emit(self.buffer[:end], dim=True)
                if self.enable_color:
                    sys.stdout.write(self.RESET)
                self.buffer = self.buffer[end + len(self.CLOSE):]
                self._in_think = False
            else:
                start = self.buffer.find(self.OPEN)
                if start == -1:
                    safe = len(self.buffer) if final else max(0, len(self.buffer) - self.HOLDBACK)
                    if safe:
                        self._emit(self.buffer[:safe], dim=False)
                        self.buffer = self.buffer[safe:]
                    return
                if start:
                    self._emit(self.buffer[:start], dim=False)
                self.buffer = self.buffer[start + len(self.OPEN):]
                self._in_think = True
                if self.enable_color:
                    sys.stdout.write(self.DIM)

    def _emit(self, text: str, dim: bool) -> None:
        if not text:
            return
        if dim and self.enable_color:
            sys.stdout.write(self.DIM + text + self.RESET)
        else:
            sys.stdout.write(text)
        sys.stdout.flush()


def _handle_chat(state: ChatState, line: str) -> None:
    provider_name = state.config.model.provider
    try:
        harness = TestingHarness(
            repo_path=state.repo_path or Path.cwd(),
            config=state.config,
            provider_name=provider_name,
        )
        provider = harness.provider
    except Exception as exc:  # noqa: BLE001
        _print(f"failed to init provider: {exc}", color=typer.colors.RED)
        return

    context_lines = [
        f"Current provider: {state.config.model.provider}",
        f"Current model:    {state.config.model.model_name}",
        f"Current repo:     {state.repo_path or '(not set)'}",
        f"Target coverage:  {state.config.goals.target_line_coverage}",
        f"Repair mode:      {state.config.policy.repair_mode}",
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + "\n".join(context_lines)},
    ]
    messages.extend(state.history)
    messages.append({"role": "user", "content": line})

    spinner = WaveSpinner(stream=sys.stderr, status="connecting")
    spinner.start()
    if not spinner._enabled:  # non-TTY fallback: tell user something is happening
        typer.echo("thinking…", err=True)
    printer = _StreamPrinter(enable_color=sys.stdout.isatty())
    first_delta = {"seen": False}

    def on_status(status: str) -> None:
        spinner.set_status(status)

    def on_delta(chunk: str) -> None:
        if not first_delta["seen"]:
            first_delta["seen"] = True
            spinner.set_status("streaming")
            spinner.stop()
            sys.stdout.write("\n")
            sys.stdout.flush()
        printer.feed(chunk)

    reply = ""
    try:
        reply = provider.chat_stream(messages, on_delta=on_delta, on_status=on_status)
    except ProviderError as exc:
        spinner.stop()
        _print(f"\nprovider error: {exc}", color=typer.colors.RED)
        return
    except NotImplementedError:
        spinner.stop()
        _print("this provider does not support free-form chat yet.", color=typer.colors.YELLOW)
        return
    except KeyboardInterrupt:
        spinner.stop()
        _print("\n(interrupted)", color=typer.colors.YELLOW)
        return
    finally:
        spinner.stop()

    printer.flush_rest()
    if printer.any_output:
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        _print(reply)

    state.history.append({"role": "user", "content": line})
    state.history.append({"role": "assistant", "content": reply})


def run_chat(
    initial_repo: Optional[Path],
    initial_provider: Optional[str],
    initial_model: Optional[str],
    initial_base_url: Optional[str] = None,
) -> None:
    state = ChatState()
    if initial_repo is not None:
        _cmd_repo(state, [str(initial_repo)])
    if initial_provider:
        _cmd_provider(state, [initial_provider])
    if initial_model:
        _cmd_model(state, [initial_model])
    if initial_base_url:
        _cmd_base_url(state, [initial_base_url])

    _banner(state)
    while True:
        try:
            line = input(typer.style("\nyou> ", fg=typer.colors.BRIGHT_BLUE))
        except (EOFError, KeyboardInterrupt):
            _print("")
            return
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if not _handle_slash(state, line):
                return
            continue
        _handle_chat(state, line)
