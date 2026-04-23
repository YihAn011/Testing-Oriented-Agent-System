from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import shutil
import typer

from .config import AgentConfig, default_config_path, load_config, save_config
from .harness import TestingHarness
from .schemas import RunState
from .tools import ToolContext, build_default_tool_registry

app = typer.Typer(help="CLI for the testing-oriented harness agent.")


def _load_effective_config(
    repo_path: Path,
    config_path: Optional[Path],
    provider: Optional[str],
    repair_mode: Optional[str],
    goal: Optional[str],
    target_coverage: Optional[float],
    apply_accepted: bool,
) -> AgentConfig:
    cfg_path = config_path or default_config_path(repo_path)
    config = load_config(cfg_path)
    if provider:
        config.model.provider = provider  # type: ignore[assignment]
        if provider == "openai_compatible" and config.model.model_name == "gemini-2.5-flash":
            config.model.model_name = os.environ.get("QWEN_MODEL", "qwen3")
    if repair_mode:
        config.policy.repair_mode = repair_mode  # type: ignore[assignment]
    if goal:
        config.goals.user_goal = goal
    if target_coverage is not None:
        config.goals.target_line_coverage = target_coverage
    if apply_accepted:
        config.policy.apply_accepted_patch_to_original = True
    return config


def _prompt_repair_mode(default_value: str) -> str:
    typer.echo("Choose repair behavior for this run:")
    typer.echo("  1) suggest_only  -> only localize bugs and recommend fixes")
    typer.echo("  2) auto          -> repair inside sandbox, then show diff for review")
    choice = typer.prompt("Enter 1 or 2", default="2" if default_value == "auto" else "1")
    return "auto" if choice.strip() == "2" else "suggest_only"


def _latest_run_dir(repo_path: Path) -> Path:
    root = repo_path / ".testing_agent_runs"
    if not root.exists():
        raise typer.BadParameter("No runs found for this repository.")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise typer.BadParameter("No runs found for this repository.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_state(repo_path: Path, run_id: Optional[str]) -> tuple[Path, RunState]:
    run_dir = (repo_path / ".testing_agent_runs" / run_id) if run_id else _latest_run_dir(repo_path)
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise typer.BadParameter(f"State file not found: {state_path}")
    return run_dir, RunState.load(state_path)


@app.command("init-config")
def init_config(repo_path: Path, output: Optional[Path] = None) -> None:
    """Write a starter configuration file into the target repository."""
    repo_path = repo_path.resolve()
    target = output or default_config_path(repo_path)
    config = AgentConfig()
    save_config(config, target)
    typer.echo(f"Wrote config to {target}")


@app.command("plan")
def plan_repo(
    repo_path: Path,
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config."),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini, mock, or openai_compatible (local Qwen / vLLM / Ollama)"),
    goal: Optional[str] = typer.Option(None, "--goal", help="Override testing goal."),
    target_coverage: Optional[float] = typer.Option(None, "--target-coverage", help="Override target line coverage."),
) -> None:
    repo_path = repo_path.resolve()
    config = _load_effective_config(repo_path, config_path, provider, None, goal, target_coverage, False)
    harness = TestingHarness(repo_path=repo_path, config=config)
    harness.bootstrap()
    harness.plan()
    typer.echo(f"Run ID: {harness.run_id}")
    typer.echo((harness.run_dir / 'plan.json').read_text(encoding='utf-8'))


@app.command("run")
def run_repo(
    repo_path: Path,
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to YAML config."),
    provider: Optional[str] = typer.Option(None, "--provider", help="gemini, mock, or openai_compatible (local Qwen / vLLM / Ollama)"),
    repair_mode: Optional[str] = typer.Option(None, "--repair-mode", help="ask, suggest_only, or auto"),
    goal: Optional[str] = typer.Option(None, "--goal", help="Override testing goal."),
    target_coverage: Optional[float] = typer.Option(None, "--target-coverage", help="Override target line coverage."),
    apply_accepted: bool = typer.Option(False, "--apply-accepted", help="Copy accepted sandbox changes back to the original repo."),
    show_report: bool = typer.Option(True, "--show-report/--no-show-report", help="Print final markdown report."),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-n",
        help="Skip prompts (repair mode choice and sandbox patch confirmation).",
    ),
) -> None:
    repo_path = repo_path.resolve()
    config = _load_effective_config(repo_path, config_path, provider, repair_mode, goal, target_coverage, apply_accepted)
    if non_interactive and (config.policy.repair_mode == "ask" or repair_mode == "ask"):
        config.policy.repair_mode = "auto"
    elif config.policy.repair_mode == "ask" or repair_mode == "ask":
        config.policy.repair_mode = _prompt_repair_mode(config.policy.repair_mode)
    harness = TestingHarness(repo_path=repo_path, config=config)
    state = harness.run_full()
    typer.echo(f"Run ID: {state.run_id}")
    typer.echo(f"Artifacts: {harness.run_dir}")
    if show_report and state.report_markdown:
        typer.echo(state.report_markdown)
    diff_payload = harness.diff_workspace()
    if diff_payload["changed_files"]:
        typer.echo("\nSandbox diff:\n")
        for item in diff_payload["diffs"]:
            typer.echo(item["diff"])
        if non_interactive:
            accepted = True
            typer.echo("Non-interactive mode: auto-accepting sandbox patch review (sandbox only unless --apply-accepted).")
        else:
            accepted = typer.confirm("Accept this sandbox patch as the recommended result?", default=True)
        if accepted:
            typer.echo(f"Accepted patch remains in sandbox at {state.sandbox_path}")
            if config.policy.apply_accepted_patch_to_original:
                _apply_sandbox_to_original(repo_path, Path(state.sandbox_path), diff_payload["changed_files"])
                typer.echo("Accepted changes were copied back to the original repository.")
        else:
            typer.echo("Patch not accepted. Original repository remains unchanged.")


@app.command("report")
def show_report(repo_path: Path, run_id: Optional[str] = typer.Option(None, "--run-id")) -> None:
    repo_path = repo_path.resolve()
    _, state = _load_state(repo_path, run_id)
    if not state.report_markdown:
        raise typer.BadParameter("Run has no markdown report yet.")
    typer.echo(state.report_markdown)


@app.command("diff")
def show_diff(repo_path: Path, run_id: Optional[str] = typer.Option(None, "--run-id")) -> None:
    repo_path = repo_path.resolve()
    _, state = _load_state(repo_path, run_id)
    run_dir = repo_path / ".testing_agent_runs" / state.run_id
    tools = build_default_tool_registry()
    ctx = ToolContext(config=AgentConfig(), state=state, run_dir=run_dir)
    payload = tools.invoke("diff_workspace", ctx, {})
    if not payload["changed_files"]:
        typer.echo("No workspace diff for this run.")
        return
    for item in payload["diffs"]:
        typer.echo(item["diff"])


@app.command("mcp-server")
def run_mcp_server(
    repo_path: Path,
    config_path: Optional[Path] = typer.Option(None, "--config"),
    provider: Optional[str] = typer.Option("mock", "--provider", help="gemini, mock, or openai_compatible"),
) -> None:
    """Start the experimental stdio MCP-style bridge for tools and skill resources."""
    from .mcp_server import MCPBridge

    repo_path = repo_path.resolve()
    config = _load_effective_config(repo_path, config_path, provider, None, None, None, False)
    bridge = MCPBridge(repo_path=repo_path, config=config)
    bridge.run_stdio()


@app.command("chat")
def chat_cmd(
    repo_path: Optional[Path] = typer.Argument(None, help="Optional repo to target from the start."),
    provider: Optional[str] = typer.Option(None, "--provider", help="mock, gemini, or openai_compatible"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model_name (e.g. qwen3)."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="OpenAI-compatible API root (ends with /v1)."),
) -> None:
    """Start an interactive chat session (like Codex CLI). Use slash commands to drive the harness."""
    from .chat import run_chat

    repo = repo_path.resolve() if repo_path is not None else None
    run_chat(initial_repo=repo, initial_provider=provider, initial_model=model, initial_base_url=base_url)


@app.command("resume")
def resume_run(repo_path: Path, run_id: Optional[str] = typer.Option(None, "--run-id")) -> None:
    """Load an existing run state and print the final report if available."""
    repo_path = repo_path.resolve()
    _, state = _load_state(repo_path, run_id)
    typer.echo(f"Loaded run {state.run_id}")
    if state.report_markdown:
        typer.echo(state.report_markdown)
    else:
        typer.echo("Run has not finished reporting yet.")


def _apply_sandbox_to_original(repo_path: Path, sandbox_path: Path, changed_files: list[str]) -> None:
    for rel in changed_files:
        source = sandbox_path / rel
        target = repo_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.copy2(source, target)
        elif target.exists():
            target.unlink()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
