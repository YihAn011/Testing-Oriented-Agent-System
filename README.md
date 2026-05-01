# Testing-Oriented Agent System

Repository: `https://github.com/YihAn011/Testing-Oriented-Agent-System.git`

This repository contains `testing-agent-harness`, a command-line testing-oriented agent harness for repository-level Python testing. The system treats testing as a staged process rather than a one-shot prompt. It scans a repository, builds a plan, runs baseline tests, parses failures, localizes likely bug files, optionally applies a sandboxed repair, refreshes coverage, and writes reproducible artifacts for later inspection.

The final project report is the source of truth for the system description and evaluation. This README is aligned to that report and organized for repository handoff: what is in the repository, how to build and run it, what external software it builds on, and how to reproduce the evaluation.

## What Is In The Repository

- `README.md`: setup, run, and evaluation overview.
- `ARCHITECTURE.md`: short architecture summary for the harness and workflow.
- `testing_agent_harness/`: harness source code, CLI, tools, provider layer, schemas, and prompts.
- `testing_agent_harness/prompts/`: YAML skill definitions and structured prompt files.
- `examples/buggy_calc/`: bundled benchmark repository with a small arithmetic bug.
- `examples/buggy_snake/`: bundled benchmark repository with gameplay logic bugs.
- `tests/`: pytest suite for normalization, provider behavior, sandboxing, repair behavior, MCP bridge behavior, and related harness logic.
- `docs/evaluation.md`: replication guide aligned to the final paper.
- `docs/ai_usage.md`: AI tooling disclosure aligned to the final paper.

## Repository Layout

### Main package

- `testing_agent_harness/cli.py`: Typer CLI entry points for `init-config`, `plan`, `run`, `report`, `diff`, `chat`, `resume`, and `mcp-server`.
- `testing_agent_harness/chat.py`: interactive chat interface with commands such as `/repo`, `/plan`, `/run`, `/report`, and `/diff`.
- `testing_agent_harness/harness.py`: orchestration layer for typed run state, budgets, sandboxing, rollback, routing, repair, and reporting.
- `testing_agent_harness/tools.py`: deterministic tools such as `project_scan`, `run_tests`, `run_coverage`, `failure_parse`, `apply_changes`, and `diff_workspace`.
- `testing_agent_harness/models.py`: provider implementations for `openai_compatible`, `gemini`, and `mock`.
- `testing_agent_harness/schemas.py`: Pydantic schemas for plans, route decisions, failures, patches, reports, and run state.
- `testing_agent_harness/config.py`: YAML-backed runtime configuration.
- `testing_agent_harness/mcp_server.py`: experimental MCP-style stdio bridge.

### Bundled benchmarks

- `examples/buggy_calc/`: one failing test at baseline; repair changes `return a - b` to `return a + b`.
- `examples/buggy_snake/`: two failing tests at baseline; repair changes wall-boundary logic and snake-growth logic in `src/terminal_snake/game.py`.

### Run artifacts

Each harness run writes artifacts under:

```text
<repo-under-test>/.testing_agent_runs/<run_id>/
```

Typical artifacts include:

- `state.json`
- `events.jsonl`
- `plan.json`
- `reports/final_report.md`
- `reports/final_report.json`
- `sandbox/`

These artifacts are part of the project’s reproducibility story and are explicitly discussed in the paper.

## External Software And Artifacts Built On

The project builds on standard open-source Python testing and CLI infrastructure plus local model-serving software:

- `pytest`
- `coverage.py`
- `Typer`
- `Pydantic`
- `PyYAML`
- local OpenAI-compatible LLM servers such as `Ollama`

The final paper emphasizes local `Qwen3` served through an OpenAI-compatible endpoint such as Ollama. `Gemini` compatibility remains in the code from earlier experiments, but the final design emphasizes local Qwen3 and the deterministic mock provider for reproducible testing.

## Build

### Recommended setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This is the setup sequence shown in the paper’s deliverables and reproducibility sections.

### Convenience bootstrap

The repository also includes:

```bash
./start.sh
```

`start.sh` creates the virtual environment if needed, installs the package in editable mode, loads `.env`, and starts the interactive CLI path.

### Test the harness repository itself

The final paper reports that after the final modifications the harness pytest suite reports:

```text
40 passed
```

Run it with:

```bash
pytest -q
```

Pytest may also report three collection warnings for classes whose names begin with `Test`; the paper notes that these warnings do not affect execution.

## Run

### Non-interactive benchmark run

The paper uses this form as the canonical example:

```bash
python -m testing_agent_harness.cli run examples/buggy_calc --provider mock --repair-mode auto -n
```

This runs the full workflow non-interactively on the bundled benchmark.

### Interactive chat mode

You can also launch the interactive interface and use commands such as:

```text
/repo examples/buggy_calc
/plan
/run
/report
/diff
```

### Local Qwen3 configuration

For local-Qwen runs, the paper describes using an OpenAI-compatible server such as Ollama at:

```text
http://127.0.0.1:11434/v1
```

An example configuration in the paper is:

```yaml
model:
  provider: openai_compatible
  model_name: qwen3
  base_url: http://127.0.0.1:11434/v1
budget:
  fast_mode: true
  max_iterations: 1
policy:
  repair_mode: auto
  apply_accepted_patch_to_original: false
```

By default, accepted changes remain in the sandbox unless the user explicitly asks to copy them back to the original repository.

## Evaluation Overview

The detailed replication guide is in [`docs/evaluation.md`](docs/evaluation.md).

The final paper evaluates the system on two bundled Python benchmarks:

- `examples/buggy_calc`
- `examples/buggy_snake`

It compares three conditions on the same repositories:

1. existing tests only
2. suggest-only harness
3. full harness

The reported result is:

- existing tests only: `0/2` repositories fixed
- suggest-only harness: `0/2` repositories fixed
- full harness: `2/2` repositories fixed

The paper’s case-study results also report:

- `buggy_calc`: failures reduced from `1` to `0`, final coverage `47.62%`
- `buggy_snake`: failures reduced from `2` to `0`, final coverage `77.59%`

## Safety And Control Behavior

The final paper emphasizes the following design properties:

- sandbox-first execution so the original repository is preserved by default
- typed run state and persisted artifacts for reproducibility
- explicit repair modes: `suggest_only`, `ask`, and `auto`
- path validation for model-generated file paths
- regression testing after repair
- rollback or rejection behavior for bad patches
- normalization layers and deterministic fallbacks around LLM outputs

The key contribution is not just another test-generation prompt. It is a process-driven harness in which deterministic tools own execution, safety, and state, while the LLM is limited to constrained reasoning tasks with structured outputs.

## Documentation Handoff

A third party taking over the project should start with:

1. `README.md`
2. `ARCHITECTURE.md`
3. `testing_agent_harness/`
4. `examples/buggy_calc/` and `examples/buggy_snake/`
5. `tests/`
6. `docs/evaluation.md`
7. `docs/ai_usage.md`

The repository is intended to be inspectable and reproducible, not just demoable from a terminal session.
