# Testing-Oriented Agent System

A CLI-based, process-driven agent that **plans, generates, runs, localizes, repairs, and reports** Python tests end-to-end, powered by a **local Qwen3** model through an OpenAI-compatible server (Ollama / vLLM / LM Studio / llama.cpp). No cloud key required.

> **One repo, one command**:
> ```bash
> ./start.sh
> ```
> spins up the venv, installs deps, loads `.env`, and drops you into an interactive chat where `/run` drives the whole pipeline against any repository.

---

## Why this shape

This is **not** a free-form ReAct agent. The harness is the source of truth and the LLM is a constrained scheduler / reviewer. The paper-inspired pipeline is:

1. **Environment & reproducibility** — build a manifest of how to run the repo.
2. **Plan & goals** — structured objectives, budgets, workflows.
3. **Workflow routing** — decide what to do next based on test state.
4. **Test generation & iterative improvement** — grow coverage without regressions.
5. **Execution feedback** — capture stdout/stderr/stack, trim, normalize.
6. **Failure & bug localization** — rank suspects with path validation.
7. **Repair** — minimal patch, rerun tests, **auto-revert if it doesn't help**.
8. **Report** — Markdown + JSON artifacts + sandbox diff.

Every meaningful action is either a **deterministic tool** (Python) or a **skill** (LLM prompt with a strict output schema). The harness owns state, budgets, sandboxing, rollback, and logging.

---

## Architecture at a glance

```
┌──────────────────────────────────────────────────────────────┐
│                         Harness                              │
│  state · budgets · sandbox · diff · rollback · logs          │
└───────────────┬───────────────────────────┬──────────────────┘
                │                           │
        ┌───────▼────────┐         ┌────────▼────────┐
        │   Tools (py)   │         │  Skills (LLM)   │
        │ project_scan   │         │ plan_builder    │
        │ run_tests      │         │ workflow_router │
        │ run_coverage   │         │ test_generation │
        │ failure_parse  │         │ bug_localizer   │
        │ apply_changes  │         │ repair_patch    │
        │ diff_workspace │         │ final_judge     │
        │ ...            │         │ report_writer   │
        └────────────────┘         └─────────────────┘
                                            │
                                   ┌────────▼────────┐
                                   │    Provider     │
                                   │ openai_compat.  │ ← Ollama / vLLM / LM Studio
                                   │ gemini (opt.)   │
                                   │ mock (tests)    │
                                   └─────────────────┘
```

---

## Quick start

### 1. One-click (recommended)

```bash
./start.sh
```

What it does:

- creates `.venv/` if missing
- `pip install -e .`
- loads `.env`
- launches the interactive chat REPL (provider = `openai_compatible`, model = `qwen3` by default)

Inside the chat:

```
you> /repo examples/buggy_calc
you> /run
```

You'll see per-stage headers, **live-streamed LLM tokens** (dimmed, `<think>…</think>` folded into grey), and a final Markdown report with sandbox diff.

### 2. Manual

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # fill in values (local Qwen3 defaults work out of the box with Ollama)
python -m testing_agent_harness.cli chat --repo examples/buggy_calc
```

---

## Interactive chat commands

Inside the REPL (`start.sh` or `cli chat`):

| Command | What it does |
|---|---|
| `/repo <path>` | switch the target repository |
| `/plan` | only build a plan (no code changes) |
| `/run` | full pipeline: plan → baseline → route → localize → repair → iterate → report |
| `/report` | show the latest Markdown report |
| `/diff` | show the latest sandbox diff |
| `/model` | print active provider + model |
| `/ls [path]` | list files inside the current repo |
| `/history` | show previous messages in this session |
| `/help` | list every command |
| `/quit` | exit |

Free-form text also works — anything that isn't a `/command` is sent to the LLM as a normal chat turn, streamed in real-time.

---

## Connecting a local Qwen3 (default path)

The harness speaks the standard **OpenAI Chat Completions** protocol, so any of these work without code changes:

| Server | Typical `base_url` | Notes |
|---|---|---|
| Ollama | `http://127.0.0.1:11434/v1` | default; run `ollama pull qwen3` first |
| vLLM | `http://127.0.0.1:8000/v1` | set `model_name` to the served checkpoint id |
| LM Studio | `http://127.0.0.1:1234/v1` | enable the "Local Server" tab |
| llama.cpp server | `http://127.0.0.1:8080/v1` | `./server -m qwen3.gguf --api-server` |

Config lives in the target repo's `.testing_agent.yaml`:

```yaml
model:
  provider: openai_compatible
  model_name: qwen3              # or qwen3:4b, qwen3:8b, etc.
  base_url: http://127.0.0.1:11434/v1
  api_key_env: OPENAI_API_KEY    # leave unset if server has no auth
  timeout_seconds: 300
  max_output_tokens: 1536
  num_ctx: 4096

budget:
  fast_mode: true                # replace narrative skills with deterministic logic
  max_iterations: 1
  max_runtime_minutes: 5

policy:
  repair_mode: auto              # suggest_only | ask | auto
  stop_when_tests_pass: true
```

### Why `fast_mode`

Small local models (e.g. `qwen3:4b`) are too slow to drive every orchestration step. When `fast_mode: true`, the harness replaces purely-narrative skills (`plan_reviewer`, `workflow_router`, `failure_localizer`, `repair_decider`, `final_judge`, `report_writer`) with deterministic Python logic, and only uses the LLM where it actually matters (`plan_builder`, `test_generation`, `bug_localizer`, `repair_patch`). Runs drop from 30+ min to well under 5.

---

## Safe repair behavior

Every run happens in an isolated **sandbox copy** of your repo under `.testing_agent_runs/<run_id>/sandbox/`. The original source is never mutated automatically.

Built-in safety rails:

- **Path validation** — every path the LLM produces (for localization or patching) is resolved against `project_scan.source_files`. Hallucinated paths (e.g. `src/core.py` when the real file is `src/buggy_calc/core.py`) are either remapped by basename or dropped.
- **No net-new source files from `repair_patch`** — repair can only overwrite files that already exist in the sandbox.
- **Auto-revert** — after a patch, tests rerun. If the repair didn't strictly reduce failures, the harness reverts every touched file from its pre-patch snapshot. `failed_repair_count` gets incremented so the report stays honest.
- **Regression gate** for `iterative_improvement` — generated tests are only kept if the full suite still passes.
- **Full diff** appears in the final report so you can choose to copy changes back manually.

---

## End-to-end example (included)

`examples/buggy_calc/` is a tiny Python package with a real bug. Demo:

```bash
# 1. Start the chat
./start.sh

# 2. Inside the chat
you> /repo examples/buggy_calc
you> /run
```

The included `core.py` has `add(a, b)` returning `a - b`. Expected output (condensed):

```
● planning          ▸ plan_builder      {…streamed JSON plan…}  ✓
● baseline_execution                                             ✓
● routing                                                        ✓
● localization      ▸ bug_localizer     {"candidates":[{"path":"src/buggy_calc/core.py","confidence":0.8,…}]}  ✓
● repair            ▸ repair_patch      {"changes":[{"path":"src/buggy_calc/core.py","content":"…a + b…"}]}    ✓
● iterative_improvement ▸ test_generation   {"files":[{"path":"tests/test_core_generated.py",…}]}              ✓
● finalization                                                   ✓

Goal achieved: True
Tests passed · coverage 100.00% · 1 repair · 1 generated test file
```

---

## Full CLI (non-interactive)

```bash
# Initialize a config file in the target repo
python -m testing_agent_harness.cli init-config /path/to/repo

# Plan only (no code changes)
python -m testing_agent_harness.cli plan /path/to/repo --provider openai_compatible

# Full run
python -m testing_agent_harness.cli run  /path/to/repo --provider openai_compatible --repair-mode auto

# Latest report / diff
python -m testing_agent_harness.cli report /path/to/repo
python -m testing_agent_harness.cli diff   /path/to/repo

# MCP-style stdio bridge (exposes tools + skill prompts over the MCP transport)
python -m testing_agent_harness.cli mcp-server /path/to/repo --provider openai_compatible
```

Gemini is still supported for backwards compatibility:

```bash
export GEMINI_API_KEY=...
python -m testing_agent_harness.cli run /path/to/repo --provider gemini --repair-mode auto
```

---

## Environment variables

Everything lives in `.env` (gitignored). Copy from `.env.example`:

```bash
cp .env.example .env
```

| Variable | Purpose | Default |
|---|---|---|
| `OLLAMA_BASE_URL` | OpenAI-compatible endpoint for the local server | `http://127.0.0.1:11434/v1` |
| `QWEN_MODEL` | Model id passed to the server | `qwen3` |
| `OPENAI_API_KEY` | Only set if your local server enforces a bearer token | *(unset)* |
| `GEMINI_API_KEY` | Only needed for the optional `gemini` provider | *(unset)* |
| `TEST_AGENT_PROVIDER` | Default provider for `start.sh` | `openai_compatible` |
| `TEST_AGENT_MODE` | `chat` or `run` for `start.sh` | `chat` |
| `TEST_AGENT_REPO` | Optional default repo path for `start.sh` | *(current dir / example)* |

---

## Layout

```
testing_agent_harness/
├── cli.py            # CLI entry points (typer)
├── chat.py           # interactive REPL + live token streaming
├── harness.py        # orchestration, state, sandbox, rollback
├── tools.py          # deterministic tools
├── models.py         # provider impls (openai_compatible, gemini, mock)
├── prompts/          # skill prompt + schema YAMLs
├── schemas.py        # pydantic models for plans, results, diffs
├── config.py         # user-facing config (model/budget/policy/goals)
└── mcp_server.py     # stdio MCP-style bridge (optional)
examples/buggy_calc/  # walk-through fixture
tests/                # harness + provider unit tests
start.sh              # one-click launcher
.env.example          # template (real .env is gitignored)
```

---

## Artifacts per run

```
<repo>/.testing_agent_runs/<run_id>/
├── state.json                 # full run state snapshot
├── events.jsonl               # every stage/skill/tool event
├── plan.json                  # the generated plan
├── sandbox/                   # isolated working copy
└── reports/
    ├── final_report.md
    └── final_report.json
```

---

## Tests

```bash
python -m pytest
```

Covers: harness orchestration, path validation, auto-revert, mock provider, OpenAI-compatible provider (mocked HTTP), MCP bridge.

---

## Status

- Local Qwen3 (Ollama) — **validated end-to-end** against `examples/buggy_calc` (bug injection → localization → patch → green).
- Gemini provider — ships with exponential-backoff retry, but live API validation must be done from a network-enabled environment.
- Mock provider — fully deterministic, used by the test suite.
