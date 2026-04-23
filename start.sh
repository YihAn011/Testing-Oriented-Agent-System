#!/usr/bin/env bash
# One-shot: create venv, install this package, init config if needed,
# and launch the chat CLI wired to your local Qwen3 via Ollama's OpenAI-compatible API.
#
# Usage: ./start.sh
# Optional env:
#   PYTHON               python executable (default: python3)
#   TEST_AGENT_REPO      path to repo under test (default: examples/buggy_calc in this project)
#   TEST_AGENT_PROVIDER  openai_compatible | gemini | mock (default: openai_compatible)
#   TEST_AGENT_MODE      chat | run (default: chat)
#   OLLAMA_BASE_URL      OpenAI-compatible root (default: http://127.0.0.1:11434/v1)
#   QWEN_MODEL           model id served by Ollama/vLLM (default: qwen3)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Load project-local .env (gitignored). Variables in the current shell
# take precedence over values in the file.
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"

if ! command -v "$PYTHON" &>/dev/null; then
  echo "error: '$PYTHON' not found on PATH" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating virtualenv: $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"

echo "==> Upgrading pip (if available)"
python -m pip install -q --upgrade pip 2>/dev/null || true

echo "==> Installing testing-agent-harness (editable)"
python -m pip install -q -e .

REPO="${TEST_AGENT_REPO:-$ROOT/examples/buggy_calc}"
PROVIDER="${TEST_AGENT_PROVIDER:-openai_compatible}"
MODE="${TEST_AGENT_MODE:-chat}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen3}"
export QWEN_MODEL

if [[ ! -d "$REPO" ]]; then
  echo "error: repo directory not found: $REPO" >&2
  exit 1
fi

if [[ ! -f "$REPO/.testing_agent.yaml" ]]; then
  echo "==> Writing default config: $REPO/.testing_agent.yaml"
  python -m testing_agent_harness.cli init-config "$REPO"
fi

if [[ "$PROVIDER" == "openai_compatible" ]]; then
  # Warn early if the local server is not reachable (Ollama / vLLM).
  if ! curl -fsS --max-time 2 "$OLLAMA_BASE_URL/models" >/dev/null 2>&1; then
    echo "warning: cannot reach $OLLAMA_BASE_URL/models (is Ollama/vLLM running?)" >&2
    echo "         start Ollama with:  ollama serve   and pull a model with:  ollama pull $QWEN_MODEL" >&2
  fi
fi

CHAT_ARGS=("$REPO" --provider "$PROVIDER")
if [[ "$PROVIDER" == "openai_compatible" ]]; then
  CHAT_ARGS+=(--model "$QWEN_MODEL" --base-url "$OLLAMA_BASE_URL")
fi

case "$MODE" in
  chat)
    echo "==> Starting chat (repo=$REPO provider=$PROVIDER model=$QWEN_MODEL base_url=$OLLAMA_BASE_URL)"
    exec python -m testing_agent_harness.cli chat "${CHAT_ARGS[@]}"
    ;;
  run)
    echo "==> Running harness (repo=$REPO provider=$PROVIDER)"
    RUN_ARGS=("$REPO" --provider "$PROVIDER" --repair-mode auto --non-interactive)
    exec python -m testing_agent_harness.cli run "${RUN_ARGS[@]}"
    ;;
  *)
    echo "error: unknown TEST_AGENT_MODE=$MODE (expected: chat | run)" >&2
    exit 1
    ;;
esac
