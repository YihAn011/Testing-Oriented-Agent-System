# Evaluation

This document describes how to replicate the evaluation, what repository assets are used, and what comparison system is included.

## 1. How To Replicate The Evaluation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
```

Then run the benchmark repositories:

```bash
python -m testing_agent_harness.cli run examples/buggy_calc --provider mock --repair-mode auto -n
python -m testing_agent_harness.cli run examples/buggy_snake --provider mock --repair-mode auto -n
```

To reproduce the suggest-only comparison condition:

```bash
python -m testing_agent_harness.cli run examples/buggy_calc --provider mock --repair-mode suggest_only -n
python -m testing_agent_harness.cli run examples/buggy_snake --provider mock --repair-mode suggest_only -n
```

To reproduce the baseline existing-tests-only condition:

```bash
cd examples/buggy_calc && pytest -q
cd examples/buggy_snake && pytest -q
```

For each run, inspect:

- `.testing_agent_runs/<run_id>/state.json`
- `.testing_agent_runs/<run_id>/events.jsonl`
- `.testing_agent_runs/<run_id>/plan.json`
- `.testing_agent_runs/<run_id>/reports/final_report.md`
- `.testing_agent_runs/<run_id>/reports/final_report.json`
- `.testing_agent_runs/<run_id>/sandbox/`

## 2. Test Cases, Benchmarks, And Data Included In The Repository

The repository already includes the materials needed for the evaluation:

- `examples/buggy_calc/`: bundled benchmark repository
- `examples/buggy_snake/`: bundled benchmark repository
- `tests/`: pytest suite for the harness itself

These bundled examples are the benchmark/test cases used in the final evaluation.

## 3. Alternative System Used In The Evaluation

The evaluation compares the full harness against at least one alternative system:

- `existing tests only`: run the benchmark repository’s tests directly, without harness repair

The repository also includes another comparison condition used in the evaluation:

- `suggest-only harness`: the harness localizes and reports failures but does not repair code

The full evaluated system is:

- `full harness`: sandboxed localization, repair, rerun, and reporting
