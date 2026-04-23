# Architecture Mapping to the Paper

This project directly maps the paper's seven structural requirements into code.

## 1. Environment and Reproducibility

Implemented by:

- `snapshot_sandbox` tool
- `project_scan` tool
- `build_manifest` tool
- `ReproducibilityManifest` schema

Artifacts:

- `state.json`
- `plan.json`
- isolated sandbox under each run

## 2. Test Planning and Goal Definition

Implemented by:

- `plan_builder` skill
- `plan_reviewer` skill
- `TestPlan` schema

The plan is generated before the long iterative flow starts.

## 3. Targeted Workflow Selection and Stopping Conditions

Implemented by:

- `workflow_router` skill
- harness loop in `TestingHarness.run_full`
- iteration and failed repair budgets in config
- stagnation detection in `iterative_improvement`

## 4. Test Case Generation and Iterative Improvement

Implemented by:

- `run_coverage` tool
- `coverage_gap_analysis` tool
- `test_generation` skill
- `test_quality_critic` skill
- `apply_changes` tool

This is a closed loop guided by coverage plus test quality signals.

## 5. Execution Feedback, Failure Localization, Bug Localization, and Repair Control

Implemented by:

- `run_tests` tool
- `failure_parse` tool
- `failure_localizer` skill
- `suspect_files` tool
- `bug_localizer` skill
- `repair_decider` skill
- `repair_patch` skill

Safety boundaries:

- sandbox only writes during autonomous flow
- diff for every accepted modification
- regression run after repair
- rollback tool available

## 6. Full Logging and Traceability

Implemented by:

- `EventLogger`
- `events.jsonl`
- stage and skill logs
- persisted `state.json`

Every stage emits started and completed events.
Every skill emits started and completed events.
Every tool call triggered by the LLM can be logged through the skill runner.

## 7. Report Generation

Implemented by:

- `final_judge` skill
- `report_writer` skill
- `write_report` tool

Artifacts:

- `reports/final_report.md`
- `reports/final_report.json`

## MCP decision

MCP is included as an optional transport layer only.

Reason:

- the harness should own the workflow, budgets, and safety boundaries
- MCP is useful for exposing the internal tool and skill contracts to other runtimes
- using MCP as the core runtime would blur the line between transport and orchestration

So the design is:

- harness = execution authority
- tools and skills = capability contracts
- MCP bridge = optional discovery and invocation adapter
