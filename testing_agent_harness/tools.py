from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import ast
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

from .config import AgentConfig
from .schemas import (
    CommandRecord,
    CoverageFileGap,
    CoverageSnapshot,
    DependencyRecord,
    FailureItem,
    FailureSummary,
    ProjectScan,
    ReproducibilityManifest,
    RunState,
    TestRunResult,
)
from .utils import color_unified_diff, copy_repo_to_sandbox, ensure_dir, relative_files, safe_read_text, truncate, write_json


@dataclass(slots=True)
class ToolContext:
    config: AgentConfig
    state: RunState
    run_dir: Path

    @property
    def repo_path(self) -> Path:
        return Path(self.state.repo_path)

    @property
    def sandbox_path(self) -> Path:
        return Path(self.state.sandbox_path)


class ToolError(RuntimeError):
    pass


class Tool:
    name: str = "tool"
    description: str = ""
    schema: dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> Any:
        raise NotImplementedError


class RepoTreeTool(Tool):
    name = "repo_tree"
    description = "List files under the sandboxed repository with filtering and truncation."
    schema = {
        "type": "object",
        "properties": {
            "max_files": {"type": "integer", "default": 300},
            "suffixes": {"type": "array", "items": {"type": "string"}},
        },
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        max_files = int(args.get("max_files", 300))
        suffixes = set(args.get("suffixes", []))
        files = []
        for path in sorted(ctx.sandbox_path.rglob("*")):
            if not path.is_file():
                continue
            rel = str(path.relative_to(ctx.sandbox_path))
            if any(part in {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build"} for part in path.parts):
                continue
            if suffixes and path.suffix not in suffixes:
                continue
            files.append(rel)
            if len(files) >= max_files:
                break
        return {"files": files, "count": len(files)}


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a text file from the sandbox."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_chars": {"type": "integer", "default": 12000},
        },
        "required": ["path"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        rel = args["path"]
        target = (ctx.sandbox_path / rel).resolve()
        if ctx.sandbox_path not in target.parents and target != ctx.sandbox_path:
            raise ToolError("Path escapes sandbox")
        return {"path": rel, "content": safe_read_text(target, int(args.get("max_chars", 12000)))}


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write a text file inside the sandbox, creating parent directories if needed."
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        rel = args["path"]
        target = (ctx.sandbox_path / rel).resolve()
        if ctx.sandbox_path not in target.parents and target != ctx.sandbox_path:
            raise ToolError("Path escapes sandbox")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return {"path": rel, "bytes": len(args["content"].encode("utf-8"))}


class SearchCodeTool(Tool):
    name = "search_code"
    description = "Search the sandbox for a regex or literal pattern in text files."
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "literal": {"type": "boolean", "default": False},
            "suffixes": {"type": "array", "items": {"type": "string"}},
            "max_results": {"type": "integer", "default": 50},
        },
        "required": ["pattern"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        suffixes = set(args.get("suffixes", []))
        max_results = int(args.get("max_results", 50))
        pattern_text = args["pattern"]
        if args.get("literal", False):
            matcher: Callable[[str], Any] = lambda line: pattern_text in line
        else:
            regex = re.compile(pattern_text)
            matcher = lambda line: regex.search(line)
        results: list[dict[str, Any]] = []
        for path in sorted(ctx.sandbox_path.rglob("*")):
            if not path.is_file():
                continue
            if suffixes and path.suffix not in suffixes:
                continue
            if any(part in {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for idx, line in enumerate(text.splitlines(), start=1):
                if matcher(line):
                    results.append({
                        "path": str(path.relative_to(ctx.sandbox_path)),
                        "line_number": idx,
                        "line": line[:300],
                    })
                    if len(results) >= max_results:
                        return {"results": results}
        return {"results": results}


class ASTSummaryTool(Tool):
    name = "ast_summary"
    description = "Summarize Python modules, classes, and functions from a source file."
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        rel = args["path"]
        target = ctx.sandbox_path / rel
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                })
            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({"name": item.name, "lineno": item.lineno})
                classes.append({"name": node.name, "lineno": node.lineno, "methods": methods})
        return {"path": rel, "functions": functions, "classes": classes}


class SnapshotSandboxTool(Tool):
    name = "snapshot_sandbox"
    description = "Create or reset an isolated sandbox copy of the original repository."
    schema = {"type": "object", "properties": {}}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        sandbox_root = ensure_dir(ctx.run_dir / "sandbox")
        sandbox = copy_repo_to_sandbox(ctx.repo_path, sandbox_root)
        ctx.state.sandbox_path = str(sandbox)
        return {"sandbox_path": str(sandbox)}


class ProjectScanTool(Tool):
    name = "project_scan"
    description = "Inspect the sandbox and infer project type, key files, and testing hints."
    schema = {"type": "object", "properties": {}}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        root = ctx.sandbox_path
        root_files = [p.name for p in root.iterdir() if p.is_file()]
        test_files = [str(p.relative_to(root)) for p in root.rglob("test*.py")] + [str(p.relative_to(root)) for p in root.rglob("*_test.py")]
        source_files = [str(p.relative_to(root)) for p in root.rglob("*.py") if "tests" not in p.parts]
        config_files = [name for name in root_files if name in {"pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini", "requirements.txt", "requirements-dev.txt"}]
        package_manager = None
        project_type = "unknown"
        python_modules = []
        discovered_commands = []
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "setup.cfg").exists():
            project_type = "python"
            package_manager = "pip"
            discovered_commands.extend(["python -m pytest -q", "python -m coverage run -m pytest -q"])
        if (root / "tox.ini").exists():
            discovered_commands.append("python -m tox -q")
        if (root / "noxfile.py").exists():
            discovered_commands.append("python -m nox")
        for path in root.rglob("*.py"):
            if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            python_modules.append(str(path.relative_to(root)))
        summary = f"Detected {project_type} project with {len(source_files)} source files and {len(test_files)} test files."
        scan = ProjectScan(
            project_type=project_type,
            package_manager=package_manager,
            root_files=sorted(root_files),
            test_files=sorted(set(test_files)),
            source_files=sorted(source_files),
            config_files=sorted(config_files),
            python_modules=sorted(python_modules),
            discovered_commands=discovered_commands,
            summary=summary,
        )
        ctx.state.project_scan = scan
        return scan.model_dump()


class BuildManifestTool(Tool):
    name = "build_manifest"
    description = "Build a reproducibility manifest for a Python project, including dependencies and commands."
    schema = {"type": "object", "properties": {}}

    def _parse_requirements(self, path: Path) -> list[DependencyRecord]:
        items: list[DependencyRecord] = []
        if not path.exists():
            return items
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[<>=!~]", line)[0].strip()
            items.append(DependencyRecord(name=name, version=line if name != line else None, source=path.name))
        return items

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        root = ctx.sandbox_path
        dependencies: list[DependencyRecord] = []
        install_commands: list[CommandRecord] = []
        test_commands: list[CommandRecord] = []
        coverage_commands: list[CommandRecord] = []
        test_entry_points: list[str] = []

        dependencies.extend(self._parse_requirements(root / "requirements.txt"))
        dependencies.extend(self._parse_requirements(root / "requirements-dev.txt"))
        dependencies.extend(self._parse_requirements(root / "test-requirements.txt"))

        if (root / "pyproject.toml").exists() or (root / "setup.py").exists() or (root / "setup.cfg").exists():
            install_commands.append(CommandRecord(name="editable_install", command="python -m pip install -e .", status="candidate"))
        for req_name in ["requirements.txt", "requirements-dev.txt", "test-requirements.txt"]:
            if (root / req_name).exists():
                install_commands.append(CommandRecord(name=f"install_{req_name}", command=f"python -m pip install -r {req_name}", status="candidate"))

        pytest_target = "tests" if (root / "tests").exists() else "."
        test_commands.append(CommandRecord(name="pytest", command=f"python -m pytest -q {pytest_target}", status="candidate"))
        coverage_commands.append(CommandRecord(name="coverage_pytest", command=f"python -m coverage run -m pytest -q {pytest_target}", status="candidate"))
        test_entry_points.append(pytest_target)

        manifest = ReproducibilityManifest(
            repo_path=str(ctx.repo_path),
            sandbox_path=str(ctx.sandbox_path),
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            dependencies=dependencies,
            install_commands=install_commands,
            test_commands=test_commands,
            coverage_commands=coverage_commands,
            environment_variables=[],
            test_entry_points=test_entry_points,
            notes=["Sandboxed copy is the only writable workspace.", "All diffs are computed against the original repo."],
        )
        ctx.state.manifest = manifest
        return manifest.model_dump()


class InstallDependenciesTool(Tool):
    name = "install_dependencies"
    description = "Run one or more dependency installation commands inside the sandbox."
    schema = {
        "type": "object",
        "properties": {"commands": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "integer", "default": 1200}},
        "required": ["commands"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        results = []
        for command in args["commands"]:
            result = ShellExecTool().run(ctx, {"command": command, "timeout": args.get("timeout", 1200)})
            results.append(result)
            if result["exit_code"] != 0:
                break
        return {"results": results}


class ShellExecTool(Tool):
    name = "shell_exec"
    description = "Execute a shell command within the sandbox and capture stdout, stderr, exit code, and runtime."
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "default": 600},
        },
        "required": ["command"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        command = args["command"]
        timeout = int(args.get("timeout", 600))
        start = time.time()
        env = os.environ.copy()
        env.setdefault("PYTHONHASHSEED", str(ctx.state.manifest.random_seed if ctx.state.manifest else 42))
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=ctx.sandbox_path,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        duration = time.time() - start
        return {
            "command": command,
            "exit_code": proc.returncode,
            "stdout": truncate(proc.stdout, 12000),
            "stderr": truncate(proc.stderr, 12000),
            "duration_seconds": duration,
        }


class RunTestsTool(Tool):
    name = "run_tests"
    description = "Run the selected test command and return a structured result."
    schema = {
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 900}},
        "required": ["command"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        result = ShellExecTool().run(ctx, args)
        payload = TestRunResult(
            command=result["command"],
            exit_code=result["exit_code"],
            passed=result["exit_code"] == 0,
            stdout=result["stdout"],
            stderr=result["stderr"],
            duration_seconds=float(result["duration_seconds"]),
        )
        ctx.state.latest_test_result = payload
        return payload.model_dump()


class RunCoverageTool(Tool):
    name = "run_coverage"
    description = "Run coverage.py with pytest, save JSON coverage output, and return structured coverage data."
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "json_path": {"type": "string", "default": "artifacts/coverage.json"},
            "timeout": {"type": "integer", "default": 1200},
        },
        "required": ["command"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        json_path = args.get("json_path", "artifacts/coverage.json")
        cov_file = ctx.run_dir / json_path
        cov_file.parent.mkdir(parents=True, exist_ok=True)
        command = args["command"]
        if "coverage json" not in command:
            command = f"{command} && python -m coverage json -o {cov_file.as_posix()}"
        result = ShellExecTool().run(ctx, {"command": command, "timeout": args.get("timeout", 1200)})
        coverage = CoverageSnapshot(total_percent=0.0, files=[], raw_json_path=str(cov_file))
        if cov_file.exists():
            data = json.loads(cov_file.read_text(encoding="utf-8"))
            totals = data.get("totals", {})
            coverage.total_percent = float(totals.get("percent_covered", 0.0)) / 100.0
            files: list[CoverageFileGap] = []
            for rel, item in data.get("files", {}).items():
                missing_lines = list(item.get("missing_lines", []))
                percent = item.get("summary", {}).get("percent_covered")
                files.append(CoverageFileGap(path=rel, missing_lines=missing_lines, covered_percent=(float(percent) / 100.0) if percent is not None else None))
            coverage.files = sorted(files, key=lambda entry: (-len(entry.missing_lines), entry.path))
        payload = TestRunResult(
            command=command,
            exit_code=result["exit_code"],
            passed=result["exit_code"] == 0,
            stdout=result["stdout"],
            stderr=result["stderr"],
            duration_seconds=float(result["duration_seconds"]),
            coverage=coverage,
        )
        ctx.state.latest_test_result = payload
        return payload.model_dump()


class CoverageGapTool(Tool):
    name = "coverage_gap_analysis"
    description = "Map coverage missing lines to Python functions for targeted test generation."
    schema = {"type": "object", "properties": {}, "required": []}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        latest = ctx.state.latest_test_result
        if not latest or not latest.coverage:
            return {"files": [], "summary": "No coverage data available."}
        enriched = []
        for item in latest.coverage.files:
            target = ctx.sandbox_path / item.path
            if not target.exists() or target.suffix != ".py":
                enriched.append(item.model_dump())
                continue
            try:
                tree = ast.parse(target.read_text(encoding="utf-8"))
            except SyntaxError:
                enriched.append(item.model_dump())
                continue
            missing_functions: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start = node.lineno
                    end = getattr(node, "end_lineno", node.lineno)
                    if any(start <= line <= end for line in item.missing_lines):
                        missing_functions.append(node.name)
            enriched.append({
                **item.model_dump(),
                "missing_functions": sorted(set(missing_functions)),
            })
        return {
            "files": enriched,
            "summary": f"Found {sum(1 for item in enriched if item.get('missing_lines'))} files with missing lines.",
        }


class FailureParseTool(Tool):
    name = "failure_parse"
    description = "Parse pytest output into structured failures and distinguish likely environment failures."
    schema = {"type": "object", "properties": {}, "required": []}

    FAILURE_PATTERN = re.compile(r"FAILED\s+(.+?)\s+-\s+(.+)")
    TRACE_FILE_PATTERN = re.compile(r"^(.*\.py):(\d+):\s+(?:in\s+(.+))?")

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        latest = ctx.state.latest_test_result
        if not latest:
            return FailureSummary().model_dump()
        text = "\n".join([latest.stdout, latest.stderr])
        failures: list[FailureItem] = []
        current_stack: list[str] = []
        current_name: str | None = None
        current_message: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip("\n")
            match = self.FAILURE_PATTERN.search(line)
            if match:
                if current_name and current_message:
                    failures.append(FailureItem(test_name=current_name, message=current_message, stack_excerpt=current_stack[:12], failure_type=self._infer_failure_type(current_message)))
                current_name = match.group(1)
                current_message = match.group(2)
                current_stack = []
                continue
            if current_name and line:
                current_stack.append(line)
        if current_name and current_message:
            failures.append(FailureItem(test_name=current_name, message=current_message, stack_excerpt=current_stack[:12], failure_type=self._infer_failure_type(current_message)))
        suspected_environment_issue = any(token in text.lower() for token in ["modulenotfounderror", "no module named", "command not found", "could not find", "importerror"])
        summary = FailureSummary(
            failures=failures,
            error_count=len(failures),
            suspected_environment_issue=suspected_environment_issue,
            notes=["Environment issue likely" if suspected_environment_issue else "Application or test failure likely"],
        )
        ctx.state.latest_failures = summary
        return summary.model_dump()

    def _infer_failure_type(self, message: str) -> str:
        lowered = message.lower()
        if "assert" in lowered:
            return "assertion"
        if "zerodivisionerror" in lowered or "exception" in lowered:
            return "exception"
        if "timeout" in lowered:
            return "timeout"
        return "unknown"


class SuspectFilesTool(Tool):
    name = "suspect_files"
    description = "Rank likely buggy files from failure traces and coverage gaps."
    schema = {"type": "object", "properties": {}, "required": []}

    FILE_PATTERN = re.compile(r"([A-Za-z0-9_./\\-]+\.py)")

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        score: dict[str, float] = {}
        failures = ctx.state.latest_failures.failures if ctx.state.latest_failures else []
        for failure in failures:
            for line in failure.stack_excerpt:
                for match in self.FILE_PATTERN.findall(line):
                    normalized = match.replace("\\", "/")
                    if normalized.startswith("/"):
                        try:
                            normalized = str(Path(normalized).resolve().relative_to(ctx.sandbox_path))
                        except Exception:
                            continue
                    if normalized.endswith(".py"):
                        score[normalized] = score.get(normalized, 0.0) + 2.0
        latest = ctx.state.latest_test_result
        if latest and latest.coverage:
            for file_gap in latest.coverage.files[:10]:
                if file_gap.missing_lines:
                    score[file_gap.path] = score.get(file_gap.path, 0.0) + min(3.0, len(file_gap.missing_lines) / 10.0)
        ranked = [{"path": path, "score": round(value, 2)} for path, value in sorted(score.items(), key=lambda item: item[1], reverse=True)]
        return {"candidates": ranked[:10]}


class QualityCheckTool(Tool):
    name = "test_quality_check"
    description = "Evaluate generated test files for assertion quality, runtime cost, duplication, and flaky risk."
    schema = {
        "type": "object",
        "properties": {"files": {"type": "array", "items": {"type": "object"}}},
        "required": ["files"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        assertion_score = 0.0
        runtime_risk = 0.0
        flaky_risk = 0.0
        duplication_risk = 0.0
        notes: list[str] = []
        existing_tests = "\n".join(
            safe_read_text(path, max_chars=4000)
            for path in [ctx.sandbox_path / rel for rel in relative_files(ctx.sandbox_path) if rel.startswith("tests/") and rel.endswith(".py")]
        )
        input_files = [item for item in args.get("files", []) if isinstance(item, dict)]
        valid_files = [item for item in input_files if isinstance(item.get("content"), str) and item["content"].strip()]
        skipped = len(input_files) - len(valid_files)
        if skipped:
            notes.append(f"skipped {skipped} file(s) with no 'content' field (got metadata only)")
        total_files = max(1, len(valid_files))
        for item in valid_files:
            content = item["content"]
            assertion_score += min(1.0, content.count("assert ") / 3.0)
            if "assert True" in content or "pass" in content:
                notes.append(f"{item['path']}: contains weak assertion placeholder")
                assertion_score -= 0.4
            if any(token in content for token in ["sleep(", "time.sleep", "random.", "datetime.now", "uuid.uuid4"]):
                flaky_risk += 0.7
                notes.append(f"{item['path']}: possible flaky source found")
            if content.count("@pytest.mark.parametrize") > 4 or len(content.splitlines()) > 220:
                runtime_risk += 0.6
                notes.append(f"{item['path']}: elevated runtime risk")
            if content[:500] and content[:500] in existing_tests:
                duplication_risk += 0.8
                notes.append(f"{item['path']}: possible duplication with existing tests")
        assertion_quality = max(0.0, min(1.0, assertion_score / total_files))
        runtime_risk = max(0.0, min(1.0, runtime_risk / total_files))
        flaky_risk = max(0.0, min(1.0, flaky_risk / total_files))
        duplication_risk = max(0.0, min(1.0, duplication_risk / total_files))
        accepted = assertion_quality >= 0.4 and runtime_risk <= 0.8 and flaky_risk <= 0.8 and duplication_risk <= 0.8
        return {
            "assertion_quality": round(assertion_quality, 3),
            "runtime_risk": round(runtime_risk, 3),
            "flaky_risk": round(flaky_risk, 3),
            "duplication_risk": round(duplication_risk, 3),
            "accepted": accepted,
            "notes": notes,
        }


class ApplyChangesTool(Tool):
    name = "apply_changes"
    description = "Apply a list of file changes to the sandbox while enforcing policy boundaries."
    schema = {
        "type": "object",
        "properties": {"changes": {"type": "array", "items": {"type": "object"}}},
        "required": ["changes"],
    }

    def _is_allowed_path(self, ctx: ToolContext, relative_path: str) -> bool:
        if relative_path.startswith("/"):
            return False
        if any(relative_path.startswith(prefix + "/") or relative_path == prefix for prefix in [".git", ".venv", "venv", "dist", "build", "__pycache__"]):
            return False
        if relative_path.startswith("tests/"):
            return ctx.config.policy.allow_test_modification
        return ctx.config.policy.allow_code_repair

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        applied: list[str] = []
        for item in args["changes"]:
            rel = item["path"]
            if not self._is_allowed_path(ctx, rel):
                raise ToolError(f"Modification blocked by policy: {rel}")
            target = (ctx.sandbox_path / rel).resolve()
            if ctx.sandbox_path not in target.parents and target != ctx.sandbox_path:
                raise ToolError("Path escapes sandbox")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item["content"], encoding="utf-8")
            applied.append(rel)
        return {"applied": applied}


class DiffWorkspaceTool(Tool):
    name = "diff_workspace"
    description = "Compute unified diffs between the original repo and sandbox workspace."
    schema = {"type": "object", "properties": {}}

    def _is_binary(self, path: Path) -> bool:
        try:
            sample = path.read_bytes()[:4096]
        except Exception:
            return True
        return b"\x00" in sample

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        original_files = set(relative_files(ctx.repo_path))
        sandbox_files = set(relative_files(ctx.sandbox_path))
        all_files = sorted(original_files | sandbox_files)
        diffs: list[dict[str, Any]] = []
        for rel in all_files:
            if rel in {".coverage"} or rel.startswith(".pytest_cache/") or rel.startswith("__pycache__/") or rel.startswith(".testing_agent_runs/"):
                continue
            orig = ctx.repo_path / rel
            sand = ctx.sandbox_path / rel
            if orig.exists() and self._is_binary(orig):
                continue
            if sand.exists() and self._is_binary(sand):
                continue
            before = safe_read_text(orig, max_chars=200000) if orig.exists() else ""
            after = safe_read_text(sand, max_chars=200000) if sand.exists() else ""
            if before == after:
                continue
            diff_text = color_unified_diff(before, after, f"a/{rel}", f"b/{rel}")
            diffs.append({"path": rel, "diff": diff_text})
        return {"changed_files": [item["path"] for item in diffs], "diffs": diffs}


class RollbackSandboxTool(Tool):
    name = "rollback_sandbox"
    description = "Discard sandbox changes and restore a fresh copy from the original repo."
    schema = {"type": "object", "properties": {}}

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        sandbox_root = ensure_dir(ctx.run_dir / "sandbox")
        sandbox = copy_repo_to_sandbox(ctx.repo_path, sandbox_root)
        ctx.state.sandbox_path = str(sandbox)
        return {"sandbox_path": str(sandbox), "rolled_back": True}


class WriteReportTool(Tool):
    name = "write_report"
    description = "Write markdown and JSON report artifacts for the run."
    schema = {
        "type": "object",
        "properties": {
            "markdown": {"type": "string"},
            "json_payload": {"type": "object"},
        },
        "required": ["markdown", "json_payload"],
    }

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        reports_dir = ensure_dir(ctx.run_dir / "reports")
        md_path = reports_dir / "final_report.md"
        json_path = reports_dir / "final_report.json"
        md_path.write_text(args["markdown"], encoding="utf-8")
        write_json(json_path, args["json_payload"])
        return {"markdown_path": str(md_path), "json_path": str(json_path)}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def list_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        tools = [self._tools[name] for name in names] if names else list(self._tools.values())
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema,
            }
            for tool in tools
        ]

    def invoke(self, name: str, ctx: ToolContext, args: dict[str, Any]) -> Any:
        return self.get(name).run(ctx, args)



def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in [
        RepoTreeTool(),
        ReadFileTool(),
        WriteFileTool(),
        SearchCodeTool(),
        ASTSummaryTool(),
        SnapshotSandboxTool(),
        ProjectScanTool(),
        BuildManifestTool(),
        InstallDependenciesTool(),
        ShellExecTool(),
        RunTestsTool(),
        RunCoverageTool(),
        CoverageGapTool(),
        FailureParseTool(),
        SuspectFilesTool(),
        QualityCheckTool(),
        ApplyChangesTool(),
        DiffWorkspaceTool(),
        RollbackSandboxTool(),
        WriteReportTool(),
    ]:
        registry.register(tool)
    return registry
