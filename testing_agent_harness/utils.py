from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import textwrap


ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "run"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: object) -> Path:
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return p


def read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def copy_repo_to_sandbox(repo_path: str | Path, sandbox_root: str | Path) -> Path:
    src = Path(repo_path).resolve()
    dst = Path(sandbox_root).resolve() / src.name
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", ".pytest_cache", "dist", "build", ".testing_agent_runs")
    shutil.copytree(src, dst, ignore=ignore)
    return dst


def relative_files(root: str | Path) -> list[str]:
    base = Path(root)
    items: list[str] = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            items.append(str(path.relative_to(base)))
    return items


def safe_read_text(path: str | Path, max_chars: int = 12000) -> str:
    p = Path(path)
    try:
        data = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        data = p.read_text(encoding="latin-1")
    if len(data) <= max_chars:
        return data
    return data[:max_chars] + "\n...[truncated]..."


def color_unified_diff(before: str, after: str, fromfile: str, tofile: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=fromfile,
        tofile=tofile,
        lineterm="",
    )
    colored: list[str] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            colored.append(line)
        elif line.startswith("+"):
            colored.append(f"{ANSI_GREEN}{line}{ANSI_RESET}")
        elif line.startswith("-"):
            colored.append(f"{ANSI_RED}{line}{ANSI_RESET}")
        else:
            colored.append(line)
    return "\n".join(colored)


def wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def repo_display_name(path: str | Path) -> str:
    return Path(path).resolve().name


def env_preview(keys: Iterable[str]) -> list[str]:
    present = []
    for key in keys:
        if os.environ.get(key):
            present.append(key)
    return present


def truncate(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]..."


def terminal_supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM") not in {None, "dumb"}
