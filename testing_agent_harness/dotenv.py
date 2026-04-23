"""Tiny, dependency-free ``.env`` loader.

Semantics:
- Lines starting with ``#`` and blank lines are ignored.
- ``KEY=VALUE`` pairs are parsed; surrounding single/double quotes are stripped.
- Existing process environment variables are NOT overwritten (explicit shell
  exports win over .env values).
- Walks up from ``start_dir`` to find ``.env`` anywhere on the way to the
  filesystem root, plus the current working directory.
"""

from __future__ import annotations

from pathlib import Path
import os


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _candidate_files(start_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in {start_dir.resolve(), Path.cwd().resolve()}:
        seen: set[Path] = set()
        cur: Path | None = base
        while cur is not None and cur not in seen:
            seen.add(cur)
            candidate = cur / ".env"
            if candidate not in candidates:
                candidates.append(candidate)
            cur = cur.parent if cur != cur.parent else None
    return candidates


def load_dotenv(start_dir: Path | None = None) -> list[Path]:
    """Load the first ``.env`` found walking up from ``start_dir``.

    Returns the list of files that were actually read (empty if none).
    Existing environment variables are preserved.
    """
    start = start_dir or Path(__file__).resolve().parent
    loaded: list[Path] = []
    for path in _candidate_files(start):
        if path.is_file():
            values = _parse_env_file(path)
            for k, v in values.items():
                if k not in os.environ:
                    os.environ[k] = v
            loaded.append(path)
            break
    return loaded
