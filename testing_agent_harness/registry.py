from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import yaml

from .schemas import EventRecord, RunState
from .tools import ToolContext, ToolRegistry
from .utils import ensure_dir


@dataclass(slots=True)
class SkillSpec:
    name: str
    description: str
    prompt: str
    allowed_tools: list[str]
    output_schema: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SkillSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            description=data["description"],
            prompt=data["prompt"],
            allowed_tools=data.get("allowed_tools", []),
            output_schema=data["output_schema"],
        )


class EventLogger:
    def __init__(self, events_path: str | Path, run_id: str) -> None:
        self.events_path = Path(events_path)
        self.run_id = run_id
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, kind: str, name: str, status: str, payload: dict[str, Any]) -> None:
        record = EventRecord(run_id=self.run_id, stage=stage, kind=kind, name=name, status=status, payload=payload)
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")


class SkillRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._skills: dict[str, SkillSpec] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self.root.glob("*.yaml")):
            spec = SkillSpec.from_yaml(path)
            self._skills[spec.name] = spec

    def get(self, name: str) -> SkillSpec:
        if name not in self._skills:
            raise KeyError(f"Unknown skill: {name}")
        return self._skills[name]

    def list(self) -> list[SkillSpec]:
        return list(self._skills.values())


class SkillRunner:
    def __init__(self, provider: Any, tools: ToolRegistry, skills: SkillRegistry, logger: EventLogger) -> None:
        self.provider = provider
        self.tools = tools
        self.skills = skills
        self.logger = logger

    def run(self, skill_name: str, stage: str, ctx: ToolContext, payload: dict[str, Any]) -> dict[str, Any]:
        spec = self.skills.get(skill_name)
        self.logger.emit(stage, "skill", skill_name, "started", {"input": payload})
        try:
            output = self.provider.run_skill(spec=spec, payload=payload, tool_registry=self.tools, ctx=ctx, logger=self.logger, stage=stage)
        except Exception as exc:  # pragma: no cover - safety logging
            self.logger.emit(stage, "skill", skill_name, "failed", {"error": str(exc)})
            raise
        self.logger.emit(stage, "skill", skill_name, "completed", {"output": output})
        return output
