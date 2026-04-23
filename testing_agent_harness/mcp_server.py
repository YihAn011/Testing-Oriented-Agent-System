from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

from .config import AgentConfig
from .harness import TestingHarness


class MCPBridge:
    """Experimental MCP-style bridge.

    The harness remains the source of truth. This bridge simply exposes the same
    tool registry and skill prompt files over a light stdio JSON-RPC interface so
    external LLM runtimes can discover and call them.
    """

    def __init__(self, repo_path: str | Path, config: AgentConfig) -> None:
        self.harness = TestingHarness(repo_path=repo_path, config=config, provider_name="mock")
        self.harness.tools.invoke("snapshot_sandbox", self.harness.ctx, {})
        self.harness.tools.invoke("project_scan", self.harness.ctx, {})
        self.harness.tools.invoke("build_manifest", self.harness.ctx, {})

    def run_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
            except Exception as exc:  # pragma: no cover - stdio error path
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(exc)}}
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "serverInfo": {"name": "testing-agent-harness", "version": "0.1.0"},
                    "capabilities": {"tools": {}, "resources": {}},
                },
            }
        if method == "tools/list":
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.schema,
                }
                for tool in self.harness.tools._tools.values()  # pylint: disable=protected-access
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
        if method == "tools/call":
            name = params["name"]
            arguments = params.get("arguments", {})
            result = self.harness.tools.invoke(name, self.harness.ctx, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
            }
        if method == "resources/list":
            resources = [
                {
                    "uri": f"skill://{spec.name}",
                    "name": spec.name,
                    "description": spec.description,
                    "mimeType": "application/yaml",
                }
                for spec in self.harness.skills.list()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": resources}}
        if method == "resources/read":
            uri = params["uri"]
            if not uri.startswith("skill://"):
                raise ValueError("Unsupported resource URI")
            skill_name = uri.split("skill://", 1)[1]
            spec = self.harness.skills.get(skill_name)
            text = json.dumps(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "allowed_tools": spec.allowed_tools,
                    "output_schema": spec.output_schema,
                    "prompt": spec.prompt,
                },
                indent=2,
                ensure_ascii=False,
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": text}]},
            }
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}
