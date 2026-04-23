from __future__ import annotations

from pathlib import Path
import shutil

from testing_agent_harness.config import AgentConfig
from testing_agent_harness.mcp_server import MCPBridge


EXAMPLE_REPO = Path(__file__).resolve().parents[1] / "examples" / "buggy_calc"


def _copy_example(tmp_path: Path) -> Path:
    target = tmp_path / "buggy_calc"
    shutil.copytree(EXAMPLE_REPO, target)
    return target


def test_mcp_bridge_lists_tools_and_resources(tmp_path: Path) -> None:
    repo = _copy_example(tmp_path)
    bridge = MCPBridge(repo_path=repo, config=AgentConfig())

    init_resp = bridge.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp["result"]["serverInfo"]["name"] == "testing-agent-harness"

    tools_resp = bridge.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tool_names = {item["name"] for item in tools_resp["result"]["tools"]}
    assert "run_tests" in tool_names
    assert "build_manifest" in tool_names

    resources_resp = bridge.handle_request({"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}})
    uris = {item["uri"] for item in resources_resp["result"]["resources"]}
    assert "skill://plan_builder" in uris

    read_resp = bridge.handle_request({"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "skill://plan_builder"}})
    assert "Build a testing plan" in read_resp["result"]["contents"][0]["text"]
