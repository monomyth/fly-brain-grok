from __future__ import annotations

import base64
import os
from pathlib import Path

import pytest

from rebot_adapter.episode import default_mcp_binary
from rebot_adapter.teacher import CAPTURE_CAMERAS
from train.dataset import VIEW_CAMERAS


def _grok_socket() -> Path:
    uid = os.getuid()
    tmp = os.environ.get("TMPDIR", "/tmp")
    return Path(tmp) / f"rebot-motionlab-grok-{uid}" / "control.sock"


def _live_client():
    pytest.importorskip("rebot_adapter.mcp")
    sock = _grok_socket()
    if not sock.exists():
        pytest.skip(f"Grok lab socket not running: {sock}")
    mcp = default_mcp_binary()
    if not mcp.is_file():
        pytest.skip(f"ReBotMCP not found: {mcp}")
    from rebot_adapter.mcp import MCPClient

    os.environ["REBOT_MCP_NO_LAUNCH"] = "1"
    client = MCPClient(mcp, launch_lab=False)
    try:
        client.initialize("flybrain-camera-smoke")
        client.wait_ready(timeout=6)
        return client
    except (RuntimeError, TimeoutError) as exc:
        client.close()
        pytest.skip(f"Grok lab not ready: {exc}")


def test_capture_cameras_are_front_and_gripper():
    assert CAPTURE_CAMERAS == ("Front", "Gripper")
    assert VIEW_CAMERAS == ("Front", "Gripper")
    assert "Top" not in CAPTURE_CAMERAS


def test_live_mcp_front_gripper_capture_does_not_steal_view():
    client = _live_client()
    try:
        tools = client.rpc("tools/list")["tools"]
        by_name = {t["name"]: t for t in tools}
        for tool_name in ("rebot_set_view", "rebot_capture_view"):
            choices = by_name[tool_name]["inputSchema"]["properties"]["camera"]["enum"]
            assert choices == ["Orbit", "Front", "Top", "Gripper"]
            assert "Front" in choices and "Gripper" in choices
        kept = client.state()["view"]["camera"]
        for name in ("Orbit", "Front", "Top", "Gripper"):
            shot = client.call(
                "rebot_capture_view",
                {"camera": name, "width": 160, "height": 120, "apply": False},
            )
            assert shot["camera"] == name
            assert shot["width"] == 160 and shot["height"] == 120
            jpeg = base64.b64decode(shot["jpeg_base64"])
            assert jpeg[:2] == b"\xff\xd8"
        assert client.state()["view"]["camera"] == kept
    finally:
        client.close()
