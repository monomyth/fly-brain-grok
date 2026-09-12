from __future__ import annotations

import base64
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from malecns_cache.paths import datasets_dir

from .mcp import MCPClient

CAPTURE_CAMERAS = ("Front", "Gripper")


def cube(state: dict) -> dict:
    return state["objects"]["cube"]


def _save_jpeg(folder: Path, name: str, payload: dict) -> str:
    path = folder / f"{name}.jpg"
    path.write_bytes(base64.b64decode(payload["jpeg_base64"]))
    return str(path)


def record_pick(
    client: MCPClient,
    run_dir: Path | None = None,
    x_mm: float = 280.0,
    y_mm: float = 0.0,
    size_mm: float = 40.0,
    hold_s: float = 2.0,
) -> dict:
    """Scripted teacher. Logs frames and commands for imitation, not MaleCNS."""
    run_dir = run_dir or datasets_dir("rebot-teacher") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frames = run_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"
    log = []

    def note(step: str, state: dict, command: dict | None = None, photo: str | None = None, photos: dict | None = None) -> None:
        row = {
            "t": time.time(),
            "step": step,
            "command": command,
            "tcp_mm": state.get("tcp_mm"),
            "tcp_rpy_deg": state.get("tcp_rpy_deg"),
            "gripper_mm": state.get("gripper_mm"),
            "cube": cube(state),
            "tcp_level": state.get("tcp_level"),
            "photo": photo,
            "photos": photos or {},
            "observation_mode": "state-assisted",
        }
        log.append(row)
        with log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    client.call("rebot_set_control_mode", {"mode": "scripted"})
    client.call("rebot_playback", {"action": "reset"})
    state = client.wait_stopped()
    client.call("rebot_apply_preset", {"name": "Folded"})
    state = client.wait_stopped()
    def capture_views(stem: str) -> dict[str, str]:
        photos: dict[str, str] = {}
        for name in CAPTURE_CAMERAS:
            shot = client.call(
                "rebot_capture_view",
                {"camera": name, "width": 320, "height": 240, "apply": False},
            )
            photos[name] = _save_jpeg(frames, f"{stem}-{name.lower()}", shot)
        return photos

    client.call("rebot_set_view", {"grid": True, "tool_axes": True})
    place = {"present": True, "x_mm": x_mm, "y_mm": y_mm, "size_mm": size_mm}
    client.call("rebot_set_cube", place)
    state = client.state()
    photos = capture_views("00-spawn")
    note("spawn", state, place, photos.get("Front"), photos)

    client.call("rebot_apply_preset", {"name": "Ready"})
    state = client.wait_stopped()
    client.call("rebot_set_gripper", {"opening_mm": 90})
    state = client.wait_stopped()
    spawn = cube(state)
    cx, cy = spawn["center_mm"]["x"], spawn["center_mm"]["y"]
    size = spawn["size_mm"]
    approach = {"x_mm": cx, "y_mm": cy, "z_mm": 48, "keep_level": True}
    client.call("rebot_move_to_pose", approach)
    state = client.wait_stopped()
    yaw = math.radians(state.get("tcp_rpy_deg", {}).get("yaw", 0))
    grasp = {"x_mm": cx + 35 * math.cos(yaw), "y_mm": cy + 35 * math.sin(yaw), "z_mm": 48, "keep_level": True}
    client.call("rebot_move_to_pose", grasp)
    state = client.wait_stopped()
    photos = capture_views("01-approach")
    note("approach", state, grasp, photos.get("Front"), photos)

    close = {"opening_mm": float(size) + 2.0}
    client.call("rebot_set_gripper", close)
    state = client.wait_stopped()
    photos = capture_views("02-pinch")
    note("close", state, close, photos.get("Front"), photos)
    if not cube(state)["attached"]:
        raise RuntimeError(f"teacher failed to attach: {cube(state)}")
    if state["gripper_mm"] < size - 3:
        raise RuntimeError(f"gripper closed through the cube: {state['gripper_mm']}")

    lift = {"x_mm": cx, "y_mm": cy, "z_mm": 160, "keep_level": True}
    client.call("rebot_move_to_pose", lift)
    state = client.wait_stopped()
    photos = capture_views("03-hold")
    note("lift", state, lift, photos.get("Front"), photos)
    if not cube(state)["attached"] or cube(state)["center_mm"]["z"] < 80:
        raise RuntimeError(f"teacher failed to lift: {cube(state)}")

    from .pick import pick_success, qualifying

    hold_t0 = time.monotonic()
    held = state
    while time.monotonic() - hold_t0 < max(hold_s, 0.0):
        held = client.state()
        c = cube(held)
        if not qualifying(
            attached=bool(c["attached"]),
            cube_z_mm=float(c["center_mm"]["z"]),
            tcp_level=bool(held.get("tcp_level")),
        ):
            break
        time.sleep(0.1)
    measured_hold = time.monotonic() - hold_t0
    if not qualifying(
        attached=bool(cube(held)["attached"]),
        cube_z_mm=float(cube(held)["center_mm"]["z"]),
        tcp_level=bool(held.get("tcp_level")),
    ):
        measured_hold = 0.0
    photos = capture_views("03b-hold2s")
    note("hold", held, lift, photos.get("Front"), photos)

    picked = pick_success(
        attached=bool(cube(held)["attached"]),
        cube_z_mm=float(cube(held)["center_mm"]["z"]),
        tcp_level=bool(held.get("tcp_level")),
        hold_s=measured_hold,
        min_hold_s=hold_s,
    )
    open_cmd = {"opening_mm": 90}
    client.call("rebot_set_gripper", open_cmd)
    state = client.wait_stopped()
    deadline = time.monotonic() + 3
    landed = state
    while time.monotonic() < deadline:
        landed = client.state()
        c = cube(landed)
        if not c["attached"] and not c.get("falling") and c["center_mm"]["z"] < 25:
            break
        time.sleep(0.05)
    photos = capture_views("04-landed")
    note("landed", landed, open_cmd, photos.get("Front"), photos)
    summary = {
        "ok": picked,
        "picked": picked,
        "attached": bool(cube(held)["attached"]),
        "peak_cube_z_mm": float(cube(held)["center_mm"]["z"]),
        "tcp_level": bool(held.get("tcp_level")),
        "hold_s": measured_hold,
        "cube_xy": [x_mm, y_mm],
        "cube_size_mm": size_mm,
        "run_dir": str(run_dir),
        "steps": [row["step"] for row in log],
        "observation_mode": "state-assisted-teacher",
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
