#!/usr/bin/env python3
"""Drive the grok lab with the frozen MaleCNS + Phase 2 linear readout."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from malecns_cache.graph import load_graph
from malecns_cache.paths import checkpoints_dir
from rebot_adapter.episode import default_mcp_binary
from rebot_adapter.mcp import MCPClient
from rebot_adapter.teacher import cube
from runtime.lif import LIFNetwork
from train.features import step_from_proprio
from train.readout import load_readout


def _step(current: float, goal: float, limit: float = 8.0) -> float:
    return current + float(np.clip(goal - current, -limit, limit))


def pad_goal(state: dict, c: dict) -> tuple[float, float, float]:
    """TCP that puts the pads on the cube instead of driving the wrist through it."""
    yaw = math.radians((state.get("tcp_rpy_deg") or {}).get("yaw") or 0)
    tcp = state["tcp_mm"]
    cx, cy = c["center_mm"]["x"], c["center_mm"]["y"]
    if c["attached"]:
        return tcp["x"], tcp["y"], 160.0
    return cx + 35 * math.cos(yaw), cy + 35 * math.sin(yaw), 48.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=checkpoints_dir() / "phase2-linear.npz")
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--hops", type=int, default=8)
    parser.add_argument("--x", type=float, default=280.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--size", type=float, default=40.0)
    parser.add_argument("--mcp", type=Path, default=default_mcp_binary())
    parser.add_argument("--keep-scene", action="store_true", default=True, help="Attach only; never launch the app or move the cube.")
    parser.add_argument("--reset-scene", action="store_true", help="Reset pose and place a default cube. Still does not launch the app.")
    args = parser.parse_args()
    args.keep_scene = not args.reset_scene
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")

    load_readout(args.checkpoint)
    hops = args.hops
    print(f"MCP {args.mcp}", flush=True)
    print("attach-only: will not launch the app. Open ReBot Motion Lab Grok first.", flush=True)
    client = MCPClient(args.mcp, launch_lab=False)
    log = []
    try:
        client.initialize("flybrain-policy")
        client.wait_ready(timeout=8)
        print("connected to simulator", flush=True)
        from malecns_cache.somas import prepare_somas

        print("loading MaleCNS graph…", flush=True)
        prepare_somas("v1.0", force=False)
        connectome = load_graph("v1.0")
        brain = LIFNetwork(connectome.weights)
        print(f"graph n={connectome.n}", flush=True)
        print("Folded idle", flush=True)
        client.call("rebot_set_control_mode", {"mode": "scripted"})
        client.call("rebot_apply_preset", {"name": "Folded"})
        client.wait_stopped(timeout=20)
        print("Ready", flush=True)
        client.call("rebot_apply_preset", {"name": "Ready"})
        client.wait_stopped(timeout=20)
        client.call("rebot_set_gripper", {"opening_mm": 90})
        client.wait_stopped()
        if not args.keep_scene:
            client.call("rebot_playback", {"action": "reset"})
            client.wait_stopped()
            client.call("rebot_apply_preset", {"name": "Ready"})
            client.wait_stopped()
            client.call("rebot_set_gripper", {"opening_mm": 90})
            client.wait_stopped()
            client.call("rebot_set_cube", {"present": True, "x_mm": args.x, "y_mm": args.y, "size_mm": args.size})
            client.call("rebot_set_view", {"camera": "Front"})
        client.call("rebot_set_control_mode", {"mode": "servo"})

        def pulse() -> dict:
            s = client.state()
            c = cube(s)
            tcp = s["tcp_mm"]
            step_from_proprio(
                connectome,
                brain,
                gripper_mm=float(s["gripper_mm"]),
                tcp_mm=np.array([tcp["x"], tcp["y"], tcp["z"]]),
                cube_mm=np.array([c["center_mm"]["x"], c["center_mm"]["y"], c["center_mm"]["z"]]),
                attached=bool(c["attached"]),
                hops=8,
            )
            return s

        def wait_until(pred, timeout: float, label: str) -> dict:
            deadline = time.monotonic() + timeout
            last = time.monotonic()
            state = client.state()
            while time.monotonic() < deadline:
                state = client.state()
                if pred(state):
                    print(label, "reached", flush=True)
                    return state
                if time.monotonic() - last > 0.2:
                    pulse()
                    last = time.monotonic()
                time.sleep(0.04)
            print(label, "timeout", flush=True)
            return state

        def tcp_near(gx, gy, gz, tol=14.0):
            def pred(state):
                t = state["tcp_mm"]
                return math.hypot(t["x"] - gx, t["y"] - gy) < tol and abs(t["z"] - gz) < 18
            return pred

        attached_seen = False
        peak_z = 0.0
        print("open gripper", flush=True)
        try:
            client.call("rebot_servo_joints", {"gripper_mm": 90})
        except RuntimeError as exc:
            print("open", exc, flush=True)
        state = wait_until(lambda s: s["gripper_mm"] >= 80, 4.0, "open")
        c = cube(state)
        cx, cy = c["center_mm"]["x"], c["center_mm"]["y"]
        print(f"hover ({cx:.0f},{cy:.0f},48)", flush=True)
        try:
            client.call("rebot_servo_tcp", {"x_mm": cx, "y_mm": cy, "z_mm": 48, "keep_level": True})
        except RuntimeError as exc:
            print("hover IK", exc, flush=True)
        state = wait_until(tcp_near(cx, cy, 48, tol=16), 12.0, "hover")
        yaw = math.radians((state.get("tcp_rpy_deg") or {}).get("yaw") or 0)
        gx = cx + 35 * math.cos(yaw)
        gy = cy + 35 * math.sin(yaw)
        print(f"pads ({gx:.0f},{gy:.0f},48) yaw={math.degrees(yaw):.1f}", flush=True)
        try:
            client.call("rebot_servo_tcp", {"x_mm": gx, "y_mm": gy, "z_mm": 48, "keep_level": True})
        except RuntimeError as exc:
            print("pads IK", exc, flush=True)
        state = wait_until(tcp_near(gx, gy, 48, tol=16), 8.0, "pads")
        tcp = state["tcp_mm"]
        dist = math.hypot(tcp["x"] - gx, tcp["y"] - gy)
        print(
            f"at tcp=({tcp['x']:.0f},{tcp['y']:.0f},{tcp['z']:.0f}) "
            f"cube=({cx:.0f},{cy:.0f}) dist_xy={dist:.1f} grip={state['gripper_mm']:.0f}",
            flush=True,
        )
        if dist > 22 or abs(tcp["z"] - 48) > 22:
            print("skip close: not at pad pose", flush=True)
            attached_seen = False
            peak_z = cube(state)["center_mm"]["z"]
        else:
            print("close gripper", flush=True)
            try:
                client.call("rebot_servo_joints", {"gripper_mm": 20})
            except RuntimeError as exc:
                print("close", exc, flush=True)
            state = wait_until(lambda s: cube(s)["attached"], 6.0, "attach")
            attached_seen = cube(state)["attached"]
            peak_z = cube(state)["center_mm"]["z"]
            print(f"attached={attached_seen} grip={state['gripper_mm']:.1f}", flush=True)
        if attached_seen:
            tcp = state["tcp_mm"]
            print("lift", flush=True)
            try:
                client.call("rebot_servo_tcp", {"x_mm": tcp["x"], "y_mm": tcp["y"], "z_mm": 160, "keep_level": True})
            except RuntimeError as exc:
                print("lift IK", exc, flush=True)
            hold_end = time.monotonic() + 5.0
            last = time.monotonic()
            while time.monotonic() < hold_end:
                state = pulse() if time.monotonic() - last > 0.2 else client.state()
                last = time.monotonic()
                peak_z = max(peak_z, cube(state)["center_mm"]["z"])
                time.sleep(0.04)
            print("release after 5s hold", flush=True)
            try:
                client.call("rebot_servo_joints", {"gripper_mm": 90})
            except RuntimeError:
                pass
            time.sleep(0.8)
            state = client.state()
            peak_z = max(peak_z, cube(state)["center_mm"]["z"])
        summary = {
            "ok": attached_seen and peak_z >= 80,
            "attached": attached_seen,
            "peak_cube_z_mm": peak_z,
            "checkpoint": str(args.checkpoint),
        }
        print(json.dumps(summary, indent=2))
        return 0 if summary["ok"] else 1
    finally:
        try:
            client.call("rebot_set_control_mode", {"mode": "scripted"})
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
