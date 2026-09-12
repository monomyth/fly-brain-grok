#!/usr/bin/env python3
"""Scripted pick; brain watches. JPEG → R1–R6, PPL101 / KC→MBON on correct phases."""
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from pathlib import Path

import numpy as np

from malecns_cache.graph import load_graph
from malecns_cache.paths import checkpoints_dir
from malecns_cache.somas import prepare_somas
from rebot_adapter.episode import default_mcp_binary
from rebot_adapter.mcp import MCPClient
from rebot_adapter.teacher import cube
from runtime.broadcast import publish_activity
from runtime.encoder import decode_jpeg, encode_frame
from train.features import drive_kenyon_from_image
from runtime.lif import LIFNetwork
from runtime.plasticity import KCToMBON


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp", type=Path, default=default_mcp_binary())
    parser.add_argument("--hops", type=int, default=16)
    parser.add_argument("--shuffle-reward", action="store_true")
    parser.add_argument("--tag", default="learn")
    args = parser.parse_args()
    prepare_somas("v1.0")
    connectome = load_graph("v1.0")
    brain = LIFNetwork(connectome.weights)
    memory = KCToMBON(connectome, brain)
    r1 = connectome.indices("photoreceptors_r1r6")
    r8 = connectome.indices("photoreceptors_r8")
    print(f"learn-on-script n={connectome.n} kc-mbon={memory.slots.size}", flush=True)

    client = MCPClient(args.mcp, launch_lab=False)
    rewards: list[float] = []
    prev_grid: np.ndarray | None = None
    last_dump: np.ndarray | None = None
    DA = 0.15
    SHUFFLE_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0)

    def see(reward: float, label: str) -> dict:
        nonlocal prev_grid, last_dump
        state = client.state()
        try:
            shot = client.call("rebot_capture_view", {"camera": "Front", "width": 160, "height": 120, "apply": False})
            rgb = decode_jpeg(base64.b64decode(shot["jpeg_base64"]))
            vis, chroma = encode_frame(rgb, int(r1.size), int(r8.size))
            brain.i_ext.fill(0)
            if r1.size:
                brain.i_ext[r1] = vis
            if r8.size:
                brain.i_ext[r8] = np.clip(chroma, 0, 4)
            prev_grid = drive_kenyon_from_image(brain, memory.kc, rgb, prev_grid=prev_grid)
        except RuntimeError:
            if memory.kc.size:
                brain.i_ext[memory.kc] = np.float32(60.0)
        dump = abs(reward) >= DA
        if args.shuffle_reward and dump:
            applied = float(SHUFFLE_SIGNS[len(rewards) % len(SHUFFLE_SIGNS)] * abs(reward))
        else:
            applied = reward
        if dump:
            memory.pulse(80.0)
        brain.step(args.hops)
        if dump:
            n = memory.update(applied, eta=1e-2)
            now = brain.duty[memory.kc] if memory.kc.size else np.zeros(0)
            cos = 0.0
            if last_dump is not None and now.size and float(np.linalg.norm(now)) and float(np.linalg.norm(last_dump)):
                cos = float(now @ last_dump / (np.linalg.norm(now) * np.linalg.norm(last_dump)))
            last_dump = now.copy()
            on = float((now > 0.05).mean()) if now.size else 0.0
            print(
                f"{label} r={applied:+.2f}{' shuffle' if args.shuffle_reward else ''} "
                f"edges={n} kc_on={on:.3f} vs_prev={cos:.3f}",
                flush=True,
            )
            rewards.append(applied)
        else:
            memory.accumulate()
        publish_activity(brain)
        return state

    try:
        client.initialize("flybrain-learn")
        client.wait_ready(timeout=8)
        print("connected — script picks, brain watches", flush=True)

        def idle_then_ready() -> None:
            print("Folded idle", flush=True)
            client.call("rebot_set_control_mode", {"mode": "scripted"})
            client.call("rebot_apply_preset", {"name": "Folded"})
            client.wait_stopped(timeout=20)
            print("Ready", flush=True)
            client.call("rebot_apply_preset", {"name": "Ready"})
            client.wait_stopped(timeout=20)
            client.call("rebot_set_cube", {"present": True, "x_mm": 280, "y_mm": 0, "size_mm": 40})
            client.call("rebot_set_gripper", {"opening_mm": 90})
            client.wait_stopped()

        def watch_stopped(timeout: float, label: str, reward_ok: float) -> dict:
            time.sleep(0.12)
            deadline = time.monotonic() + timeout
            last = 0.0
            state = client.state()
            while time.monotonic() < deadline:
                state = client.state()
                if state.get("playback") == "stopped" and not state.get("manual_motion"):
                    see(reward_ok, f"{label} reached")
                    return state
                if time.monotonic() - last > 0.25:
                    see(0.0, label)
                    last = time.monotonic()
                time.sleep(0.04)
            see(-0.4, f"{label} timeout")
            print(label, "timeout", flush=True)
            return state

        idle_then_ready()
        brain.reset_state()

        print("open gripper", flush=True)
        client.call("rebot_set_gripper", {"opening_mm": 90})
        state = watch_stopped(8.0, "open", 0.2)

        c = cube(state)
        cx, cy = c["center_mm"]["x"], c["center_mm"]["y"]
        print(f"hover ({cx:.0f},{cy:.0f},48)", flush=True)
        client.call("rebot_move_to_pose", {"x_mm": cx, "y_mm": cy, "z_mm": 48, "keep_level": True})
        state = watch_stopped(20.0, "hover", 0.5)

        yaw = math.radians((state.get("tcp_rpy_deg") or {}).get("yaw") or 0)
        gx = cx + 35 * math.cos(yaw)
        gy = cy + 35 * math.sin(yaw)
        print(f"pads ({gx:.0f},{gy:.0f},48) yaw={math.degrees(yaw):.1f}", flush=True)
        client.call("rebot_move_to_pose", {"x_mm": gx, "y_mm": gy, "z_mm": 48, "keep_level": True})
        state = watch_stopped(20.0, "pads", 0.7)

        tcp = state["tcp_mm"]
        dist = math.hypot(tcp["x"] - gx, tcp["y"] - gy)
        print(
            f"at tcp=({tcp['x']:.0f},{tcp['y']:.0f},{tcp['z']:.0f}) "
            f"cube=({cx:.0f},{cy:.0f},{c['center_mm']['z']:.0f}) dist_xy={dist:.1f} grip={state['gripper_mm']:.0f}",
            flush=True,
        )
        attached = False
        peak_z = cube(state)["center_mm"]["z"]
        if dist > 22 or abs(tcp["z"] - 48) > 22:
            print(f"skip close dist_xy={dist:.1f}", flush=True)
            see(-0.5, "missed pads")
        else:
            print("close gripper", flush=True)
            client.call("rebot_set_gripper", {"opening_mm": 20})
            state = watch_stopped(12.0, "attach", 1.6)
            attached = cube(state)["attached"]
            peak_z = cube(state)["center_mm"]["z"]
            print(f"attached={attached} grip={state['gripper_mm']:.1f}", flush=True)

        if attached:
            tcp = state["tcp_mm"]
            print("lift", flush=True)
            client.call("rebot_move_to_pose", {"x_mm": tcp["x"], "y_mm": tcp["y"], "z_mm": 160, "keep_level": True})
            state = watch_stopped(20.0, "lift", 1.2)
            peak_z = max(peak_z, cube(state)["center_mm"]["z"])
            hold_end = time.monotonic() + 5.0
            last = 0.0
            while time.monotonic() < hold_end:
                state = client.state()
                peak_z = max(peak_z, cube(state)["center_mm"]["z"])
                if time.monotonic() - last > 0.25:
                    see(0.0, "hold")
                    last = time.monotonic()
                time.sleep(0.04)
            print("release after 5s hold", flush=True)
            client.call("rebot_set_gripper", {"opening_mm": 90})
            state = watch_stopped(8.0, "release", 0.4)
            peak_z = max(peak_z, cube(state)["center_mm"]["z"])

        print("Folded idle after pick", flush=True)
        try:
            client.call("rebot_set_control_mode", {"mode": "scripted"})
            client.call("rebot_apply_preset", {"name": "Folded"})
            client.wait_stopped(timeout=20)
        except RuntimeError as exc:
            print("fold after", exc, flush=True)

        dest = checkpoints_dir("rebot-pickup") / f"kc-mbon-{args.tag}.npz"
        np.savez(
            dest,
            data=brain.weights.data.copy(),
            slots=memory.slots,
            rest=memory.rest,
            pre=memory.pre,
            post=memory.post,
        )
        delta = brain.weights.data[memory.slots] - memory.rest if memory.slots.size else np.zeros(0)
        summary = {
            "ok": attached and peak_z >= 80,
            "attached": attached,
            "peak_cube_z_mm": peak_z,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "kc_mbon": int(memory.slots.size),
            "l2_delta": float(np.linalg.norm(delta)),
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "shuffle_reward": bool(args.shuffle_reward),
            "checkpoint": str(dest),
        }
        print(json.dumps(summary, indent=2), flush=True)
        return 0 if summary["ok"] else 1
    finally:
        try:
            client.call("rebot_set_control_mode", {"mode": "scripted"})
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
