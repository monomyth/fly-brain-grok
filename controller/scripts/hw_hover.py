#!/usr/bin/env python3
"""Lift TCP a few millimetres on isengard. Does not run from the Mac."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebot_adapter.arm_host import OffHostError, require_arm_host
from rebot_adapter.b601 import ARM_JOINTS, CAN_JOINTS, FailClosed, fk_pose, ik_arm, load_gripper_curve

DZ_MM = 15.0
MAX_DZ_MM = 25.0
STEP_DEG = 2.0
RATE_HZ = 10.0
IK_ERR_MM = 5.0


def plan_hover(arm_deg: np.ndarray, dz_mm: float, *, keep_level: bool = False) -> dict:
    if dz_mm <= 0 or dz_mm > MAX_DZ_MM:
        raise FailClosed(f"dz_mm must be in (0, {MAX_DZ_MM}]")
    pose = fk_pose(arm_deg)
    target = {"x": pose.x_mm, "y": pose.y_mm, "z": pose.z_mm + float(dz_mm)}
    used = bool(keep_level)
    q1, err = ik_arm(target, arm_deg, keep_level=used)
    if (not np.all(np.isfinite(q1)) or err > IK_ERR_MM) and used:
        used = False
        q1, err = ik_arm(target, arm_deg, keep_level=False)
    if not np.all(np.isfinite(q1)) or err > IK_ERR_MM:
        raise FailClosed(
            f"IK failed err_mm={err} q0_deg={[round(float(x),2) for x in arm_deg]} "
            f"xyz_mm={[round(pose.x_mm,1), round(pose.y_mm,1), round(pose.z_mm,1)]}"
        )
    return {
        "from_mm": pose.as_mm(),
        "to_mm": target,
        "q0_deg": [float(x) for x in arm_deg],
        "q1_deg": [float(x) for x in q1],
        "ik_err_mm": float(err),
        "dz_mm": float(dz_mm),
        "keep_level": used,
    }


def _joints_from_obs(obs: dict) -> np.ndarray:
    missing = [n for n in ARM_JOINTS if f"{n}.pos" not in obs]
    if missing:
        raise FailClosed(f"missing joint feedback: {missing}")
    return np.array([float(obs[f"{n}.pos"]) for n in ARM_JOINTS], dtype=np.float64)


def _action(arm_deg: np.ndarray, grip: float) -> dict:
    out = {f"{n}.pos": float(v) for n, v in zip(ARM_JOINTS, arm_deg)}
    out["gripper.pos"] = float(grip)
    return out


def _interp(q0: np.ndarray, q1: np.ndarray) -> list[np.ndarray]:
    delta = float(np.max(np.abs(q1 - q0)))
    n = max(1, int(math_ceil(delta / STEP_DEG)))
    return [q0 + (q1 - q0) * (i / n) for i in range(1, n + 1)]


def math_ceil(x: float) -> int:
    import math

    return int(math.ceil(x))


def _make_robot():
    from lerobot.robots.rebot_b601_follower.config_rebot_b601_follower import RebotB601FollowerRobotConfig
    from lerobot.robots.rebot_b601_follower.rebot_b601_follower import RebotB601Follower

    cfg = RebotB601FollowerRobotConfig(
        port="/dev/ttyACM0",
        id="follower",
        can_adapter="damiao",
        disable_torque_on_disconnect=False,
        reset_to_calibrated_zero=False,
        control_mode="pos_vel",
        pos_vel_velocity=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 30.0],
        max_relative_target=3.0,
        cameras={},
    )
    return RebotB601Follower(cfg)


def run_hover(*, dz_mm: float, go: bool, hold_s: float) -> dict:
    require_arm_host()
    spec = load_gripper_curve("v1")
    if go:
        robot = _make_robot()
        robot.connect(calibrate=False)
        try:
            obs = robot.get_observation()
            q0 = _joints_from_obs(obs)
            grip = float(obs.get("gripper.pos", 0.0))
            plan = plan_hover(q0, dz_mm)
            plan["gripper_deg"] = grip
            plan["spec_version"] = spec.get("version")
            for q in _interp(q0, np.array(plan["q1_deg"])):
                robot.send_action(_action(q, grip))
                time.sleep(1.0 / RATE_HZ)
            time.sleep(hold_s)
            for q in _interp(np.array(plan["q1_deg"]), q0):
                robot.send_action(_action(q, grip))
                time.sleep(1.0 / RATE_HZ)
            plan["sent"] = True
            plan["returned"] = True
            return plan
        finally:
            robot.disconnect()
    return {
        "sent": False,
        "dz_mm": dz_mm,
        "note": "dry run. pass --go on isengard at the cell to enable motors and lift TCP.",
        "spec_version": spec.get("version"),
        "measured": spec.get("measured"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="B601 TCP hover on isengard")
    p.add_argument("--dz-mm", type=float, default=DZ_MM)
    p.add_argument("--go", action="store_true", help="Enable motors and move. Must be at the cell.")
    p.add_argument("--hold-s", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        result = run_hover(dz_mm=args.dz_mm, go=args.go, hold_s=args.hold_s)
    except OffHostError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FailClosed as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0 if (args.go and result.get("sent")) or (not args.go) else 2


if __name__ == "__main__":
    raise SystemExit(main())
