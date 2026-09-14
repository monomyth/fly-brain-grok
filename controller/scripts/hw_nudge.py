#!/usr/bin/env python3
"""DN-bus millimetre nudge. Plan with fly-brain Python; --go on isengard with LeRobot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rebot_adapter.arm_host import OffHostError, require_arm_host
from rebot_adapter.b601 import FOLDED_ARM_DEG, FailClosed, fk_pose, ik_arm

CAP_MM = 15.0
SCALE_TO_MM = 10.0
TINY_MM = 2.0
IK_ERR_MM = 5.0


def clip_delta(dx: float, dy: float, dz: float, *, cap: float = CAP_MM, scale_to: float = SCALE_TO_MM) -> dict:
    raw = np.array([float(dx), float(dy), float(dz)], dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise FailClosed("non-finite unpack delta")
    clipped = np.clip(raw, -cap, cap)
    n = float(np.linalg.norm(clipped))
    scaled = False
    used = clipped
    if n < 1e-6:
        raise FailClosed("unpack delta is zero")
    if n < TINY_MM:
        used = clipped * (scale_to / n)
        scaled = True
    return {
        "raw_mm": {"dx": float(raw[0]), "dy": float(raw[1]), "dz": float(raw[2])},
        "used_mm": {"dx": float(used[0]), "dy": float(used[1]), "dz": float(used[2])},
        "scaled": scaled,
        "norm_raw_mm": n if not scaled else float(np.linalg.norm(raw)),
    }


def plan_nudge(arm_deg: np.ndarray, delta: dict) -> dict:
    pose = fk_pose(arm_deg)
    u = delta["used_mm"]
    target = {"x": pose.x_mm + u["dx"], "y": pose.y_mm + u["dy"], "z": pose.z_mm + u["dz"]}
    q1, err = ik_arm(target, arm_deg, keep_level=False)
    if not np.all(np.isfinite(q1)) or err > IK_ERR_MM:
        raise FailClosed(f"IK failed err_mm={err}")
    out = dict(delta)
    out.update(
        {
            "from_mm": pose.as_mm(),
            "to_mm": target,
            "q0_deg": [float(x) for x in arm_deg],
            "q1_deg": [float(x) for x in q1],
            "ik_err_mm": float(err),
            "fly_picked": False,
        }
    )
    return out


def _cmd_from_hop(hop: dict) -> tuple[float, float, float]:
    cmd = hop.get("cube_command") or {}
    return float(cmd.get("dx_mm") or 0.0), float(cmd.get("dy_mm") or 0.0), float(cmd.get("dz_mm") or 0.0)


def make_plan(frames: Path, steps: int, stub: bool) -> dict:
    from arm.crop import build_crop, write_stub_crop
    from malecns_cache.paths import project_data
    from rebot_adapter.hw_phase import hop_search_from_dir

    if stub:
        crop = write_stub_crop(project_data() / "prepared" / "stub-hw-nudge")
    else:
        crop = build_crop()
    hop = hop_search_from_dir(crop, frames, nsteps=steps)
    if not hop.get("ok"):
        raise FailClosed("hop not ok; refuse nudge")
    dx, dy, dz = _cmd_from_hop(hop)
    delta = clip_delta(dx, dy, dz)
    plan = plan_nudge(np.array(FOLDED_ARM_DEG, dtype=np.float64), delta)
    plan["hop_ok"] = hop.get("ok")
    plan["synaptic_gain"] = hop.get("synaptic_gain")
    plan["cube_dn_mean"] = hop.get("cube_dn_mean")
    plan["feed"] = hop.get("feed")
    plan["cube_command"] = hop.get("cube_command")
    plan["q0_is_folded_guess"] = True
    return plan


def apply_plan(plan: dict, *, go: bool) -> dict:
    import hw_hover as hover

    require_arm_host()
    if not go:
        plan = dict(plan)
        plan["sent"] = False
        plan["note"] = "dry run. ./nudge --go applies the capped fly delta and stays. Cycle/fold to zero."
        return plan
    robot = hover._make_robot()
    robot.connect(calibrate=False)
    try:
        obs = robot.get_observation()
        q0 = hover._joints_from_obs(obs)
        grip = float(obs.get("gripper.pos", 0.0))
        live = plan_nudge(q0, {"used_mm": plan["used_mm"], "raw_mm": plan.get("raw_mm", plan["used_mm"]), "scaled": plan.get("scaled", False)})
        q1 = np.array(live["q1_deg"])
        for q in hover._interp(q0, q1):
            robot.send_action(hover._action(q, grip))
            hover.time.sleep(1.0 / hover.RATE_HZ)
        hover._settle(robot, q1, grip)
        live["sent"] = True
        live["returned"] = False
        live["q_end_deg"] = [float(x) for x in hover._joints_from_obs(robot.get_observation())]
        live["gripper_deg"] = grip
        live["feed"] = plan.get("feed")
        return live
    finally:
        robot.disconnect()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fly-capped TCP nudge")
    p.add_argument("--from-dir", type=Path, default=None)
    p.add_argument("--plan-json", type=Path, default=None)
    p.add_argument("--apply", type=Path, default=None)
    p.add_argument("--go", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        if args.apply is not None:
            plan = json.loads(args.apply.read_text())
            result = apply_plan(plan, go=args.go)
        else:
            if args.from_dir is None:
                print("--from-dir is required to plan", file=sys.stderr)
                return 2
            result = make_plan(args.from_dir, steps=args.steps, stub=args.stub)
            result["sent"] = False
    except (OffHostError, FailClosed) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
