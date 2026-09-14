#!/usr/bin/env python3
"""Log proposed B601 joints from JPEGs. Never opens CAN."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebot_adapter.arm_host import OffHostError, open_physical_can, require_arm_host
from rebot_adapter.b601 import READY_ARM_DEG, ik_arm
from rebot_adapter.hw_phase import write_json


def shadow_from_dir(folder: Path, stub: bool, steps: int) -> dict:
    import numpy as np
    from arm.crop import build_crop, write_stub_crop
    from arm.lif_crop import CropLIF
    from arm.unpack import unpack
    from malecns_cache.paths import project_data
    from rebot_adapter.hw_phase import hop_search_from_dir

    if stub:
        crop = write_stub_crop(project_data() / "prepared" / "stub-hw-shadow")
    else:
        crop = build_crop()
    hop = hop_search_from_dir(crop, folder, nsteps=steps)
    cmd = hop.get("cube_command") or {}
    q0 = np.array(READY_ARM_DEG, dtype=np.float64)
    from rebot_adapter.b601 import fk_pose

    pose = fk_pose(q0)
    target = {
        "x": float(pose.x_mm) + float(cmd.get("dx_mm") or 0.0),
        "y": float(pose.y_mm) + float(cmd.get("dy_mm") or 0.0),
        "z": float(pose.z_mm) + float(cmd.get("dz_mm") or 0.0),
    }
    q1, err = ik_arm(target, q0, keep_level=bool(cmd.get("keep_level", True)))
    joints = {name: float(v) for name, v in zip(
        ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"), q1
    )}
    joints["gripper"] = 0.0
    return {
        "sent": False,
        "can_open": False,
        "connect_called": False,
        "joints_estimated": True,
        "ik_err_mm": float(err),
        "joints_deg": joints,
        "tcp_command": cmd,
        "hop_ok": hop.get("ok"),
        "synaptic_gain": hop.get("synaptic_gain"),
        "cube_dn_mean": hop.get("cube_dn_mean"),
        "fly_picked": False,
        "t": time.time(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Shadow DN-bus → IK log (no CAN)")
    p.add_argument("--from-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--arm-live", action="store_true", help="Refuse unless on isengard; still does not open CAN")
    args = p.parse_args(argv)
    if args.arm_live:
        try:
            require_arm_host()
        except OffHostError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            open_physical_can()
        except OffHostError:
            pass
    row = shadow_from_dir(args.from_dir, stub=args.stub, steps=args.steps)
    write_json(args.out, row)
    print(json.dumps({k: row[k] for k in ("sent", "can_open", "hop_ok", "ik_err_mm", "fly_picked")}))
    return 0 if row["sent"] is False and row["can_open"] is False else 2


if __name__ == "__main__":
    raise SystemExit(main())
