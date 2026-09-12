from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from train.dataset import Sample

PAD_DEPTH_MM = 35.0
PAD_Z_MM = 48.0
LIFT_Z_MM = 160.0
DEFAULT_CUBE_XY = (280.0, 0.0)
DEFAULT_CUBE_SIZE_MM = 20.0
READY_TCP_MM = (542.905, 0.0, 409.289)
HOLD_WEIGHT = 12.0
CLOSE_WEIGHT = 16.0


def pad_xy(cube_x: float, cube_y: float, yaw_rad: float, depth: float = PAD_DEPTH_MM) -> tuple[float, float]:
    return cube_x + depth * math.cos(yaw_rad), cube_y + depth * math.sin(yaw_rad)


@dataclass
class OverlayCommand:
    action: np.ndarray
    phase: str
    pad_xyz: tuple[float, float, float]
    dist_pad_xy: float


def overlay_command(
    tcp: dict,
    gripper_mm: float,
    attached: bool,
    yaw: float,
    cube_xy: tuple[float, float] = DEFAULT_CUBE_XY,
    cube_size_mm: float = DEFAULT_CUBE_SIZE_MM,
    pad_lock: tuple[float, float, float] | None = None,
    step: float = 8.0,
    lift_z: float = LIFT_Z_MM,
    pad_depth: float = PAD_DEPTH_MM,
    pad_z: float = PAD_Z_MM,
) -> OverlayCommand:
    """Close only when TCP is in the jaw box."""
    hx, hy, hz = cube_xy[0], cube_xy[1], pad_z
    if pad_lock is not None:
        gx, gy, gz = pad_lock
    else:
        gx, gy = pad_xy(hx, hy, yaw, depth=pad_depth)
        gz = pad_z
    tx, ty, tz = float(tcp["x"]), float(tcp["y"]), float(tcp["z"])
    dist_pad = float(math.hypot(tx - gx, ty - gy))
    dist_hover = float(math.hypot(tx - hx, ty - hy))
    if attached:
        dz = float(np.clip(lift_z - tz, -step, step))
        if tz >= lift_z - 2.0:
            dz = 0.0
        phase = "hold" if tz >= 140.0 else "lift"
        action = np.array([0.0, 0.0, max(dz, 0.0), 0.0], dtype=np.float64)
        return OverlayCommand(action, phase, (gx, gy, gz), dist_pad)
    near_pad = dist_pad < 18.0 and abs(tz - gz) < 18.0
    if near_pad:
        dgrip = float(np.clip((cube_size_mm + 2.0) - gripper_mm, -step, step))
        action = np.array(
            [
                float(np.clip(gx - tx, -step, step)),
                float(np.clip(gy - ty, -step, step)),
                float(np.clip(gz - tz, -step, step)),
                dgrip,
            ],
            dtype=np.float64,
        )
        return OverlayCommand(action, "close", (gx, gy, gz), dist_pad)
    keep_open = 0.0 if gripper_mm >= 80.0 else float(np.clip(90.0 - gripper_mm, 0.0, step))
    at_hover = dist_hover < 20.0 and abs(tz - hz) < 22.0
    ax, ay, az = (gx, gy, gz) if at_hover else (hx, hy, hz)
    action = np.array(
        [
            float(np.clip(ax - tx, -step, step)),
            float(np.clip(ay - ty, -step, step)),
            float(np.clip(az - tz, -step, step)),
            keep_open,
        ],
        dtype=np.float64,
    )
    phase = "align" if at_hover else "approach"
    return OverlayCommand(action, phase, (gx, gy, gz), dist_pad)


def expert_action(sample: Sample, step: float = 8.0, hold_z: float = LIFT_Z_MM) -> np.ndarray:
    """Grok pad-grasp expert. Close is negative dgrip. Hold keeps dgrip at 0."""
    gx, gy = pad_xy(float(sample.cube_mm[0]), float(sample.cube_mm[1]), float(sample.yaw_rad))
    tcp = sample.tcp_mm
    size = float(sample.cube_size_mm or DEFAULT_CUBE_SIZE_MM)
    if sample.attached:
        dz = float(np.clip(hold_z - tcp[2], -step, step))
        if tcp[2] >= hold_z - 2.0:
            dz = 0.0
        return np.array([0.0, 0.0, max(dz, 0.0), 0.0], dtype=np.float64)
    dx = float(np.clip(gx - tcp[0], -step, step))
    dy = float(np.clip(gy - tcp[1], -step, step))
    dz = float(np.clip(PAD_Z_MM - tcp[2], -step, step))
    dist = float(math.hypot(tcp[0] - gx, tcp[1] - gy))
    near = dist < 40.0 and abs(float(tcp[2]) - PAD_Z_MM) < 30.0
    if near:
        dgrip = float(np.clip((size + 2.0) - sample.gripper_mm, -step, step))
    else:
        dgrip = 0.0 if sample.gripper_mm >= 80.0 else float(np.clip(90.0 - sample.gripper_mm, 0.0, step))
    return np.array([dx, dy, dz, dgrip], dtype=np.float64)


def overlay_action(sample: Sample, step: float = 8.0, hold_z: float = LIFT_Z_MM) -> np.ndarray:
    tcp = {"x": float(sample.tcp_mm[0]), "y": float(sample.tcp_mm[1]), "z": float(sample.tcp_mm[2])}
    cube_xy = (float(sample.cube_mm[0]), float(sample.cube_mm[1]))
    return overlay_command(
        tcp,
        float(sample.gripper_mm),
        bool(sample.attached),
        float(sample.yaw_rad),
        cube_xy=cube_xy,
        cube_size_mm=float(sample.cube_size_mm or DEFAULT_CUBE_SIZE_MM),
        step=step,
        lift_z=hold_z,
    ).action


def _row_weight(phase: str, attached: bool, dgrip: float) -> float:
    if attached or phase in {"hold", "lift"}:
        return HOLD_WEIGHT
    if phase == "close" or dgrip < -0.5:
        return CLOSE_WEIGHT
    if phase == "align":
        return 10.0
    return 1.0


def imitation_actions(
    samples: list[Sample],
    step: float = 8.0,
    hold_z: float = LIFT_Z_MM,
    hold_bonus: bool = True,
    relabel: str = "hold",
    hold_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Labels: overlay pad traces, hold-only expert, or logged deltas."""
    actions = np.zeros((len(samples), 4), dtype=np.float64)
    weights = np.ones(len(samples), dtype=np.float64)
    mode = str(relabel or "hold")
    attached_weight = HOLD_WEIGHT if mode == "overlay" else 6.0
    if hold_weight is not None:
        attached_weight = float(hold_weight)
    for i, sample in enumerate(samples):
        raw = np.asarray(sample.action, dtype=np.float64).reshape(-1)
        if raw.size < 4:
            raw = np.zeros(4, dtype=np.float64)
        raw = np.clip(raw[:4], -step, step)
        if mode == "overlay":
            if sample.attached:
                actions[i] = raw
                weights[i] = attached_weight
            else:
                actions[i] = overlay_action(sample, step=step, hold_z=hold_z)
                weights[i] = _row_weight("overlay", False, float(actions[i, 3]))
            continue
        if hold_bonus and sample.attached:
            actions[i] = expert_action(sample, step=step, hold_z=hold_z)
            weights[i] = attached_weight
        else:
            actions[i] = raw
            weights[i] = 1.0
    return actions, weights


def overlay_rollout(
    start_tcp: tuple[float, float, float] = READY_TCP_MM,
    start_grip: float = 90.0,
    cube_xy: tuple[float, float] = DEFAULT_CUBE_XY,
    cube_size_mm: float = DEFAULT_CUBE_SIZE_MM,
    yaw_rad: float = 0.0,
    step: float = 8.0,
    max_steps: int = 90,
    attached: bool = False,
) -> list[dict]:
    tcp = {"x": float(start_tcp[0]), "y": float(start_tcp[1]), "z": float(start_tcp[2])}
    grip = float(start_grip)
    attached_now = bool(attached)
    pad_lock: tuple[float, float, float] | None = None
    rows: list[dict] = []
    for _ in range(max(int(max_steps), 1)):
        decision = overlay_command(
            tcp,
            grip,
            attached_now,
            yaw_rad,
            cube_xy=cube_xy,
            cube_size_mm=cube_size_mm,
            pad_lock=pad_lock,
            step=step,
        )
        if pad_lock is None and decision.phase in {"align", "close"}:
            pad_lock = decision.pad_xyz
        action = np.asarray(decision.action, dtype=np.float64)
        rows.append(
            {
                "tcp": np.array([tcp["x"], tcp["y"], tcp["z"]], dtype=np.float64),
                "grip": grip,
                "attached": attached_now,
                "action": action,
                "phase": decision.phase,
                "weight": _row_weight(decision.phase, attached_now, float(action[3])),
                "yaw_rad": float(yaw_rad),
                "cube_xy": cube_xy,
            }
        )
        tcp["x"] += float(action[0])
        tcp["y"] += float(action[1])
        tcp["z"] = max(32.0, tcp["z"] + float(action[2]))
        if decision.phase == "close":
            grip = float(cube_size_mm) + 2.0
            attached_now = True
        elif attached_now:
            grip = float(cube_size_mm) + 2.0
        else:
            grip = float(np.clip(grip + float(action[3]), 0.0, 90.0))
        if decision.phase == "hold" and tcp["z"] >= LIFT_Z_MM - 2.0:
            break
    return rows


def overlay_training_rows(
    cube_xy: tuple[float, float] = DEFAULT_CUBE_XY,
    cube_size_mm: float = DEFAULT_CUBE_SIZE_MM,
    step: float = 8.0,
    val_traces: int = 4,
) -> tuple[list[dict], list[dict]]:
    """Train/val overlay traces; val = last N whole traces."""
    yaws = (0.0, math.radians(56.6), math.radians(50.0), math.radians(60.0))
    starts = (
        READY_TCP_MM,
        (500.0, 0.0, 350.0),
        (400.0, 0.0, 200.0),
        (350.0, 0.0, 100.0),
        (280.0, 0.0, 80.0),
        (280.0, 0.0, 48.0),
        (315.0, 0.0, 48.0),
        (303.0, 26.0, 48.0),
        (320.0, 5.0, 60.0),
        (260.0, -5.0, 70.0),
        (295.0, 13.0, 65.0),
        (290.0, 10.0, 70.0),
        (300.0, 20.0, 55.0),
        (270.0, 24.0, 59.0),
        (280.0, -35.0, 48.0),
        (280.0, 35.0, 48.0),
        (300.0, -30.0, 48.0),
        (300.0, 30.0, 48.0),
        (302.0, -16.0, 134.0),
        (275.0, -2.0, 115.0),
        (294.0, -20.0, 100.0),
        (295.0, -19.0, 75.0),
        (307.0, -5.0, 55.0),
        (308.0, -4.0, 54.0),
        (325.0, -16.0, 114.0),
        (259.0, -5.0, 94.0),
        (278.0, -2.0, 137.0),
    )
    traces: list[list[dict]] = []
    for yaw in yaws:
        for start in starts:
            traces.append(
                overlay_rollout(
                    start_tcp=start,
                    start_grip=90.0,
                    cube_xy=cube_xy,
                    cube_size_mm=cube_size_mm,
                    yaw_rad=yaw,
                    step=step,
                    attached=False,
                )
            )
        gx, gy = pad_xy(cube_xy[0], cube_xy[1], yaw)
        traces.append(
            overlay_rollout(
                start_tcp=(gx, gy, PAD_Z_MM),
                start_grip=90.0,
                cube_xy=cube_xy,
                cube_size_mm=cube_size_mm,
                yaw_rad=yaw,
                step=step,
                attached=False,
            )
        )
        traces.append(
            overlay_rollout(
                start_tcp=(gx, gy, PAD_Z_MM),
                start_grip=cube_size_mm + 2.0,
                cube_xy=cube_xy,
                cube_size_mm=cube_size_mm,
                yaw_rad=yaw,
                step=step,
                attached=True,
            )
        )
    n_val = min(max(int(val_traces), 1), max(len(traces) - 1, 1))
    val = [row for trace in traces[-n_val:] for row in trace]
    train = [row for trace in traces[:-n_val] for row in trace]
    close_hold = [row for row in train if row["phase"] in {"close", "hold", "lift"}]
    train.extend(close_hold)
    train.extend(close_hold)
    align = [row for row in train if row["phase"] == "align"]
    train.extend(align)
    return train, val
