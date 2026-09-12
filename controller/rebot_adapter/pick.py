from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from runtime.decoder import ArmCommand
from train.expert import (
    DEFAULT_CUBE_SIZE_MM,
    DEFAULT_CUBE_XY,
    LIFT_Z_MM,
    PAD_DEPTH_MM,
    PAD_Z_MM,
    overlay_command,
    pad_xy,
)


def pick_success(
    *,
    attached: bool,
    cube_z_mm: float,
    tcp_level: bool,
    hold_s: float,
    min_z_mm: float = 100.0,
    min_hold_s: float = 2.0,
) -> bool:
    return bool(attached) and float(cube_z_mm) >= min_z_mm and bool(tcp_level) and float(hold_s) >= min_hold_s


def qualifying(*, attached: bool, cube_z_mm: float, tcp_level: bool, min_z_mm: float = 100.0) -> bool:
    return bool(attached) and float(cube_z_mm) >= min_z_mm and bool(tcp_level)


def score_pick(state: dict, hold_s: float, min_hold_s: float = 2.0) -> bool:
    cube = state["objects"]["cube"]
    return pick_success(
        attached=bool(cube.get("attached")),
        cube_z_mm=float(cube["center_mm"]["z"]),
        tcp_level=bool(state.get("tcp_level")),
        hold_s=hold_s,
        min_hold_s=min_hold_s,
    )


def dist_xy(tcp: dict, cube_xy: tuple[float, float]) -> float:
    return float(math.hypot(tcp["x"] - cube_xy[0], tcp["y"] - cube_xy[1]))


def yaw_rad(state: dict) -> float:
    return math.radians(float((state.get("tcp_rpy_deg") or {}).get("yaw") or 0.0))


def servo_stalled(cmd: ArmCommand, before: dict, after: dict, min_frac: float = 0.25) -> bool:
    commanded = abs(cmd.dx_mm) + abs(cmd.dy_mm) + abs(cmd.dz_mm)
    if commanded < 1.0:
        return False
    moved = abs(after["x"] - before["x"]) + abs(after["y"] - before["y"]) + abs(after["z"] - before["z"])
    return moved < min_frac * commanded


@dataclass
class OverlayDecision:
    command: ArmCommand
    phase: str
    pad_xyz: tuple[float, float, float]
    dist_pad_xy: float


# Close to cube width; lab floor stops the pads on the solid.
PAD_Y_TOL_MM = 2.0
# Grasp.jawPad (m) and overlay near_pad (mm). inJaws radial is half + this.
JAW_PAD_MM = 14.0
NEAR_PAD_MM = 18.0


def pad_y_tol_mm(size_mm: float) -> float:
    if float(size_mm) <= 25.0:
        return 1.0
    leftover = 0.5 * (float(size_mm) + 2.0) - 0.5 * float(size_mm)
    return max(float(PAD_Y_TOL_MM), leftover)


def clip_spawn_dy(
    cmd: ArmCommand,
    tcp: dict,
    cube_y: float = 0.0,
    max_abs_y: float = 40.0,
    step: float = 8.0,
) -> ArmCommand:
    """Keep |Y| ≤ max_abs_y on a Y≈0 cube spawn."""
    if abs(float(cube_y)) > 5.0:
        return cmd
    y = float(tcp["y"])
    limited = float(np.clip(y + cmd.dy_mm, -max_abs_y, max_abs_y))
    dy = float(np.clip(limited - y, -step, step))
    return ArmCommand(cmd.dx_mm, dy, cmd.dz_mm, cmd.dgrip_mm, cmd.keep_level)


def hold_pad_dy(
    cmd: ArmCommand,
    tcp: dict,
    cube_y: float = 0.0,
    size_mm: float = 20.0,
    step: float = 8.0,
) -> ArmCommand:
    """Keep TCP Y on the cube so both pads can meet the solid. Does not rewrite dX/dZ/dgrip."""
    y = float(tcp["y"])
    if float(size_mm) <= 25.0:
        dy = float(np.clip(float(cube_y) - y, -step, step))
        return ArmCommand(cmd.dx_mm, dy, cmd.dz_mm, cmd.dgrip_mm, cmd.keep_level)
    tol = pad_y_tol_mm(size_mm)
    lo, hi = float(cube_y) - tol, float(cube_y) + tol
    limited = float(np.clip(y + cmd.dy_mm, lo, hi))
    dy = float(np.clip(limited - y, -step, step))
    return ArmCommand(cmd.dx_mm, dy, cmd.dz_mm, cmd.dgrip_mm, cmd.keep_level)


def hold_pad_dx(
    cmd: ArmCommand,
    tcp: dict,
    pad_x: float,
    cube_x: float | None = None,
    size_mm: float = 20.0,
    step: float = 8.0,
) -> ArmCommand:
    """Keep TCP X on the pad band; 20 mm fingertip pinch stays on cube X."""
    x = float(tcp["x"])
    if float(size_mm) <= 25.0 and cube_x is not None:
        dx = float(np.clip(float(cube_x) - x, -step, step))
        return ArmCommand(dx, cmd.dy_mm, cmd.dz_mm, cmd.dgrip_mm, cmd.keep_level)
    else:
        lo, hi = float(pad_x) - NEAR_PAD_MM, float(pad_x) + NEAR_PAD_MM
        if cube_x is not None:
            face = float(cube_x) - 0.5 * float(size_mm)
            hi = max(hi, face + 3.0)
    limited = float(np.clip(x + cmd.dx_mm, lo, hi))
    dx = float(np.clip(limited - x, -step, step))
    return ArmCommand(dx, cmd.dy_mm, cmd.dz_mm, cmd.dgrip_mm, cmd.keep_level)


def cube_rest_z_mm(size_mm: float) -> float:
    """Table-rest cube center: 20 mm → 9 mm (matches live cube_z)."""
    return 0.5 * float(size_mm) - 1.0


def tip_grasp_z_mm(size_mm: float) -> float:
    """20 mm: fingertips down near the table so pads straddle the solid. 40 mm: keep_level pad height."""
    if float(size_mm) <= 25.0:
        return 8.0
    return float(PAD_Z_MM)


def pad_aim_x_mm(cube_x: float, size_mm: float, depth: float = PAD_DEPTH_MM) -> float:
    """20 mm pinches at the fingertips (TCP XY = cube). Larger cubes use pad depth."""
    if float(size_mm) <= 25.0:
        return float(cube_x)
    return float(cube_x) - float(depth)


def jaw_z_floor_mm(size_mm: float, *, attached: bool = False) -> float:
    """Minimum TCP z. Does not force a lift after attach — U must produce +Z."""
    if float(size_mm) <= 25.0:
        return tip_grasp_z_mm(size_mm)
    if attached:
        return 32.0
    return max(32.0, PAD_Z_MM - 4.0)


def in_jaw_box(
    tcp: dict,
    cube_xy: tuple[float, float] = DEFAULT_CUBE_XY,
    size_mm: float = DEFAULT_CUBE_SIZE_MM,
    cube_z_mm: float | None = None,
    pad_x: float | None = None,
) -> bool:
    """True when cube can sit in Grasp.inJaws at this TCP (keep_level, jaws ±Y)."""
    radial = 0.5 * float(size_mm) + JAW_PAD_MM
    cz = cube_rest_z_mm(size_mm) if cube_z_mm is None else float(cube_z_mm)
    y_ok = abs(float(tcp["y"]) - float(cube_xy[1])) <= radial
    z_ok = abs(float(tcp["z"]) - cz) <= radial
    x_near = abs(float(tcp["x"]) - float(cube_xy[0])) <= radial
    hang_x = float(cube_xy[0]) - PAD_DEPTH_MM if pad_x is None else float(pad_x)
    x_hang = abs(float(tcp["x"]) - hang_x) <= NEAR_PAD_MM
    # 20 mm cube: pad z=48 is outside cz±radial (9±24); hanging jaws still enclose it.
    if x_hang and cz <= float(tcp["z"]) <= float(PAD_Z_MM) + NEAR_PAD_MM:
        z_ok = True
    return bool(y_ok and z_ok and (x_near or x_hang))


def readout_xyz_diverged(raw: ArmCommand, sent: ArmCommand, atol: float = 1e-3) -> bool:
    """True when sent TCP Δ was rewritten vs the ungated readout (Y envelope excluded)."""
    return abs(float(raw.dx_mm) - float(sent.dx_mm)) > atol or abs(float(raw.dz_mm) - float(sent.dz_mm)) > atol


def overlay_attached_lift_policy(rows: list[dict], min_ticks: int = 6) -> bool:
    """True when the raise is the overlay attached template (0, 0, +8, 0)."""
    raise_rows: list[dict] = []
    for row in rows:
        if not bool(row.get("attached")):
            continue
        if float(row.get("cube_z") or 0.0) < 28.0:
            continue
        if float(row.get("dz") or 0.0) <= 1.0:
            if raise_rows and float(row.get("cube_z") or 0.0) >= 100.0:
                break
            continue
        raise_rows.append(row)
    if len(raise_rows) < int(min_ticks):
        return False

    def template(row: dict) -> bool:
        return (
            abs(float(row.get("dx") or 0.0)) < 0.2
            and abs(float(row.get("dy") or 0.0)) < 0.2
            and abs(float(row.get("dz") or 0.0) - 8.0) < 0.2
            and abs(float(row.get("dgrip") or 0.0)) < 0.2
        )

    return all(template(row) for row in raise_rows)


def acting_map_changed(rows: list[dict]) -> bool:
    """True when the live readout swapped maps mid-episode (kernel vs linear)."""
    maps = {str(row.get("map")) for row in rows if str(row.get("map") or "") in {"kernel", "linear"}}
    return len(maps) > 1


def clipped_plus8_hold_policy(rows: list[dict], min_ticks: int = 6) -> bool:
    """True when a map/grip/z switch precedes a saturated +8 raise."""
    if not acting_map_changed(rows):
        return False
    raise_rows = [
        row
        for row in rows
        if bool(row.get("attached")) and float(row.get("cube_z") or 0.0) >= 28.0 and float(row.get("dz") or 0.0) > 1.0
    ]
    if len(raise_rows) < int(min_ticks):
        return False
    return all(abs(float(row.get("dz") or 0.0) - 8.0) < 0.2 for row in raise_rows)


def episode_outcome(
    *,
    overlay: bool,
    readout: bool,
    quintic_ran: bool,
    attached: bool,
    cube_z_mm: float,
    tcp_level: bool,
    hold_s: float,
    min_hold_s: float = 2.0,
    xyz_diverged: bool = False,
    overlay_lift_policy: bool = False,
    map_changed: bool = False,
    plus8_hold_policy: bool = False,
    acting_map: str | None = None,
    mean_dn_hz: float = 0.0,
    black_dn_hz: float = 0.0,
    dn_l2: float | None = None,
    g_trained: bool = False,
    ablation_changed: bool | None = None,
) -> dict:
    from arm.score import ActingMap, parse_acting_map, score_episode

    if quintic_ran:
        controller = "quintic-teacher"
        inferred = ActingMap.overlay
    elif overlay:
        controller = "overlay-servo"
        inferred = ActingMap.overlay
    elif readout:
        controller = "readout"
        inferred = ActingMap.knn
    else:
        controller = "hop"
        inferred = ActingMap.idle
    amap = parse_acting_map(acting_map) if acting_map is not None else inferred
    success = pick_success(
        attached=attached,
        cube_z_mm=cube_z_mm,
        tcp_level=tcp_level,
        hold_s=hold_s,
        min_hold_s=min_hold_s,
    )
    readout_acting = controller == "readout"
    overlay_acting = controller == "overlay-servo"
    readout_ok = (
        readout_acting
        and not overlay_acting
        and not quintic_ran
        and not bool(xyz_diverged)
        and not bool(overlay_lift_policy)
        and not bool(map_changed)
        and not bool(plus8_hold_policy)
    )
    teacher_in = overlay_acting or bool(quintic_ran) or amap in {ActingMap.overlay, ActingMap.knn, ActingMap.teacher_distill}
    scored = score_episode(
        acting_map=amap,
        kinematic_success=success,
        teacher_in_path=teacher_in,
        mean_dn_hz=mean_dn_hz,
        black_dn_hz=black_dn_hz,
        dn_l2=dn_l2,
        g_trained=g_trained,
        ablation_changed=ablation_changed,
    )
    return {
        "controller": controller,
        "readout": readout_acting,
        "overlay": overlay_acting,
        "quintic_ran": bool(quintic_ran),
        "xyz_diverged": bool(xyz_diverged),
        "overlay_lift_policy": bool(overlay_lift_policy),
        "map_changed": bool(map_changed),
        "plus8_hold_policy": bool(plus8_hold_policy),
        "picked": bool(success) and (readout_ok or overlay_acting),
        "readout_picked": bool(success) and readout_ok,
        "fly_picked": bool(scored.fly_picked),
        "lab_picked": bool(scored.lab_picked),
        "da_learned": bool(scored.da_learned),
        "acting_map": amap.value,
        "teacher_ok": bool(success) and bool(quintic_ran),
    }


def optical_close_gate(
    cmd: ArmCommand,
    feat: np.ndarray,
    names: list[str] | tuple[str, ...] | None = None,
    tcp_z: float = 0.0,
    attached: bool = False,
    yaw: float = 0.0,
    tcp: dict | None = None,
    cube_xy: tuple[float, float] | None = None,
    size_mm: float = DEFAULT_CUBE_SIZE_MM,
    cube_z_mm: float | None = None,
    pad_x: float | None = None,
) -> ArmCommand:
    """Zero negative dgrip while TCP is still high or off the jaw box. Does not write XY/Z."""
    del attached, yaw, feat, names
    if cmd.dgrip_mm >= 0.0:
        return cmd
    if float(tcp_z) > 58.0:
        return ArmCommand(cmd.dx_mm, cmd.dy_mm, cmd.dz_mm, 0.0, cmd.keep_level)
    if tcp is not None and cube_xy is not None and not in_jaw_box(
        tcp, cube_xy, size_mm, cube_z_mm=cube_z_mm, pad_x=pad_x
    ):
        return ArmCommand(cmd.dx_mm, cmd.dy_mm, cmd.dz_mm, 0.0, cmd.keep_level)
    return cmd


def spawn_pad_overlay(
    tcp: dict,
    gripper_mm: float,
    attached: bool,
    yaw: float,
    cube_xy: tuple[float, float] = DEFAULT_CUBE_XY,
    cube_size_mm: float = DEFAULT_CUBE_SIZE_MM,
    lift_z: float = LIFT_Z_MM,
    pad_depth: float = PAD_DEPTH_MM,
    pad_z: float = PAD_Z_MM,
    step: float = 8.0,
    pad_lock: tuple[float, float, float] | None = None,
) -> OverlayDecision:
    """Close only when TCP is in the jaw box."""
    out = overlay_command(
        tcp,
        gripper_mm,
        attached,
        yaw,
        cube_xy=cube_xy,
        cube_size_mm=cube_size_mm,
        pad_lock=pad_lock,
        step=step,
        lift_z=lift_z,
        pad_depth=pad_depth,
        pad_z=pad_z,
    )
    cmd = ArmCommand(float(out.action[0]), float(out.action[1]), float(out.action[2]), float(out.action[3]), True)
    return OverlayDecision(cmd, out.phase, out.pad_xyz, out.dist_pad_xy)


def scripted_pad_pick(
    client,
    x_mm: float,
    y_mm: float,
    size_mm: float,
    hold_s: float = 2.0,
    lift_z: float = LIFT_Z_MM,
    pad_depth: float = PAD_DEPTH_MM,
    pad_z: float = PAD_Z_MM,
) -> dict:
    client.call("rebot_set_control_mode", {"mode": "scripted"})
    client.call("rebot_move_to_pose", {"x_mm": x_mm, "y_mm": y_mm, "z_mm": pad_z, "keep_level": True})
    state = client.wait_stopped(timeout=25)
    yaw = math.radians(float((state.get("tcp_rpy_deg") or {}).get("yaw") or 0.0))
    gx, gy = pad_xy(x_mm, y_mm, yaw, depth=pad_depth)
    client.call("rebot_move_to_pose", {"x_mm": gx, "y_mm": gy, "z_mm": pad_z, "keep_level": True})
    state = client.wait_stopped(timeout=20)
    opening = float(size_mm) + 2.0
    client.call("rebot_set_gripper", {"opening_mm": opening})
    state = client.wait_stopped(timeout=15)
    cube = state["objects"]["cube"]
    if state["gripper_mm"] < size_mm - 3:
        raise RuntimeError(f"gripper closed through the cube: {state['gripper_mm']} mm vs {size_mm} mm")
    if not cube.get("attached"):
        return state
    tcp = state["tcp_mm"]
    client.call("rebot_move_to_pose", {"x_mm": tcp["x"], "y_mm": tcp["y"], "z_mm": lift_z, "keep_level": True})
    state = client.wait_stopped(timeout=20)
    deadline = time.monotonic() + max(hold_s, 0.0)
    while time.monotonic() < deadline:
        state = client.state()
        cube = state["objects"]["cube"]
        if not qualifying(
            attached=bool(cube.get("attached")),
            cube_z_mm=float(cube["center_mm"]["z"]),
            tcp_level=bool(state.get("tcp_level")),
        ):
            return state
        time.sleep(0.1)
    return state
