"""Sealed episode flags. fly_picked / da_learned / lab_picked are set only here."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActingMap(str, Enum):
    overlay = "overlay"
    knn = "knn"
    teacher_distill = "teacher_distill"
    dn_bus = "dn_bus"
    leg_mn = "leg_mn"
    idle = "idle"


LAB_MAPS = frozenset(
    {
        ActingMap.overlay,
        ActingMap.knn,
        ActingMap.teacher_distill,
    }
)
TEACHER_MAPS = LAB_MAPS
DN_HZ_EPS = 0.05
DN_VAR_EPS = 1.0


def parse_acting_map(value: object) -> ActingMap:
    if isinstance(value, ActingMap):
        return value
    raw = str(getattr(value, "value", value) or "idle")
    aliases = {
        "linear": ActingMap.knn,
        "kernel": ActingMap.knn,
        "readout": ActingMap.knn,
        "quintic": ActingMap.overlay,
        "quintic-teacher": ActingMap.overlay,
        "overlay-servo": ActingMap.overlay,
        "hop": ActingMap.idle,
        "leg_mn": ActingMap.leg_mn,
        "dn_bus": ActingMap.dn_bus,
        "teacher_distill": ActingMap.teacher_distill,
        "overlay": ActingMap.overlay,
        "knn": ActingMap.knn,
        "idle": ActingMap.idle,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return ActingMap(raw)
    except ValueError:
        return ActingMap.idle


@dataclass(frozen=True)
class EpisodeScore:
    acting_map: ActingMap
    kinematic_success: bool
    fly_picked: bool
    lab_picked: bool
    da_learned: bool
    mean_dn_hz: float
    black_dn_hz: float
    dn_l2: float
    g_trained: bool
    teacher_in_path: bool


def rates_vary(mean_dn_hz: float, black_dn_hz: float, dn_l2: float | None = None) -> bool:
    if float(mean_dn_hz) <= DN_HZ_EPS:
        return False
    if dn_l2 is not None:
        return float(dn_l2) >= DN_VAR_EPS
    return abs(float(mean_dn_hz) - float(black_dn_hz)) >= DN_VAR_EPS


def _row_command(row: dict) -> dict:
    cmd = row.get("command")
    if isinstance(cmd, dict):
        return cmd
    return {
        "dx_mm": float(row.get("dx_mm") or row.get("dx") or 0.0),
        "dy_mm": float(row.get("dy_mm") or row.get("dy") or 0.0),
        "dz_mm": float(row.get("dz_mm") or row.get("dz") or 0.0),
        "dgrip_mm": float(row.get("dgrip_mm") or row.get("dgrip") or 0.0),
    }


def _row_cube_z(row: dict) -> float:
    return float(row.get("cube_z_mm") or row.get("cube_z") or 0.0)


def maps_are_dn_bus(ticks: list[dict]) -> bool:
    if not ticks:
        return False
    return all(parse_acting_map(r.get("acting_map")) is ActingMap.dn_bus for r in ticks)


def open_gripper_tick0_attach(ticks: list[dict]) -> bool:
    """Tick 0 already welded with jaws ~open is pad graze, not a bus pinch."""
    if not ticks:
        return False
    r0 = ticks[0]
    return bool(r0.get("attached")) and float(r0.get("gripper_mm") or 0.0) >= 70.0


def overlay_plus8_lift(ticks: list[dict], min_ticks: int = 6) -> bool:
    n = 0
    for row in ticks:
        if not bool(row.get("attached")):
            continue
        cmd = _row_command(row)
        if (
            abs(float(cmd.get("dz_mm") or 0.0) - 8.0) < 0.25
            and abs(float(cmd.get("dx_mm") or 0.0)) < 0.25
            and abs(float(cmd.get("dy_mm") or 0.0)) < 0.25
            and abs(float(cmd.get("dgrip_mm") or 0.0)) < 0.25
        ):
            n += 1
    return n >= int(min_ticks)


def bus_commanded_plus_z(ticks: list[dict]) -> bool:
    for row in ticks:
        if not bool(row.get("attached")):
            continue
        if float(_row_command(row).get("dz_mm") or 0.0) > 0.5:
            return True
    return False


def lift_without_plus_z(ticks: list[dict]) -> bool:
    """Cube rose to pick height while logged U.dz never went positive."""
    if not ticks:
        return False
    z0 = _row_cube_z(ticks[0])
    z1 = max(_row_cube_z(r) for r in ticks)
    if z1 < 100.0 or (z1 - z0) < 40.0:
        return False
    return not bus_commanded_plus_z(ticks)


def tcp_rose_while_dz_negative(ticks: list[dict]) -> bool:
    """TCP climbed after attach while logged dz stayed negative (`|dz|` rewrite)."""
    prev = None
    n = 0
    for row in ticks:
        tcp = row.get("tcp") or {}
        z = float(tcp.get("z") or row.get("tcp_z") or 0.0)
        dz = float(_row_command(row).get("dz_mm") or 0.0)
        if bool(row.get("attached")) and prev is not None and z > prev + 1.5 and dz < -0.1:
            n += 1
        prev = z
    return n >= 4


def trace_blocks_fly(ticks: list[dict] | None) -> bool:
    if not ticks:
        return False
    if not maps_are_dn_bus(ticks):
        return True
    if open_gripper_tick0_attach(ticks):
        return True
    if overlay_plus8_lift(ticks):
        return True
    if lift_without_plus_z(ticks):
        return True
    if tcp_rose_while_dz_negative(ticks):
        return True
    return False


def fly_picked(
    *,
    acting_map: ActingMap | str,
    kinematic_success: bool,
    teacher_in_path: bool,
    mean_dn_hz: float,
    black_dn_hz: float,
    g_trained: bool,
    dn_l2: float | None = None,
    abort_only: bool = False,
    ticks: list[dict] | None = None,
) -> bool:
    """Live fly rates, trained g, dn_bus, kinematic gates. Never overlay/k-NN."""
    amap = parse_acting_map(acting_map)
    if amap is not ActingMap.dn_bus:
        return False
    if teacher_in_path or amap in TEACHER_MAPS:
        return False
    if not kinematic_success:
        return False
    if not g_trained:
        return False
    if abort_only:
        return False
    if not rates_vary(mean_dn_hz, black_dn_hz, dn_l2):
        return False
    if ticks is not None and trace_blocks_fly(ticks):
        return False
    if ticks is not None and kinematic_success and not bus_commanded_plus_z(ticks):
        return False
    return True


def lab_picked(*, acting_map: ActingMap | str, kinematic_success: bool) -> bool:
    if not kinematic_success:
        return False
    return parse_acting_map(acting_map) in LAB_MAPS


def da_learned(*, ablation_changed: bool | None) -> bool:
    """True only when a KC→MBON freeze/shuffle/erase changes pick or abort rate."""
    return bool(ablation_changed)


def score_episode(
    *,
    acting_map: ActingMap | str,
    kinematic_success: bool,
    teacher_in_path: bool = False,
    mean_dn_hz: float = 0.0,
    black_dn_hz: float = 0.0,
    dn_l2: float | None = None,
    g_trained: bool = False,
    ablation_changed: bool | None = None,
    abort_only: bool = False,
    ticks: list[dict] | None = None,
) -> EpisodeScore:
    amap = parse_acting_map(acting_map)
    teacher = bool(teacher_in_path) or amap in TEACHER_MAPS or trace_blocks_fly(ticks)
    fly = fly_picked(
        acting_map=amap,
        kinematic_success=kinematic_success,
        teacher_in_path=teacher,
        mean_dn_hz=mean_dn_hz,
        black_dn_hz=black_dn_hz,
        g_trained=g_trained,
        dn_l2=dn_l2,
        abort_only=abort_only,
        ticks=ticks,
    )
    lab = lab_picked(acting_map=amap, kinematic_success=kinematic_success)
    return EpisodeScore(
        acting_map=amap,
        kinematic_success=bool(kinematic_success),
        fly_picked=bool(fly),
        lab_picked=bool(lab),
        da_learned=da_learned(ablation_changed=ablation_changed),
        mean_dn_hz=float(mean_dn_hz),
        black_dn_hz=float(black_dn_hz),
        dn_l2=float(dn_l2) if dn_l2 is not None else abs(float(mean_dn_hz) - float(black_dn_hz)),
        g_trained=bool(g_trained),
        teacher_in_path=teacher,
    )
