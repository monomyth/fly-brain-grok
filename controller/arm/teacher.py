"""Overlay/quintic teacher. If it commands, acting_map is overlay — never fly_picked."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rebot_adapter.pick import OverlayDecision, spawn_pad_overlay
from runtime.decoder import ArmCommand
from train.expert import overlay_command

from .score import ActingMap
from .unpack import command_dict


@dataclass
class TeacherCommand:
    command: ArmCommand
    acting_map: ActingMap
    phase: str
    action: np.ndarray


def teacher_target(
    tcp: dict,
    gripper_mm: float,
    attached: bool,
    yaw: float,
    cube_xy: tuple[float, float] = (280.0, 0.0),
    cube_size_mm: float = 40.0,
    pad_lock: tuple[float, float, float] | None = None,
) -> TeacherCommand:
    decision: OverlayDecision = spawn_pad_overlay(
        tcp,
        gripper_mm,
        attached,
        yaw,
        cube_xy=cube_xy,
        cube_size_mm=cube_size_mm,
        pad_lock=pad_lock,
    )
    action = np.array(
        [decision.command.dx_mm, decision.command.dy_mm, decision.command.dz_mm, decision.command.dgrip_mm],
        dtype=np.float64,
    )
    return TeacherCommand(
        command=decision.command,
        acting_map=ActingMap.overlay,
        phase=decision.phase,
        action=action,
    )


def teacher_from_state(
    state: dict,
    cube_xy: tuple[float, float] | None = None,
    pad_lock: tuple[float, float, float] | None = None,
) -> TeacherCommand:
    tcp = state["tcp_mm"]
    cube = state.get("objects", {}).get("cube") or state.get("cube") or {}
    attached = bool(cube.get("attached"))
    yaw = float(np.radians((state.get("tcp_rpy_deg") or {}).get("yaw") or 0.0))
    center = cube.get("center_mm") or {}
    xy = cube_xy or (float(center.get("x") or 280.0), float(center.get("y") or 0.0))
    size = float(cube.get("size_mm") or 20.0)
    return teacher_target(
        tcp,
        float(state.get("gripper_mm") or 90.0),
        attached,
        yaw,
        cube_xy=xy,
        cube_size_mm=size,
        pad_lock=pad_lock,
    )


def overlay_action_vec(*args, **kwargs) -> np.ndarray:
    out = overlay_command(*args, **kwargs)
    return np.asarray(out.action, dtype=np.float64)


def as_log(cmd: TeacherCommand) -> dict:
    row = command_dict(cmd.command)
    row["acting_map"] = ActingMap.overlay.value
    row["phase"] = cmd.phase
    return row
