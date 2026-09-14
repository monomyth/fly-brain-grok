"""Offline B601 joint plant. Default lab remains MCP TCP; this profile never talks to motors."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera_contract import HW_CAMERAS, HW_CAPTURE_WH, CameraContractError, TimedFrame, encoder_pair, stamp_frame

CAN_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS = CAN_JOINTS[:-1]
GRIPPER_JOINT = "gripper"

# URDF ReBot_Arm_DM revolute joints, same order as CAN_JOINTS[0:6].
_ARM_XYZ = (
    (-8.416e-05, 0.0, 0.08465),
    (0.020084, 0.031625, 0.05555),
    (-0.264, 0.0, 0.0),
    (0.2426, -0.054, -0.001625),
    (0.078308, -0.0375, -0.03),
    (0.028008, 0.0, 0.04),
)
_ARM_RPY = (
    (0.0, 0.0, 0.0),
    (-1.5708, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (-1.5708, 0.0, 0.0),
    (0.0, 1.5708, 0.0),
)
_ARM_AXIS = (
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
)
_ARM_LIMITS_RAD = (
    (-2.8, 2.8),
    (-3.14, 0.0),
    (-3.14, 0.0),
    (-1.87, 1.57),
    (-1.57, 1.57),
    (-3.14, 3.14),
)
_END_XYZ = (0.0, 0.0, 0.15539)
_END_RPY = (0.0, -1.5708, 3.1415)

READY_ARM_DEG = (0.0, -95.0, -95.0, 10.0, 0.0, 0.0)
FOLDED_ARM_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
LEVEL_COS = math.cos(math.radians(5.0))
HOME_RATE_DEG_S = 20.0
TABLE_Z_MM = 0.0
TABLE_TCP_CLEAR_MM = 8.0
GRAVITY_MM_S2 = 9.81 * 1000.0
SLIP_TILT_DEG = 15.0

COMMAND_HZ = 2.0
CAMERA_HZ = 15.0
COMMAND_PERIOD_S = 1.0 / COMMAND_HZ
CAMERA_PERIOD_S = 1.0 / CAMERA_HZ

GRIPPER_CURVE_VERSION = "fake-v0"
GRIPPER_CLOSED_DEG = 0.0
GRIPPER_OPEN_DEG = -270.0
LAB_CLOSED_MM = 0.0
LAB_OPEN_MM = 90.0
FORCE_POS_RATIO = 0.07

FAKE_V0_CURVE = {
    "version": GRIPPER_CURVE_VERSION,
    "measured": False,
    "closed_deg": GRIPPER_CLOSED_DEG,
    "open_deg": GRIPPER_OPEN_DEG,
    "lab_closed_mm": LAB_CLOSED_MM,
    "lab_open_mm": LAB_OPEN_MM,
    "force_pos_ratio": FORCE_POS_RATIO,
    "open_table": [[GRIPPER_CLOSED_DEG, LAB_CLOSED_MM], [GRIPPER_OPEN_DEG, LAB_OPEN_MM]],
    "close_table": [[GRIPPER_CLOSED_DEG, LAB_CLOSED_MM], [GRIPPER_OPEN_DEG, LAB_OPEN_MM]],
}


class FailClosed(RuntimeError):
    """Refuse the command rather than guess units, joints, or cameras."""


class UnitMixError(FailClosed):
    """Millimetres and degrees mixed on the gripper channel."""


class MissingJointError(FailClosed):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"missing joint feedback: {name}")


class IncompleteActionError(FailClosed):
    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"incomplete joint action, missing {self.missing}")


class NotArmedError(FailClosed):
    """Torque is off; connect() must not enable it."""


def _rot_rpy(rpy: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _axis_angle(axis: tuple[float, float, float], ang: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _origin(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rot_rpy(rpy)
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def _motion(axis: tuple[float, float, float], q_rad: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _axis_angle(axis, q_rad)
    return t


def _rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    sy = -float(R[2, 0])
    cy = math.sqrt(float(R[0, 0]) ** 2 + float(R[1, 0]) ** 2)
    if cy > 1e-8:
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        pitch = math.atan2(sy, cy)
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    else:
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
        pitch = math.atan2(sy, cy)
        roll = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def arm_limits_deg() -> tuple[tuple[float, float], ...]:
    return tuple((math.degrees(lo), math.degrees(hi)) for lo, hi in _ARM_LIMITS_RAD)


def clamp_arm_deg(q_deg: np.ndarray) -> np.ndarray:
    lim = arm_limits_deg()
    out = np.asarray(q_deg, dtype=np.float64).reshape(-1)[:6].copy()
    for i, (lo, hi) in enumerate(lim):
        out[i] = float(np.clip(out[i], lo, hi))
    return out


@dataclass(frozen=True)
class TcpPose:
    x_mm: float
    y_mm: float
    z_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    tool_x: tuple[float, float, float]
    tool_y: tuple[float, float, float]
    tool_z: tuple[float, float, float]
    level: bool

    def as_mm(self) -> dict:
        return {"x": self.x_mm, "y": self.y_mm, "z": self.z_mm}

    def rpy_deg(self) -> dict:
        return {"roll": self.roll_deg, "pitch": self.pitch_deg, "yaw": self.yaw_deg}


def fk_matrix(arm_deg: np.ndarray) -> np.ndarray:
    q = clamp_arm_deg(arm_deg)
    t = np.eye(4)
    for i in range(6):
        t = t @ _origin(_ARM_XYZ[i], _ARM_RPY[i]) @ _motion(_ARM_AXIS[i], math.radians(float(q[i])))
    return t @ _origin(_END_XYZ, _END_RPY)


def fk_pose(arm_deg: np.ndarray) -> TcpPose:
    t = fk_matrix(arm_deg)
    p = t[:3, 3] * 1000.0
    tool_x = tuple(float(v) for v in t[:3, 0])
    tool_y = tuple(float(v) for v in t[:3, 1])
    tool_z = tuple(float(v) for v in t[:3, 2])
    rpy = _rpy_deg(t[:3, :3])
    level = float(tool_z[2]) >= LEVEL_COS
    return TcpPose(
        x_mm=float(p[0]),
        y_mm=float(p[1]),
        z_mm=float(p[2]),
        roll_deg=rpy[0],
        pitch_deg=rpy[1],
        yaw_deg=rpy[2],
        tool_x=tool_x,
        tool_y=tool_y,
        tool_z=tool_z,
        level=level,
    )


def _fk_pos_m(arm_deg: np.ndarray) -> np.ndarray:
    return fk_matrix(arm_deg)[:3, 3].copy()


def _orientation_residual(arm_deg: np.ndarray, target_m: np.ndarray, fingers_down: bool) -> np.ndarray:
    pose = fk_pose(arm_deg)
    pos = np.array([pose.x_mm, pose.y_mm, pose.z_mm]) / 1000.0 - target_m
    if fingers_down:
        x_err = np.array(pose.tool_x) - np.array([0.0, 0.0, -1.0])
        ori = np.array([x_err[0], x_err[2], float(np.dot(pose.tool_y, (1.0, 0.0, 0.0)))])
    else:
        ori = np.array(pose.tool_z) - np.array([0.0, 0.0, 1.0])
    return np.concatenate([pos, ori])


def ik_arm(
    target_mm: dict | tuple[float, float, float],
    initial_deg: np.ndarray,
    *,
    keep_level: bool = False,
    fingers_down: bool = False,
    iterations: int = 300,
) -> tuple[np.ndarray, float]:
    """Damped least-squares IK. keep_level/fingers_down are explicit; default is free orientation."""
    if isinstance(target_mm, dict):
        target_m = np.array([float(target_mm["x"]), float(target_mm["y"]), float(target_mm["z"])]) / 1000.0
    else:
        target_m = np.asarray(target_mm, dtype=np.float64).reshape(3) / 1000.0
    if not np.all(np.isfinite(target_m)):
        q = clamp_arm_deg(initial_deg)
        return q, float("inf")
    q = clamp_arm_deg(initial_deg)
    lim = arm_limits_deg()
    constrained = bool(keep_level) or bool(fingers_down)
    n_it = iterations if not constrained else max(int(iterations), 400)
    if not constrained:
        for _ in range(n_it):
            p = _fk_pos_m(q)
            err = target_m - p
            if float(np.linalg.norm(err)) < 0.001:
                break
            cols = []
            for i in range(6):
                h = 0.01 if q[i] + 0.01 <= lim[i][1] else -0.01
                trial = q.copy()
                trial[i] += h
                cols.append((_fk_pos_m(trial) - p) / math.radians(h))
            J = np.column_stack(cols)
            a = J @ J.T + 0.0004 * np.eye(3)
            step = np.linalg.solve(a, err)
            dq = np.array([float(np.clip(np.dot(cols[i], step), -0.12, 0.12)) for i in range(6)])
            q = clamp_arm_deg(q + np.degrees(dq))
        err_m = float(np.linalg.norm(_fk_pos_m(q) - target_m))
        return q, err_m * 1000.0
    ori_w = 0.05
    for _ in range(n_it):
        r = _orientation_residual(q, target_m, fingers_down)
        if float(np.linalg.norm(r[:3])) < 0.001 and float(np.linalg.norm(r[3:])) < 0.03:
            break
        J = np.zeros((6, 6))
        for i in range(6):
            h = 0.01 if q[i] + 0.01 <= lim[i][1] else -0.01
            trial = q.copy()
            trial[i] += h
            J[:, i] = (_orientation_residual(trial, target_m, fingers_down) - r) / math.radians(h)
        w = np.array([1.0, 1.0, 1.0, ori_w, ori_w, ori_w])
        A = np.diag(np.full(6, 0.0004))
        b = np.zeros(6)
        for k in range(6):
            wk = w[k]
            b += J[k] * r[k] * wk
            A += np.outer(J[k], J[k]) * wk
        try:
            step = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        q = clamp_arm_deg(q - np.degrees(np.clip(step, -0.12, 0.12)))
    r = _orientation_residual(q, target_m, fingers_down)
    return q, float(np.linalg.norm(r[:3]) * 1000.0)


def _interp_table(table: list[list[float]], x: float) -> float:
    xs = [float(p[0]) for p in table]
    ys = [float(p[1]) for p in table]
    if not xs:
        raise UnitMixError("empty gripper table")
    # Gripper opens toward −270°; tables are listed closed→open.
    order = np.argsort(xs)
    xs_s = [xs[i] for i in order]
    ys_s = [ys[i] for i in order]
    if x <= xs_s[0]:
        return ys_s[0]
    if x >= xs_s[-1]:
        return ys_s[-1]
    for i in range(len(xs_s) - 1):
        if xs_s[i] <= x <= xs_s[i + 1]:
            t = 0.0 if xs_s[i + 1] == xs_s[i] else (x - xs_s[i]) / (xs_s[i + 1] - xs_s[i])
            return ys_s[i] + t * (ys_s[i + 1] - ys_s[i])
    return ys_s[-1]


def _invert_table(table: list[list[float]], y: float) -> float:
    flipped = [[float(p[1]), float(p[0])] for p in table]
    return _interp_table(flipped, y)


def calibration_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "calibration"


def load_gripper_curve(version: str = GRIPPER_CURVE_VERSION) -> dict:
    """fake-v0 is in-module. Any other identity must exist on disk; do not substitute."""
    if version == GRIPPER_CURVE_VERSION:
        path = calibration_dir() / f"b601-{version}.json"
        if path.is_file():
            blob = json.loads(path.read_text())
            if str(blob.get("version") or "") != version:
                raise FailClosed(f"calibration version mismatch in {path}")
            return blob
        return dict(FAKE_V0_CURVE)
    path = calibration_dir() / f"b601-{version}.json"
    if not path.is_file():
        raise FailClosed(f"measured calibration {version} missing: {path}")
    blob = json.loads(path.read_text())
    if str(blob.get("version") or "") != version:
        raise FailClosed(f"calibration version mismatch in {path}")
    return blob


def gripper_deg_to_mm(deg: float, curve: dict | None = None) -> float:
    curve = curve or FAKE_V0_CURVE
    table = curve.get("open_table") or FAKE_V0_CURVE["open_table"]
    return float(np.clip(_interp_table(table, float(deg)), LAB_CLOSED_MM, LAB_OPEN_MM))


def gripper_mm_to_deg(mm: float, curve: dict | None = None) -> float:
    curve = curve or FAKE_V0_CURVE
    table = curve.get("open_table") or FAKE_V0_CURVE["open_table"]
    return float(np.clip(_invert_table(table, float(mm)), GRIPPER_OPEN_DEG, GRIPPER_CLOSED_DEG))


def _looks_like_mm_gripper(value: float) -> bool:
    # Legal hardware aperture is 0° shut … −270° open. A positive value is a mm mix-up.
    return float(value) > 0.5


def _is_mm_grip_key(key: object) -> bool:
    s = str(key)
    if s in {"opening_mm", "gripper_mm", "dgrip_mm"}:
        return True
    return "grip" in s.lower() and s.endswith("_mm")


def parse_action(action: dict) -> dict[str, float]:
    if not isinstance(action, dict):
        raise IncompleteActionError(list(CAN_JOINTS))
    has_mm = any(_is_mm_grip_key(k) for k in action)
    parsed: dict[str, float] = {}
    for name in CAN_JOINTS:
        if name in action:
            parsed[name] = float(action[name])
        elif f"{name}.pos" in action:
            parsed[name] = float(action[f"{name}.pos"])
        elif name == GRIPPER_JOINT and "gripper_deg" in action:
            parsed[name] = float(action["gripper_deg"])
    if has_mm and GRIPPER_JOINT in parsed:
        raise UnitMixError("gripper_mm and gripper degrees both present")
    if has_mm and GRIPPER_JOINT not in parsed:
        raise UnitMixError("hardware action expects gripper degrees, not millimetres")
    missing = [name for name in CAN_JOINTS if name not in parsed]
    if missing:
        raise IncompleteActionError(missing)
    if _looks_like_mm_gripper(parsed[GRIPPER_JOINT]):
        raise UnitMixError("positive gripper command is millimetres, not B601 degrees")
    if parsed[GRIPPER_JOINT] < GRIPPER_OPEN_DEG - 1.0:
        raise UnitMixError("gripper command outside hardware aperture")
    for name in ARM_JOINTS:
        if not math.isfinite(parsed[name]):
            raise FailClosed(f"non-finite joint {name}")
    parsed[GRIPPER_JOINT] = float(np.clip(parsed[GRIPPER_JOINT], GRIPPER_OPEN_DEG, GRIPPER_CLOSED_DEG))
    arm = clamp_arm_deg(np.array([parsed[n] for n in ARM_JOINTS]))
    for name, val in zip(ARM_JOINTS, arm):
        parsed[name] = float(val)
    return parsed


def action_from_joints(joints: dict[str, float]) -> dict[str, float]:
    return {f"{name}.pos": float(joints[name]) for name in CAN_JOINTS}


def iter_schedule(duration_s: float, *, command_hz: float = COMMAND_HZ, camera_hz: float = CAMERA_HZ):
    """Camera and command events on independent periods. Not one LIF tick per frame."""
    if command_hz <= 0 or camera_hz <= 0:
        raise FailClosed("cadence Hz must be positive")
    n_cam = int(math.floor(float(duration_s) * float(camera_hz) + 1e-12))
    n_cmd = int(math.floor(float(duration_s) * float(command_hz) + 1e-12))
    events = [("camera", i / float(camera_hz)) for i in range(n_cam)]
    events += [("command", i / float(command_hz)) for i in range(n_cmd)]
    events.sort(key=lambda item: (item[1], 0 if item[0] == "camera" else 1))
    yield from events


@dataclass
class PlantClock:
    command_hz: float = COMMAND_HZ
    camera_hz: float = CAMERA_HZ
    t: float = 0.0
    last_command_t: float = -1e9
    last_camera_t: float = -1e9

    @property
    def command_period_s(self) -> float:
        return 1.0 / float(self.command_hz)

    @property
    def camera_period_s(self) -> float:
        return 1.0 / float(self.camera_hz)

    def advance(self, dt: float) -> None:
        self.t += float(dt)

    def camera_due(self) -> bool:
        return (self.t - self.last_camera_t) + 1e-12 >= self.camera_period_s

    def command_due(self) -> bool:
        return (self.t - self.last_command_t) + 1e-12 >= self.command_period_s

    def mark_camera(self) -> None:
        self.last_camera_t = self.t

    def mark_command(self) -> None:
        self.last_command_t = self.t


@dataclass
class CubeState:
    x_mm: float = 280.0
    y_mm: float = 0.0
    z_mm: float = 9.0
    size_mm: float = 20.0
    vz_mm_s: float = 0.0
    present: bool = True

    def rest_z_mm(self) -> float:
        return 0.5 * float(self.size_mm) - 1.0

    def center_mm(self) -> dict:
        return {"x": self.x_mm, "y": self.y_mm, "z": self.z_mm}


def _blank_hw_rgb() -> np.ndarray:
    w, h = HW_CAPTURE_WH
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:, :] = (0.12, 0.12, 0.12)
    return img


def _paint_cube(img: np.ndarray, cube: CubeState, *, wrist: bool) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    sx = float(np.clip((cube.x_mm - 180.0) / 200.0, 0.05, 0.95))
    sy = float(np.clip(0.5 + cube.y_mm / 160.0, 0.08, 0.92))
    if wrist:
        sy = float(np.clip(0.72 - cube.z_mm / 400.0, 0.2, 0.9))
    cx = int(sx * (w - 1))
    cy = int(sy * (h - 1))
    rw = max(8, int(w * 0.04))
    rh = max(8, int(h * 0.06))
    y0, y1 = max(0, cy - rh), min(h, cy + rh)
    x0, x1 = max(0, cx - rw), min(w, cx + rw)
    out[y0:y1, x0:x1, 0] = 0.85
    out[y0:y1, x0:x1, 1] = 0.35
    out[y0:y1, x0:x1, 2] = 0.08
    return out


class B601Plant:
    """Seven-channel joint plant. Joints move independently; cube is not welded."""

    profile = "hardware"

    def __init__(
        self,
        *,
        joints: dict[str, float] | None = None,
        cube: CubeState | None = None,
        camera_age_s: float = 0.0,
        camera_skew_s: float = 0.0,
        curve: dict | None = None,
    ):
        self.curve = curve or load_gripper_curve(GRIPPER_CURVE_VERSION)
        self.joints = {name: 0.0 for name in CAN_JOINTS}
        if joints:
            for name in CAN_JOINTS:
                if name in joints:
                    self.joints[name] = float(joints[name])
        self.cube = cube or CubeState()
        self.connected = False
        self.torque_enabled = False
        self.armed = False
        self.homing = False
        self.table_collision = False
        self._dropped: set[str] = set()
        self.clock = PlantClock()
        self.camera_age_s = float(camera_age_s)
        self.camera_skew_s = float(camera_skew_s)
        self.last_command_time: float | None = None
        self.last_frames: dict[str, TimedFrame] = {}

    @classmethod
    def ready(cls, **kwargs) -> "B601Plant":
        joints = {name: float(v) for name, v in zip(ARM_JOINTS, READY_ARM_DEG)}
        joints[GRIPPER_JOINT] = GRIPPER_OPEN_DEG
        return cls(joints=joints, **kwargs)

    def connect(self) -> None:
        """Open the bus. Does not enable torque and does not move."""
        self.connected = True

    def arm(self) -> None:
        if not self.connected:
            raise NotArmedError("arm() before connect()")
        self.torque_enabled = True
        self.armed = True

    def hold(self) -> None:
        if not self.armed:
            raise NotArmedError("hold() before arm()")
        self.homing = False

    def drop_joint(self, name: str) -> None:
        if name not in CAN_JOINTS:
            raise FailClosed(f"unknown joint {name}")
        self._dropped.add(name)

    def restore_joint(self, name: str) -> None:
        self._dropped.discard(name)

    def get_joint_positions(self, *, strict: bool = True) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in CAN_JOINTS:
            if name in self._dropped:
                if strict:
                    raise MissingJointError(name)
                out[name] = 0.0
                continue
            out[name] = float(self.joints[name])
        return out

    def present_pos(self, *, strict: bool = False) -> dict[str, float]:
        """Follower missing-state path. strict=False paints 0°; that is not a folded pose."""
        return self.get_joint_positions(strict=strict)

    def tcp(self) -> TcpPose:
        return fk_pose(np.array([self.joints[n] for n in ARM_JOINTS]))

    def gripper_mm(self) -> float:
        return gripper_deg_to_mm(self.joints[GRIPPER_JOINT], self.curve)

    def tilt_deg(self) -> float:
        pose = self.tcp()
        c = max(-1.0, min(1.0, float(pose.tool_z[2])))
        return math.degrees(math.acos(c))

    def estimate_grasp(self) -> dict:
        pose = self.tcp()
        opening = self.gripper_mm()
        cube = self.cube
        near_xy = math.hypot(pose.x_mm - cube.x_mm, pose.y_mm - cube.y_mm) <= (0.5 * cube.size_mm + 22.0)
        near_z = abs(pose.z_mm - cube.z_mm) <= (0.5 * cube.size_mm + 30.0)
        opening_ok = (cube.size_mm - 3.0) <= opening <= (cube.size_mm + 4.0)
        untilted = self.tilt_deg() <= SLIP_TILT_DEG
        holding = bool(cube.present and near_xy and near_z and opening_ok and untilted)
        return {
            "grasp_estimated": holding,
            "grasp_label": "estimated",
            "near_xy": near_xy,
            "near_z": near_z,
            "opening_ok": opening_ok,
            "untilted": untilted,
        }

    def _apply_joints(self, parsed: dict[str, float], *, homing: bool) -> None:
        arm = np.array([parsed[n] for n in ARM_JOINTS])
        pose = fk_pose(arm)
        self.table_collision = False
        if (not homing) and pose.z_mm < TABLE_Z_MM + TABLE_TCP_CLEAR_MM:
            self.table_collision = True
            return
        for name in CAN_JOINTS:
            self.joints[name] = float(parsed[name])

    def send_action(self, action: dict, *, fill_missing: bool = False) -> dict[str, float]:
        if not self.armed or not self.torque_enabled:
            raise NotArmedError("send_action while torque off")
        try:
            parsed = parse_action(action)
        except IncompleteActionError as exc:
            if not fill_missing:
                raise
            # Driver substitute: missing keys become 0°. Adapter must not use this path.
            filled = dict(action)
            for name in exc.missing:
                filled[f"{name}.pos"] = 0.0
            parsed = parse_action(filled)
        self._apply_joints(parsed, homing=self.homing)
        self.last_command_time = self.clock.t
        self.clock.mark_command()
        return dict(self.joints)

    def _home_step(self, dt: float) -> None:
        step = HOME_RATE_DEG_S * max(float(dt), 0.0)
        nxt = dict(self.joints)
        done = True
        for name in CAN_JOINTS:
            q = float(nxt[name])
            if abs(q) <= step:
                nxt[name] = 0.0
            else:
                nxt[name] = q - math.copysign(step, q)
                done = False
        self._apply_joints(nxt, homing=True)
        if done:
            self.homing = False

    def home(self, dt: float | None = None) -> None:
        if not self.armed:
            raise NotArmedError("home() before arm()")
        self.homing = True
        if dt is not None:
            self._home_step(dt)

    def reset(self) -> None:
        """Drive joints toward calibrated zero. Does not teleport the cube."""
        if self.armed:
            self.homing = True

    def place_cube(self, x_mm: float, y_mm: float, size_mm: float = 20.0) -> None:
        self.cube = CubeState(x_mm=float(x_mm), y_mm=float(y_mm), z_mm=0.5 * float(size_mm) - 1.0, size_mm=float(size_mm))

    def step(self, dt: float) -> None:
        dt = float(dt)
        self.clock.advance(dt)
        if self.homing and self.armed:
            self._home_step(dt)
        grasp = self.estimate_grasp()
        cube = self.cube
        if not cube.present:
            return
        if grasp["grasp_estimated"]:
            pose = self.tcp()
            hang = 0.5 * cube.size_mm + 8.0
            cube.x_mm = pose.x_mm
            cube.y_mm = pose.y_mm
            cube.z_mm = max(cube.rest_z_mm(), pose.z_mm - hang)
            cube.vz_mm_s = 0.0
            return
        cube.vz_mm_s -= GRAVITY_MM_S2 * dt
        cube.z_mm += cube.vz_mm_s * dt
        rest = cube.rest_z_mm()
        if cube.z_mm <= rest:
            cube.z_mm = rest
            cube.vz_mm_s = 0.0

    def capture(self, name: str) -> TimedFrame:
        if name not in HW_CAMERAS:
            raise CameraContractError(f"hardware cameras are {HW_CAMERAS}, not {name!r}")
        rgb = _paint_cube(_blank_hw_rgb(), self.cube, wrist=(name == "wrist"))
        capture_t = self.clock.t
        if name == "overview":
            capture_t = self.clock.t - abs(self.camera_skew_s)
        receive_t = capture_t + max(0.0, self.camera_age_s)
        frame = stamp_frame(
            name,
            rgb,
            capture_time=capture_t,
            receive_time=receive_t,
            command_time=self.last_command_time,
        )
        self.last_frames[name] = frame
        return frame

    def capture_all(self) -> dict[str, TimedFrame]:
        frames = {name: self.capture(name) for name in HW_CAMERAS}
        self.clock.mark_camera()
        return frames

    def state(self) -> dict:
        pose = self.tcp()
        grasp = self.estimate_grasp()
        cube = {
            "present": self.cube.present,
            "center_mm": self.cube.center_mm(),
            "size_mm": self.cube.size_mm,
            "grasp_estimated": grasp["grasp_estimated"],
            "grasp_label": "estimated",
        }
        return {
            "profile": self.profile,
            "connected": self.connected,
            "torque_enabled": self.torque_enabled,
            "armed": self.armed,
            "homing": self.homing,
            "table_collision": self.table_collision,
            "joints_deg": dict(self.joints),
            "tcp_mm": pose.as_mm(),
            "tcp_rpy_deg": pose.rpy_deg(),
            "tcp_level": pose.level,
            "gripper_deg": self.joints[GRIPPER_JOINT],
            "gripper_mm": self.gripper_mm(),
            "gripper_curve": str(self.curve.get("version") or GRIPPER_CURVE_VERSION),
            "gripper_curve_measured": bool(self.curve.get("measured")),
            "objects": {"cube": cube},
            "clock": {
                "t": self.clock.t,
                "command_hz": self.clock.command_hz,
                "camera_hz": self.clock.camera_hz,
                "command_period_s": self.clock.command_period_s,
                "camera_period_s": self.clock.camera_period_s,
                "last_command_time": self.last_command_time,
            },
        }


@dataclass
class CommandResult:
    sent: bool
    withheld: bool
    reason: str | None = None
    action: dict | None = None
    fail_closed: bool = False


@dataclass
class HardwareAdapter:
    """TCP Δmm → seven CAN degrees. Withholds on missing joints, mixed units, or missing cameras."""

    plant: B601Plant
    withheld: bool = False
    last_reason: str | None = None
    last_obs: dict | None = None
    fail_closed: bool = False

    def connect(self) -> None:
        self.plant.connect()

    def arm(self) -> None:
        self.plant.arm()

    def hold(self) -> None:
        self.plant.hold()

    def home(self, dt: float | None = None) -> None:
        self.plant.home(dt)

    def reset(self) -> None:
        self.plant.reset()

    def _withhold(self, reason: str, *, sticky: bool = True) -> CommandResult:
        self.withheld = True
        self.last_reason = reason
        if sticky:
            self.fail_closed = True
        return CommandResult(sent=False, withheld=True, reason=reason, fail_closed=self.fail_closed)

    def observe(self) -> dict:
        try:
            joints = self.plant.get_joint_positions(strict=True)
            joint_fault = None
        except MissingJointError as exc:
            painted = self.plant.present_pos(strict=False)
            self.last_obs = {"joints_deg": painted, "fault": f"missing:{exc.name}", "painted_zero": True}
            self.fail_closed = True
            self.withheld = True
            self.last_reason = f"missing:{exc.name}"
            raise
        frames = self.plant.capture_all()
        try:
            encoder_pair({name: frames[name].rgb for name in frames}, "hardware")
        except CameraContractError:
            self.fail_closed = True
            raise
        times = {
            name: {
                "capture_time": frames[name].capture_time,
                "receive_time": frames[name].receive_time,
                "command_time": frames[name].command_time,
                "age_s": frames[name].age_s,
            }
            for name in HW_CAMERAS
        }
        skew = float(frames["wrist"].capture_time) - float(frames["overview"].capture_time)
        obs = {
            "joints_deg": joints,
            "tcp_mm": self.plant.tcp().as_mm(),
            "frames": frames,
            "times": times,
            "camera_skew_s": skew,
            "fault": joint_fault,
            "grasp": self.plant.estimate_grasp(),
        }
        self.last_obs = obs
        return obs

    def send_joints(self, action: dict) -> CommandResult:
        if self.fail_closed:
            return self._withhold(self.last_reason or "fail_closed")
        if not self.plant.armed:
            return self._withhold("not_armed")
        try:
            self.plant.get_joint_positions(strict=True)
        except MissingJointError as exc:
            return self._withhold(f"missing:{exc.name}")
        try:
            parsed = parse_action(action)
        except (UnitMixError, IncompleteActionError, FailClosed) as exc:
            return self._withhold(str(exc))
        self.plant.send_action(action_from_joints(parsed), fill_missing=False)
        self.withheld = False
        self.last_reason = None
        return CommandResult(sent=True, withheld=False, action=action_from_joints(parsed), fail_closed=False)

    def command_tcp(
        self,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        dgrip_mm: float,
        *,
        keep_level: bool = False,
        fingers_down: bool = False,
        dgrip_deg: float | None = None,
    ) -> CommandResult:
        if dgrip_deg is not None:
            return self._withhold("dgrip_deg mixed into millimetre TCP command")
        if not self.plant.armed:
            return self._withhold("not_armed")
        try:
            self.plant.get_joint_positions(strict=True)
        except MissingJointError as exc:
            return self._withhold(f"missing:{exc.name}")
        pose = self.plant.tcp()
        target = {"x": pose.x_mm + float(dx_mm), "y": pose.y_mm + float(dy_mm), "z": pose.z_mm + float(dz_mm)}
        q, err = ik_arm(
            target,
            np.array([self.plant.joints[n] for n in ARM_JOINTS]),
            keep_level=bool(keep_level),
            fingers_down=bool(fingers_down),
        )
        if not math.isfinite(err) or err > 8.0:
            return self._withhold(f"ik_failed:{err}", sticky=False)
        new_mm = float(np.clip(self.plant.gripper_mm() + float(dgrip_mm), LAB_CLOSED_MM, LAB_OPEN_MM))
        grip_deg = gripper_mm_to_deg(new_mm, self.plant.curve)
        joints = {name: float(v) for name, v in zip(ARM_JOINTS, q)}
        joints[GRIPPER_JOINT] = grip_deg
        return self.send_joints(action_from_joints(joints))
