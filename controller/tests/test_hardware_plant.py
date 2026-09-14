from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from rebot_adapter.b601 import (
    ARM_JOINTS,
    CAMERA_HZ,
    CAN_JOINTS,
    COMMAND_HZ,
    GRIPPER_CLOSED_DEG,
    GRIPPER_JOINT,
    GRIPPER_OPEN_DEG,
    HOME_RATE_DEG_S,
    OVERVIEW_MEAN,
    B601Plant,
    FailClosed,
    HardwareAdapter,
    MissingJointError,
    NotArmedError,
    TableCollisionError,
    UnitMixError,
    action_from_joints,
    fk_pose,
    gripper_deg_to_mm,
    gripper_mm_to_deg,
    ik_arm,
    iter_schedule,
    load_gripper_curve,
)
from rebot_adapter.camera_contract import (
    HW_CAPTURE_WH,
    CameraContractError,
    crop_rectify_to_encoder,
    encoder_pair,
    select_named,
)
from rebot_adapter.episode import capture_rgbs
from tests.test_mcp_fake import FakeRobot


def _armed_ready(**kwargs) -> B601Plant:
    plant = B601Plant.ready(**kwargs)
    plant.connect()
    plant.arm()
    return plant


def test_can_joint_order_and_names():
    assert CAN_JOINTS == (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    )


def test_fk_ik_roundtrip_tcp_mm():
    q0 = np.array([0.0, -95.0, -95.0, 10.0, 0.0, 0.0])
    pose = fk_pose(q0)
    assert math.isfinite(pose.x_mm) and math.isfinite(pose.z_mm)
    target = {"x": pose.x_mm + 12.0, "y": pose.y_mm - 6.0, "z": pose.z_mm + 8.0}
    q1, err = ik_arm(target, q0, keep_level=False)
    assert err < 2.5
    back = fk_pose(q1)
    dist = math.hypot(back.x_mm - target["x"], back.y_mm - target["y"])
    dist = math.hypot(dist, back.z_mm - target["z"])
    assert dist < 2.5


def test_adapter_tcp_delta_uses_seven_named_joints():
    plant = _armed_ready()
    before = plant.tcp()
    adapter = HardwareAdapter(plant)
    result = adapter.command_tcp(10.0, 0.0, 0.0, 0.0, keep_level=False)
    assert result.sent and not result.withheld
    assert set(result.action) == {f"{name}.pos" for name in CAN_JOINTS}
    after = plant.tcp()
    assert abs(after.x_mm - (before.x_mm + 10.0)) < 4.0
    assert abs(after.y_mm - before.y_mm) < 4.0


def test_gripper_curve_roundtrip_and_measured_flag(tmp_path, monkeypatch):
    curve = load_gripper_curve("fake-v0")
    assert curve["measured"] is False
    assert gripper_mm_to_deg(0.0, curve) == pytest.approx(GRIPPER_CLOSED_DEG)
    assert gripper_mm_to_deg(90.0, curve) == pytest.approx(GRIPPER_OPEN_DEG)
    assert gripper_deg_to_mm(GRIPPER_OPEN_DEG, curve) == pytest.approx(90.0)
    assert gripper_deg_to_mm(-135.0, curve) == pytest.approx(45.0)
    with pytest.raises(FailClosed):
        load_gripper_curve("v1-absent")
    monkeypatch.setattr("rebot_adapter.b601.calibration_dir", lambda: tmp_path)
    with pytest.raises(FailClosed):
        load_gripper_curve("v1")
    assert load_gripper_curve("fake-v0")["measured"] is False


def test_positive_mm_gripper_does_not_clip_to_closed():
    plant = _armed_ready()
    plant.joints[GRIPPER_JOINT] = GRIPPER_OPEN_DEG
    action = action_from_joints(plant.joints)
    action["gripper.pos"] = 90.0
    before = plant.joints[GRIPPER_JOINT]
    with pytest.raises(UnitMixError):
        plant.send_action(action)
    assert plant.joints[GRIPPER_JOINT] == pytest.approx(before)
    assert plant.joints[GRIPPER_JOINT] != pytest.approx(GRIPPER_CLOSED_DEG)

    mixed = action_from_joints({**plant.joints, GRIPPER_JOINT: GRIPPER_OPEN_DEG})
    mixed["gripper_mm"] = 90.0
    with pytest.raises(UnitMixError):
        plant.send_action(mixed)
    assert plant.joints[GRIPPER_JOINT] == pytest.approx(GRIPPER_OPEN_DEG)


def test_hardware_state_has_estimated_grasp_not_attach_oracle():
    plant = _armed_ready()
    st = plant.state()
    cube = st["objects"]["cube"]
    assert "attached" not in cube
    assert cube["grasp_label"] == "estimated"
    assert cube["grasp_estimated"] is False
    plant.place_cube(plant.tcp().x_mm, plant.tcp().y_mm, size_mm=20.0)
    plant.cube.z_mm = plant.tcp().z_mm
    plant.joints[GRIPPER_JOINT] = gripper_mm_to_deg(20.0, plant.curve)
    grasp = plant.estimate_grasp()
    assert grasp["grasp_label"] == "estimated"
    assert grasp["grasp_estimated"] is True


def test_joints_move_independently_without_keep_level():
    plant = _armed_ready()
    tilt0 = plant.tilt_deg()
    z0 = plant.tcp().tool_z
    action = action_from_joints(plant.joints)
    action["wrist_flex.pos"] = 50.0
    plant.send_action(action)
    assert plant.joints["wrist_flex"] == pytest.approx(50.0)
    assert plant.tilt_deg() > tilt0 + 15.0
    assert plant.tcp().tool_z != z0
    assert plant.tcp().level is False


def test_table_collision_blocks_command_not_home():
    plant = _armed_ready()
    pose = plant.tcp()
    q, err = ik_arm({"x": pose.x_mm, "y": pose.y_mm, "z": 2.0}, np.array([plant.joints[n] for n in ARM_JOINTS]))
    assert err < 8.0
    before = dict(plant.joints)
    action = action_from_joints({**plant.joints, **{n: float(v) for n, v in zip(ARM_JOINTS, q)}})
    with pytest.raises(TableCollisionError):
        plant.send_action(action)
    assert plant.table_collision is True
    assert plant.joints["shoulder_lift"] == pytest.approx(before["shoulder_lift"])
    adapter = HardwareAdapter(plant)
    result = adapter.command_tcp(0.0, 0.0, 2.0 - pose.z_mm, 0.0, keep_level=True)
    assert result.sent is False
    assert result.withheld is True
    assert result.reason == "table_collision"
    assert plant.joints["shoulder_lift"] == pytest.approx(before["shoulder_lift"])
    assert plant.tcp().z_mm == pytest.approx(pose.z_mm, abs=0.5)
    plant.home(dt=0.5)
    assert plant.joints["shoulder_lift"] != pytest.approx(before["shoulder_lift"])


def test_crop_rectify_does_not_stretch_16x9_to_4x3():
    h, w = HW_CAPTURE_WH[1], HW_CAPTURE_WH[0]
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.ogrid[:h, :w]
    cy, cx, r = h // 2, w // 2, 80
    rgb[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 1.0

    def aspect(img: np.ndarray) -> float:
        m = img.mean(axis=2) > 0.5
        ys, xs = np.where(m)
        return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)

    stretched = np.asarray(
        Image.fromarray((rgb * 255).astype(np.uint8)).resize((160, 120), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    cropped = crop_rectify_to_encoder(rgb)
    assert cropped.shape == (120, 160, 3)
    assert abs(aspect(stretched) - 1.0) > 0.15
    assert abs(aspect(cropped) - 1.0) < 0.08


def test_encoder_pair_requires_hardware_names_and_size():
    rgb = np.zeros((480, 848, 3), dtype=np.float32)
    lab = np.zeros((120, 160, 3), dtype=np.float32)
    pair = encoder_pair({"overview": rgb, "wrist": rgb}, "hardware")
    assert len(pair) == 2
    assert pair[0].shape == (120, 160, 3)
    with pytest.raises(CameraContractError):
        encoder_pair({"Front": rgb, "Gripper": rgb}, "hardware")
    with pytest.raises(CameraContractError):
        encoder_pair({"Front": lab, "Gripper": lab}, "hardware")
    with pytest.raises(CameraContractError):
        encoder_pair({"overview": rgb, "wrist": rgb}, "lab")
    with pytest.raises(CameraContractError):
        encoder_pair({"Front": rgb, "Gripper": rgb}, "lab")


def test_missing_camera_does_not_use_first_feed():
    rgb_a = np.zeros((480, 848, 3), dtype=np.float32)
    rgb_a[:, :, 0] = 1.0
    frames = {"overview": rgb_a}
    with pytest.raises(CameraContractError, match="missing"):
        select_named(frames, ("wrist", "overview"))
    with pytest.raises(CameraContractError):
        encoder_pair(frames, "hardware")
    first = next(iter(frames.values()))
    assert first is rgb_a


def test_capture_rgbs_fails_on_wrong_name():
    class SwapRobot(FakeRobot):
        def call(self, name, arguments=None, retries=1, timeout=12):
            shot = super().call(name, arguments, retries, timeout)
            if name == "rebot_capture_view":
                shot = dict(shot)
                shot["camera"] = "Orbit"
            return shot

    with pytest.raises(CameraContractError):
        capture_rgbs(SwapRobot(), ("Front", "Gripper"), width=32, height=24)


def test_spec_v1_curve_loads():
    from rebot_adapter.b601 import load_gripper_curve

    blob = load_gripper_curve("v1")
    assert blob["version"] == "v1"
    assert blob["measured"] is False
    assert blob["closed_deg"] == 0.0
    assert blob["open_deg"] == -270.0


def test_plan_hover_lifts_z():
    import importlib.util
    from pathlib import Path

    from rebot_adapter.b601 import READY_ARM_DEG, FailClosed, fk_pose

    path = Path(__file__).resolve().parents[1] / "scripts" / "hw_hover.py"
    spec = importlib.util.spec_from_file_location("hw_hover", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    q0 = np.array(READY_ARM_DEG, dtype=np.float64)
    z0 = fk_pose(q0).z_mm
    plan = mod.plan_hover(q0, 15.0)
    assert plan["to_mm"]["z"] == pytest.approx(z0 + 15.0)
    assert plan["ik_err_mm"] < 5.0
    q_lift = np.array([0.0, -20.0, 0.0, 0.0, 0.0, 0.0])
    plan2 = mod.plan_hover(q_lift, 15.0)
    assert plan2["ik_err_mm"] < 5.0
    assert plan2["keep_level"] is False
    with pytest.raises(FailClosed):
        mod.plan_hover(q0, -5.0)


def test_plant_cameras_match_live_photometry():
    plant = B601Plant.ready()
    frames = plant.capture_all()
    ov = np.asarray(frames["overview"].rgb)
    wr = np.asarray(frames["wrist"].rgb)
    assert ov.shape == (480, 848, 3)
    assert wr.shape == (480, 848, 3)
    assert float(ov.mean()) < 0.05
    assert float(wr.mean()) > 0.2
    assert abs(float(ov.mean()) - OVERVIEW_MEAN) < 0.02
    orange = wr[:, :, 0] - np.maximum(wr[:, :, 1], wr[:, :, 2])
    assert float(orange.max()) > 0.4
    ov_orange = ov[:, :, 0] - np.maximum(ov[:, :, 1], ov[:, :, 2])
    assert float(ov_orange.max()) < 0.05


def test_timed_frames_age_and_skew():
    plant = _armed_ready(camera_age_s=0.25, camera_skew_s=0.08)
    plant.clock.t = 1.0
    frames = plant.capture_all()
    wrist, overview = frames["wrist"], frames["overview"]
    assert 0.1 <= wrist.age_s <= 0.5
    assert 0.1 <= overview.age_s <= 0.5
    assert wrist.age_s == pytest.approx(0.25)
    assert abs(wrist.capture_time - overview.capture_time) == pytest.approx(0.08)
    adapter = HardwareAdapter(plant)
    obs = adapter.observe()
    assert "capture_time" in obs["times"]["wrist"]
    assert "receive_time" in obs["times"]["wrist"]
    assert "command_time" in obs["times"]["wrist"]
    assert abs(obs["camera_skew_s"]) == pytest.approx(0.08)


def test_missing_joint_zero_fill_is_withheld():
    plant = _armed_ready()
    lift0 = plant.joints["shoulder_lift"]
    assert lift0 == pytest.approx(-95.0)
    plant.drop_joint("shoulder_lift")
    with pytest.raises(MissingJointError):
        plant.get_joint_positions(strict=True)
    painted = plant.present_pos(strict=False)
    assert painted["shoulder_lift"] == 0.0
    adapter = HardwareAdapter(plant)
    naive = action_from_joints(painted)
    result = adapter.send_joints(naive)
    assert result.withheld and result.fail_closed
    assert plant.joints["shoulder_lift"] == pytest.approx(lift0)
    tcp = adapter.command_tcp(5.0, 0.0, 0.0, 0.0)
    assert tcp.withheld
    assert plant.joints["shoulder_lift"] == pytest.approx(lift0)


def test_incomplete_action_does_not_zero_wrist_yaw():
    plant = _armed_ready()
    plant.joints["wrist_yaw"] = 20.0
    incomplete = action_from_joints(plant.joints)
    del incomplete["wrist_yaw.pos"]
    adapter = HardwareAdapter(plant)
    result = adapter.send_joints(incomplete)
    assert result.withheld
    assert plant.joints["wrist_yaw"] == pytest.approx(20.0)
    plant.send_action(incomplete, fill_missing=True)
    assert plant.joints["wrist_yaw"] == pytest.approx(0.0)


def test_connect_does_not_enable_torque_or_move():
    plant = B601Plant.ready()
    q0 = dict(plant.joints)
    plant.connect()
    assert plant.connected is True
    assert plant.torque_enabled is False
    assert plant.armed is False
    action = action_from_joints(plant.joints)
    action["shoulder_pan.pos"] = 10.0
    with pytest.raises(NotArmedError):
        plant.send_action(action)
    assert plant.joints == q0
    adapter = HardwareAdapter(plant)
    refused = adapter.command_tcp(5.0, 0.0, 0.0, 0.0)
    assert refused.withheld
    assert refused.reason == "not_armed"
    assert plant.joints == q0
    adapter.arm()
    assert plant.torque_enabled is True
    sent = adapter.command_tcp(5.0, 0.0, 0.0, 0.0)
    assert sent.sent and not sent.withheld
    assert plant.tcp().x_mm != pytest.approx(fk_pose(np.array([q0[n] for n in ARM_JOINTS])).x_mm)


def test_home_interpolates_toward_zero_at_20dps():
    plant = _armed_ready()
    plant.joints[GRIPPER_JOINT] = GRIPPER_OPEN_DEG
    lift0 = plant.joints["shoulder_lift"]
    plant.home(dt=1.0)
    assert plant.joints["shoulder_lift"] == pytest.approx(lift0 + HOME_RATE_DEG_S)
    assert plant.joints[GRIPPER_JOINT] == pytest.approx(GRIPPER_OPEN_DEG + HOME_RATE_DEG_S)
    assert HOME_RATE_DEG_S == 20.0
    held = dict(plant.joints)
    plant.hold()
    plant.step(1.0)
    assert plant.homing is False
    assert plant.joints["shoulder_lift"] == pytest.approx(held["shoulder_lift"])


def test_reset_and_home_do_not_respawn_cube():
    plant = _armed_ready()
    plant.place_cube(200.0, 12.0, size_mm=20.0)
    lift0 = plant.joints["shoulder_lift"]
    plant.reset()
    assert plant.homing is True
    assert plant.cube.x_mm == pytest.approx(200.0)
    assert plant.cube.y_mm == pytest.approx(12.0)
    plant.step(1.0)
    assert plant.joints["shoulder_lift"] == pytest.approx(lift0 + HOME_RATE_DEG_S)
    assert plant.cube.x_mm == pytest.approx(200.0)
    plant.home(dt=1.0)
    assert plant.cube.x_mm == pytest.approx(200.0)
    assert plant.cube.y_mm == pytest.approx(12.0)
    assert plant.joints["shoulder_lift"] == pytest.approx(lift0 + 2.0 * HOME_RATE_DEG_S)
    assert abs(plant.cube.x_mm - 280.0) > 1.0


def test_gravity_and_slip_not_boolean_attach():
    plant = _armed_ready()
    pose = plant.tcp()
    plant.place_cube(pose.x_mm, pose.y_mm, size_mm=20.0)
    plant.cube.z_mm = pose.z_mm
    plant.joints[GRIPPER_JOINT] = gripper_mm_to_deg(20.0, plant.curve)
    assert plant.estimate_grasp()["grasp_estimated"] is True
    z_before = plant.cube.z_mm
    adapter = HardwareAdapter(plant)
    lifted = adapter.command_tcp(0.0, 0.0, 40.0, 0.0, keep_level=True)
    assert lifted.sent
    plant.step(0.05)
    hang = 0.5 * plant.cube.size_mm + 8.0
    assert plant.estimate_grasp()["grasp_estimated"] is True
    assert plant.cube.z_mm == pytest.approx(plant.tcp().z_mm - hang, abs=2.0)
    assert plant.cube.z_mm > z_before + 20.0
    assert "attached" not in plant.state()["objects"]["cube"]

    open_grip = action_from_joints({**plant.joints, GRIPPER_JOINT: GRIPPER_OPEN_DEG})
    plant.send_action(open_grip)
    for _ in range(40):
        plant.step(0.02)
    assert plant.cube.z_mm == pytest.approx(plant.cube.rest_z_mm(), abs=1.0)

    plant.place_cube(plant.tcp().x_mm, plant.tcp().y_mm, size_mm=20.0)
    plant.cube.z_mm = plant.tcp().z_mm
    plant.joints[GRIPPER_JOINT] = gripper_mm_to_deg(20.0, plant.curve)
    plant.step(0.0)
    tilt = action_from_joints({**plant.joints, "wrist_flex": 50.0})
    plant.send_action(tilt)
    plant.step(0.02)
    assert plant.estimate_grasp()["grasp_estimated"] is False
    for _ in range(40):
        plant.step(0.02)
    assert plant.cube.z_mm == pytest.approx(plant.cube.rest_z_mm(), abs=1.0)


def test_cadence_is_2hz_command_15hz_camera():
    assert COMMAND_HZ == 2.0
    assert CAMERA_HZ == 15.0
    events = list(iter_schedule(1.0))
    n_cam = sum(kind == "camera" for kind, _ in events)
    n_cmd = sum(kind == "command" for kind, _ in events)
    assert n_cam == 15
    assert n_cmd == 2
    assert n_cam != n_cmd
    plant = B601Plant.ready()
    assert plant.clock.command_period_s == pytest.approx(0.5)
    assert plant.clock.camera_period_s == pytest.approx(1.0 / 15.0)
    assert plant.clock.command_period_s != plant.clock.camera_period_s
