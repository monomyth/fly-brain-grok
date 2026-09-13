import stat
from pathlib import Path

from rebot_adapter.mcp import MCPClient, app_bundle_for, lab_binary_for
from rebot_adapter.teacher import cube


def _jpeg_b64(color=(10, 10, 10)) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


_SET_CUBE_KEYS = frozenset({"present", "x_mm", "y_mm", "z_mm", "size_mm", "yaw_deg", "attached"})


class FakeRobot:
    def __init__(self, delay_gripper: bool = False):
        self.captures: list[str] = []
        self.calls: list[tuple[str, dict]] = []
        self.delay_gripper = bool(delay_gripper)
        self._pending_gripper = None
        self._state = {
            "scene_ready": True,
            "playback": "stopped",
            "manual_motion": False,
            "control_mode": "scripted",
            "gripper_mm": 0,
            "tcp_mm": {"x": 280.0, "y": 0.0, "z": 80.0},
            "tcp_rpy_deg": {"roll": 0, "pitch": 0, "yaw": 0},
            "tcp_level": True,
            "view": {"camera": "Orbit", "grid": True, "trace": False, "tool_axes": False},
            "objects": {
                "cube": {
                    "present": True,
                    "attached": False,
                    "falling": False,
                    "size_mm": 40,
                    "center_mm": {"x": 280, "y": 0, "z": 19},
                }
            },
        }

    def _apply_gripper(self, opening: float) -> None:
        self._state["gripper_mm"] = opening
        cube_w = self._state["objects"]["cube"]["size_mm"]
        tcp = self._state["tcp_mm"]
        c = self._state["objects"]["cube"]["center_mm"]
        near = abs(tcp["x"] - c["x"]) < 40 and abs(tcp["z"] - 48) < 30
        if opening <= cube_w + 2 and near:
            self._state["objects"]["cube"]["attached"] = True
        if opening >= cube_w + 12:
            self._state["objects"]["cube"]["attached"] = False
            self._state["objects"]["cube"]["center_mm"]["z"] = float(cube_w) / 2.0 - 1.0

    def _flush_gripper(self) -> None:
        if self._pending_gripper is None:
            return
        opening = self._pending_gripper
        self._pending_gripper = None
        self._apply_gripper(opening)

    def call(self, name, arguments=None, retries=1, timeout=12):
        del retries, timeout
        arguments = dict(arguments or {})
        if name != "rebot_get_state":
            self.calls.append((name, arguments))
        if name == "rebot_get_state":
            return dict(self._state)
        if name == "rebot_set_control_mode":
            self._state["control_mode"] = arguments["mode"]
            return {"accepted": True, "state": self._state}
        if name in {"rebot_servo_tcp", "rebot_move_to_pose"}:
            self._state["tcp_mm"] = {"x": arguments["x_mm"], "y": arguments["y_mm"], "z": arguments["z_mm"]}
            if self._state["objects"]["cube"]["attached"]:
                self._state["objects"]["cube"]["center_mm"]["z"] = arguments["z_mm"] - 29
            return {"accepted": True, "state": self._state}
        if name in {"rebot_servo_joints", "rebot_set_gripper"}:
            opening = arguments.get("gripper_mm", arguments.get("opening_mm"))
            if opening is None:
                raise KeyError(name)
            opening = float(opening)
            if self.delay_gripper:
                self._pending_gripper = opening
            else:
                self._apply_gripper(opening)
            return {"accepted": True, "state": self._state}
        if name == "rebot_set_cube":
            extra = set(arguments) - _SET_CUBE_KEYS
            if extra:
                raise KeyError(name)
            c = self._state["objects"]["cube"]
            if "present" in arguments:
                c["present"] = bool(arguments["present"])
            if "attached" in arguments:
                c["attached"] = bool(arguments["attached"])
            if "x_mm" in arguments:
                c["center_mm"]["x"] = arguments["x_mm"]
            if "y_mm" in arguments:
                c["center_mm"]["y"] = arguments["y_mm"]
            if "size_mm" in arguments:
                c["size_mm"] = arguments["size_mm"]
            if not c.get("attached"):
                c["falling"] = False
                c["center_mm"]["z"] = float(c.get("size_mm") or 40) / 2.0 - 1.0
            return {"accepted": True, "state": self._state}
        if name == "rebot_playback":
            if arguments.get("action") == "reset":
                self._state["playback"] = "stopped"
                self._state["gripper_mm"] = 0
            return {"accepted": True, "state": self._state}
        if name == "rebot_apply_preset":
            if arguments.get("name") == "Folded":
                self._state["gripper_mm"] = 0.0
            return {"accepted": True, "state": self._state}
        if name == "rebot_set_view":
            if "camera" in arguments:
                self._state["view"]["camera"] = arguments["camera"]
            return {"accepted": True, "state": self._state}
        if name == "rebot_capture_view":
            camera = arguments.get("camera", "Orbit")
            self.captures.append(camera)
            if arguments.get("apply"):
                self._state["view"]["camera"] = camera
            color = {"Front": (200, 80, 40), "Gripper": (40, 80, 200), "Top": (40, 200, 80)}.get(camera, (10, 10, 10))
            return {
                "camera": camera,
                "width": int(arguments.get("width", 320)),
                "height": int(arguments.get("height", 240)),
                "jpeg_base64": _jpeg_b64(color),
                "cube_in_view": True,
                "tcp_in_view": True,
            }
        raise KeyError(name)

    def state(self, retries=1, timeout=12):
        self._flush_gripper()
        return self.call("rebot_get_state")

    def try_state(self, timeout=8):
        return self.state()

    def wait_stopped(self, timeout=20):
        self.calls.append(("wait_stopped", {"timeout": timeout}))
        if not self.delay_gripper:
            self._flush_gripper()
        return dict(self._state)

    def initialize(self, name="flybrain-dn-bus"):
        self.calls.append(("initialize", {"name": name}))
        return {"ok": True}

    def wait_ready(self, timeout=20):
        return dict(self._state)

    def apply_preset(self, name, timeout=40):
        return self.call("rebot_apply_preset", {"name": name})

    def wait_servo_step(self, target=None, timeout=3.0):
        del target, timeout
        return self.try_state()

    def close(self):
        self.calls.append(("close", {}))


def test_lab_binary_sits_next_to_mcp():
    mcp = Path("/tmp/Debug/ReBotMCP")
    assert lab_binary_for(mcp) == Path("/tmp/Debug/ReBotMotionLab")
    assert app_bundle_for(Path("/tmp/ReBot Motion Lab Grok.app/Contents/MacOS/ReBotMCP")) == Path("/tmp/ReBot Motion Lab Grok.app").resolve()


def test_fake_servo_attaches_and_lifts():
    robot = FakeRobot()
    robot.call("rebot_set_control_mode", {"mode": "servo"})
    robot.call("rebot_servo_tcp", {"x_mm": 280, "y_mm": 0, "z_mm": 48, "keep_level": True})
    robot.call("rebot_servo_joints", {"gripper_mm": 40})
    assert cube(robot.state())["attached"]
    robot.call("rebot_servo_tcp", {"x_mm": 280, "y_mm": 0, "z_mm": 160, "keep_level": True})
    assert cube(robot.state())["center_mm"]["z"] > 80
    robot.call("rebot_servo_joints", {"gripper_mm": 90})
    assert not cube(robot.state())["attached"]


def test_spawn_pad_overlay_picks_on_fake():
    from rebot_adapter.pick import pick_success, spawn_pad_overlay

    robot = FakeRobot()
    robot.call("rebot_set_control_mode", {"mode": "servo"})
    robot.call("rebot_servo_tcp", {"x_mm": 543, "y_mm": 0, "z_mm": 409, "keep_level": True})
    robot.call("rebot_servo_joints", {"gripper_mm": 90})
    hold_s = 0.0
    picked = False
    for _ in range(120):
        state = robot.state()
        tcp = state["tcp_mm"]
        attached = cube(state)["attached"]
        decision = spawn_pad_overlay(tcp, float(state["gripper_mm"]), attached, 0.0)
        cmd = decision.command
        z = max(32.0, tcp["z"] + cmd.dz_mm)
        if attached and tcp["z"] < 140:
            z = 160.0
        robot.call("rebot_servo_tcp", {"x_mm": tcp["x"] + cmd.dx_mm, "y_mm": tcp["y"] + cmd.dy_mm, "z_mm": z, "keep_level": True})
        grip = float(state["gripper_mm"] + cmd.dgrip_mm)
        if decision.phase == "close":
            grip = 42.0
        robot.call("rebot_servo_joints", {"gripper_mm": grip})
        after = robot.state()
        c = cube(after)
        if c["attached"] and c["center_mm"]["z"] >= 100 and after["tcp_level"]:
            hold_s += 0.25
        else:
            hold_s = 0.0
        if pick_success(attached=c["attached"], cube_z_mm=c["center_mm"]["z"], tcp_level=after["tcp_level"], hold_s=hold_s):
            picked = True
            break
    assert picked
    assert cube(robot.state())["attached"]
    assert cube(robot.state())["center_mm"]["z"] >= 100


def test_scripted_pad_pick_closes_to_size_plus_two():
    from rebot_adapter.pick import scripted_pad_pick

    robot = FakeRobot()
    robot.call("rebot_set_control_mode", {"mode": "scripted"})
    robot.call("rebot_servo_tcp", {"x_mm": 280, "y_mm": 0, "z_mm": 80, "keep_level": True})
    robot.call("rebot_set_gripper", {"opening_mm": 90})
    after = scripted_pad_pick(robot, 280.0, 0.0, 40.0, hold_s=0.0)
    assert after["gripper_mm"] == 42.0
    assert cube(after)["attached"]
    assert cube(after)["center_mm"]["z"] >= 100


def test_capture_rgbs_requests_front_and_gripper():
    from rebot_adapter.episode import capture_rgbs
    from rebot_adapter.teacher import CAPTURE_CAMERAS

    robot = FakeRobot()
    frames = capture_rgbs(robot, CAPTURE_CAMERAS, width=32, height=24)
    assert len(frames) == 2
    assert robot.captures == ["Front", "Gripper"]
    assert all(frame.ndim == 3 and frame.shape[2] == 3 for frame in frames)
    assert robot._state["view"]["camera"] == "Orbit"


def test_record_pick_writes_front_and_gripper(tmp_path):
    from rebot_adapter.teacher import CAPTURE_CAMERAS, record_pick
    from train.dataset import VIEW_CAMERAS, load_teacher_run

    assert CAPTURE_CAMERAS == VIEW_CAMERAS == ("Front", "Gripper")
    robot = FakeRobot()
    summary = record_pick(robot, run_dir=tmp_path / "ep", hold_s=0.0)
    assert summary["ok"]
    frames = tmp_path / "ep" / "frames"
    names = sorted(p.name for p in frames.glob("*.jpg"))
    assert any(name.endswith("-front.jpg") for name in names)
    assert any(name.endswith("-gripper.jpg") for name in names)
    assert not any("top" in name.lower() for name in names)
    assert set(robot.captures) == {"Front", "Gripper"}
    assert "Top" not in robot.captures
    samples = load_teacher_run(tmp_path / "ep")
    cameras = {s.camera for s in samples}
    assert cameras == {"Front", "Gripper"}
    log = (tmp_path / "ep" / "log.jsonl").read_text()
    assert '"Front"' in log and '"Gripper"' in log
    assert "Top" not in log


def test_cleanup_episode_opens_then_folds():
    from rebot_adapter.pick import cleanup_episode
    from rebot_adapter.teacher import cube

    robot = FakeRobot(delay_gripper=True)
    robot.call("rebot_set_gripper", {"opening_mm": 20})
    robot._flush_gripper()
    robot._state["objects"]["cube"]["attached"] = True
    robot._state["objects"]["cube"]["size_mm"] = 20
    robot._state["objects"]["cube"]["center_mm"] = {"x": 200.0, "y": -54.0, "z": 120.0}
    robot._state["tcp_mm"] = {"x": 280.0, "y": 0.0, "z": 48.0}
    cleanup_episode(robot, x_mm=280.0, y_mm=0.0, size_mm=20.0)
    seq = [name for name, _ in robot.calls]
    assert "rebot_set_gripper" in seq
    fold_i = seq.index("rebot_apply_preset")
    assert robot.calls[fold_i][1].get("name") == "Folded"
    assert seq[fold_i + 1] == "wait_stopped"
    assert "rebot_set_cube" in seq[fold_i + 2 :]
    open_i = next(i for i, (name, args) in enumerate(robot.calls) if name == "rebot_set_gripper" and args.get("opening_mm") == 90)
    assert open_i < fold_i
    cube_args = next(args for name, args in robot.calls if name == "rebot_set_cube")
    assert set(cube_args) <= _SET_CUBE_KEYS
    assert "falling" not in cube_args
    assert any(name == "rebot_set_gripper" and args.get("opening_mm") == 90 for name, args in robot.calls)
    assert robot._state["gripper_mm"] == 0
    c = cube(robot._state)
    assert c["attached"] is False
    assert c.get("falling") is False
    assert abs(float(c["center_mm"]["x"]) - 280.0) < 1e-9
    assert abs(float(c["center_mm"]["y"]) - 0.0) < 1e-9
    assert float(c["center_mm"]["z"]) < 20.0
    assert robot._state["control_mode"] == "scripted"


def test_second_episode_sees_table_cube_at_spawn():
    from rebot_adapter.pick import cleanup_episode, scene_cube_xy
    from rebot_adapter.teacher import cube

    robot = FakeRobot(delay_gripper=True)
    robot.call("rebot_set_gripper", {"opening_mm": 20})
    robot._flush_gripper()
    robot._state["objects"]["cube"]["attached"] = True
    robot._state["objects"]["cube"]["size_mm"] = 20
    robot._state["objects"]["cube"]["center_mm"] = {"x": 200.0, "y": -54.0, "z": 120.0}
    cleanup_episode(robot, x_mm=280.0, y_mm=0.0, size_mm=20.0)
    st = robot.state()
    c = cube(st)
    x, y, already = scene_cube_xy(st, 280.0, 0.0)
    assert c["attached"] is False
    assert c.get("falling") is False
    assert float(c["center_mm"]["z"]) < 20.0
    assert already is True
    assert abs(x - 280.0) < 1e-9
    assert abs(y - 0.0) < 1e-9
    assert abs(float(c["center_mm"]["x"]) - 280.0) < 1e-9
    assert abs(float(c["center_mm"]["y"]) - 0.0) < 1e-9


def test_cleanup_skips_fold_when_gripper_stays_shut():
    from rebot_adapter.pick import cleanup_episode

    class StuckGrip(FakeRobot):
        def _apply_gripper(self, opening: float) -> None:
            del opening

    robot = StuckGrip()
    robot._state["gripper_mm"] = 20.0
    robot._state["objects"]["cube"]["attached"] = True
    cleanup_episode(robot, x_mm=280.0, y_mm=0.0, size_mm=20.0, open_s=0.0)
    seq = [name for name, _ in robot.calls]
    assert "rebot_apply_preset" not in seq
    assert "rebot_set_cube" in seq
    assert abs(float(robot._state["gripper_mm"]) - 20.0) < 1e-9


def test_fake_set_cube_rejects_falling():
    robot = FakeRobot()
    try:
        robot.call("rebot_set_cube", {"x_mm": 280.0, "y_mm": 0.0, "size_mm": 20.0, "falling": False})
    except KeyError:
        return
    raise AssertionError("falling must be rejected")


def _fake_mcp(tmp_path, delay=0.4, error=False) -> tuple[Path, Path]:
    log = tmp_path / "calls.log"
    script = tmp_path / "fake-mcp"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        f"log_path = {str(log)!r}\n"
        f"delay = {delay!r}\n"
        f"error = {error!r}\n"
        "def reply(req, result=None, is_error=False):\n"
        "    if is_error:\n"
        "        body = {'jsonrpc': '2.0', 'id': req['id'], 'result': {'isError': True, 'content': [{'text': 'No IK'}]}}\n"
        "    else:\n"
        "        body = {'jsonrpc': '2.0', 'id': req['id'], 'result': result}\n"
        "    sys.stdout.write(json.dumps(body) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    req = json.loads(line)\n"
        "    method = req.get('method')\n"
        "    if method == 'initialize':\n"
        "        reply(req, {'protocolVersion': '2025-11-25', 'serverInfo': {'name': 'fake'}, 'capabilities': {}})\n"
        "        continue\n"
        "    if method == 'notifications/initialized':\n"
        "        continue\n"
        "    if method == 'tools/call':\n"
        "        with open(log_path, 'a') as handle:\n"
        "            handle.write(req['params']['name'] + '\\n')\n"
        "        if error:\n"
        "            reply(req, is_error=True)\n"
        "            continue\n"
        "        time.sleep(delay)\n"
        "        reply(req, {'structuredContent': {'ok': True, 'scene_ready': True, 'playback': 'stopped', 'manual_motion': False}})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, log


def test_pending_timeout_does_not_return_other_tool_result(tmp_path):
    script, log = _fake_mcp(tmp_path, delay=0.4)
    client = MCPClient(script, launch_lab=False)
    try:
        client.initialize()
        try:
            client.call("rebot_servo_tcp", {"x_mm": 1}, timeout=0.05, retries=1)
            raise AssertionError("expected timeout")
        except TimeoutError:
            pass
        assert client._pending_id is not None
        try:
            client.call("rebot_get_state", timeout=0.05, retries=1)
        except TimeoutError:
            pass
        names = log.read_text().strip().splitlines()
        assert names[0] == "rebot_servo_tcp"
        assert names.count("rebot_get_state") <= 1
        client.drain_pending(timeout=1.0)
        state = client.call("rebot_get_state", timeout=2.0, retries=1)
        assert "scene_ready" in state or "ok" in state
    finally:
        client.close()


def test_late_reply_after_short_drain_does_not_desync(tmp_path):
    script, log = _fake_mcp(tmp_path, delay=0.35)
    client = MCPClient(script, launch_lab=False)
    try:
        client.initialize()
        try:
            client.call("rebot_get_state", timeout=0.05, retries=1)
            raise AssertionError("expected timeout")
        except TimeoutError:
            pass
        assert client._pending_id is not None
        assert client.drain_pending(timeout=0.05) is None
        assert client._pending_id is not None
        late = client.drain_pending(timeout=1.0)
        assert late is not None
        assert late.get("id") == late.get("_pending_id")
        client.call("rebot_get_state", timeout=2.0, retries=1)
        assert log.read_text().strip().splitlines() == ["rebot_get_state", "rebot_get_state"]
    finally:
        client.close()


def test_timeout_drains_late_reply_without_second_call(tmp_path):
    script, log = _fake_mcp(tmp_path, delay=0.4)
    client = MCPClient(script, launch_lab=False)
    try:
        client.initialize()
        try:
            client.call("rebot_get_state", timeout=0.05, retries=1)
            raise AssertionError("expected timeout")
        except TimeoutError:
            pass
        assert log.read_text().strip().splitlines() == ["rebot_get_state"]
        client.drain_pending(timeout=1.0)
        assert log.read_text().strip().splitlines() == ["rebot_get_state"]
        client.call("rebot_get_state", timeout=2.0, retries=1)
        assert log.read_text().strip().splitlines() == ["rebot_get_state", "rebot_get_state"]
    finally:
        client.close()


def test_wait_servo_step_waits_until_target(tmp_path):
    log = tmp_path / "calls.log"
    script = tmp_path / "fake-mcp-tcp"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"log_path = {str(log)!r}\n"
        "xs = [100.0, 104.0, 108.0, 108.0]\n"
        "i = {'n': 0}\n"
        "def reply(req, result):\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'structuredContent': result}}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    req = json.loads(line)\n"
        "    method = req.get('method')\n"
        "    if method == 'initialize':\n"
        "        reply(req, {'protocolVersion': '2025-11-25'})\n"
        "        continue\n"
        "    if method == 'notifications/initialized':\n"
        "        continue\n"
        "    if method == 'tools/call':\n"
        "        name = req['params']['name']\n"
        "        with open(log_path, 'a') as handle:\n"
        "            handle.write(name + '\\n')\n"
        "        if name != 'rebot_get_state':\n"
        "            reply(req, {'ok': True})\n"
        "            continue\n"
        "        x = xs[min(i['n'], len(xs) - 1)]\n"
        "        i['n'] += 1\n"
        "        reply(req, {'tcp_mm': {'x': x, 'y': 0.0, 'z': 80.0}, 'scene_ready': True, 'playback': 'stopped', 'manual_motion': False})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    client = MCPClient(script, launch_lab=False)
    try:
        client.initialize()
        target = {"x_mm": 108.0, "y_mm": 0.0, "z_mm": 80.0}
        last = client.wait_servo_step(target=target, timeout=2.0)
        assert last is not None
        assert abs(last["tcp_mm"]["x"] - 108.0) <= 2.5
        names = log.read_text().strip().splitlines()
        assert names.count("rebot_get_state") >= 3
    finally:
        client.close()


def test_wait_servo_step_does_not_return_on_first_motion(tmp_path):
    log = tmp_path / "calls.log"
    script = tmp_path / "fake-mcp-tcp2"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"log_path = {str(log)!r}\n"
        "xs = [100.0, 102.0, 102.0, 102.0, 102.0]\n"
        "i = {'n': 0}\n"
        "def reply(req, result):\n"
        "    sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': req['id'], 'result': {'structuredContent': result}}) + '\\n')\n"
        "    sys.stdout.flush()\n"
        "for line in sys.stdin:\n"
        "    if not line.strip():\n"
        "        continue\n"
        "    req = json.loads(line)\n"
        "    method = req.get('method')\n"
        "    if method == 'initialize':\n"
        "        reply(req, {'protocolVersion': '2025-11-25'})\n"
        "        continue\n"
        "    if method == 'notifications/initialized':\n"
        "        continue\n"
        "    if method == 'tools/call':\n"
        "        name = req['params']['name']\n"
        "        with open(log_path, 'a') as handle:\n"
        "            handle.write(name + '\\n')\n"
        "        x = xs[min(i['n'], len(xs) - 1)]\n"
        "        i['n'] += 1\n"
        "        reply(req, {'tcp_mm': {'x': x, 'y': 0.0, 'z': 80.0}, 'scene_ready': True})\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    client = MCPClient(script, launch_lab=False)
    try:
        client.initialize()
        last = client.wait_servo_step(target={"x_mm": 108.0, "y_mm": 0.0, "z_mm": 80.0}, timeout=0.6)
        assert last is not None
        assert abs(last["tcp_mm"]["x"] - 108.0) > 2.5
        assert last["tcp_mm"]["x"] >= 102.0
        assert log.read_text().strip().splitlines().count("rebot_get_state") >= 2
    finally:
        client.close()


def test_tool_error_is_not_retried(tmp_path):
    script, log = _fake_mcp(tmp_path, delay=0.0, error=True)
    client = MCPClient(script, launch_lab=False)
    try:
        client.initialize()
        try:
            client.call("rebot_servo_tcp", {"x_mm": 0}, retries=3, timeout=2.0)
            raise AssertionError("expected tool error")
        except RuntimeError as exc:
            assert "No IK" in str(exc)
        assert log.read_text().strip().splitlines() == ["rebot_servo_tcp"]
    finally:
        client.close()
