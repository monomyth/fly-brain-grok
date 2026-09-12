from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from malecns_cache.graph import Connectome
from runtime.decoder import decode_dn_rates
from runtime.encoder import decode_jpeg, encode_frame
from runtime.lif import LIFNetwork

from .mcp import MCPClient
from .teacher import CAPTURE_CAMERAS, cube


def capture_rgbs(
    client: MCPClient,
    cameras: tuple[str, ...] = CAPTURE_CAMERAS,
    width: int = 160,
    height: int = 120,
    save_dir: Path | None = None,
    stem: str | None = None,
) -> list:
    """JPEG pixels from each training camera. apply is always false."""
    frames = []
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    for name in cameras:
        shot = client.call(
            "rebot_capture_view",
            {"camera": name, "width": width, "height": height, "apply": False},
        )
        raw = base64.b64decode(shot["jpeg_base64"])
        if save_dir is not None:
            (save_dir / f"{stem or 'tick'}-{name.lower()}.jpg").write_bytes(raw)
        frames.append(decode_jpeg(raw))
    return frames


@dataclass
class Tick:
    attached: bool
    tcp: dict
    command: dict
    cube_z: float


class ServoLoop:
    """Closed-loop fly tick. Lab keeps IK, floor, and grasp."""

    def __init__(self, client: MCPClient, connectome: Connectome, steps_per_tick: int = 100):
        self.client = client
        self.connectome = connectome
        self.brain = LIFNetwork(connectome.weights)
        self.steps_per_tick = steps_per_tick
        self.r1 = connectome.indices("photoreceptors_r1r6")
        self.r8 = connectome.indices("photoreceptors_r8")

    def setup(self, x_mm: float = 280, y_mm: float = 0, size_mm: float = 40) -> dict:
        self.client.call("rebot_playback", {"action": "reset"})
        self.client.wait_stopped()
        self.client.call("rebot_apply_preset", {"name": "Ready"})
        self.client.wait_stopped()
        self.client.call("rebot_set_gripper", {"opening_mm": 60})
        self.client.wait_stopped()
        self.client.call("rebot_set_cube", {"present": True, "x_mm": x_mm, "y_mm": y_mm, "size_mm": size_mm})
        self.client.call("rebot_set_view", {"camera": "Front"})
        self.client.call("rebot_set_control_mode", {"mode": "servo"})
        self.brain.reset_state()
        return self.client.state()

    def tick(self) -> Tick:
        frames = capture_rgbs(self.client, CAPTURE_CAMERAS, width=320, height=240)
        currents = []
        for rgb in frames:
            r1, r8 = encode_frame(rgb, int(self.r1.size), int(self.r8.size))
            currents.append((r1, r8))
        current = np.zeros(self.brain.n, dtype=np.float32)
        if currents:
            r1 = np.mean([c[0] for c in currents], axis=0)
            r8 = np.mean([c[1] for c in currents], axis=0)
            if self.r1.size:
                current[self.r1] = r1
            if self.r8.size:
                current[self.r8] = r8
        self.brain.i_ext = current
        self.brain.step(self.steps_per_tick)
        cmd = decode_dn_rates(self.connectome, self.brain.rate_hz)
        state = self.client.state()
        tcp = state["tcp_mm"]
        target = {
            "x_mm": tcp["x"] + cmd.dx_mm,
            "y_mm": tcp["y"] + cmd.dy_mm,
            "z_mm": tcp["z"] + cmd.dz_mm,
            "keep_level": True,
        }
        self.client.call("rebot_servo_tcp", target)
        grip = float(np.clip(state["gripper_mm"] + cmd.dgrip_mm, 0, 90))
        self.client.call("rebot_servo_joints", {"gripper_mm": grip})
        after = self.client.state()
        c = cube(after)
        return Tick(attached=bool(c["attached"]), tcp=after["tcp_mm"], command=target, cube_z=c["center_mm"]["z"])


def default_mcp_binary() -> Path:
    root = Path(__file__).resolve().parents[2] / "rebot-motion-lab-grok"
    for name in ("ReBot Motion Lab Grok.app", "ReBot Motion Lab.app"):
        bundled = root / "dist" / name / "Contents/MacOS/ReBotMCP"
        if bundled.is_file():
            return bundled
    return root / ".build/out/Products/Debug/ReBotMCP"
