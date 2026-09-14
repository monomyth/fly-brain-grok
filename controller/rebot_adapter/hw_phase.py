"""Phase 0 ownership and Phase 1 camera hop. USB I/O runs on the robot host."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from arm.hop_probe import MIN_CUBE_DN_HZ, _vision_run, live_go, phase_scramble
from arm.unpack import command_is_abort
from .arm_host import ssh_base
from .camera_contract import HW_CAMERAS, HW_CAPTURE_WH, encoder_pair, require_size

USB_305 = "2bc5:0840"
USB_336L = "2bc5:0807"
SERIAL_WRIST = "CV2L761000FA"
SERIAL_OVERVIEW = "CPC6463000PZ"
MOTOR_MARKERS = (
    "lerobot-calibrate",
    "lerobot-teleoperate",
    "lerobot-record",
    "lerobot-rollout",
    "run-dual-policy",
    "open-gripper",
    "close-gripper",
    "record-dual-five",
    "collect-and-train-dual-five",
)
CAMERA_MARKERS = ("dual_rgb_bridge", "start-dual-cameras", "rgb_bridge.py", "hw_orbbec_grab")
RIG_ID = "b601-orbbec-848x480-v0"


class HwPhaseError(RuntimeError):
    """Phase 0/1 failed closed."""


def _has_marker(text: str, markers: tuple[str, ...]) -> list[str]:
    hits = []
    for line in (text or "").splitlines():
        if "grep" in line:
            continue
        for m in markers:
            if m in line:
                hits.append(line.strip())
                break
    return hits


def ownership_report(
    *,
    lsusb_text: str,
    fuser_acm: str,
    fuser_usb: str,
    ps_text: str,
    tty_acm: bool,
    tty_usb: bool,
    crash_log: bool,
    host: str,
) -> dict:
    """Parse host snapshots. Does not open serial or call connect()."""
    usb = lsusb_text or ""
    cam_305 = USB_305.lower() in usb.lower()
    cam_336 = USB_336L.lower() in usb.lower()
    motor_hits = _has_marker(ps_text, MOTOR_MARKERS)
    cam_hits = _has_marker(ps_text, CAMERA_MARKERS)
    acm_busy = any(tok.isdigit() for tok in (fuser_acm or "").replace(":", " ").split())
    leader_busy = any(tok.isdigit() for tok in (fuser_usb or "").replace(":", " ").split())
    capture_ok = cam_305 and cam_336 and not motor_hits and not cam_hits
    out = {
        "phase": 0,
        "host": host,
        "herdr_is_estop": False,
        "ssh_drop_is_estop": False,
        "connect_called": False,
        "gemini_305_usb": cam_305,
        "gemini_336l_usb": cam_336,
        "tty_acm0": bool(tty_acm),
        "tty_usb0": bool(tty_usb),
        "tty_acm0_busy": acm_busy,
        "tty_usb0_busy": leader_busy,
        "motor_processes": motor_hits,
        "camera_processes": cam_hits,
        "memory_crash_log": bool(crash_log),
        "capture_ok": bool(capture_ok),
        "policy_qualify": False,
        "note": "e-stop is physical. Herdr/SSH are not. Do not call follower connect(). Phase 0 never qualifies a policy.",
    }
    if crash_log:
        out["qualify_block"] = "unrepaired memory-subsystem log present"
    elif motor_hits or acm_busy:
        out["qualify_block"] = "motor process or CAN device busy"
    elif not capture_ok:
        out["qualify_block"] = "cameras missing or busy"
    else:
        out["qualify_block"] = "phase 0 does not qualify a policy"
    return out


def _run(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=False, capture_output=True, text=True, timeout=timeout)


def collect_ownership_local(host: str = "localhost", robotics: Path | None = None) -> dict:
    lsusb = _run(["lsusb"])
    f_acm = _run(["fuser", "-v", "/dev/ttyACM0"])
    f_usb = _run(["fuser", "-v", "/dev/ttyUSB0"])
    ps = _run(["ps", "-u", os.environ.get("USER", "monomyth"), "-o", "pid,cmd"])
    crash = False
    root = robotics or Path.home() / "robotics"
    crash = (root / "logs" / "crash-investigation-20260906.md").is_file()
    return ownership_report(
        lsusb_text=lsusb.stdout + lsusb.stderr,
        fuser_acm=f_acm.stdout + f_acm.stderr,
        fuser_usb=f_usb.stdout + f_usb.stderr,
        ps_text=ps.stdout + ps.stderr,
        tty_acm=Path("/dev/ttyACM0").exists(),
        tty_usb=Path("/dev/ttyUSB0").exists(),
        crash_log=crash,
        host=host,
    )


def collect_ownership_ssh(host: str, robotics: str = "/home/monomyth/robotics") -> dict:
    remote = r"""
set +e
echo '---LSUSB---'
lsusb 2>&1
echo '---FUSER_ACM---'
fuser -v /dev/ttyACM0 2>&1
echo '---FUSER_USB---'
fuser -v /dev/ttyUSB0 2>&1
echo '---PS---'
ps -u "$USER" -o pid,cmd 2>&1
echo '---TTY---'
test -e /dev/ttyACM0 && echo ACM=1 || echo ACM=0
test -e /dev/ttyUSB0 && echo USB=1 || echo USB=0
echo '---CRASH---'
test -f ROBOTICS/logs/crash-investigation-20260906.md && echo CRASH=1 || echo CRASH=0
""".replace("ROBOTICS", robotics)
    proc = subprocess.run(
        [*ssh_base(host), "bash", "-lc", remote],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = proc.stdout + proc.stderr
    def sect(name: str) -> str:
        key = f"---{name}---"
        if key not in text:
            return ""
        rest = text.split(key, 1)[1]
        nxt = rest.find("\n---")
        return rest if nxt < 0 else rest[:nxt]
    return ownership_report(
        lsusb_text=sect("LSUSB"),
        fuser_acm=sect("FUSER_ACM"),
        fuser_usb=sect("FUSER_USB"),
        ps_text=sect("PS"),
        tty_acm="ACM=1" in sect("TTY"),
        tty_usb="USB=1" in sect("TTY"),
        crash_log="CRASH=1" in sect("CRASH"),
        host=host,
    )


def load_hw_jpeg(path: Path, name: str) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return require_size(rgb, HW_CAPTURE_WH, name)


def rig_id(serials: dict, jpeg_wh: tuple[int, int] = HW_CAPTURE_WH) -> str:
    blob = json.dumps({"serials": serials, "wh": list(jpeg_wh), "crop": "center-then-scale", "encoder": [160, 120]}, sort_keys=True)
    return RIG_ID + "-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def hop_hardware_frames(lif, overview: np.ndarray, wrist: np.ndarray, nsteps: int = 150) -> dict:
    """Crop 848×480 pair and compare table vs black. fly_picked stays false."""
    frames = {"overview": overview, "wrist": wrist}
    front, grip = encoder_pair(frames, "hardware")
    black = np.zeros_like(front)
    t0 = time.perf_counter()
    cube_r = _vision_run(lif, front, grip, nsteps)
    black_r = _vision_run(lif, black, black, nsteps)
    scr_f = phase_scramble(front, np.random.default_rng(1))
    scr_g = phase_scramble(grip, np.random.default_rng(2))
    scr_r = _vision_run(lif, scr_f, scr_g, nsteps)
    elapsed = time.perf_counter() - t0
    from arm.bus import l2, cosine

    dn_c = np.asarray(cube_r["dn_vec"], dtype=np.float64)
    dn_b = np.asarray(black_r["dn_vec"], dtype=np.float64)
    dn_s = np.asarray(scr_r["dn_vec"], dtype=np.float64)
    live = {
        "split_cameras": True,
        "cube_dn_mean": cube_r["dn_mean"],
        "empty_dn_mean": black_r["dn_mean"],
        "scramble_dn_mean": scr_r["dn_mean"],
        "l2_cube_empty": l2(dn_c, dn_b),
        "l2_cube_scramble": l2(dn_c, dn_s),
        "cos_cube_empty": cosine(dn_c, dn_b),
        "cos_cube_scramble": cosine(dn_c, dn_s),
        "cube_r1": cube_r["r1"],
        "scramble_r1": scr_r["r1"],
        "gf_hz_cube": float(cube_r["bus"][4]) if cube_r["bus"] else 0.0,
        "cube_command": cube_r["cmd"],
        "cube_is_abort": command_is_abort(cube_r["cmd"]),
        "silence_command_zero": True,
        "empty": {"dn_mean": black_r["dn_mean"]},
        "black_is_pixels": True,
        "empty_table_note": "black is zeros, not an empty-table capture",
        "elapsed_s": elapsed,
        "encoder_wh": [160, 120],
        "capture_wh": list(HW_CAPTURE_WH),
        "rig_id": RIG_ID,
    }
    live["ok"] = bool(live_go(live) and float(live["cube_dn_mean"]) >= MIN_CUBE_DN_HZ)
    live["fly_picked"] = False
    live["da_learned"] = False
    return live


def hop_from_dir(lif, folder: Path, nsteps: int = 150) -> dict:
    folder = Path(folder)
    paths = {name: folder / f"{name}.jpg" for name in HW_CAMERAS}
    missing = [n for n, p in paths.items() if not p.is_file()]
    if missing:
        raise HwPhaseError(f"missing JPEGs: {missing}")
    overview = load_hw_jpeg(paths["overview"], "overview")
    wrist = load_hw_jpeg(paths["wrist"], "wrist")
    meta_path = folder / "capture.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    hop = hop_hardware_frames(lif, overview, wrist, nsteps=nsteps)
    hop["capture"] = meta
    hop["paths"] = {k: str(v) for k, v in paths.items()}
    serials = (meta.get("serials") or {"wrist": SERIAL_WRIST, "overview": SERIAL_OVERVIEW})
    hop["rig_id"] = rig_id(serials)
    return hop


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
