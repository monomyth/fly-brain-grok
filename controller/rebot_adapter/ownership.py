"""Phase 0 device ownership. Stdlib only — no MaleCNS, no pyarrow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .arm_host import ssh_base

USB_305 = "2bc5:0840"
USB_336L = "2bc5:0807"
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
