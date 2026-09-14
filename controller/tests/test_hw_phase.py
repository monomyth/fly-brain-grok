from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from arm.crop import write_stub_crop
from arm.lif_crop import CropLIF
from rebot_adapter.camera_contract import CameraContractError, encoder_pair
from rebot_adapter.hw_phase import (
    USB_305,
    USB_336L,
    hop_from_dir,
    hop_hardware_frames,
    ownership_report,
    ssh_base,
)


def _lsusb_ok() -> str:
    return (
        f"Bus 006 Device 005: ID {USB_305} Orbbec Gemini 305\n"
        f"Bus 006 Device 006: ID {USB_336L} Orbbec Gemini 336L\n"
    )


def test_ownership_capture_ok_never_qualifies_policy():
    r = ownership_report(
        lsusb_text=_lsusb_ok(),
        fuser_acm="",
        fuser_usb="",
        ps_text="  1 bash\n",
        tty_acm=True,
        tty_usb=True,
        crash_log=True,
        host="isengard.local",
    )
    assert r["capture_ok"] is True
    assert r["gemini_305_usb"] is True
    assert r["gemini_336l_usb"] is True
    assert r["policy_qualify"] is False
    assert r["connect_called"] is False
    assert r["herdr_is_estop"] is False


def test_ownership_motor_process_blocks_capture():
    r = ownership_report(
        lsusb_text=_lsusb_ok(),
        fuser_acm="/dev/ttyACM0:  4321",
        fuser_usb="",
        ps_text=" 4321 python lerobot-rollout --robot.type=rebot_b601_follower\n",
        tty_acm=True,
        tty_usb=True,
        crash_log=False,
        host="isengard.local",
    )
    assert r["capture_ok"] is False
    assert r["tty_acm0_busy"] is True
    assert r["motor_processes"]
    assert r["policy_qualify"] is False


def test_ssh_base_is_argv():
    argv = ssh_base("isengard.local")
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert argv[-1] == "isengard.local"
    assert all(";" not in a and "|" not in a for a in argv)


def _orange(w=848, h=480) -> np.ndarray:
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[120:360, 200:648, 0] = 220
    rgb[120:360, 200:648, 1] = 90
    rgb[120:360, 200:648, 2] = 20
    return rgb


def test_hw_hop_cube_vs_black_stub(tmp_path):
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, nsteps=100)
    hop = hop_hardware_frames(lif, _orange(), _orange(), nsteps=100)
    assert hop["fly_picked"] is False
    assert hop["da_learned"] is False
    assert hop["capture_wh"] == [848, 480]
    assert hop["encoder_wh"] == [160, 120]
    assert float(hop["cube_dn_mean"]) > float(hop["empty_dn_mean"])
    assert float(hop["l2_cube_empty"]) > 0.0


def test_hop_from_dir_requires_named_jpegs(tmp_path):
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, nsteps=100)
    folder = tmp_path / "frames"
    folder.mkdir()
    Image.fromarray(_orange()).save(folder / "wrist.jpg")
    try:
        hop_from_dir(lif, folder, nsteps=100)
    except Exception as exc:
        assert "overview" in str(exc).lower() or "missing" in str(exc).lower()
    else:
        raise AssertionError("missing overview must fail")
    Image.fromarray(_orange()).save(folder / "overview.jpg")
    hop = hop_from_dir(lif, folder, nsteps=100)
    assert hop["fly_picked"] is False
    assert "wrist" in hop["paths"]


def test_hardware_profile_rejects_lab_names():
    try:
        encoder_pair({"Front": _orange(), "Gripper": _orange()}, "hardware")
    except CameraContractError:
        return
    raise AssertionError("lab names must not be accepted on hardware profile")
