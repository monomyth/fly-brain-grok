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
    hop_search,
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


def test_sync_rsync_argv_is_not_can():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "hw_sync.py"
    spec = importlib.util.spec_from_file_location("hw_sync", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cmds = mod.rsync_argv("isengard.local", "/home/monomyth/fly-brain-grok")
    flat = " ".join(tok for cmd in cmds for tok in cmd)
    assert "rsync" in flat
    assert "/dev/ttyACM0" not in flat
    assert all(cmd[0] == "rsync" for cmd in cmds)
    assert any(tok.endswith(".json") for cmd in cmds for tok in cmd)


def test_hop_search_wrist_only_cube_on_plant(tmp_path):
    from rebot_adapter.b601 import B601Plant

    crop = write_stub_crop(tmp_path / "crop")
    plant = B601Plant.ready()
    frames = plant.capture_all()
    hop = hop_search(crop, frames["overview"].rgb, frames["wrist"].rgb, nsteps=100, gains=(1.5, 2.5))
    assert hop["fly_picked"] is False
    assert hop["ok"] is True
    assert float(hop["cube_dn_mean"]) > float(hop["empty_dn_mean"])


def test_hop_search_picks_working_gain_on_stub(tmp_path):
    crop = write_stub_crop(tmp_path / "crop")
    hop = hop_search(crop, _orange(), _orange(), nsteps=100, gains=(1.5, 2.5))
    assert hop["ok"] is True
    assert hop["fly_picked"] is False
    assert hop["synaptic_gain"] == 1.5
    assert hop["gain_sweep"][0]["ok"] is True


def test_hardware_profile_rejects_lab_names():
    try:
        encoder_pair({"Front": _orange(), "Gripper": _orange()}, "hardware")
    except CameraContractError:
        return
    raise AssertionError("lab names must not be accepted on hardware profile")
