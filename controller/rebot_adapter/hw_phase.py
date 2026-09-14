"""Phase 0 ownership and Phase 1 camera hop. USB I/O runs on the robot host."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from arm.hop_probe import MIN_CUBE_DN_HZ, _vision_run, live_go, phase_scramble
from arm.unpack import command_is_abort
from .arm_host import ssh_base
from .camera_contract import (
    HW_CAMERAS,
    HW_CAPTURE_WH,
    encoder_pair,
    pad_frac,
    pad_like,
    pad_roi_encoder,
    require_size,
)
from .ownership import (
    CAMERA_MARKERS,
    MOTOR_MARKERS,
    USB_305,
    USB_336L,
    collect_ownership_local,
    collect_ownership_ssh,
    ownership_report,
)

SERIAL_WRIST = "CV2L761000FA"
SERIAL_OVERVIEW = "CPC6463000PZ"
RIG_ID = "b601-orbbec-848x480-v0"


class HwPhaseError(RuntimeError):
    """Phase 0/1 failed closed."""


def load_hw_jpeg(path: Path, name: str) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return require_size(rgb, HW_CAPTURE_WH, name)


def rig_id(serials: dict, jpeg_wh: tuple[int, int] = HW_CAPTURE_WH) -> str:
    blob = json.dumps({"serials": serials, "wh": list(jpeg_wh), "crop": "center-then-scale", "encoder": [160, 120]}, sort_keys=True)
    return RIG_ID + "-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def hop_hardware_frames(lif, overview: np.ndarray, wrist: np.ndarray, nsteps: int = 150) -> dict:
    """Crop 848×480 pair and compare table vs black. fly_picked stays false."""
    ov_pad = pad_frac(overview)
    wr_pad = pad_frac(wrist)
    if ov_pad >= 0.12 and wr_pad < 0.5 * ov_pad:
        front = pad_roi_encoder(overview)
        grip = np.zeros_like(front)
        feed = "overview_pad"
    else:
        frames = {"overview": overview, "wrist": wrist}
        front, grip = encoder_pair(frames, "hardware")
        feed = "wrist_grip"
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
        "cube_L1": cube_r.get("L1"),
        "cube_Mi1": cube_r.get("Mi1"),
        "cube_T4": cube_r.get("T4"),
        "cube_LC10": cube_r.get("LC10"),
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
        "feed": feed,
    }
    live["ok"] = bool(live_go(live) and float(live["cube_dn_mean"]) >= MIN_CUBE_DN_HZ)
    live["fly_picked"] = False
    live["da_learned"] = False
    return live


HW_GAIN_GRID = (1.5, 1.55, 1.58, 1.6, 1.62, 1.65, 1.7, 1.8, 2.0, 2.5)


def hop_search(crop, overview: np.ndarray, wrist: np.ndarray, nsteps: int = 150, gains: tuple = HW_GAIN_GRID) -> dict:
    """First gain that live_go's vs black. Default 1.5 is silent on the dim real table."""
    from arm.lif_crop import CropLIF

    last = None
    sweep = []
    ov_pad = pad_frac(overview)
    wr_pad = pad_frac(wrist)
    pad = ov_pad >= 0.12 and wr_pad < 0.5 * ov_pad
    for g in gains:
        if pad:
            lif = CropLIF(crop, synaptic_gain=float(g), nsteps=nsteps, luma_scale=4.0, chroma_scale=0.0)
        else:
            lif = CropLIF(crop, synaptic_gain=float(g), nsteps=nsteps)
        hop = hop_hardware_frames(lif, overview, wrist, nsteps=nsteps)
        hop["synaptic_gain"] = float(g)
        sweep.append(
            {
                "gain": float(g),
                "cube_dn_mean": hop["cube_dn_mean"],
                "empty_dn_mean": hop["empty_dn_mean"],
                "gf_hz_cube": hop["gf_hz_cube"],
                "cube_is_abort": hop["cube_is_abort"],
                "ok": hop["ok"],
            }
        )
        last = hop
        if hop["ok"]:
            last["gain_sweep"] = sweep
            return last
    if last is None:
        raise HwPhaseError("gain grid is empty")
    last["gain_sweep"] = sweep
    return last


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


def hop_search_from_dir(crop, folder: Path, nsteps: int = 150) -> dict:
    folder = Path(folder)
    paths = {name: folder / f"{name}.jpg" for name in HW_CAMERAS}
    missing = [n for n, p in paths.items() if not p.is_file()]
    if missing:
        raise HwPhaseError(f"missing JPEGs: {missing}")
    overview = load_hw_jpeg(paths["overview"], "overview")
    wrist = load_hw_jpeg(paths["wrist"], "wrist")
    meta_path = folder / "capture.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    hop = hop_search(crop, overview, wrist, nsteps=nsteps)
    hop["capture"] = meta
    hop["paths"] = {k: str(v) for k, v in paths.items()}
    serials = meta.get("serials") or {"wrist": SERIAL_WRIST, "overview": SERIAL_OVERVIEW}
    hop["rig_id"] = rig_id(serials)
    return hop


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
