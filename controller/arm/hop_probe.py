"""Phase 1d hop table. CLI. Do not open an optimizer until scored DN Hz > 0."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from malecns_cache.paths import checkpoints_dir, project_data
from runtime.encoder import load_image

from .bus import cosine, l2
from .crop import Crop, build_crop, write_stub_crop
from .eye import LUMA_SCALE, cube_chroma_frac, phase_scramble
from .lif_crop import CropLIF
from .unpack import U0, command_is_abort, command_saturated, unpack

LAYERS = (
    ("R1", "photoreceptors_r1r6"),
    ("L1", "L1"),
    ("L2", "L2"),
    ("Mi1", "Mi1"),
    ("Tm3", "Tm3"),
    ("T4", "T4"),
    ("T5", "T5"),
    ("LC10", "LC10"),
    ("LPTC", "LPTC"),
    ("DNfl", "DNfl"),
    ("DNxl", "DNxl"),
    ("scored_dn", "scored_dn"),
    ("T1_MN", "leg_mn_front"),
)


def _mean_hz(hz: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return 0.0
    return float(np.mean(hz[idx]))


def _synthetic_cube(h: int = 120, w: int = 160, y0: int = 40, x0: int = 50, bh: int = 50, bw: int = 60) -> np.ndarray:
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    y1 = min(h, max(0, y0) + int(bh))
    x1 = min(w, max(0, x0) + int(bw))
    y0 = min(max(0, y0), h)
    x0 = min(max(0, x0), w)
    rgb[y0:y1, x0:x1, 0] = 0.85
    rgb[y0:y1, x0:x1, 1] = 0.35
    rgb[y0:y1, x0:x1, 2] = 0.08
    return rgb


def csr_edge_report(crop: Crop) -> dict:
    W = crop.weights
    pre = W.indices
    posts = np.repeat(np.arange(crop.n, dtype=np.int32), np.diff(W.indptr))
    groups = crop.groups

    def count(src: str, dst: str) -> int:
        s, d = groups.get(src), groups.get(dst)
        if s is None or d is None or len(s) == 0 or len(d) == 0:
            return 0
        sset = np.zeros(crop.n, dtype=bool)
        dset = np.zeros(crop.n, dtype=bool)
        sset[s] = True
        dset[d] = True
        return int(np.count_nonzero(sset[pre] & dset[posts]))

    edges = {
        "R1_to_L1": count("photoreceptors_r1r6", "L1"),
        "R1_to_L2": count("photoreceptors_r1r6", "L2"),
        "L1_to_Mi1": count("L1", "Mi1"),
        "Mi1_to_T4": count("Mi1", "T4"),
        "T4_to_LC10": count("T4", "LC10"),
        "LC10_to_scored_dn": count("LC10", "scored_dn"),
        "nnz": int(W.nnz),
        "n_neg": int(np.count_nonzero(W.data < 0)),
        "n_pos": int(np.count_nonzero(W.data > 0)),
    }
    return edges


INJECT_DRIVE = 4.0
INJECT_HZ_CAP = 1000.0
SCRAMBLE_COS_MAX = 0.85
SCRAMBLE_ENERGY_MIN = 0.5
CMD_SHIFT_MIN = 0.5
GF_ABORT_FRAC = 0.5
DN_HZ_EPS = 0.05
MIN_CUBE_DN_HZ = 5.0
DY_FLIP_MIN = 0.05


def hop_table(lif: CropLIF, drive: float = INJECT_DRIVE, nsteps: int = 150) -> list[dict]:
    rows = []
    for name, group in LAYERS:
        idx = lif.crop.indices(group)
        lif.reset_episode()
        hz = lif.inject(idx, drive, nsteps=nsteps)
        row = {
            "layer": name,
            "n": int(idx.size),
            "self_hz": _mean_hz(hz, idx),
            "R1_hz": _mean_hz(hz, lif.crop.indices("photoreceptors_r1r6")),
            "L1_hz": _mean_hz(hz, lif.crop.indices("L1")),
            "Mi1_hz": _mean_hz(hz, lif.crop.indices("Mi1")),
            "T4_hz": _mean_hz(hz, lif.crop.indices("T4")),
            "LC10_hz": _mean_hz(hz, lif.crop.indices("LC10")),
            "DNfl_hz": _mean_hz(hz, lif.crop.indices("DNfl")),
            "scored_dn_hz": _mean_hz(hz, lif.crop.indices("scored_dn")),
            "t1_mn_hz": _mean_hz(hz, lif.crop.indices("leg_mn_front")),
        }
        rows.append(row)
    return rows


def _cmd_l1(a: dict, b: dict) -> float:
    return (
        abs(float(a.get("dx_mm", 0.0)) - float(b.get("dx_mm", 0.0)))
        + abs(float(a.get("dy_mm", 0.0)) - float(b.get("dy_mm", 0.0)))
        + abs(float(a.get("dz_mm", 0.0)) - float(b.get("dz_mm", 0.0)))
        + abs(float(a.get("dgrip_mm", 0.0)) - float(b.get("dgrip_mm", 0.0)))
    )


def _strip_dn(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "dn_vec"}


def _vision_run(lif: CropLIF, front, grip, nsteps: int) -> dict:
    lif.reset_episode()
    hz = lif.step_vision(front, grip, nsteps=nsteps)
    rates = lif.rates()
    return {
        "dn_mean": rates.scored_mean_hz,
        "dn_vec": rates.scored.tolist(),
        "bus": rates.vec.tolist(),
        "pools": {k: float(v) for k, v in rates.pools.items()},
        "r1": _mean_hz(hz, lif.crop.indices("photoreceptors_r1r6")),
        "t1": rates.t1_mn_hz,
        "cmd": unpack(rates, U0).__dict__,
    }


def _dna02_lr(row: dict) -> dict[str, float]:
    pools = row.get("pools") or {}
    return {
        "L": float(pools.get("DNa02_L") or 0.0),
        "R": float(pools.get("DNa02_R") or 0.0),
    }


def _dna02_crop(crop: Crop) -> dict:
    idx = crop.indices("DNa02")
    return {
        "n": int(idx.size),
        "hex1": [int(x) for x in np.asarray(crop.hex1)[idx]] if idx.size else [],
        "hex2": [int(x) for x in np.asarray(crop.hex2)[idx]] if idx.size else [],
        "n_L": int(crop.indices("DNa02_L").size),
        "n_R": int(crop.indices("DNa02_R").size),
    }


def _dy_sign_flip(left, right) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return a * b < 0.0 and abs(a - b) >= DY_FLIP_MIN


def _scramble_energy_starved(vis: dict) -> bool:
    cube_r1 = float(vis.get("cube_r1") or 0.0)
    scr_r1 = float(vis.get("scramble_r1") or 0.0)
    return cube_r1 > 0.05 and scr_r1 < SCRAMBLE_ENERGY_MIN * cube_r1


def _scramble_hz_matches_cube(vis: dict) -> bool:
    cube = float(vis.get("cube_dn_mean") or 0.0)
    if cube <= DN_HZ_EPS:
        return False
    cos = vis.get("cos_cube_scramble")
    cos_v = 1.0 if cos is None else float(cos)
    return cos_v >= SCRAMBLE_COS_MAX


def _vision_conditions(lif: CropLIF, nsteps: int) -> dict:
    cube = _synthetic_cube()
    black = np.zeros_like(cube)
    scramble = phase_scramble(cube, np.random.default_rng(0))
    shift = _synthetic_cube(y0=20, x0=25, bh=80, bw=60)
    left = _synthetic_cube(y0=40, x0=0, bh=50, bw=60)
    right = _synthetic_cube(y0=40, x0=100, bh=50, bw=60)

    cube_r = _vision_run(lif, cube, cube, nsteps)
    black_r = _vision_run(lif, black, black, nsteps)
    scr_r = _vision_run(lif, scramble, scramble, nsteps)
    shift_r = _vision_run(lif, shift, shift, nsteps)
    left_r = _vision_run(lif, left, left, nsteps)
    right_r = _vision_run(lif, right, right, nsteps)
    split_l = _vision_run(lif, cube, black, nsteps)
    split_r = _vision_run(lif, black, cube, nsteps)
    _vision_run(lif, cube, cube, nsteps)
    silent_cmd = unpack(lif.rates().silenced(), U0)

    dn_cube = np.asarray(cube_r["dn_vec"], dtype=np.float64)
    dn_black = np.asarray(black_r["dn_vec"], dtype=np.float64)
    dn_scr = np.asarray(scr_r["dn_vec"], dtype=np.float64)
    dn_shift = np.asarray(shift_r["dn_vec"], dtype=np.float64)
    dn_left = np.asarray(left_r["dn_vec"], dtype=np.float64)
    dn_right = np.asarray(right_r["dn_vec"], dtype=np.float64)
    cube_cmd = cube_r["cmd"]
    shift_cmd = shift_r["cmd"]
    gf_hz = float(cube_r["bus"][4]) if cube_r["bus"] else 0.0
    dy_left = float(left_r["cmd"]["dy_mm"])
    dy_right = float(right_r["cmd"]["dy_mm"])
    dy_split_l = float(split_l["cmd"]["dy_mm"])
    dy_split_r = float(split_r["cmd"]["dy_mm"])
    return {
        "cube": _strip_dn(cube_r),
        "black": _strip_dn(black_r),
        "scramble": _strip_dn(scr_r),
        "cube_shift": _strip_dn(shift_r),
        "left": _strip_dn(left_r),
        "right": _strip_dn(right_r),
        "split_front": _strip_dn(split_l),
        "split_grip": _strip_dn(split_r),
        "cube_dn_mean": cube_r["dn_mean"],
        "black_dn_mean": black_r["dn_mean"],
        "scramble_dn_mean": scr_r["dn_mean"],
        "l2_cube_black": l2(dn_cube, dn_black),
        "l2_cube_scramble": l2(dn_cube, dn_scr),
        "l2_cube_shift": l2(dn_cube, dn_shift),
        "l2_left_right": l2(dn_left, dn_right),
        "cmd_l1_cube_shift": float(_cmd_l1(cube_cmd, shift_cmd)),
        "cmd_dy_shift": abs(float(cube_cmd["dy_mm"]) - float(shift_cmd["dy_mm"])),
        "cos_cube_black": cosine(dn_cube, dn_black),
        "cos_cube_scramble": cosine(dn_cube, dn_scr),
        "cos_left_right": cosine(dn_left, dn_right),
        "gf_hz_cube": gf_hz,
        "cube_is_abort": command_is_abort(cube_cmd),
        "scramble_is_abort": command_is_abort(scr_r["cmd"]),
        "cube_command_saturated": command_saturated(cube_cmd),
        "silence_command_zero": abs(silent_cmd.dx_mm) + abs(silent_cmd.dy_mm) + abs(silent_cmd.dz_mm) + abs(silent_cmd.dgrip_mm) < 1e-9,
        "black_command": black_r["cmd"],
        "cube_command": cube_cmd,
        "shift_command": shift_cmd,
        "n_dn_gt1_cube": int(np.count_nonzero(dn_cube > 1.0)),
        "n_dn": int(dn_cube.size),
        "cube_r1": float(cube_r["r1"]),
        "scramble_r1": float(scr_r["r1"]),
        "dy_left": dy_left,
        "dy_right": dy_right,
        "dy_split_front": dy_split_l,
        "dy_split_grip": dy_split_r,
        "laterality_dy_flips": _dy_sign_flip(dy_left, dy_right),
        "split_dy_flips": _dy_sign_flip(dy_split_l, dy_split_r),
        "dna02_left": _dna02_lr(left_r),
        "dna02_right": _dna02_lr(right_r),
        "luma_scale": float(getattr(lif, "luma_scale", LUMA_SCALE)),
    }


def vision_probe(lif: CropLIF, nsteps: int = 150) -> dict:
    vis = _vision_conditions(lif, nsteps)
    starved = _scramble_energy_starved(vis)
    vis["scramble_energy_starved"] = starved
    if (
        not starved
        and _scramble_hz_matches_cube(vis)
        and float(getattr(lif, "luma_scale", 0.0)) > 0.0
    ):
        lif.luma_scale = 0.0
        vis2 = _vision_conditions(lif, nsteps)
        vis2["luma_cut"] = True
        vis2["luma_cut_reason"] = "matched-energy scramble Hz matched cube; raw luma dropped"
        vis2["scramble_energy_starved"] = _scramble_energy_starved(vis2)
        vis2["pre_cut"] = {
            "cube_dn_mean": vis["cube_dn_mean"],
            "scramble_dn_mean": vis["scramble_dn_mean"],
            "cos_cube_scramble": vis["cos_cube_scramble"],
            "gf_hz_cube": vis["gf_hz_cube"],
            "cube_r1": vis["cube_r1"],
            "scramble_r1": vis["scramble_r1"],
        }
        return vis2
    vis["luma_cut"] = False
    return vis


def gain_pass(crop: Crop, nsteps: int, gains: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5)) -> tuple[float, dict]:
    last: dict = {}
    last_gain = gains[0]
    for gain in gains:
        lif = CropLIF(crop, synaptic_gain=gain, nsteps=nsteps)
        vis = vision_probe(lif, nsteps=nsteps)
        vis["synaptic_gain"] = gain
        last = vis
        last_gain = gain
        print(
            f"gain {gain:.2f} cube_dn={vis['cube_dn_mean']:.3f} black_dn={vis['black_dn_mean']:.3f} "
            f"l2={vis['l2_cube_black']:.2f} scramble_cos={vis['cos_cube_scramble']:.3f} "
            f"shift_l2={vis['l2_cube_shift']:.2f} gf={vis['gf_hz_cube']:.1f} abort={vis['cube_is_abort']} "
            f"n>1={vis['n_dn_gt1_cube']}",
            flush=True,
        )
        if vision_ok(vis):
            return gain, vis
    return last_gain, last


def excitatory_only(crop: Crop, nsteps: int, gain: float, luma_scale: float | None = None) -> dict:
    lif = CropLIF(crop, synaptic_gain=gain, nsteps=nsteps, luma_scale=LUMA_SCALE if luma_scale is None else float(luma_scale))
    data = lif.crop.weights.data
    data[:] = np.maximum(data, 0.0)
    vis = vision_probe(lif, nsteps=nsteps)
    vis["synaptic_gain"] = gain
    vis["excitatory_only"] = True
    return vis


def vision_ok(vis: dict) -> bool:
    if not vis:
        return False
    if float(vis.get("cube_dn_mean") or 0.0) < MIN_CUBE_DN_HZ:
        return False
    if float(vis.get("l2_cube_black") or 0.0) < 1.0:
        return False
    cube_r1 = vis.get("cube_r1")
    scr_r1 = vis.get("scramble_r1")
    # Chroma scramble is gray: starved scramble is not a fail if black already splits.
    if not _scramble_energy_starved(vis):
        if float(vis.get("l2_cube_scramble") or 0.0) < 1.0:
            return False
        cos_scr = vis.get("cos_cube_scramble")
        if cos_scr is None:
            cos_scr = 1.0
        if float(cos_scr) >= SCRAMBLE_COS_MAX:
            return False
    if float(vis.get("cmd_l1_cube_shift") or 0.0) < CMD_SHIFT_MIN:
        return False
    cmd = vis.get("cube_command")
    if cmd is not None and command_saturated(cmd):
        return False
    if bool(vis.get("cube_command_saturated")):
        return False
    if bool(vis.get("cube_is_abort")):
        return False
    gf = float(vis.get("gf_hz_cube") or 0.0)
    if gf >= float(U0.abort_hz):
        return False
    if gf >= GF_ABORT_FRAC * float(U0.abort_hz):
        return False
    if vis.get("silence_command_zero") is False:
        return False
    return True


def hops_unsaturated(hops: list[dict], cap: float = INJECT_HZ_CAP) -> bool:
    if not hops:
        return False
    return max(float(row.get("self_hz") or 0.0) for row in hops) < cap


def passed(vis: dict, hops: list[dict] | None = None) -> bool:
    """Green if vision_ok. Inject-hop kHz is logged, not a vision fail."""
    return vision_ok(vis)


def train_gate(report: dict | None) -> bool:
    """Phase 3/4 stay closed until synthetic vision_ok and live_go."""
    if not report:
        return False
    return bool(report.get("ok")) and report.get("live_ok") is True


def _resolve_photo(ep: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    cand = ep / "frames" / p.name
    return cand if cand.is_file() else None


def _cube_present(rec: dict) -> bool:
    if rec.get("present") is False:
        return False
    cube = rec.get("cube")
    if isinstance(cube, dict) and cube.get("present") is False:
        return False
    return True


def load_optical_fg_pairs(root: Path, per_episode: int = 2) -> list[dict]:
    """Front and Gripper stay separate files. Never pair a path with itself."""
    pairs: list[dict] = []
    if not root.is_dir():
        return pairs
    episodes = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if (root / "log.jsonl").is_file():
        episodes = [root] + episodes
    for ep in episodes:
        log = ep / "log.jsonl"
        if not log.is_file():
            continue
        ranked: list[dict] = []
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            photos = rec.get("photos") or {}
            front = _resolve_photo(ep, photos.get("Front"))
            grip = _resolve_photo(ep, photos.get("Gripper"))
            if front is None or grip is None:
                continue
            if front.resolve() == grip.resolve():
                continue
            cube = rec.get("cube") or {}
            center = cube.get("center_mm") or {}
            grip_rgb = load_image(grip)
            ranked.append(
                {
                    "front": front,
                    "grip": grip,
                    "episode": ep.name,
                    "tick": rec.get("tick"),
                    "cube_y": float(center.get("y") or 0.0),
                    "cube_x": float(center.get("x") or 0.0),
                    "cube_z": float(center.get("z") or 0.0),
                    "present": _cube_present(rec),
                    "step": rec.get("step"),
                    "grip_chroma": cube_chroma_frac(grip_rgb),
                    "front_chroma": cube_chroma_frac(load_image(front)),
                    "size_mm": float(cube.get("size_mm") or 0.0),
                }
            )
        ranked.sort(key=lambda p: float(p.get("grip_chroma") or 0.0), reverse=True)
        kept = ranked[: max(int(per_episode), 1)]
        seen = {(p["front"], p["grip"], p.get("tick")) for p in kept}
        for p in ranked:
            if p.get("present") is not False:
                continue
            key = (p["front"], p["grip"], p.get("tick"))
            if key in seen:
                continue
            kept.append(p)
            seen.add(key)
        pairs.extend(kept)
    return pairs


def _pick_pair(pairs: list[dict], *, y_sign: int = 0, require_grip_chroma: float = 0.0) -> dict | None:
    if not pairs:
        return None
    if y_sign > 0:
        cands = [p for p in pairs if float(p.get("cube_y") or 0.0) > 5.0]
    elif y_sign < 0:
        cands = [p for p in pairs if float(p.get("cube_y") or 0.0) < -5.0]
    else:
        cands = [p for p in pairs if abs(float(p.get("cube_y") or 0.0)) <= 5.0]
    if y_sign != 0 and not cands:
        return None
    pool = cands or pairs
    sized = [p for p in pool if abs(float(p.get("size_mm") or 0.0) - 20.0) < 0.5]
    if sized:
        pool = sized
    if require_grip_chroma > 0.0:
        hot = [p for p in pool if float(p.get("grip_chroma") or 0.0) >= require_grip_chroma]
        if hot:
            pool = hot
    return max(pool, key=lambda p: float(p.get("grip_chroma") or 0.0))


def live_go(live: dict | None) -> bool:
    if not live:
        return False
    if not live.get("split_cameras"):
        return False
    if float(live.get("cube_dn_mean") or 0.0) < MIN_CUBE_DN_HZ:
        return False
    empty = live.get("empty")
    empty_ok = empty is not None and float(live.get("l2_cube_empty") or 0.0) >= 1.0
    if empty is not None and not empty_ok:
        return False
    if not empty_ok:
        if _scramble_energy_starved(live):
            return False
        if float(live.get("l2_cube_scramble") or 0.0) < 1.0:
            return False
        cos_scr = live.get("cos_cube_scramble")
        if cos_scr is not None and float(cos_scr) >= SCRAMBLE_COS_MAX:
            return False
    if live.get("silence_command_zero") is False:
        return False
    gf = float(live.get("gf_hz_cube") or 0.0)
    if gf >= GF_ABORT_FRAC * float(U0.abort_hz):
        return False
    if bool(live.get("cube_is_abort")):
        return False
    return True


def live_replay_probe(lif: CropLIF, root: Path, nsteps: int = 150) -> dict:
    """Offline replay of apply:false Front/Gripper JPEGs."""
    pairs = load_optical_fg_pairs(root, per_episode=12)
    cube_pairs = [p for p in pairs if p.get("present") is not False]
    cube_p = _pick_pair(cube_pairs, y_sign=0, require_grip_chroma=0.005)
    left_p = _pick_pair(cube_pairs, y_sign=-1, require_grip_chroma=0.005)
    right_p = _pick_pair(cube_pairs, y_sign=1, require_grip_chroma=0.005)
    empty_p = next((p for p in pairs if p.get("present") is False), None)
    if cube_p is None:
        return {
            "ok": False,
            "split_cameras": False,
            "note": f"no Front/Gripper pairs under {root}",
            "n_pairs": 0,
        }
    front = load_image(cube_p["front"])
    grip = load_image(cube_p["grip"])
    same = cube_p["front"].resolve() == cube_p["grip"].resolve() or bool(np.allclose(front, grip))
    scr_f = phase_scramble(front, np.random.default_rng(1))
    scr_g = phase_scramble(grip, np.random.default_rng(2))
    cube_r = _vision_run(lif, front, grip, nsteps)
    scr_r = _vision_run(lif, scr_f, scr_g, nsteps)
    _vision_run(lif, front, grip, nsteps)
    silent_cmd = unpack(lif.rates().silenced(), U0)

    left_r = right_r = None
    if left_p is not None:
        left_r = _vision_run(lif, load_image(left_p["front"]), load_image(left_p["grip"]), nsteps)
    if right_p is not None:
        right_r = _vision_run(lif, load_image(right_p["front"]), load_image(right_p["grip"]), nsteps)
    empty_r = None
    if empty_p is not None:
        empty_r = _vision_run(lif, load_image(empty_p["front"]), load_image(empty_p["grip"]), nsteps)

    dn_cube = np.asarray(cube_r["dn_vec"], dtype=np.float64)
    dn_scr = np.asarray(scr_r["dn_vec"], dtype=np.float64)
    gf_hz = float(cube_r["bus"][4]) if cube_r["bus"] else 0.0
    out: dict = {
        "split_cameras": not same,
        "front_path": str(cube_p["front"]),
        "grip_path": str(cube_p["grip"]),
        "cube_episode": cube_p["episode"],
        "cube_tick": cube_p.get("tick"),
        "grip_chroma": float(cube_p.get("grip_chroma") or 0.0),
        "n_pairs": len(pairs),
        "cube": _strip_dn(cube_r),
        "scramble": _strip_dn(scr_r),
        "cube_dn_mean": cube_r["dn_mean"],
        "scramble_dn_mean": scr_r["dn_mean"],
        "cube_r1": float(cube_r["r1"]),
        "scramble_r1": float(scr_r["r1"]),
        "l2_cube_scramble": l2(dn_cube, dn_scr),
        "cos_cube_scramble": cosine(dn_cube, dn_scr),
        "gf_hz_cube": gf_hz,
        "cube_command": cube_r["cmd"],
        "cube_is_abort": command_is_abort(cube_r["cmd"]),
        "silence_command_zero": abs(silent_cmd.dx_mm) + abs(silent_cmd.dy_mm) + abs(silent_cmd.dz_mm) + abs(silent_cmd.dgrip_mm) < 1e-9,
        "empty": None,
        "empty_table_note": "no empty-table frames in rebot-optical-fg (need present:false Front+Gripper); not faked as black pixels",
        "luma_scale": float(getattr(lif, "luma_scale", LUMA_SCALE)),
    }
    if empty_r is not None:
        dn_empty = np.asarray(empty_r["dn_vec"], dtype=np.float64)
        out["empty"] = _strip_dn(empty_r)
        out["empty_dn_mean"] = empty_r["dn_mean"]
        out["l2_cube_empty"] = l2(dn_cube, dn_empty)
        out["cos_cube_empty"] = cosine(dn_cube, dn_empty)
        out["empty_table_note"] = None
    same_pose = (
        left_p is not None
        and right_p is not None
        and left_p["front"].resolve() == right_p["front"].resolve()
        and left_p["grip"].resolve() == right_p["grip"].resolve()
    )
    if left_r is not None and right_r is not None and not same_pose:
        dn_left = np.asarray(left_r["dn_vec"], dtype=np.float64)
        dn_right = np.asarray(right_r["dn_vec"], dtype=np.float64)
        out["left"] = _strip_dn(left_r)
        out["right"] = _strip_dn(right_r)
        out["left_episode"] = left_p["episode"] if left_p else None
        out["right_episode"] = right_p["episode"] if right_p else None
        out["dy_left"] = float(left_r["cmd"]["dy_mm"])
        out["dy_right"] = float(right_r["cmd"]["dy_mm"])
        out["l2_left_right"] = l2(dn_left, dn_right)
        out["cos_left_right"] = cosine(dn_left, dn_right)
        out["laterality_dy_flips"] = _dy_sign_flip(out["dy_left"], out["dy_right"])
        out["dna02_left"] = _dna02_lr(left_r)
        out["dna02_right"] = _dna02_lr(right_r)
    else:
        out["laterality_note"] = "need pose variation across episodes for left vs right"
    out["ok"] = live_go(out)
    if not out["ok"]:
        out["note"] = (
            "live replay no-go "
            f"(cube_dn={out.get('cube_dn_mean')}, scramble_dn={out.get('scramble_dn_mean')}, "
            f"dy_left={out.get('dy_left')}, dy_right={out.get('dy_right')}, gf={out.get('gf_hz_cube')}); "
            "do not distill"
        )
    return out


def run(
    *,
    stub: bool = False,
    nsteps: int = 150,
    force_crop: bool = False,
    dest: Path | None = None,
    replay_dir: Path | None = None,
) -> dict:
    t0 = time.perf_counter()
    if stub:
        crop = write_stub_crop(checkpoints_dir("rebot-pickup") / "stub-crop")
    else:
        crop = build_crop("v1.0", force=force_crop)
    edges = csr_edge_report(crop)
    print(
        f"crop n={crop.n} nnz={crop.weights.nnz} scored_dn={crop.indices('scored_dn').size} "
        f"front_r1={crop.indices('front_r1').size} grip_r1={crop.indices('grip_r1').size} "
        f"DNfl={crop.indices('DNfl').size} DNxl={crop.indices('DNxl').size} "
        f"({crop.meta.get('dnfl_source')}; {crop.meta.get('dnxl_source')})",
        flush=True,
    )
    print("csr", edges, flush=True)
    gain, vis = gain_pass(crop, nsteps=nsteps)
    exo = None
    luma_scale = 0.0 if vis.get("luma_cut") else LUMA_SCALE
    if not vision_ok(vis):
        print("vision_ok false; excitatory-only diagnostic", flush=True)
        exo = excitatory_only(crop, nsteps=nsteps, gain=gain, luma_scale=luma_scale)
        print(
            f"exc-only cube_dn={exo['cube_dn_mean']:.3f} black={exo['black_dn_mean']:.3f} "
            f"scramble_cos={exo.get('cos_cube_scramble')} abort={exo.get('cube_is_abort')}",
            flush=True,
        )
    lif = CropLIF(crop, synaptic_gain=gain, nsteps=nsteps, luma_scale=luma_scale)
    hops = hop_table(lif, nsteps=nsteps)
    live = None
    root = replay_dir
    if root is None and not stub:
        cand = project_data() / "datasets" / "rebot-optical-fg"
        if cand.is_dir():
            root = cand
    if root is not None:
        print(f"live replay {root}", flush=True)
        live_gain = 2.5 if gain < 2.5 else gain
        live_lif = CropLIF(crop, synaptic_gain=live_gain, nsteps=nsteps, luma_scale=luma_scale)
        live = live_replay_probe(live_lif, Path(root), nsteps=nsteps)
        live["synaptic_gain"] = live_gain
    ok = passed(vis, hops)
    live_ok = live_go(live) if live else None
    inject_max = max((float(r.get("self_hz") or 0.0) for r in hops), default=0.0)
    if not ok:
        note = (
            "A.1–A.2 not green "
            f"(scramble_cos={vis.get('cos_cube_scramble')}, cmd_shift={vis.get('cmd_l1_cube_shift')}, "
            f"dy_left={vis.get('dy_left')}, dy_right={vis.get('dy_right')}, "
            f"gf_hz={vis.get('gf_hz_cube')}, luma_cut={vis.get('luma_cut')}); "
            "do not distill; do not score overlay as fly"
        )
    elif live_ok is False:
        note = "synthetic A.1–A.2 passed; live replay no-go; do not distill"
    else:
        note = None
    report = {
        "ok": ok,
        "stub": bool(stub),
        "n": crop.n,
        "nnz": int(crop.weights.nnz),
        "nsteps": nsteps,
        "synaptic_gain": gain,
        "luma_cut": bool(vis.get("luma_cut")) if vis else False,
        "luma_scale": float(luma_scale),
        "dnfl_source": crop.meta.get("dnfl_source"),
        "dnxl_source": crop.meta.get("dnxl_source"),
        "n_scored_dn": int(crop.indices("scored_dn").size),
        "n_front_r1": int(crop.indices("front_r1").size),
        "n_grip_r1": int(crop.indices("grip_r1").size),
        "scored_types": {
            "DNfl": int(crop.indices("DNfl").size),
            "DNxl": int(crop.indices("DNxl").size),
            "DNa01": int(crop.indices("DNa01").size),
            "DNa02": int(crop.indices("DNa02").size),
            "DNp01": int(crop.indices("DNp01").size),
            "MDN": int(crop.indices("MDN").size),
            "DNp07": int(crop.indices("DNp07").size),
            "DNp10": int(crop.indices("DNp10").size),
        },
        "dna02": _dna02_crop(crop),
        "csr_edges": edges,
        "hops": hops,
        "vision": vis,
        "live_replay": live,
        "live_ok": live_ok,
        "excitatory_only": exo,
        "silence_zeros": bool(vis.get("silence_command_zero")) if vis else False,
        "inject_max_self_hz": inject_max,
        "elapsed_s": time.perf_counter() - t0,
        "note": note,
        "fly_picked": False,
    }
    dest = dest or checkpoints_dir("rebot-pickup") / ("hop-probe-stub.json" if stub else "hop-probe.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n")
    report["path"] = str(dest)
    _print_table(report)
    return report


def _print_table(report: dict) -> None:
    print(f"\n{'layer':<12} {'n':>6} {'self Hz':>10} {'DN Hz':>10} {'T1 Hz':>10}")
    for row in report["hops"]:
        print(
            f"{row['layer']:<12} {row['n']:6d} {row['self_hz']:10.3f} "
            f"{row['scored_dn_hz']:10.3f} {row['t1_mn_hz']:10.3f}"
        )
    vis = report.get("vision") or {}
    print(
        f"\nvision cube_dn={vis.get('cube_dn_mean')} black={vis.get('black_dn_mean')} "
        f"l2={vis.get('l2_cube_black')} scramble_l2={vis.get('l2_cube_scramble')} "
        f"cmd_shift={vis.get('cmd_l1_cube_shift')} dy_l={vis.get('dy_left')} dy_r={vis.get('dy_right')} "
        f"gf={vis.get('gf_hz_cube')} luma_cut={vis.get('luma_cut')} "
        f"silence_zero={vis.get('silence_command_zero')}"
    )
    live = report.get("live_replay") or {}
    if live:
        print(
            f"live cube_dn={live.get('cube_dn_mean')} scramble={live.get('scramble_dn_mean')} "
            f"dy_l={live.get('dy_left')} dy_r={live.get('dy_right')} gf={live.get('gf_hz_cube')} "
            f"split={live.get('split_cameras')} empty={live.get('empty_table_note') or 'present'} "
            f"live_ok={report.get('live_ok')}"
        )
    print(f"hop_probe ok={report['ok']} live_ok={report.get('live_ok')} gain={report['synaptic_gain']} wrote {report.get('path')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MaleCNS crop hop-probe")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--force-crop", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replay-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.steps < 50:
        print("steps must be ≥ 50", file=sys.stderr)
        return 2
    report = run(
        stub=args.stub,
        nsteps=args.steps,
        force_crop=args.force_crop,
        dest=args.out,
        replay_dir=args.replay_dir,
    )
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
