"""Phase 3: distill overlay teacher into crop gains g. Teacher is not in the scored path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from malecns_cache.paths import checkpoints_dir, datasets_dir
from runtime.encoder import load_image

from train.expert import PAD_DEPTH_MM, PAD_Z_MM

from .crop import GAIN_CLASSES, build_crop, write_stub_crop
from .hop_probe import train_gate
from .lif_crop import CropLIF
from .log import g_hash, write_json, write_skip_checkpoint
from .score import ActingMap
from .teacher import teacher_from_state
from .unpack import U0, UParams, command_is_abort, unpack

ABORT_PEN = 250.0
U_LO = np.array([0.5, 0.5, 0.5, 0.5, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, 0.0], dtype=np.float64)
U_HI = np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0], dtype=np.float64)


def _load_hop(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def teacher_frames(root: Path, limit: int | None = 8) -> list[dict]:
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for ep in sorted(p for p in root.iterdir() if p.is_dir()):
        log = ep / "log.jsonl"
        if not log.is_file():
            continue
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            photos = rec.get("photos") or {}
            front = photos.get("Front")
            grip = photos.get("Gripper")
            if not front or not grip:
                continue
            fp, gp = Path(front), Path(grip)
            if not fp.is_file():
                fp = ep / "frames" / Path(front).name
            if not gp.is_file():
                gp = ep / "frames" / Path(grip).name
            if not fp.is_file() or not gp.is_file():
                continue
            rec["_front"] = fp
            rec["_grip"] = gp
            rec["_dir"] = ep
            rows.append(rec)
    if limit is None or limit < 0:
        return rows
    return rows[: max(int(limit), 0)]


def _state_from_log(rec: dict) -> dict:
    cube = rec.get("cube") or {"attached": False, "center_mm": {"x": 280, "y": 0, "z": 10}, "size_mm": 20}
    return {
        "tcp_mm": rec.get("tcp_mm") or {"x": 280.0, "y": 0.0, "z": 80.0},
        "gripper_mm": rec.get("gripper_mm") or 90.0,
        "tcp_rpy_deg": rec.get("tcp_rpy_deg") or {"yaw": 0.0},
        "objects": {"cube": cube},
    }


def _live_pad_lock(rec: dict) -> tuple[float, float, float]:
    cube = rec.get("cube") or {}
    center = cube.get("center_mm") or {}
    return (
        float(center.get("x") or 280.0) - PAD_DEPTH_MM,
        float(center.get("y") or 0.0),
        PAD_Z_MM,
    )


def _frame_size_mm(rec: dict) -> float:
    cube = rec.get("cube") or {}
    return float(cube.get("size_mm") or 20.0)


def _is_present(rec: dict) -> bool:
    cube = rec.get("cube") or {}
    if rec.get("present") is False or cube.get("present") is False:
        return False
    return True


def _is_pad_row(rec: dict) -> bool:
    step = str(rec.get("step") or "")
    z = float((rec.get("tcp_mm") or {}).get("z") or 99.0)
    return "pad" in step or abs(z - PAD_Z_MM) < 8.0 or step in {"close", "pinch", "align"}


def _is_lift_row(rec: dict) -> bool:
    """Attached pinch with jaws near the solid, still below hold height."""
    cube = rec.get("cube") or {}
    if not bool(cube.get("attached")):
        return False
    if float(rec.get("gripper_mm") or 90.0) >= 75.0:
        return False
    z = float((rec.get("tcp_mm") or {}).get("z") or 0.0)
    return z <= 150.0


def distill_frames(limit: int = 8) -> list[dict]:
    """Pad/close plus attached lift ticks so U can learn +Z after pinch."""
    rows: list[dict] = []
    for name in ("rebot-optical-fg", "rebot-teacher-fg"):
        rows.extend(teacher_frames(datasets_dir(name), limit=None))
    if not rows:
        return rows
    uniq: list[dict] = []
    seen: set[tuple] = set()
    for rec in rows:
        key = (str(rec.get("_front")), str(rec.get("_grip")), rec.get("tick"), rec.get("step"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(rec)
    if limit is None or limit < 0:
        return uniq
    limit = int(limit)
    pad20 = [r for r in uniq if _is_present(r) and abs(_frame_size_mm(r) - 20.0) < 0.5 and _is_pad_row(r) and not _is_lift_row(r)]
    pad = [r for r in uniq if _is_present(r) and _is_pad_row(r) and not _is_lift_row(r)]
    lift = [r for r in uniq if _is_present(r) and _is_lift_row(r)]
    n_lift = min(len(lift), limit // 2)
    if lift and limit >= 2:
        n_lift = min(len(lift), max(n_lift, 2), limit)
    chosen: list[dict] = list(lift[:n_lift])
    for pool in (pad20, pad, uniq):
        for rec in pool:
            if rec in chosen:
                continue
            chosen.append(rec)
            if len(chosen) >= limit:
                return chosen
    return chosen[:limit]


def _clip_action(act: np.ndarray, step: float) -> np.ndarray:
    return np.clip(np.asarray(act, dtype=np.float64).reshape(-1)[:4], -step, step)


def loss_on_frames(lif: CropLIF, frames: list[dict], u: UParams, lam_g: float = 0.05, lam_r: float = 1e-4) -> tuple[float, dict]:
    err = 0.0
    excess = []
    n_abort = 0
    axis_err = np.zeros(4, dtype=np.float64)
    for rec in frames:
        lif.reset_episode()
        front = load_image(rec["_front"])
        grip = load_image(rec["_grip"])
        lif.step_vision(front, grip)
        rates = lif.rates()
        attached = bool((rec.get("cube") or {}).get("attached"))
        cmd = unpack(rates, u, attached=attached)
        tgt = teacher_from_state(_state_from_log(rec), pad_lock=_live_pad_lock(rec)).action
        tgt_c = _clip_action(tgt, u.step)
        excess.append(float(np.linalg.norm(np.asarray(tgt, dtype=np.float64).reshape(-1)[:4] - tgt_c)))
        pred = np.array([cmd.dx_mm, cmd.dy_mm, cmd.dz_mm, cmd.dgrip_mm], dtype=np.float64)
        axis_err += (pred - tgt_c) ** 2
        w = 3.0 if attached else 1.0
        err += w * float(np.sum((pred - tgt_c) ** 2))
        err += lam_r * float(np.abs(rates.vec).sum())
        if rates.abort or command_is_abort(cmd):
            err += ABORT_PEN
            n_abort += 1
    err += lam_g * float(np.sum((lif.g - 1.0) ** 2))
    n = max(len(frames), 1)
    stats = {
        "clip_step_mm": float(u.step),
        "mean_tgt_excess_over_clip": float(np.mean(excess)) if excess else 0.0,
        "axis_mse": {"dx": float(axis_err[0] / n), "dy": float(axis_err[1] / n), "dz": float(axis_err[2] / n), "dgrip": float(axis_err[3] / n)},
        "n_abort_frames": int(n_abort),
    }
    return err / n, stats


def sequential_abort_loss(lif: CropLIF, frames: list[dict], u: UParams, n_ticks: int = 3) -> tuple[float, int]:
    """n_ticks vision steps without reset_episode."""
    n_abort = 0
    def is_pad(rec: dict) -> bool:
        step = str(rec.get("step") or "")
        z = float((rec.get("tcp_mm") or {}).get("z") or 99.0)
        return "pad" in step or abs(z - PAD_Z_MM) < 8.0

    seq = [rec for rec in frames if is_pad(rec) and abs(_frame_size_mm(rec) - 20.0) < 0.5][:2]
    if not seq:
        seq = [rec for rec in frames if is_pad(rec)][:2]
    if not seq and frames:
        seq = frames[:1]
    if not seq:
        from .hop_probe import _synthetic_cube

        rgb = _synthetic_cube()
        lif.reset_episode()
        for _ in range(n_ticks):
            lif.step_vision(rgb, rgb)
            rates = lif.rates()
            cmd = unpack(rates, u, attached=False)
            if rates.abort or command_is_abort(cmd):
                n_abort += 1
        return ABORT_PEN * n_abort, n_abort
    for rec in seq:
        lif.reset_episode()
        front = load_image(rec["_front"])
        grip = load_image(rec["_grip"])
        for _ in range(n_ticks):
            lif.step_vision(front, grip)
            rates = lif.rates()
            cmd = unpack(rates, u, attached=bool((rec.get("cube") or {}).get("attached")))
            if rates.abort or command_is_abort(cmd):
                n_abort += 1
    return ABORT_PEN * n_abort, n_abort


def _freeze_u_abort(u: UParams) -> UParams:
    u.abort_hz = float(U0.abort_hz)
    u.abort_dx = float(U0.abort_dx)
    u.abort_dz = float(U0.abort_dz)
    u.abort_open = float(U0.abort_open)
    u.step = 8.0
    u.rate_div = float(U0.rate_div)
    return u


def _gf_quiet_g(g0: np.ndarray) -> np.ndarray:
    g = g0.copy()
    names = list(GAIN_CLASSES)
    for name, val in (("DNp01", 0.01), ("LC10", 0.3)):
        if name in names:
            g[names.index(name)] = val
    return g


def _close_u_vec() -> np.ndarray:
    u = UParams()
    u.dgrip_scale = 8.0
    u.dz_scale = 8.0
    u.w_xl = -2.0
    u.w_contact = 2.5
    return u.as_vector()


def distill(
    *,
    stub: bool = False,
    hop_path: Path | None = None,
    n_gen: int = 5,
    pop: int = 4,
    frames: int = 8,
    seed: int = 0,
) -> dict:
    hop_path = hop_path or checkpoints_dir("rebot-pickup") / ("hop-probe-stub.json" if stub else "hop-probe.json")
    hop = _load_hop(hop_path)
    dest = checkpoints_dir("rebot-pickup") / ("g-distill-stub.npz" if stub else "g-distill.npz")
    if not stub and not train_gate(hop):
        return write_skip_checkpoint(
            json_path=dest.with_suffix(".json"),
            npz=dest,
            payload={
                "reason": "hop_probe not green or live_go false; optimizer not opened",
                "acting_map": ActingMap.dn_bus.value,
                "g_hash": None,
                "da_learned": False,
            },
        )
    data = distill_frames(limit=frames)
    if not data:
        return write_skip_checkpoint(
            json_path=dest.with_suffix(".json"),
            npz=dest,
            payload={
                "reason": "no teacher Front+Gripper frames; optimizer not opened",
                "acting_map": ActingMap.dn_bus.value,
                "g_hash": None,
                "da_learned": False,
            },
        )
    crop = write_stub_crop(checkpoints_dir("rebot-pickup") / "stub-crop") if stub else build_crop("v1.0")
    live = (hop.get("live_replay") if hop else None) or {}
    gain = float(live.get("synaptic_gain") or (hop.get("synaptic_gain") if hop else None) or 1.5)
    if not stub and gain < 2.5:
        gain = 2.5
    lif = CropLIF(crop, synaptic_gain=gain, nsteps=100 if stub else 150)
    g0 = lif.g.copy()
    u0_vec = U0.as_vector().copy()
    n_g = int(g0.size)
    gf_i = list(GAIN_CLASSES).index("DNp01") if "DNp01" in GAIN_CLASSES else None
    lc_i = list(GAIN_CLASSES).index("LC10") if "LC10" in GAIN_CLASSES else None

    def _clip_g(gvec: np.ndarray) -> np.ndarray:
        gvec = np.clip(gvec, 0.005, 8.0)
        if gf_i is not None:
            gvec[gf_i] = np.clip(gvec[gf_i], 0.005, 0.05)  # DNp01 ∈ [0.005, 0.05]
        if lc_i is not None:
            gvec[lc_i] = np.clip(gvec[lc_i], 0.12, 0.45)  # LC10 ∈ [0.12, 0.45]
        return gvec.astype(np.float32)

    def eval_theta(theta: np.ndarray) -> float:
        gvec = _clip_g(theta[:n_g])
        uvec = np.clip(theta[n_g:], U_LO, U_HI)
        lif.set_g(gvec)
        u = _freeze_u_abort(UParams.from_vector(uvec, base=U0))
        loss, _stats = loss_on_frames(lif, data, u)
        seq_pen, _n_seq = sequential_abort_loss(lif, data, u)
        return loss + seq_pen

    rng = np.random.default_rng(seed)
    theta0 = np.concatenate([g0.astype(np.float64), u0_vec])
    seed_g = _gf_quiet_g(g0)
    seed_u = np.clip(_close_u_vec(), U_LO, U_HI)
    theta = np.concatenate([seed_g.astype(np.float64), seed_u])
    best0 = eval_theta(theta0)
    best_s = eval_theta(theta)
    if best0 <= best_s:
        theta, best = theta0, best0
    else:
        best = best_s
    history = [{"gen": 0, "loss": best, "g_hash": g_hash(theta[:n_g].astype(np.float32))}]
    sigma_g = 0.15
    sigma_u = 0.25
    for gen in range(1, n_gen + 1):
        cand = [theta]
        losses = [best]
        for _ in range(pop):
            noise = np.concatenate(
                [
                    rng.normal(0.0, sigma_g, size=n_g),
                    rng.normal(0.0, sigma_u, size=u0_vec.size),
                ]
            )
            trial = theta + noise
            trial[:n_g] = _clip_g(trial[:n_g])
            trial[n_g:] = np.clip(trial[n_g:], U_LO, U_HI)
            cand.append(trial)
            losses.append(eval_theta(trial))
        k = int(np.argmin(losses))
        theta = cand[k]
        best = losses[k]
        history.append({"gen": gen, "loss": best, "g_hash": g_hash(theta[:n_g].astype(np.float32))})
        print(f"distill gen={gen} loss={best:.4f} g_hash={g_hash(theta[:n_g].astype(np.float32))}", flush=True)
    g = _clip_g(theta[:n_g])
    u = _freeze_u_abort(UParams.from_vector(np.clip(theta[n_g:], U_LO, U_HI), base=U0))
    lif.set_g(g)
    best, clip_stats = loss_on_frames(lif, data, u)
    seq_pen, seq_n = sequential_abort_loss(lif, data, u)
    clip_stats = dict(clip_stats)
    clip_stats["sequential_abort_ticks"] = int(seq_n)
    best = float(best + seq_pen)
    pad_abort = int(seq_n) > 0
    u_changed = not np.allclose(u.as_vector(), u0_vec, atol=1e-5)
    out = {
        "ok": g_hash(g) != g_hash(g0) or u_changed,
        "skipped": False,
        "acting_map": ActingMap.dn_bus.value,
        "teacher_off": True,
        "g_hash": g_hash(g),
        "g_hash_init": g_hash(g0),
        "g": g.tolist(),
        "g_init": g0.tolist(),
        "gain_classes": list(GAIN_CLASSES),
        "u_n_params": u.n_params(),
        "u": u.as_vector().tolist(),
        "u_abort_hz": float(u.abort_hz),
        "loss": best,
        "history": history,
        "n_frames": len(data),
        "synaptic_gain": gain,
        "clip_stats": clip_stats,
        "pad_abort": pad_abort,
        "sequential_abort_ticks": int(seq_n),
        "freeze_g_required": True,
        "fly_picked": False,
        "note": "g+tiny U ES; W frozen; abort_hz frozen; pad close at size+2; contact term distilled; freeze-g ablation required for fly_picked",
    }
    dest = checkpoints_dir("rebot-pickup") / ("g-distill-stub.npz" if stub else "g-distill.npz")
    np.savez(
        dest,
        g=g,
        g_init=g0,
        synaptic_gain=np.array([gain], dtype=np.float32),
        u=u.as_vector(),
    )
    write_json(dest.with_suffix(".json"), out)
    out["path"] = str(dest)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--gens", type=int, default=5)
    parser.add_argument("--pop", type=int, default=4)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args(argv)
    result = distill(stub=args.stub, n_gen=args.gens, pop=args.pop, frames=args.frames)
    print(json.dumps({k: result[k] for k in result if k != "g" and k != "g_init" and k != "history"}, indent=2))
    if result.get("skipped"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
