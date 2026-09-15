#!/usr/bin/env python3
"""Replay teleop pick videos through MaleCNS hop. No motors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebot_adapter.b601 import fk_pose
from rebot_adapter.hw_phase import hop_search, write_json


def tcp_row(q6: np.ndarray, grip: float, i: int) -> dict:
    p = fk_pose(q6)
    return {
        "i": int(i),
        "grip_deg": float(grip),
        "q_deg": [float(x) for x in q6],
        "xyz_mm": {"x": p.x_mm, "y": p.y_mm, "z": p.z_mm},
    }


def dxyz(a: dict, b: dict) -> dict:
    return {k: float(b["xyz_mm"][k] - a["xyz_mm"][k]) for k in ("x", "y", "z")}


def hop_jpegs(crop, overview: Path, wrist: Path, nsteps: int) -> dict:
    from PIL import Image

    ov = np.asarray(Image.open(overview).convert("RGB"))
    wr = np.asarray(Image.open(wrist).convert("RGB"))
    hop = hop_search(crop, ov, wr, nsteps=nsteps)
    return {
        "ok": hop.get("ok"),
        "feed": hop.get("feed"),
        "synaptic_gain": hop.get("synaptic_gain"),
        "cube_dn_mean": hop.get("cube_dn_mean"),
        "gf_hz_cube": hop.get("gf_hz_cube"),
        "cube_is_abort": hop.get("cube_is_abort"),
        "cube_command": hop.get("cube_command"),
        "cube_bus": hop.get("cube_bus"),
        "fly_picked": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline hop + FK teacher from a pick episode")
    p.add_argument("--states", type=Path, required=True, help="(N,7) npy observation.state")
    p.add_argument("--frames-dir", type=Path, required=True, help="subdirs or files overview_I.jpg wrist_I.jpg")
    p.add_argument("--indices", default="0,120,218,360")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    st = np.load(args.states)
    if st.ndim != 2 or st.shape[1] < 6:
        print("states must be (N, >=6)", file=sys.stderr)
        return 2
    idxs = [int(x) for x in args.indices.split(",") if x.strip()]
    from arm.crop import build_crop, write_stub_crop
    from malecns_cache.paths import project_data

    crop = write_stub_crop(project_data() / "prepared" / "stub-hw-replay") if args.stub else build_crop()
    teacher = [tcp_row(st[i, :6], float(st[i, 6]) if st.shape[1] > 6 else 0.0, i) for i in range(0, len(st), max(1, len(st) // 12))]
    hops = []
    for i in idxs:
        if i < 0 or i >= len(st):
            continue
        ov = args.frames_dir / f"overview_{i}.jpg"
        wr = args.frames_dir / f"wrist_{i}.jpg"
        row = {"i": i, "tcp": tcp_row(st[i, :6], float(st[i, 6]) if st.shape[1] > 6 else 0.0, i)}
        if ov.is_file() and wr.is_file():
            hop = hop_jpegs(crop, ov, wr, args.steps)
            row["hop"] = hop
            nxt = min(i + 15, len(st) - 1)
            row["teacher_dxyz_1s_mm"] = dxyz(row["tcp"], tcp_row(st[nxt, :6], 0.0, nxt))
            cmd = hop.get("cube_command") or {}
            row["fly_dxyz_mm"] = {
                "x": float(cmd.get("dx_mm") or 0.0),
                "y": float(cmd.get("dy_mm") or 0.0),
                "z": float(cmd.get("dz_mm") or 0.0),
            }
        else:
            row["hop"] = None
            row["missing_jpegs"] = True
        hops.append(row)
    payload = {
        "n_states": int(len(st)),
        "teacher_track": teacher,
        "hops": hops,
        "fly_picked": False,
        "note": "Offline replay. fly command is one LIF tick; teacher_dxyz_1s_mm is ~1 s of teleop TCP.",
    }
    buses = [np.asarray(h["hop"]["cube_bus"], dtype=np.float64) for h in hops if h.get("hop") and h["hop"].get("cube_bus")]
    ys = []
    for h in hops:
        t = h.get("teacher_dxyz_1s_mm") or {}
        ys.append([float(t.get("x") or 0.0), float(t.get("y") or 0.0), float(t.get("z") or 0.0)])
    fit = {"rank": None, "bus_std": None, "note": "need cube_bus on hops"}
    if len(buses) >= 2:
        X = np.stack(buses)
        Y = np.asarray(ys[: len(buses)], dtype=np.float64)
        fit["bus_std"] = [float(x) for x in np.std(X, axis=0)]
        fit["bus_mean"] = [float(x) for x in np.mean(X, axis=0)]
        fit["teacher_std"] = [float(x) for x in np.std(Y, axis=0)]
        xc = X - X.mean(axis=0)
        yc = Y - Y.mean(axis=0)
        fit["rank"] = int(np.linalg.matrix_rank(xc, tol=1e-3))
        # y ≈ W x  (no bias in centered)
        w, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = X @ w
        fit["mse"] = float(np.mean((pred - Y) ** 2))
        fit["pred"] = pred.round(2).tolist()
        fit["teacher"] = Y.round(2).tolist()
        fit["note"] = "linear bus→1s TCP. Low bus_std means U cannot see the pick."
    payload["fit"] = fit
    write_json(args.out, payload)
    print(json.dumps({"out": str(args.out), "hops": len(hops), "ok": [bool(h.get("hop") and h["hop"].get("ok")) for h in hops], "fit": fit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
