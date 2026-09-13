#!/usr/bin/env python3
"""Phase 4 ablations. Offline MSE plus optional live lab rollouts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from malecns_cache.paths import checkpoints_dir, home, project_data
from rebot_adapter.episode import default_mcp_binary
from train.features import FEATURE_NAMES
from train.readout import apply_linear, load_readout, random_readout


def _mse(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    err = apply_linear(weights, x) - y
    return float(np.mean(err * err))


def _channel(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {}
    err = apply_linear(weights, x) - y
    names = ["dx_mm", "dy_mm", "dz_mm", "dgrip_mm"]
    return {name: float(np.mean(err[:, i] * err[:, i])) for i, name in enumerate(names)}


def load_learn_vs_shuffle(root: Path) -> dict:
    path = root / "learn-vs-shuffle.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def offline_table(root: Path, readout_path: Path, skip_shuffled: bool = False) -> dict:
    weights, meta = load_readout(readout_path) if readout_path.is_file() else (None, {})
    cache = root / "phase2-linear-features.npz"
    blob = np.load(cache) if cache.is_file() else None
    x_val = np.asarray(blob["x_val"], dtype=np.float64) if blob is not None else np.zeros((0, len(FEATURE_NAMES)))
    y_val = np.asarray(blob["y_val"], dtype=np.float64) if blob is not None else np.zeros((0, 4))
    x_train = np.asarray(blob["x_train"], dtype=np.float64) if blob is not None else np.zeros((0, len(FEATURE_NAMES)))
    y_train = np.asarray(blob["y_train"], dtype=np.float64) if blob is not None else np.zeros((0, 4))
    x = x_val if x_val.shape[0] else x_train
    y = y_val if y_val.shape[0] else y_train
    n_feat = x.shape[1] if x.ndim == 2 and x.shape[1] else len(FEATURE_NAMES)
    random_w = random_readout(n_feat, 4, seed=0, scale=0.05)
    trained_mse = _mse(weights, x, y) if weights is not None else None
    random_mse = _mse(random_w, x, y) if x.size else None
    black = x.copy() if x.size else x
    if black.size:
        black[:, :10] = 0.0
    novision_mse = _mse(weights, black, y) if weights is not None and black.size else None
    shuffled_mse = None
    shuffled_path = root / "phase2-shuffled-features.npz"
    if weights is not None and shuffled_path.is_file() and not skip_shuffled:
        sh = np.load(shuffled_path)
        sx = np.asarray(sh["x_val"] if sh["x_val"].shape[0] else sh["x_train"], dtype=np.float64)
        sy = np.asarray(sh["y_val"] if sh["y_val"].shape[0] else sh["y_train"], dtype=np.float64)
        shuffled_mse = _mse(weights, sx, sy)
    learn = load_learn_vs_shuffle(root)
    return {
        "n_eval": int(x.shape[0]),
        "trained_mse": trained_mse,
        "random_mse": random_mse,
        "novision_mse": novision_mse,
        "shuffled_edge_same_readout_mse": shuffled_mse,
        "trained_channel_mse": _channel(weights, x, y) if weights is not None else {},
        "readout_meta": {
            k: meta.get(k)
            for k in ("train_mse", "val_mse", "raw_log_train_mse", "mean_pred_dgrip_on_close", "mean_pred_dz_on_hold", "train_samples")
            if k in meta
        },
        "plasticity_learn_vs_shuffle": learn,
    }


def _python() -> str:
    venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else sys.executable


def run_live(condition: str, extra: list[str], env: dict[str, str], mcp: Path | None = None) -> dict:
    controller = Path(__file__).resolve().parents[1]
    py = _python()
    if condition == "teacher":
        cmd = [py, str(controller / "scripts" / "run_teacher.py")]
        if mcp is not None:
            cmd.append(str(mcp))
    else:
        cmd = [py, str(controller / "scripts" / "run_dn_arm.py"), *extra]
        if mcp is not None:
            cmd.extend(["--mcp", str(mcp)])
    print("live", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(controller), env=env, check=False)
    return {"returncode": proc.returncode, "cmd": cmd}


def _live_brief(payload: dict) -> dict:
    keys = (
        "ok",
        "picked",
        "readout_picked",
        "fly_picked",
        "xyz_diverged",
        "teacher_ok",
        "controller",
        "attached",
        "cube_z_mm",
        "peak_cube_z_mm",
        "tcp_level",
        "hold_s",
        "closest_dist_xy_mm",
        "travel_xy_mm",
        "gripper_mm",
        "end_tcp",
        "ticks",
        "quintic_ran",
        "overlay",
        "offline_mse",
    )
    return {k: payload.get(k) for k in keys if k in payload}


def render_report(path: Path, offline: dict, live: dict) -> None:
    learn = offline.get("plasticity_learn_vs_shuffle") or {}
    rows = [
        ("Teacher script", "Environment is solvable", live.get("teacher", {})),
        ("Overlay-servo (pad expert, no quintic teacher)", "Engineered pad geometry on servo steps", live.get("overlay", {})),
        ("Frozen graph + trained readout", "Connectome features help", {**live.get("trained", {}), "offline_mse": offline.get("trained_mse")}),
        ("Frozen graph + random readout", "Chance", {**live.get("random", {}), "offline_mse": offline.get("random_mse")}),
        ("No-vision (black frames)", "Not cheating via proprioception", {**live.get("novision", {}), "offline_mse": offline.get("novision_mse")}),
        ("Shuffled-edge graph + same readout", "Wiring, not just neuron count", {**live.get("shuffled", {}), "offline_mse": offline.get("shuffled_edge_same_readout_mse")}),
        ("Plasticity on/off", "Whether KC→MBON changes do anything", learn),
    ]
    lines = [
        "# MaleCNS arm-control evaluation",
        "",
        "DN→TCP maps are **engineered**, not proven motor identity. Grasp is kinematic pad attach.",
        "A pick counts only if cube `attached`, live center z ≥ 100 mm, `tcp_level` true, and hold ≥ 2 s.",
        "`picked` is only True for readout/overlay-servo controllers. Quintic lab grasps are `teacher_ok`.",
        "`fly_picked` / `readout_picked` is only True when the acting controller is the MaleCNS readout.",
        "Overlay-servo is a yaw-locked pad expert on servo steps — **not** MaleCNS motor output.",
        "",
        "| Condition | What it tests | Result |",
        "|---|---|---|",
    ]
    for name, tests, payload in rows:
        lines.append(f"| {name} | {tests} | {json.dumps(_live_brief(payload) if isinstance(payload, dict) else payload, sort_keys=True)} |")
    lines += [
        "",
        "## Notes",
        "",
        "- Phase 2 readout is a 16→4 linear map on frozen MaleCNS rates plus proprioception.",
        "- Live trained/random/novision/shuffled rows use `--readout --no-overlay --frozen` (no quintic handoff).",
        "- Overlay is pad geometry, not connectome motor identity.",
        f"- KC→MBON Learn vs shuffled-reward cosine={learn.get('cosine')} (existing Phase 3 numbers; no new plasticity runs).",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--mcp", type=Path, default=default_mcp_binary())
    parser.add_argument("--skip-shuffled-features", action="store_true")
    parser.add_argument("--write-report", type=Path, default=None, help="Write a report to this path. Does not replace eval-report.md unless set.")
    args = parser.parse_args()
    root = checkpoints_dir("rebot-pickup")
    readout = root / "phase2-linear.npz"
    offline = offline_table(root, readout, skip_shuffled=args.skip_shuffled_features)
    live: dict = {}
    if args.live:
        env = {
            **os.environ,
            "REBOT_MCP_NO_LAUNCH": "1",
            "MALECNS_HOME": os.environ.get("MALECNS_HOME") or str(home()),
            "FLYBRAIN_DATA": os.environ.get("FLYBRAIN_DATA", str(project_data())),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        }
        frozen = ["--no-overlay", "--frozen", "--mcp", str(args.mcp)]
        teacher = run_live("teacher", [], env, mcp=args.mcp)
        live["teacher"] = _read_json(root / "teacher-eval.json") | teacher
        overlay = run_live(
            "overlay",
            ["--overlay", "--frozen", "--tag", "eval-overlay", "--ticks", "160", "--mcp", str(args.mcp)],
            env,
            mcp=args.mcp,
        )
        live["overlay"] = _read_json(root / "eval-overlay.json") | overlay
        trained = run_live("trained", ["--readout", "--tag", "eval-trained", "--ticks", "140", *frozen], env, mcp=args.mcp)
        live["trained"] = _read_json(root / "eval-trained.json") | trained
        random_path = root / "phase2-random.npz"
        if readout.is_file():
            w, meta = load_readout(readout)
            from train.readout import save_readout

            save_readout(random_path, random_readout(w.shape[0], w.shape[1], seed=0), {**meta, "random": True})
        ablation = ["--ticks", "80", *frozen]
        rand = run_live(
            "random",
            ["--readout", "--readout-path", str(random_path), "--tag", "eval-random", *ablation],
            env,
            mcp=args.mcp,
        )
        live["random"] = _read_json(root / "eval-random.json") | rand
        nov = run_live("novision", ["--readout", "--blind", "--tag", "eval-novision", *ablation], env, mcp=args.mcp)
        live["novision"] = _read_json(root / "eval-novision.json") | nov
        shuf = run_live("shuffled", ["--readout", "--shuffle-edges", "--tag", "eval-shuffled", *ablation], env, mcp=args.mcp)
        live["shuffled"] = _read_json(root / "eval-shuffled.json") | shuf
    report = {"offline": offline, "live": live}
    (root / "eval-offline.json").write_text(json.dumps({"offline": offline}, indent=2) + "\n")
    if args.write_report is not None:
        dest = args.write_report
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2) + "\n")
        md = dest.with_suffix(".md") if dest.suffix != ".md" else dest
        render_report(md, offline, live)
        print(f"wrote {dest}")
    print(json.dumps(report, indent=2))
    return 0


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
