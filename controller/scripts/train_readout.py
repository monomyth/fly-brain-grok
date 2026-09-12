#!/usr/bin/env python3
"""Phase 2: freeze MaleCNS, fit a linear readout to overlay/teacher pad traces."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from malecns_cache.graph import load_graph, load_stub
from malecns_cache.paths import checkpoints_dir
from runtime.lif import LIFNetwork
from train.dataset import VIEW_CAMERAS, for_imitation, load_dataset, pair_view_samples
from train.expert import imitation_actions, overlay_training_rows
from train.features import BIAS_INDEX, FEATURE_NAMES, VISUAL_INDEXES, neural_features, proprio_feature_row
from train.readout import absorb_constant_columns, apply_linear, fit_linear, save_readout
from train.shuffle import shuffle_edges


def matrix(connectome, brain, samples, hops: int, proprio: bool) -> np.ndarray:
    xs = []
    started = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        xs.append(neural_features(connectome, brain, sample, hops=hops, proprio=proprio))
        if i % 25 == 0 or i == len(samples):
            print(f"  features {i}/{len(samples)}  elapsed_s={time.perf_counter() - started:.1f}", flush=True)
    return np.vstack(xs)


def overlay_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.vstack(
        [
            proprio_feature_row(
                row["grip"],
                row["tcp"],
                bool(row["attached"]),
                yaw_rad=float(row.get("yaw_rad") or 0.0),
            )
            for row in rows
        ]
    )
    y = np.vstack([np.asarray(row["action"], dtype=np.float64) for row in rows])
    w = np.array([float(row["weight"]) for row in rows], dtype=np.float64)
    return x, y, w


def mse(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    err = apply_linear(weights, x) - y
    return float(np.mean(err * err))


def channel_mse(weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = apply_linear(weights, x) - y
    names = ["dx_mm", "dy_mm", "dz_mm", "dgrip_mm"]
    return {name: float(np.mean(err[:, i] * err[:, i])) for i, name in enumerate(names)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional logged episodes. Default is overlay pad traces only (pilot-v1 pinches at z≈4 mm).",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        default=None,
        help="Optional extra logs. Teacher waypoint logs omit yaw and fight pad traces.",
    )
    parser.add_argument("--max-episodes", type=int, default=16)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--l2", type=float, default=1e-2)
    parser.add_argument("--no-proprio", action="store_true")
    parser.add_argument("--stub", action="store_true", help="Use the tiny test graph, not MaleCNS v1.0")
    parser.add_argument("--shuffle-edges", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--overlay-traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--neural",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run LIF on logged JPEGs so r1/r8 enter the fit. Default on when a dataset is given.",
    )
    parser.add_argument("--tag", default="phase2-linear")
    parser.add_argument(
        "--cameras",
        default=",".join(VIEW_CAMERAS),
        help="Comma-separated cameras to keep from logged episodes (default Front,Gripper).",
    )
    args = parser.parse_args()
    proprio = not args.no_proprio
    wanted = tuple(c.strip() for c in args.cameras.split(",") if c.strip()) or VIEW_CAMERAS

    def keep_views(samples):
        return [s for s in samples if s.camera in wanted]

    train_samples = []
    val_samples = []
    if args.dataset is not None:
        train_samples = pair_view_samples(
            keep_views(
                for_imitation(
                    load_dataset(args.dataset, split="train", stride=args.stride, max_episodes=args.max_episodes)
                )
            )
        )
        val_samples = pair_view_samples(
            keep_views(
                for_imitation(
                    load_dataset(
                        args.dataset, split="validation", stride=args.stride, max_episodes=max(2, args.max_episodes // 4)
                    )
                )
            )
        )
    if args.extra and args.extra.exists():
        extra = pair_view_samples(keep_views(for_imitation(load_dataset(args.extra, stride=1))))
        train_samples = extra + train_samples
    use_neural = bool(train_samples) if args.neural is None else bool(args.neural)

    overlay_train: list[dict] = []
    overlay_val: list[dict] = []
    if args.overlay_traces:
        overlay_train, overlay_val = overlay_training_rows()
        print(f"overlay traces train={len(overlay_train)} val={len(overlay_val)}", flush=True)
    x_over = y_over = w_over = None
    if overlay_train:
        x_over, y_over, w_over = overlay_matrix(overlay_train)

    x_logged = y_logged = w_logged = None
    x_logged_val = y_logged_val = None
    if train_samples and use_neural:
        if args.stub:
            connectome = load_stub(Path("/tmp/malecns-stub-train"))
        else:
            connectome = load_graph("v1.0")
        if args.shuffle_edges:
            connectome = shuffle_edges(connectome, seed=args.shuffle_seed)
        brain = LIFNetwork(connectome.weights)
        print(f"graph n={connectome.n} nnz={connectome.weights.nnz}  train={len(train_samples)} val={len(val_samples)}")
        y_logged, w_logged = imitation_actions(train_samples, hold_bonus=True, relabel="overlay")
        x_logged = matrix(connectome, brain, train_samples, hops=args.hops, proprio=proprio)
        if val_samples:
            y_logged_val, _ = imitation_actions(val_samples, relabel="overlay")
            x_logged_val = matrix(connectome, brain, val_samples, hops=args.hops, proprio=proprio)
    elif train_samples:
        y_logged, w_logged = imitation_actions(train_samples, hold_bonus=True, relabel="overlay")
        x_logged = np.vstack(
            [
                proprio_feature_row(s.gripper_mm, s.tcp_mm, s.attached, yaw_rad=float(s.yaw_rad or 0.0))
                for s in train_samples
            ]
        )
        print(f"logged proprio {len(train_samples)} (no LIF)", flush=True)

    blocks_x, blocks_y, blocks_w = [], [], []
    if x_over is not None:
        blocks_x.append(x_over)
        blocks_y.append(y_over)
        blocks_w.append(w_over)
    if x_logged is not None:
        blocks_x.append(x_logged)
        blocks_y.append(y_logged)
        blocks_w.append(w_logged * (8.0 if x_over is not None else 1.0))
    if not blocks_x:
        raise SystemExit("no training rows (need --overlay-traces and/or a dataset)")
    x_train = np.vstack(blocks_x)
    y_train = np.vstack(blocks_y)
    w_train = np.concatenate(blocks_w)
    weights = fit_linear(x_train, y_train, l2=args.l2, sample_weight=w_train)
    weights, visual_kept, visual_absorbed = absorb_constant_columns(
        weights, x_train, columns=tuple(VISUAL_INDEXES), bias_row=BIAS_INDEX
    )
    pred = apply_linear(weights, x_train)
    train_mse = mse(weights, x_train, y_train)
    close_mask = y_train[:, 3] < -0.5
    hold_mask = (x_train[:, FEATURE_NAMES.index("grip")] < 0.85) & (
        x_train[:, FEATURE_NAMES.index("tcp_z")] > 0.55
    )
    val_mse = None
    val_channels = None
    x_val = y_val = None
    val_approach_channels = None
    if overlay_val:
        x_val, y_val, _ = overlay_matrix(overlay_val)
        val_mse = mse(weights, x_val, y_val)
        val_channels = channel_mse(weights, x_val, y_val)
        approach = np.array([row["phase"] == "approach" for row in overlay_val])
        if np.any(approach):
            val_approach_channels = channel_mse(weights, x_val[approach], y_val[approach])
    elif x_logged_val is not None:
        x_val, y_val = x_logged_val, y_logged_val
        val_mse = mse(weights, x_val, y_val)
        val_channels = channel_mse(weights, x_val, y_val)
    elif val_samples:
        y_val, _ = imitation_actions(val_samples, relabel="overlay")
        x_val = np.vstack(
            [
                proprio_feature_row(s.gripper_mm, s.tcp_mm, s.attached, yaw_rad=float(s.yaw_rad or 0.0))
                for s in val_samples
            ]
        )
        val_mse = mse(weights, x_val, y_val)
        val_channels = channel_mse(weights, x_val, y_val)
    dest = checkpoints_dir("rebot-pickup") / f"{args.tag}.npz"
    cache = checkpoints_dir("rebot-pickup") / f"{args.tag}-features.npz"
    np.savez(
        cache,
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        w_train=w_train.astype(np.float32),
        x_val=np.asarray(x_val if x_val is not None else np.zeros((0, x_train.shape[1])), dtype=np.float32),
        y_val=np.asarray(y_val if y_val is not None else np.zeros((0, 4)), dtype=np.float32),
    )
    meta = {
        "phase": 2,
        "connectome": "shuffled-edge MaleCNS v1.0"
        if args.shuffle_edges
        else ("frozen MaleCNS v1.0" if not args.stub else "stub"),
        "shuffled_edges": bool(args.shuffle_edges),
        "shuffle_seed": int(args.shuffle_seed) if args.shuffle_edges else None,
        "dataset": str(args.dataset) if args.dataset else None,
        "extra": str(args.extra) if args.extra else None,
        "overlay_traces": int(len(overlay_train)),
        "overlay_val_traces": int(len(overlay_val)),
        "train_samples": int(x_train.shape[0]),
        "logged_train_samples": int(len(train_samples)),
        "logged_val_samples": int(len(val_samples)),
        "cameras": sorted(
            {
                cam
                for s in train_samples + val_samples
                for cam in ((s.views or {}) or {s.camera: s.image})
            }
        )
        if use_neural
        else [],
        "logged_cameras": sorted(
            {
                cam
                for s in train_samples + val_samples
                for cam in ((s.views or {}) or {s.camera: s.image})
            }
        ),
        "jpeg_in_fit": bool(use_neural and train_samples),
        "visual_columns_kept": [FEATURE_NAMES[i] for i in visual_kept],
        "visual_columns_absorbed": [FEATURE_NAMES[i] for i in visual_absorbed],
        "r1_std": float(np.std(x_train[:, 0])) if x_train.shape[0] else 0.0,
        "r8_std": float(np.std(x_train[:, 1])) if x_train.shape[0] else 0.0,
        "r1_std_logged": float(np.std(x_logged[:, 0])) if x_logged is not None else None,
        "r8_std_logged": float(np.std(x_logged[:, 1])) if x_logged is not None else None,
        "val_samples": int(x_val.shape[0]) if x_val is not None else 0,
        "hops": args.hops,
        "proprio": proprio,
        "neural": bool(use_neural),
        "l2": args.l2,
        "train_mse": train_mse,
        "logged_train_mse": mse(weights, x_logged, y_logged) if x_logged is not None else None,
        "raw_log_train_mse": None,
        "val_mse": val_mse,
        "val_channel_mse": val_channels,
        "val_approach_channel_mse": val_approach_channels,
        "train_channel_mse": channel_mse(weights, x_train, y_train),
        "mean_target_dgrip": float(y_train[:, 3].mean()),
        "mean_pred_dgrip": float(pred[:, 3].mean()),
        "frac_target_close": float(close_mask.mean()),
        "mean_pred_dgrip_on_close": float(pred[close_mask, 3].mean()) if np.any(close_mask) else None,
        "mean_pred_dgrip_on_hold": float(pred[hold_mask, 3].mean()) if np.any(hold_mask) else None,
        "mean_pred_dz_on_hold": float(pred[hold_mask, 2].mean()) if np.any(hold_mask) else None,
        "feature_names": FEATURE_NAMES,
        "action": ["dx_mm", "dy_mm", "dz_mm", "dgrip_mm"],
        "note": "Linear readout on Front+Gripper optical maps + LIF + proprio. Attached is not a feature.",
        "kernel": False,
        "kernel_rows": 0,
    }
    kx = ky = kw = None
    if x_logged is not None:
        kx, ky, kw = x_logged, np.array(y_logged, dtype=np.float64, copy=True), w_logged
        attached_rows = np.array([bool(s.attached) for s in train_samples], dtype=bool)
        if attached_rows.shape[0] == ky.shape[0]:
            high = (kx[:, FEATURE_NAMES.index("tcp_z")] * 200.0 > 90.0) & ~attached_rows
            ky[high, 1] = 0.0
            close = (ky[:, 3] < -0.5) & ~attached_rows
        else:
            close = ky[:, 3] < -0.5
        if np.any(close):
            kx = np.vstack([kx, kx[close]])
            ky = np.vstack([ky, ky[close]])
            kw = np.concatenate([kw, kw[close]])
        meta["kernel"] = True
        meta["kernel_rows"] = int(kx.shape[0])
        meta["kernel_attached_rows"] = int(np.count_nonzero(attached_rows)) if attached_rows.size == y_logged.shape[0] else 0
    save_readout(dest, weights, meta, kernel_x=kx, kernel_y=ky, kernel_w=kw)
    (dest.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
