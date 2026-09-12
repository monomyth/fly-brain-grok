from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_linear(
    features: np.ndarray,
    actions: np.ndarray,
    l2: float = 1e-3,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Ridge readout: features @ W ≈ actions. Connectome stays frozen."""
    if features.ndim != 2 or actions.ndim != 2:
        raise ValueError("features and actions must be 2-D")
    if features.shape[0] != actions.shape[0]:
        raise ValueError("row count mismatch")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(actions, dtype=np.float64)
    if sample_weight is not None:
        if sample_weight.shape[0] != x.shape[0]:
            raise ValueError("sample_weight length mismatch")
        scale = np.sqrt(np.clip(np.asarray(sample_weight, dtype=np.float64), 0.0, None))[:, None]
        x = x * scale
        y = y * scale
    n = x.shape[1]
    gram = x.T @ x + l2 * np.eye(n)
    return np.linalg.solve(gram, x.T @ y)


def random_readout(n_features: int, n_actions: int = 4, seed: int = 0, scale: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, size=(int(n_features), int(n_actions)))


def apply_linear(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    return features @ weights


def _feature_std(x: np.ndarray, min_std: float = 0.05) -> np.ndarray:
    """Standardize; keep millimetre proprio from vanishing in 35-D distance."""
    std = np.std(np.asarray(x, dtype=np.float64), axis=0)
    std = np.maximum(std, float(min_std))
    d = int(std.size)
    if d >= 9:
        # pack_features body tail: grip, tcp_x/400, tcp_y/400, tcp_z/200,
        # yaw_sin, yaw_cos, attached, bias, grip_r8_mass_lowz
        std[-9] = min(float(std[-9]), 0.05)
        std[-8] = min(float(std[-8]), 0.02)
        std[-7] = min(float(std[-7]), 0.02)
        std[-6] = min(float(std[-6]), 0.04)
    return std


def _kernel_neighbors(
    query: np.ndarray,
    tables_x: np.ndarray,
    k_neighbors: int,
    min_std: float,
    column_scale: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    x = np.asarray(tables_x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != q.size:
        raise ValueError("kernel table shape mismatch")
    std = _feature_std(x, min_std=min_std)
    z = (x - q) / std
    if column_scale is not None:
        z = z * np.asarray(column_scale, dtype=np.float64).reshape(1, -1)
    d2 = np.sum(z * z, axis=1)
    n = int(d2.size)
    k_nn = max(1, min(int(k_neighbors), n))
    nn = np.argpartition(d2, k_nn - 1)[:k_nn]
    return nn, d2[nn], q


def apply_kernel(
    query: np.ndarray,
    tables_x: np.ndarray,
    tables_y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    tau: float | None = 1.5,
    column_scale: np.ndarray | None = None,
    k_neighbors: int = 12,
    min_std: float = 0.1,
) -> np.ndarray:
    """k-NN Nadaraya–Watson in standardized MaleCNS feature space.

    Floor per-column std so near-constant optical/LIF slots cannot send a live
    query to zero kernel mass. Restrict to k neighbors so 35-D distance still
    yields a non-zero action. Cap proprio std so 8 mm of TCP still matters.
    """
    y = np.asarray(tables_y, dtype=np.float64)
    x = np.asarray(tables_x, dtype=np.float64)
    if y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("kernel table shape mismatch")
    nn, d2k, _ = _kernel_neighbors(query, x, k_neighbors, min_std, column_scale)
    if tau is None or float(tau) <= 0.0:
        tau_eff = max(float(np.sqrt(np.median(d2k))), 1e-3)
    else:
        tau_eff = float(tau)
    k = np.exp(-d2k / (2.0 * tau_eff * tau_eff))
    if sample_weight is not None:
        k = k * np.asarray(sample_weight, dtype=np.float64).reshape(-1)[nn]
    s = float(k.sum())
    if s < 1e-12:
        k = np.ones_like(k)
        s = float(k.sum())
    return (k / s) @ y[nn]


def apply_local_linear(
    query: np.ndarray,
    tables_x: np.ndarray,
    tables_y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    k_neighbors: int = 32,
    min_std: float = 0.05,
    l2: float = 1e-2,
    column_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Optical+LIF neighborhood, then ridge on proprio so off-demo XY still corrects."""
    y = np.asarray(tables_y, dtype=np.float64)
    x = np.asarray(tables_x, dtype=np.float64)
    if y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("kernel table shape mismatch")
    nn, _, q = _kernel_neighbors(query, x, k_neighbors, min_std, column_scale)
    # grip, tcp_x, tcp_y, tcp_z, yaw_sin, yaw_cos (not attached)
    body = np.array([-9, -8, -7, -6, -5, -4], dtype=np.int32)
    body = body[body + x.shape[1] >= 0]
    xb = np.concatenate([x[nn][:, body], np.ones((nn.size, 1))], axis=1)
    qb = np.concatenate([q[body], np.ones(1)])
    w = np.ones(nn.size, dtype=np.float64)
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)[nn]
        w = np.maximum(w, 1e-6)
    scale = np.sqrt(w)[:, None]
    xs = xb * scale
    ys = y[nn] * scale
    gram = xs.T @ xs + float(l2) * np.eye(xs.shape[1])
    try:
        hat = np.linalg.solve(gram, xs.T @ ys)
    except np.linalg.LinAlgError:
        return apply_kernel(
            q, x, y, sample_weight=sample_weight, tau=None, k_neighbors=min(12, int(nn.size)), min_std=min_std
        )
    return qb @ hat


def absorb_constant_columns(
    weights: np.ndarray,
    features: np.ndarray,
    columns: tuple[int, ...] = (0, 1),
    bias_row: int = 15,
    atol: float = 1e-6,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Fold constant feature columns into bias. Leave varying visual columns in the map."""
    w = np.array(weights, dtype=np.float64, copy=True)
    kept: list[int] = []
    absorbed: list[int] = []
    if features.ndim != 2 or features.shape[0] == 0:
        return w, kept, absorbed
    for c in columns:
        if c >= features.shape[1] or c >= w.shape[0] or bias_row >= w.shape[0]:
            continue
        if float(np.std(features[:, c])) <= atol:
            mean = float(features[:, c].mean())
            w[bias_row, :] = w[bias_row, :] + mean * w[c, :]
            w[c, :] = 0.0
            absorbed.append(int(c))
        else:
            kept.append(int(c))
    return w, kept, absorbed


def save_readout(
    path: Path,
    weights: np.ndarray,
    meta: dict,
    kernel_x: np.ndarray | None = None,
    kernel_y: np.ndarray | None = None,
    kernel_w: np.ndarray | None = None,
) -> None:
    payload = {
        "weights": np.asarray(weights, dtype=np.float32),
        "meta": json.dumps(meta),
    }
    if kernel_x is not None and kernel_y is not None:
        payload["kernel_x"] = np.asarray(kernel_x, dtype=np.float32)
        payload["kernel_y"] = np.asarray(kernel_y, dtype=np.float32)
        if kernel_w is not None:
            payload["kernel_w"] = np.asarray(kernel_w, dtype=np.float32)
    np.savez(path, **payload)


def load_readout(path: Path) -> tuple[np.ndarray, dict]:
    blob = np.load(path, allow_pickle=False)
    meta = json.loads(str(blob["meta"])) if "meta" in blob.files else {}
    return np.asarray(blob["weights"], dtype=np.float64), meta


def load_kernel_tables(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    blob = np.load(path, allow_pickle=False)
    if "kernel_x" not in blob.files or "kernel_y" not in blob.files:
        return None
    x = np.asarray(blob["kernel_x"], dtype=np.float64)
    y = np.asarray(blob["kernel_y"], dtype=np.float64)
    w = np.asarray(blob["kernel_w"], dtype=np.float64) if "kernel_w" in blob.files else np.ones(x.shape[0])
    return x, y, w
