"""Scored DN pool rates. T1 MNs are logged, not the joystick. GF is abort-only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POOL_ORDER = (
    "DNfl",
    "DNxl",
    "DNa01",
    "DNa02_L",
    "DNa02_R",
    "DNp01",
    "MDN",
    "DNp07",
    "DNp10",
)
BUS_NAMES = ("fl", "xl", "a01", "a02", "gf", "mdn", "p07", "p10")
DEAD_HZ_EPS = 0.05


def _mean(hz: np.ndarray, idx: np.ndarray) -> float:
    if idx is None or len(idx) == 0:
        return 0.0
    return float(np.mean(hz[np.asarray(idx, dtype=np.int32)]))


@dataclass
class BusRates:
    vec: np.ndarray
    pools: dict[str, float]
    t1_mn_hz: float
    abort: bool
    scored_mean_hz: float
    scored: np.ndarray

    def silenced(self) -> "BusRates":
        z = np.zeros_like(self.vec)
        pools = {k: 0.0 for k in self.pools}
        return BusRates(
            vec=z,
            pools=pools,
            t1_mn_hz=self.t1_mn_hz,
            abort=False,
            scored_mean_hz=0.0,
            scored=np.zeros_like(self.scored),
        )


def dead_pools(pools: dict[str, float], eps: float = DEAD_HZ_EPS) -> list[str]:
    out: list[str] = []
    for name in POOL_ORDER:
        if float(pools.get(name) or 0.0) <= float(eps):
            out.append(name)
    return out


def pool_rates(hz: np.ndarray, groups: dict[str, np.ndarray], abort_hz: float = 2000.0) -> BusRates:
    pools = {name: _mean(hz, groups.get(name, np.zeros(0, dtype=np.int32))) for name in POOL_ORDER}
    a02 = pools["DNa02_R"] - pools["DNa02_L"]
    vec = np.array(
        [
            pools["DNfl"],
            pools["DNxl"],
            pools["DNa01"],
            a02,
            pools["DNp01"],
            pools["MDN"],
            pools["DNp07"],
            pools["DNp10"],
        ],
        dtype=np.float64,
    )
    scored_idx = np.asarray(groups.get("scored_dn", np.zeros(0, dtype=np.int32)), dtype=np.int32)
    scored = hz[scored_idx] if scored_idx.size else np.zeros(0, dtype=np.float64)
    t1 = _mean(hz, groups.get("leg_mn_front", np.zeros(0, dtype=np.int32)))
    return BusRates(
        vec=vec,
        pools=pools,
        t1_mn_hz=t1,
        abort=pools["DNp01"] >= float(abort_hz),
        scored_mean_hz=float(np.mean(scored)) if scored.size else 0.0,
        scored=np.asarray(scored, dtype=np.float64),
    )


def pick_unpack_report(buses: np.ndarray, teacher_xyz: np.ndarray, fly_z: np.ndarray) -> dict:
    """Correlations only. Does not fit U when DNa01/DNp07/DNp10 never move."""
    X = np.asarray(buses, dtype=np.float64)
    Y = np.asarray(teacher_xyz, dtype=np.float64)
    fz = np.asarray(fly_z, dtype=np.float64).reshape(-1)
    fit: dict = {
        "rank": None,
        "bus_std": None,
        "can_unpack_pick": False,
        "note": "need cube_bus on hops",
    }
    if X.ndim != 2 or X.shape[0] < 2 or Y.ndim != 2 or Y.shape[0] != X.shape[0]:
        return fit
    fit["bus_std"] = [float(x) for x in np.std(X, axis=0)]
    fit["bus_mean"] = [float(x) for x in np.mean(X, axis=0)]
    fit["teacher_std"] = [float(x) for x in np.std(Y, axis=0)]
    xc = X - X.mean(axis=0)
    fit["rank"] = int(np.linalg.matrix_rank(xc, tol=1e-3))
    a01 = X[:, 2] if X.shape[1] > 2 else np.zeros(X.shape[0])
    p07 = X[:, 6] if X.shape[1] > 6 else np.zeros(X.shape[0])
    p10 = X[:, 7] if X.shape[1] > 7 else np.zeros(X.shape[0])
    silent_pick = float(a01.std()) < 1e-3 and float(p07.std()) < 1e-3 and float(p10.std()) < 1e-3
    mdn = X[:, 5] if X.shape[1] > 5 else np.zeros(X.shape[0])
    tz = Y[:, 2] if Y.shape[1] > 2 else np.zeros(Y.shape[0])
    ty = Y[:, 1] if Y.shape[1] > 1 else np.zeros(Y.shape[0])
    a02 = X[:, 3] if X.shape[1] > 3 else np.zeros(X.shape[0])
    if mdn.std() > 1e-6 and tz.std() > 1e-6:
        fit["corr_mdn_teacher_z"] = float(np.corrcoef(mdn, tz)[0, 1])
    if a02.std() > 1e-6 and ty.std() > 1e-6:
        fit["corr_a02_teacher_y"] = float(np.corrcoef(a02, ty)[0, 1])
    if fz.size == tz.size and fz.std() > 1e-6 and tz.std() > 1e-6:
        fit["corr_fly_z_teacher_z"] = float(np.corrcoef(fz, tz)[0, 1])
    corr_z = fit.get("corr_fly_z_teacher_z")
    can = (not silent_pick) and corr_z is not None and float(corr_z) > 0.2
    fit["can_unpack_pick"] = bool(can)
    if silent_pick or not can:
        fit["note"] = (
            "DNa01/DNp07/DNp10 silent or fly_z anti-correlates with teacher_z; "
            "linear U would interpolate the pick and is not fit"
        )
        return fit
    w, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ w
    fit["mse"] = float(np.mean((pred - Y) ** 2))
    fit["pred"] = pred.round(2).tolist()
    fit["teacher"] = Y.round(2).tolist()
    fit["note"] = "linear bus→TCP over --horizon frames."
    return fit


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (na * nb))
