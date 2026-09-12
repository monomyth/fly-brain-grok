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


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(x, y) / (na * nb))
