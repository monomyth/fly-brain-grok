from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from malecns_cache.graph import Connectome


@dataclass
class ArmCommand:
    dx_mm: float
    dy_mm: float
    dz_mm: float
    dgrip_mm: float
    keep_level: bool = True

    def clip(self, step: float = 5.0) -> "ArmCommand":
        return ArmCommand(
            dx_mm=float(np.clip(self.dx_mm, -step, step)),
            dy_mm=float(np.clip(self.dy_mm, -step, step)),
            dz_mm=float(np.clip(self.dz_mm, -step, step)),
            dgrip_mm=float(np.clip(self.dgrip_mm, -step, step)),
            keep_level=self.keep_level,
        )


def _mean_rate(rate_hz: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    return float(np.mean(rate_hz[indices]))


def decode_contrast(connectome: Connectome, signal: np.ndarray, step: float = 4.0) -> ArmCommand:
    """Left/right population contrast. Engineered joystick, not a natural map."""

    def pop(name: str) -> float:
        return _mean_rate(signal, connectome.indices(name))

    def lr(left: str, right: str) -> float:
        a, b = pop(left), pop(right)
        return step * (b - a) / (abs(a) + abs(b) + 1e-6)

    move = pop("DNpe017")
    floor = float(np.mean(signal)) + 1e-6
    dz = step * float(np.tanh(move / (10.0 * floor)))
    return ArmCommand(
        dx_mm=lr("DNa02_L", "DNa02_R"),
        dy_mm=lr("DNp20_L", "DNp20_R"),
        dz_mm=dz,
        dgrip_mm=-dz,
    ).clip(step)


_abs_weights = None


def hop_drive(weights, i_ext: np.ndarray, hops: int = 3) -> np.ndarray:
    """Unsigned |W|^k |i_ext|. LIF spikes do not reach DNs in tens of hops."""
    global _abs_weights
    if _abs_weights is None or _abs_weights.shape != weights.shape:
        _abs_weights = weights.tocsr(copy=True)
        _abs_weights.data = np.abs(_abs_weights.data)
    sig = np.abs(np.asarray(i_ext, dtype=np.float32))
    for _ in range(max(int(hops), 0)):
        sig = _abs_weights.dot(sig)
    return np.log1p(sig)


def decode_hop_drive(connectome: Connectome, i_ext: np.ndarray, hops: int = 3, step: float = 4.0) -> ArmCommand:
    return decode_contrast(connectome, hop_drive(connectome.weights, i_ext, hops=hops), step=step)


def decode_dn_rates(connectome: Connectome, rate_hz: np.ndarray, scale: float = 0.02) -> ArmCommand:
    """Engineered DN joystick. Not a proven natural motor map.

    DNp20 right-minus-left → base Y. DNa02 right-minus-left → base X.
    DNpe017 rate → lift and gripper close.
    """
    r_l = _mean_rate(rate_hz, connectome.indices("DNp20_L"))
    r_r = _mean_rate(rate_hz, connectome.indices("DNp20_R"))
    a_l = _mean_rate(rate_hz, connectome.indices("DNa02_L"))
    a_r = _mean_rate(rate_hz, connectome.indices("DNa02_R"))
    move = _mean_rate(rate_hz, connectome.indices("DNpe017"))
    dy = scale * (r_r - r_l)
    dx = scale * (a_r - a_l)
    dz = scale * move
    dgrip = -scale * move
    return ArmCommand(dx_mm=dx, dy_mm=dy, dz_mm=dz, dgrip_mm=dgrip).clip()


def _pool_contrast(rate_hz: np.ndarray, connectome: Connectome, plus: str, minus: str) -> float:
    a = _mean_rate(rate_hz, connectome.indices(plus))
    b = _mean_rate(rate_hz, connectome.indices(minus))
    return (a - b) / (abs(a) + abs(b) + 1e-6)


def decode_leg_mn(connectome: Connectome, rate_hz: np.ndarray, step: float = 4.0) -> ArmCommand:
    """Engineered joystick from VNC front-leg motor neurons. Not a natural B601 map."""
    dx = step * _pool_contrast(rate_hz, connectome, "Ti_extensor", "Ti_flexor")
    dy = step * _pool_contrast(rate_hz, connectome, "Tr_extensor", "Tr_flexor")
    reach = _mean_rate(rate_hz, connectome.indices("Fe_reductor"))
    st = _mean_rate(rate_hz, connectome.indices("Sternotrochanter"))
    floor = float(np.mean(rate_hz)) + 1e-6
    dz = step * float(np.tanh((reach + st) / (10.0 * floor)))
    dgrip = step * _pool_contrast(rate_hz, connectome, "Ta_depressor", "Ta_levator")
    if abs(dx) + abs(dy) + abs(dz) + abs(dgrip) < 1e-9:
        # unnamed T1 cells still count if named pools are empty or silent
        front = _mean_rate(rate_hz, connectome.indices("leg_mn_front"))
        dx = step * float(np.tanh(front / (10.0 * floor)))
    return ArmCommand(dx_mm=dx, dy_mm=dy, dz_mm=dz, dgrip_mm=dgrip).clip(step)
