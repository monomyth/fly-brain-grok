"""Thin U: pool rates → TCP delta + grip. Fail closed if r ≈ 0. No neck/flight DN → Δy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.decoder import ArmCommand

from .bus import BUS_NAMES, BusRates


@dataclass
class UParams:
    """Hand scale plus a few mix weights. Keep this tiny; topology is in MaleCNS."""

    dx_scale: float = 4.0
    dy_scale: float = 4.0
    dz_scale: float = 4.0
    dgrip_scale: float = 4.0
    w_fl: float = 1.0
    w_xl: float = 1.0
    w_a01: float = 1.0
    w_a02: float = 1.0
    w_mdn: float = 1.0
    w_p07: float = 0.5
    w_p10: float = 0.5
    w_contact: float = 0.0
    abort_hz: float = 2000.0
    abort_dx: float = -5.0
    abort_dz: float = 6.0
    abort_open: float = 8.0
    eps: float = 1e-3
    step: float = 8.0
    rate_div: float = 400.0  # not a trained U param; omitted from as_vector

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.dx_scale,
                self.dy_scale,
                self.dz_scale,
                self.dgrip_scale,
                self.w_fl,
                self.w_xl,
                self.w_a01,
                self.w_a02,
                self.w_mdn,
                self.w_p07,
                self.w_p10,
                self.w_contact,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, vec: np.ndarray, base: "UParams | None" = None) -> "UParams":
        u = cls() if base is None else cls(**base.__dict__)
        v = np.asarray(vec, dtype=np.float64).reshape(-1)
        names = [
            "dx_scale",
            "dy_scale",
            "dz_scale",
            "dgrip_scale",
            "w_fl",
            "w_xl",
            "w_a01",
            "w_a02",
            "w_mdn",
            "w_p07",
            "w_p10",
            "w_contact",
        ]
        for name, val in zip(names, v):
            setattr(u, name, float(val))
        return u

    def n_params(self) -> int:
        return int(self.as_vector().size)


U0 = UParams()


def unpack(
    rates: BusRates,
    u: UParams | None = None,
    gate: float = 1.0,
    attached: bool = False,
) -> ArmCommand:
    u = u or U0
    if float(np.linalg.norm(rates.vec)) < float(u.eps) or rates.scored_mean_hz < 1e-9:
        return ArmCommand(0.0, 0.0, 0.0, 0.0)
    if rates.abort or rates.pools.get("DNp01", 0.0) >= u.abort_hz:
        return ArmCommand(u.abort_dx, 0.0, u.abort_dz, u.abort_open).clip(u.step)
    g = float(gate)
    fl, xl, a01, a02, _gf, mdn, p07, p10 = (float(x) for x in rates.vec)
    div = max(float(u.rate_div), 1e-6)
    dx = u.dx_scale * np.tanh((u.w_a01 * a01 + u.w_fl * fl) / div) * g
    dy = u.dy_scale * np.tanh((u.w_a02 * a02) / div) * g
    # pinch Z is scored Hz so MDN descent cannot saturate the lift
    contact = 1.0 if attached else 0.0
    z_pre = u.w_p07 * p07 + u.w_p10 * p10 - u.w_mdn * mdn
    z_post = u.w_contact * float(rates.scored_mean_hz)
    z_mix = (1.0 - contact) * z_pre + contact * z_post
    dz = u.dz_scale * np.tanh(z_mix / div) * g
    dgrip = u.dgrip_scale * np.tanh((u.w_xl * xl) / div) * g
    return ArmCommand(float(dx), float(dy), float(dz), float(dgrip)).clip(u.step)


def command_dict(cmd: ArmCommand) -> dict[str, float]:
    return {
        "dx_mm": float(cmd.dx_mm),
        "dy_mm": float(cmd.dy_mm),
        "dz_mm": float(cmd.dz_mm),
        "dgrip_mm": float(cmd.dgrip_mm),
    }


def command_saturated(cmd: ArmCommand | dict, u: UParams | None = None, frac: float = 0.95) -> bool:
    """True when several TCP axes sit at tanh/clip rails (always-max, not a pose)."""
    u = u or U0
    if isinstance(cmd, dict):
        axes = (
            float(cmd.get("dx_mm", 0.0)),
            float(cmd.get("dy_mm", 0.0)),
            float(cmd.get("dz_mm", 0.0)),
            float(cmd.get("dgrip_mm", 0.0)),
        )
    else:
        axes = (float(cmd.dx_mm), float(cmd.dy_mm), float(cmd.dz_mm), float(cmd.dgrip_mm))
    caps = (
        min(float(u.step), abs(float(u.dx_scale))),
        min(float(u.step), abs(float(u.dy_scale))),
        min(float(u.step), abs(float(u.dz_scale))),
        min(float(u.step), abs(float(u.dgrip_scale))),
    )
    n_sat = sum(c > 0.0 and abs(v) >= float(frac) * c for v, c in zip(axes, caps))
    return n_sat >= 2


def command_is_abort(cmd: ArmCommand | dict, u: UParams | None = None) -> bool:
    u = u or U0
    if isinstance(cmd, dict):
        dx, dy, dz, dg = float(cmd.get("dx_mm", 0)), float(cmd.get("dy_mm", 0)), float(cmd.get("dz_mm", 0)), float(cmd.get("dgrip_mm", 0))
    else:
        dx, dy, dz, dg = float(cmd.dx_mm), float(cmd.dy_mm), float(cmd.dz_mm), float(cmd.dgrip_mm)
    adx = float(np.clip(u.abort_dx, -u.step, u.step))
    adz = float(np.clip(u.abort_dz, -u.step, u.step))
    adg = float(np.clip(u.abort_open, -u.step, u.step))
    return abs(dx - adx) < 0.25 and abs(dy) < 0.25 and abs(dz - adz) < 0.25 and abs(dg - adg) < 0.25


def grip_target_mm(gripper_mm: float, dgrip_mm: float, size_mm: float) -> float:
    """Close toward cube width. Lab floor stops the pads on the solid."""
    close_mm = float(size_mm)
    return float(np.clip(float(gripper_mm) + float(dgrip_mm), close_mm, 90.0))


assert U0.n_params() <= 100, "U grew too large"
assert list(BUS_NAMES) == ["fl", "xl", "a01", "a02", "gf", "mdn", "p07", "p10"]
