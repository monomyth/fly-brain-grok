from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class LIFNetwork:
    """Vectorized LIF on a frozen signed CSR connectome."""

    def __init__(
        self,
        weights: csr_matrix,
        dt: float = 1e-4,
        tau: float = 0.02,
        thresh: float = 1.0,
        reset: float = 0.0,
        synaptic_gain: float = 0.02,
        rate_tau: float = 0.05,
    ):
        if weights.shape[0] != weights.shape[1]:
            raise ValueError("weights must be square")
        if weights.data.dtype != np.float32:
            weights.data = np.asarray(weights.data, dtype=np.float32)
        self.weights = weights
        self.n = int(weights.shape[0])
        self.dt = float(dt)
        self.tau = float(tau)
        self.thresh = float(thresh)
        self.reset = float(reset)
        self.synaptic_gain = float(synaptic_gain)
        self.rate_tau = float(rate_tau)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.rate_hz = np.zeros(self.n, dtype=np.float32)
        self.duty = np.zeros(self.n, dtype=np.float32)
        self.i_ext = np.zeros(self.n, dtype=np.float32)

    def reset_state(self) -> None:
        self.v.fill(0)
        self.spikes.fill(0)
        self.rate_hz.fill(0)
        self.duty.fill(0)
        self.i_ext.fill(0)

    def set_current(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.i_ext.fill(0)
        if len(indices):
            self.i_ext[np.asarray(indices, dtype=np.int32)] = np.asarray(values, dtype=np.float32)

    def step(self, nsteps: int = 1) -> np.ndarray:
        leak = np.float32(self.dt / self.tau)
        rate_a = np.float32(self.dt / self.rate_tau)
        inv_dt = np.float32(1.0 / self.dt)
        hops = max(int(nsteps), 1)
        count = np.zeros(self.n, dtype=np.float32)
        for _ in range(hops):
            current = self.weights.dot(self.spikes) * self.synaptic_gain + self.i_ext
            self.v += leak * (-self.v + current)
            fired = self.v >= self.thresh
            self.v[fired] = self.reset
            self.spikes = fired.astype(np.float32)
            count += self.spikes
            inst = self.spikes * inv_dt
            self.rate_hz += rate_a * (inst - self.rate_hz)
        self.duty = count / np.float32(hops)
        if not np.isfinite(self.v).all():
            raise FloatingPointError("LIF voltage is not finite")
        return self.spikes
