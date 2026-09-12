from __future__ import annotations

import numpy as np

from malecns_cache.graph import Connectome
from runtime.lif import LIFNetwork


class KCToMBON:
    """Dopamine-gated updates on existing KC→MBON11 synapses only."""

    def __init__(self, connectome: Connectome, brain: LIFNetwork):
        self.brain = brain
        self.kc = connectome.indices("KC")
        self.mbon = connectome.indices("MBON11")
        self.ppl = connectome.indices("PPL101")
        kc_set = set(int(i) for i in self.kc)
        slots, pres, posts, rest = [], [], [], []
        W = brain.weights
        for post in self.mbon:
            post = int(post)
            for k in range(int(W.indptr[post]), int(W.indptr[post + 1])):
                pre = int(W.indices[k])
                if pre in kc_set:
                    slots.append(k)
                    pres.append(pre)
                    posts.append(post)
                    rest.append(float(W.data[k]))
        self.slots = np.asarray(slots, dtype=np.int32)
        self.pre = np.asarray(pres, dtype=np.int32)
        self.post = np.asarray(posts, dtype=np.int32)
        self.rest = np.asarray(rest, dtype=np.float32)
        self.trace = np.zeros(self.slots.size, dtype=np.float32)
        self.decay = np.float32(0.65)

    def pulse(self, amplitude: float) -> None:
        if self.ppl.size:
            self.brain.i_ext[self.ppl] = np.float32(amplitude)

    def _pre_post(self) -> tuple[np.ndarray, np.ndarray]:
        duty = getattr(self.brain, "duty", None)
        if duty is not None and duty.shape[0] == self.brain.n and float(np.max(duty)) > 0:
            pre, post = duty[self.pre], duty[self.post]
        else:
            rates = self.brain.rate_hz
            pre, post = rates[self.pre] / 40.0, rates[self.post] / 40.0
        # MBON11 is two cells and often silent in 16 hops; the KC pattern is the credit.
        return np.clip(pre, 0, 4), np.clip(np.maximum(post, 0.25), 0, 4)

    def accumulate(self) -> None:
        """Eligibility from this tick's KC × MBON duty. Call on every vision frame."""
        if self.slots.size == 0:
            return
        pre, post = self._pre_post()
        self.trace *= self.decay
        self.trace += pre * post

    def update(self, reward: float, eta: float = 2e-4) -> int:
        if self.slots.size == 0 or abs(reward) < 1e-6:
            return 0
        self.accumulate()
        duty = getattr(self.brain, "duty", self.brain.rate_hz)
        da_hz = float(np.mean(self.brain.rate_hz[self.ppl])) if self.ppl.size else 0.0
        da_duty = float(np.mean(duty[self.ppl])) if self.ppl.size and getattr(self.brain, "duty", None) is not None else 0.0
        kc_m = float(np.mean(duty[self.kc])) if self.kc.size else 0.0
        kc_s = float(np.std(duty[self.kc])) if self.kc.size else 0.0
        mb_m = float(np.mean(duty[self.mbon])) if self.mbon.size else 0.0
        print(f"  duty  KC={kc_m:.3f}±{kc_s:.3f} MBON11={mb_m:.3f} PPL_duty={da_duty:.3f} PPL_Hz={da_hz:.1f} |trace|={float(np.linalg.norm(self.trace)):.3f}", flush=True)
        # Valence is the task reward. PPL rate must not flip the sign (it saturates at kHz).
        scale = float(np.clip(0.35 + da_duty, 0.35, 1.5))
        gate = float(np.clip(reward, -2, 2)) * scale
        dw = np.float32(eta * gate) * self.trace
        W = self.brain.weights
        bound = np.maximum(np.abs(self.rest) * 8.0, 1.0).astype(np.float32)
        w = W.data[self.slots] + dw
        pos = self.rest >= 0
        w = np.where(pos, np.clip(w, 0.0, bound), np.clip(w, -bound, 0.0))
        W.data[self.slots] = w.astype(np.float32, copy=False)
        self.trace *= np.float32(0.05)
        return int(self.slots.size)
