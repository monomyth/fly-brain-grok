"""Crop LIF with type gains g. No per-frame reset. N ≥ 100 steps per camera frame."""

from __future__ import annotations

import numpy as np

from runtime.lif import LIFNetwork

from .bus import BusRates, pool_rates
from .crop import GAIN_CLASSES, Crop
from .eye import CHROMA_SCALE, COLUMN_SCALE, CONTRAST_SCALE, GRIP_Y_FRAC, LUMA_SCALE, camera_hex_spans, drive_camera
from .log import g_hash

MIN_STEPS = 100
DEFAULT_GAIN = 1.5


def window_hz(brain: LIFNetwork) -> np.ndarray:
    return brain.duty / np.float32(brain.dt)


class CropLIF:
    def __init__(
        self,
        crop: Crop,
        synaptic_gain: float = DEFAULT_GAIN,
        nsteps: int = 150,
        contrast_scale: float = CONTRAST_SCALE,
        luma_scale: float = LUMA_SCALE,
        chroma_scale: float = CHROMA_SCALE,
        column_scale: float = COLUMN_SCALE,
    ):
        if int(nsteps) < 50:
            raise ValueError("nsteps must be ≥ 50; per-frame 3-step eval cannot propagate")
        self.crop = crop
        self.nsteps = max(int(nsteps), MIN_STEPS)
        self.contrast_scale = float(contrast_scale)
        self.luma_scale = float(luma_scale)
        self.chroma_scale = float(chroma_scale)
        self.column_scale = float(column_scale)
        self.brain = LIFNetwork(crop.weights, synaptic_gain=float(synaptic_gain))
        self.g = np.ones(len(GAIN_CLASSES), dtype=np.float32)
        self.g_init = np.ones(len(GAIN_CLASSES), dtype=np.float32)
        self.da_delta = np.zeros_like(self.crop.w0)
        indptr = crop.weights.indptr
        posts = np.repeat(np.arange(crop.n, dtype=np.int32), np.diff(indptr))
        self._post_class = crop.pre_class[posts]
        self._gf_class = crop.class_id("DNp01")
        self.apply_gains()

    def _gain_scale(self) -> np.ndarray:
        pre = self.crop.weights.indices
        scale = self.g[self.crop.pre_class[pre]]
        gf = np.float32(self.g[self._gf_class])  # g[DNp01] scales synapses onto GF
        return scale * np.where(self._post_class == self._gf_class, gf, np.float32(1.0))

    def apply_gains(self) -> None:
        self.crop.weights.data[:] = self.crop.w0 * self._gain_scale() + self.da_delta

    def record_da_delta(self) -> None:
        """Keep KC→MBON (or any in-place) writes across later set_g / apply_gains."""
        base = self.crop.w0 * self._gain_scale()
        self.da_delta = np.asarray(self.crop.weights.data, dtype=np.float32) - base

    def set_g(self, g: np.ndarray) -> None:
        self.g = np.asarray(g, dtype=np.float32).reshape(-1)
        if self.g.size != len(GAIN_CLASSES):
            raise ValueError(f"g must have {len(GAIN_CLASSES)} entries")
        self.apply_gains()

    def reset_episode(self) -> None:
        self.brain.reset_state()

    @property
    def g_hash(self) -> str:
        return g_hash(self.g)

    @property
    def g_trained(self) -> bool:
        return self.g_hash != g_hash(self.g_init)

    def _hex_world(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        raw = (self.crop.meta or {}).get("hex_world")
        if not raw:
            return None
        return (float(raw[0][0]), float(raw[0][1])), (float(raw[1][0]), float(raw[1][1]))

    def _drive(self, front_rgb: np.ndarray | None, grip_rgb: np.ndarray | None) -> None:
        self.brain.i_ext.fill(0)
        front_idx = self.crop.indices("front_r1")
        grip_idx = self.crop.indices("grip_r1")
        world = self._hex_world()
        if front_rgb is not None and front_idx.size:
            s1, s2 = camera_hex_spans(self.crop.hex1[front_idx], self.crop.hex2[front_idx], world=world)
            cur = drive_camera(
                front_rgb,
                self.crop.hex1[front_idx],
                self.crop.hex2[front_idx],
                scale=self.contrast_scale,
                luma_scale=self.luma_scale,
                chroma_scale=self.chroma_scale,
                column_scale=self.column_scale,
                hex1_span=s1,
                hex2_span=s2,
            )
            self.brain.i_ext[front_idx] = cur
        if grip_rgb is not None and grip_idx.size:
            s1, s2 = camera_hex_spans(self.crop.hex1[grip_idx], self.crop.hex2[grip_idx], world=world)
            cur = drive_camera(
                grip_rgb,
                self.crop.hex1[grip_idx],
                self.crop.hex2[grip_idx],
                scale=self.contrast_scale,
                luma_scale=self.luma_scale,
                chroma_scale=self.chroma_scale,
                column_scale=self.column_scale,
                hex1_span=s1,
                hex2_span=s2,
                y_frac=GRIP_Y_FRAC,
            )
            self.brain.i_ext[grip_idx] = cur

    def step_vision(
        self,
        front_rgb: np.ndarray | None,
        grip_rgb: np.ndarray | None,
        nsteps: int | None = None,
        drive_kc: bool = False,
    ) -> np.ndarray:
        """One camera frame. Does not reset membrane. Cameras stay split."""
        self._drive(front_rgb, grip_rgb)
        if drive_kc and front_rgb is not None:
            from train.features import drive_kenyon_from_image

            kc = self.crop.indices("KC")
            if kc.size:
                drive_kenyon_from_image(self.brain, kc, front_rgb)
        self.brain.step(int(nsteps or self.nsteps))
        return window_hz(self.brain)

    def inject(self, indices: np.ndarray, value: float, nsteps: int | None = None) -> np.ndarray:
        self.brain.i_ext.fill(0)
        if len(indices):
            self.brain.i_ext[np.asarray(indices, dtype=np.int32)] = np.float32(value)
        self.brain.step(int(nsteps or self.nsteps))
        return window_hz(self.brain)

    def rates(self) -> BusRates:
        return pool_rates(window_hz(self.brain), self.crop.groups)

    def black_baseline(self, shape: tuple[int, int, int] = (120, 160, 3)) -> BusRates:
        self.reset_episode()
        black = np.zeros(shape, dtype=np.float32)
        self.step_vision(black, black)
        rates = self.rates()
        self.reset_episode()
        return rates
