from __future__ import annotations

import math

import numpy as np

from malecns_cache.graph import Connectome
from PIL import Image

from runtime.encoder import OPTICAL_NAMES, chroma_grid, load_image, optical_vector
from runtime.lif import LIFNetwork
from train.dataset import Sample

LIF_NAMES = [
    "DNp20_L",
    "DNp20_R",
    "DNa02_L",
    "DNa02_R",
    "DNpe017",
    "PPL101",
    "MBON11",
    "KC_mean",
]
BODY_NAMES = ["grip", "tcp_x", "tcp_y", "tcp_z", "yaw_sin", "yaw_cos", "attached", "bias"]
GRIP_GRID_NAMES = [f"grip_r8g_{r}{c}" for r in range(4) for c in range(4)]
FEATURE_NAMES = (
    [f"front_{n}" for n in OPTICAL_NAMES]
    + [f"grip_{n}" for n in OPTICAL_NAMES]
    + GRIP_GRID_NAMES
    + LIF_NAMES
    + BODY_NAMES
    + ["grip_r8_mass_lowz"]
)
BIAS_INDEX = FEATURE_NAMES.index("bias")
VISUAL_INDEXES = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith(("front_", "grip_")) or name in LIF_NAMES]


def _optical_blank(r1_mean: float = 0.45, r8_mean: float = 0.0) -> np.ndarray:
    return np.array([r1_mean, r8_mean, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def pack_features(
    front_opt: np.ndarray,
    grip_opt: np.ndarray,
    lif: np.ndarray,
    gripper_mm: float,
    tcp_mm: np.ndarray,
    attached: bool,
    yaw_rad: float = 0.0,
    grip_grid: np.ndarray | None = None,
) -> np.ndarray:
    del attached  # never a policy switch; slot stays 0
    tcp = np.asarray(tcp_mm, dtype=np.float64).reshape(-1)
    z_norm = float(tcp[2]) / 200.0
    grip_mass = float(np.asarray(grip_opt, dtype=np.float64).reshape(-1)[4]) if np.asarray(grip_opt).size > 4 else 0.0
    lowz = float(np.clip(1.0 - z_norm, 0.0, 1.0))
    grid = np.zeros(len(GRIP_GRID_NAMES), dtype=np.float64)
    if grip_grid is not None:
        g = np.asarray(grip_grid, dtype=np.float64).reshape(-1)
        grid[: min(g.size, grid.size)] = g[: grid.size]
    body = np.array(
        [
            float(gripper_mm) / 90.0,
            float(tcp[0]) / 400.0,
            float(tcp[1]) / 400.0,
            z_norm,
            math.sin(float(yaw_rad)),
            math.cos(float(yaw_rad)),
            0.0,
            1.0,
            grip_mass * lowz,
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            np.asarray(front_opt, dtype=np.float64).reshape(-1)[: len(OPTICAL_NAMES)],
            np.asarray(grip_opt, dtype=np.float64).reshape(-1)[: len(OPTICAL_NAMES)],
            grid,
            np.asarray(lif, dtype=np.float64).reshape(-1)[: len(LIF_NAMES)],
            body,
        ]
    )


def proprio_feature_row(
    gripper_mm: float,
    tcp_mm: np.ndarray,
    attached: bool,
    r1_mean: float = 0.45,
    r8_mean: float = 0.0,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    """Readout row with blank optical/DN slots, proprio, yaw, bias."""
    blank = _optical_blank(r1_mean, r8_mean)
    return pack_features(blank, blank, np.zeros(len(LIF_NAMES)), gripper_mm, tcp_mm, attached, yaw_rad=yaw_rad)


def _mean(rate: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return 0.0
    return float(np.mean(rate[idx]))


def _lif_row(connectome: Connectome, rate: np.ndarray) -> np.ndarray:
    return np.array(
        [
            _mean(rate, connectome.indices("DNp20_L")),
            _mean(rate, connectome.indices("DNp20_R")),
            _mean(rate, connectome.indices("DNa02_L")),
            _mean(rate, connectome.indices("DNa02_R")),
            _mean(rate, connectome.indices("DNpe017")),
            _mean(rate, connectome.indices("PPL101")),
            _mean(rate, connectome.indices("MBON11")),
            _mean(rate, connectome.indices("KC")),
        ],
        dtype=np.float64,
    )


def features_from_rgb(
    connectome: Connectome,
    brain: LIFNetwork,
    rgb: np.ndarray,
    gripper_mm: float,
    tcp_mm: np.ndarray,
    attached: bool,
    hops: int = 3,
    proprio: bool = True,
    reset: bool = True,
    publish: bool = False,
    yaw_rad: float = 0.0,
    camera: str = "Front",
) -> np.ndarray:
    r1_idx = connectome.indices("photoreceptors_r1r6")
    r8_idx = connectome.indices("photoreceptors_r8")
    opt, r1, r8 = optical_vector(rgb, int(r1_idx.size), int(r8_idx.size))
    if reset:
        brain.reset_state()
    current = np.zeros(brain.n, dtype=np.float32)
    if r1_idx.size:
        current[r1_idx] = r1
    if r8_idx.size:
        current[r8_idx] = r8
    brain.i_ext = current
    brain.step(max(1, hops))
    if publish:
        from runtime.broadcast import publish_activity

        publish_activity(brain)
    front = opt if str(camera) != "Gripper" else _optical_blank(0.0, 0.0)
    grip = opt if str(camera) == "Gripper" else _optical_blank(0.0, 0.0)
    lif = _lif_row(connectome, brain.rate_hz)
    if not proprio:
        tcp_mm = np.zeros(3)
        gripper_mm = 0.0
        attached = False
        yaw_rad = 0.0
    return pack_features(front, grip, lif, gripper_mm, tcp_mm, attached, yaw_rad=yaw_rad)


def step_from_proprio(
    connectome: Connectome,
    brain: LIFNetwork,
    gripper_mm: float,
    tcp_mm: np.ndarray,
    cube_mm: np.ndarray,
    attached: bool,
    hops: int = 8,
) -> None:
    """Live drive without a camera snapshot. Do not reset membrane state."""
    r1 = connectome.indices("photoreceptors_r1r6")
    ppl = connectome.indices("PPL101")
    dist = float(np.linalg.norm(tcp_mm - cube_mm))
    near = float(np.clip(1.4 - dist / 220.0, 0.05, 1.8))
    pulse = 0.55 + 0.45 * np.sin(time_phase())
    current = np.zeros(brain.n, dtype=np.float32)
    if r1.size:
        current[r1] = np.float32(near * pulse)
    if attached and ppl.size:
        current[ppl] = 2.4
    brain.i_ext = current
    brain.step(max(1, hops))
    from runtime.broadcast import publish_activity

    publish_activity(brain)


_phase = 0.0


def kenyon_grid(rgb: np.ndarray, n_kc: int) -> np.ndarray:
    """Area-average luma onto a √N×√N sheet so a small cube cannot miss the lattice."""
    side = max(8, int(np.ceil(np.sqrt(max(int(n_kc), 1)))))
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    pixels = np.clip(luma * 255.0, 0, 255).astype(np.uint8)
    resample = getattr(getattr(Image, "Resampling", Image), "BOX", Image.BOX)
    small = np.asarray(Image.fromarray(pixels, mode="L").resize((side, side), resample), dtype=np.float32) / 255.0
    return small.ravel()


def drive_kenyon_from_image(
    brain,
    kc_idx: np.ndarray,
    rgb: np.ndarray,
    lo: float = 0.0,
    hi: float = 80.0,
    prev_grid: np.ndarray | None = None,
    wta: float = 0.10,
) -> np.ndarray | None:
    """Retinotopic KC sheet: one cell per √N×√N luma patch, k-WTA + motion.

    Luma floor so KC→MBON still sees current on a flat non-black field; WTA ranks contrast.
    """
    if kc_idx.size == 0:
        return None
    grid = kenyon_grid(rgb, int(kc_idx.size))
    lum = float(np.clip(grid.mean(), 0.0, 1.0))
    if kc_idx.size == 1:
        brain.i_ext[np.asarray(kc_idx, dtype=np.int32)] = np.float32(lo + hi * lum)
        return grid
    contrast = grid - float(grid.mean())
    motion = np.abs(grid - prev_grid) if prev_grid is not None and prev_grid.shape == grid.shape else np.zeros_like(grid)
    signal = contrast + np.float32(2.5) * motion
    k = max(1, int(np.ceil(wta * signal.size)))
    winners = np.argpartition(signal, -k)[-k:]
    sparse = np.zeros_like(signal)
    sparse[winners] = np.maximum(signal[winners], 0.0)
    peak = float(np.max(sparse))
    bins = (np.arange(kc_idx.size) % sparse.size).astype(np.int32)
    if peak < 1e-8:
        current = np.full(kc_idx.size, np.float32(lo + hi * lum), dtype=np.float32)
    else:
        current = np.float32(lo + hi * (sparse[bins] / peak))
        if lum > 0:
            current = np.maximum(current, np.float32(lo + 0.25 * hi * lum))
    brain.i_ext[np.asarray(kc_idx, dtype=np.int32)] = current
    return grid


def time_phase() -> float:
    global _phase
    _phase += 0.35
    return _phase


def neural_features(
    connectome: Connectome,
    brain: LIFNetwork,
    sample: Sample,
    hops: int = 3,
    proprio: bool = True,
) -> np.ndarray:
    views = dict(sample.views or {})
    if not views and sample.image is not None:
        views = {sample.camera or "Front": sample.image}
    rgbs = []
    names = []
    for camera in ("Front", "Gripper"):
        path = views.get(camera)
        if path is None and (sample.camera or "Front") == camera:
            path = sample.image
        if path is None:
            continue
        rgbs.append(load_image(path))
        names.append(camera)
    if not rgbs:
        rgbs = [load_image(sample.image)]
        names = [sample.camera or "Front"]
    return features_from_rgbs(
        connectome,
        brain,
        rgbs,
        gripper_mm=sample.gripper_mm,
        tcp_mm=sample.tcp_mm,
        attached=sample.attached,
        hops=hops,
        proprio=proprio,
        yaw_rad=float(sample.yaw_rad or 0.0),
        cameras=tuple(names),
    )


def features_from_rgbs(
    connectome: Connectome,
    brain: LIFNetwork,
    rgbs: list[np.ndarray],
    gripper_mm: float,
    tcp_mm: np.ndarray,
    attached: bool,
    hops: int = 3,
    proprio: bool = True,
    reset: bool = True,
    publish: bool = False,
    yaw_rad: float = 0.0,
    cameras: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Concat Front+Gripper optical-neuron summaries. One LIF pass on mean currents."""
    r1_idx = connectome.indices("photoreceptors_r1r6")
    r8_idx = connectome.indices("photoreceptors_r8")
    names = list(cameras) if cameras is not None else ["Front", "Gripper"][: len(rgbs)]
    if not rgbs:
        blank = np.zeros((120, 160, 3), dtype=np.float32)
        return features_from_rgb(
            connectome,
            brain,
            blank,
            gripper_mm=gripper_mm,
            tcp_mm=tcp_mm,
            attached=attached,
            hops=hops,
            proprio=proprio,
            reset=reset,
            publish=publish,
            yaw_rad=yaw_rad,
            camera="Front",
        )
    front_opt = _optical_blank(0.0, 0.0)
    grip_opt = _optical_blank(0.0, 0.0)
    grip_grid = np.zeros(len(GRIP_GRID_NAMES), dtype=np.float64)
    r1_currents = []
    r8_currents = []
    for rgb, name in zip(rgbs, names):
        opt, r1, r8 = optical_vector(rgb, int(r1_idx.size), int(r8_idx.size))
        if name == "Gripper":
            grip_opt = opt
            grip_grid = chroma_grid(rgb, rows=4, cols=4)
        else:
            front_opt = opt
        if r1.size:
            r1_currents.append(r1)
        if r8.size:
            r8_currents.append(r8)
    if reset:
        brain.reset_state()
    current = np.zeros(brain.n, dtype=np.float32)
    if r1_idx.size and r1_currents:
        current[r1_idx] = np.mean(np.vstack(r1_currents), axis=0)
    if r8_idx.size and r8_currents:
        current[r8_idx] = np.mean(np.vstack(r8_currents), axis=0)
    brain.i_ext = current
    brain.step(max(1, hops))
    if publish:
        from runtime.broadcast import publish_activity

        publish_activity(brain)
    if not proprio:
        tcp_mm = np.zeros(3)
        gripper_mm = 0.0
        attached = False
        yaw_rad = 0.0
    return pack_features(
        front_opt,
        grip_opt,
        _lif_row(connectome, brain.rate_hz),
        gripper_mm,
        tcp_mm,
        attached,
        yaw_rad=yaw_rad,
        grip_grid=grip_grid,
    )
