"""JPEG → local contrast → Front / Gripper photoreceptor currents. Cameras stay split."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def luma(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError("rgb must be H×W×3")
    if x.max() > 1.5:
        x = x / 255.0
    return 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]


CONTRAST_SCALE = 30.0
LUMA_SCALE = 0.0
CHROMA_SCALE = 14.0
COLUMN_SCALE = 4.0
# Gripper JPEG: cube sits in the lower field (tarsus look-down).
GRIP_Y_FRAC = (0.55, 1.0)


def orange_chroma(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel orange vs gray table. R8-like; gray luma is ~0."""
    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError("rgb must be H×W×3")
    if x.max() > 1.5:
        x = x / 255.0
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    return np.maximum(r - np.maximum(g, b), 0.0).astype(np.float32)


def cube_chroma_frac(rgb: np.ndarray) -> float:
    """Fraction of pixels that look like the orange cube, not gray table."""
    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError("rgb must be H×W×3")
    if x.max() > 1.5:
        x = x / 255.0
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    return float(np.mean((r > 0.45) & (r > g + 0.08) & (r > b + 0.08)))


def local_contrast(rgb: np.ndarray, sigma_on: float = 1.0, sigma_off: float = 3.0) -> np.ndarray:
    gray = luma(rgb)
    return gaussian_filter(gray, sigma_on) - gaussian_filter(gray, sigma_off)


def column_local(gray: np.ndarray) -> np.ndarray:
    """ON deviation from the column mean."""
    g = np.asarray(gray, dtype=np.float32)
    mu = g.mean(axis=0, keepdims=True)
    return np.maximum(g - mu, 0.0)


def _match_luma_moments(plane: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = np.asarray(plane, dtype=np.float32)
    r = np.asarray(ref, dtype=np.float32)
    xs, xm = float(x.std()), float(x.mean())
    rs, rm = float(r.std()), float(r.mean())
    if xs < 1e-8:
        return np.full_like(x, rm)
    return (x - xm) * (rs / xs) + rm


def phase_scramble(rgb: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Destroy spatial structure; keep luma mean/std."""
    gray = luma(rgb)
    rng = rng or np.random.default_rng(0)
    spec = np.fft.rfft2(gray)
    mag = np.abs(spec)
    ang = rng.uniform(0.0, 2.0 * np.pi, size=spec.shape)
    ang[0, 0] = 0.0
    scrambled = np.fft.irfft2(mag * np.exp(1j * ang), s=gray.shape).astype(np.float32)
    scrambled = _match_luma_moments(scrambled, gray)
    lo, hi = float(gray.min()), float(gray.max())
    scrambled = np.clip(scrambled, lo, hi)
    scrambled = _match_luma_moments(scrambled, gray)
    return np.stack([scrambled, scrambled, scrambled], axis=2)


def camera_hex_spans(
    hex1: np.ndarray,
    hex2: np.ndarray,
    world: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if world is not None:
        return (float(world[0][0]), float(world[0][1])), (float(world[1][0]), float(world[1][1]))
    h1 = np.asarray(hex1, dtype=np.float32).reshape(-1)
    h2 = np.asarray(hex2, dtype=np.float32).reshape(-1)
    valid = (h1 >= 0) & (h2 >= 0)
    if not bool(np.any(valid)):
        return (1.0, 36.0), (1.0, 39.0)
    return (
        (float(h1[valid].min()), float(h1[valid].max())),
        (float(h2[valid].min()), float(h2[valid].max())),
    )


def _sample_plane(plane: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    coords = np.vstack([ys, xs])
    return map_coordinates(plane, coords, order=1, mode="nearest").astype(np.float32)


def _hex_xy(
    hex1: np.ndarray,
    hex2: np.ndarray,
    h: int,
    w: int,
    hex1_span: tuple[float, float],
    hex2_span: tuple[float, float],
    y_frac: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    n = int(hex1.size)
    h1 = np.asarray(hex1, dtype=np.float32)
    h2 = np.asarray(hex2, dtype=np.float32)
    valid = (h1 >= 0) & (h2 >= 0)
    xs = np.zeros(n, dtype=np.float32)
    ys = np.zeros(n, dtype=np.float32)
    span1 = max(hex1_span[1] - hex1_span[0], 1.0)
    span2 = max(hex2_span[1] - hex2_span[0], 1.0)
    xs[valid] = (h1[valid] - hex1_span[0]) / span1 * (w - 1)
    y0 = float(y_frac[0]) * (h - 1)
    y1 = float(y_frac[1]) * (h - 1)
    ys[valid] = (h2[valid] - hex2_span[0]) / span2 * (y1 - y0) + y0
    if not bool(np.any(valid)):
        cols = max(1, int(round(np.sqrt(n * w / max(h, 1)))))
        rows = int(np.ceil(n / cols))
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, w - 1, cols, dtype=np.float32),
            np.linspace(0, h - 1, rows, dtype=np.float32),
        )
        xs = grid_x.ravel()[:n]
        ys = grid_y.ravel()[:n]
    return ys, xs


def receptor_currents(
    contrast: np.ndarray,
    hex1: np.ndarray,
    hex2: np.ndarray,
    *,
    scale: float = CONTRAST_SCALE,
    luma_plane: np.ndarray | None = None,
    luma_scale: float = LUMA_SCALE,
    hex1_span: tuple[float, float] = (1.0, 36.0),
    hex2_span: tuple[float, float] = (1.0, 39.0),
    y_frac: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """R1–R6: ON contrast plus optional luma; clip to [0, 8]."""
    n = int(hex1.size)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    on = np.maximum(contrast, 0.0)
    h, w = int(contrast.shape[0]), int(contrast.shape[1])
    ys, xs = _hex_xy(hex1, hex2, h, w, hex1_span, hex2_span, y_frac=y_frac)
    on_s = _sample_plane(on, ys, xs)
    val = on_s * float(scale)
    if luma_plane is not None:
        val = val + _sample_plane(np.asarray(luma_plane, dtype=np.float32), ys, xs) * float(luma_scale)
    return np.clip(val, 0.0, 8.0).astype(np.float32)


def drive_camera(
    rgb: np.ndarray,
    hex1: np.ndarray,
    hex2: np.ndarray,
    scale: float = CONTRAST_SCALE,
    luma_scale: float = LUMA_SCALE,
    chroma_scale: float = CHROMA_SCALE,
    column_scale: float = COLUMN_SCALE,
    hex1_span: tuple[float, float] | None = None,
    hex2_span: tuple[float, float] | None = None,
    y_frac: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    chroma = orange_chroma(rgb)
    if float(chroma_scale) > 0.0:
        plane, pscale = chroma, float(chroma_scale)
        contrast = gaussian_filter(chroma, 1.0) - gaussian_filter(chroma, 3.0)
    elif float(luma_scale) > 0.0:
        gray = luma(rgb)
        plane, pscale = gray, float(luma_scale)
        contrast = local_contrast(rgb)
    elif float(column_scale) > 0.0:
        gray = luma(rgb)
        plane, pscale = column_local(gray), float(column_scale)
        contrast = local_contrast(rgb)
    else:
        plane, pscale = None, 0.0
        contrast = local_contrast(rgb)
    s1 = hex1_span if hex1_span is not None else (1.0, 36.0)
    s2 = hex2_span if hex2_span is not None else (1.0, 39.0)
    return receptor_currents(
        contrast,
        hex1,
        hex2,
        scale=scale,
        luma_plane=plane,
        luma_scale=pscale,
        hex1_span=s1,
        hex2_span=s2,
        y_frac=y_frac,
    )
