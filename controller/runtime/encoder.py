from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
from PIL import Image


def decode_jpeg(data: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _grid(n: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    cols = max(1, int(round(math.sqrt(n * width / max(height, 1)))))
    rows = int(math.ceil(n / cols))
    xs = np.linspace(0, width - 1, cols, dtype=np.float32)
    ys = np.linspace(0, height - 1, rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.ravel()[:n], grid_y.ravel()[:n]


def receptor_moments(
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
    positive: bool = True,
) -> tuple[float, float, float]:
    """Centroid and mass of receptor activity. Cube orange is positive R8 chroma."""
    w = np.asarray(values, dtype=np.float64).reshape(-1)
    x = np.asarray(xs, dtype=np.float64).reshape(-1)[: w.size]
    y = np.asarray(ys, dtype=np.float64).reshape(-1)[: w.size]
    w = np.maximum(w, 0.0) if positive else np.abs(w)
    mass = float(w.sum())
    if mass <= 1e-8 or w.size == 0:
        return 0.5, 0.5, 0.0
    cx = float(np.dot(x, w) / mass) / max(float(width - 1), 1.0)
    cy = float(np.dot(y, w) / mass) / max(float(height - 1), 1.0)
    return cx, cy, mass / (float(w.size) * 4.0)


def receptor_halves(
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    w = np.maximum(np.asarray(values, dtype=np.float64).reshape(-1), 0.0)
    x = np.asarray(xs, dtype=np.float64).reshape(-1)[: w.size]
    y = np.asarray(ys, dtype=np.float64).reshape(-1)[: w.size]
    mid_x = (width - 1) * 0.5
    mid_y = (height - 1) * 0.5
    scale = max(float(w.size) * 4.0, 1.0)
    left = float(w[x < mid_x].sum()) / scale
    right = float(w[x >= mid_x].sum()) / scale
    top = float(w[y < mid_y].sum()) / scale
    bot = float(w[y >= mid_y].sum()) / scale
    return left, right, top, bot


OPTICAL_NAMES = (
    "r1_mean",
    "r8_mean",
    "r8_cx",
    "r8_cy",
    "r8_mass",
    "r8_left",
    "r8_right",
    "r8_top",
    "r8_bot",
)


def optical_vector(rgb: np.ndarray, n_r1r6: int, n_r8: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R1/R8 currents plus a 9-D spatial summary of those optical neurons."""
    r1, r8 = encode_frame(rgb, n_r1r6, n_r8)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if r8.size:
        xs, ys = _grid(int(r8.size), width, height)
        cx, cy, mass = receptor_moments(r8, xs, ys, width, height)
        left, right, top, bot = receptor_halves(r8, xs, ys, width, height)
        r8_mean = float(r8.mean())
    else:
        cx = cy = 0.5
        mass = left = right = top = bot = 0.0
        r8_mean = 0.0
    vec = np.array(
        [
            float(r1.mean()) if r1.size else 0.0,
            r8_mean,
            cx,
            cy,
            mass,
            left,
            right,
            top,
            bot,
        ],
        dtype=np.float64,
    )
    return vec, r1, r8


def _sample(plane: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = plane.shape
    x = np.clip(xs, 0, w - 1)
    y = np.clip(ys, 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    top = plane[y0, x0] * (1 - dx) + plane[y0, x1] * dx
    bot = plane[y1, x0] * (1 - dx) + plane[y1, x1] * dx
    return top * (1 - dy) + bot * dy


def chroma_grid(rgb: np.ndarray, rows: int = 4, cols: int = 4, gain: float = 1.5) -> np.ndarray:
    """Coarse R8-chroma map of the gripper view so lift can see the cube, not a flag."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H×W×3")
    height, width, _ = rgb.shape
    chroma = rgb[:, :, 0] - rgb[:, :, 1]
    xs = np.linspace(0, width - 1, cols, dtype=np.float32)
    ys = np.linspace(0, height - 1, rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.clip(_sample(chroma, grid_x.ravel(), grid_y.ravel()) * gain, -2, 2).astype(np.float64)


def encode_grid(rgb: np.ndarray, rows: int = 8, cols: int = 8, gain: float = 1.5) -> np.ndarray:
    """Coarse luma grid so the readout can see *where* the cube is, not only mean brightness."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H×W×3")
    height, width, _ = rgb.shape
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    xs = np.linspace(0, width - 1, cols, dtype=np.float32)
    ys = np.linspace(0, height - 1, rows, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.clip(_sample(luma, grid_x.ravel(), grid_y.ravel()) * gain, 0, 4).astype(np.float32)


def encode_frame(rgb: np.ndarray, n_r1r6: int, n_r8: int = 0, gain: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Map an RGB image onto R1–R6 brightness and R8 chroma proxies.

    Photoreceptor positions are a regular grid over the frame, not a measured
    male eyemap. That is intentional and must stay labeled as a proxy.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H×W×3")
    height, width, _ = rgb.shape
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    chroma = rgb[:, :, 0] - rgb[:, :, 1]
    xs, ys = _grid(n_r1r6, width, height)
    r1 = np.clip(_sample(luma, xs, ys) * gain, 0, 4).astype(np.float32)
    if n_r8 <= 0:
        return r1, np.zeros(0, dtype=np.float32)
    xs8, ys8 = _grid(n_r8, width, height)
    r8 = np.clip(_sample(chroma, xs8, ys8) * gain, -2, 2).astype(np.float32)
    return r1, r8
