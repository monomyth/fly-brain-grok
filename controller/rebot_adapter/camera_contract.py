"""Lab vs hardware camera names, 848×480 capture, crop to the optic encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

LAB_CAMERAS = ("Front", "Gripper")
HW_CAMERAS = ("wrist", "overview")
# Optic pair order: overview≡Front, wrist≡Gripper.
HW_ENCODER_ORDER = ("overview", "wrist")
LAB_CAPTURE_WH = (160, 120)
HW_CAPTURE_WH = (848, 480)
ENCODER_WH = (160, 120)


class CameraContractError(RuntimeError):
    """Missing or wrong camera name/size; feeds must not be guessed."""


@dataclass(frozen=True)
class TimedFrame:
    name: str
    rgb: np.ndarray
    capture_time: float
    receive_time: float
    command_time: float | None = None

    @property
    def age_s(self) -> float:
        return float(self.receive_time) - float(self.capture_time)

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def require_capture_name(shot: dict, expected: str) -> None:
    got = shot.get("camera") if isinstance(shot, dict) else None
    if got != expected:
        raise CameraContractError(f"expected camera {expected!r}, got {got!r}")


def select_named(frames: dict, expected: tuple[str, ...]) -> list:
    """Pick frames by exact name."""
    if not expected:
        raise CameraContractError("expected camera list is empty")
    if not isinstance(frames, dict):
        raise CameraContractError("frames must be a name→image mapping")
    missing = [name for name in expected if name not in frames]
    if missing:
        raise CameraContractError(f"missing cameras: {missing}")
    return [frames[name] for name in expected]


def _as_rgb(rgb: object, name: str) -> np.ndarray:
    if isinstance(rgb, TimedFrame):
        arr = np.asarray(rgb.rgb)
        name = rgb.name or name
    else:
        arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise CameraContractError(f"{name} must be H×W×3")
    return arr


def require_size(rgb: np.ndarray, wh: tuple[int, int], name: str) -> np.ndarray:
    arr = _as_rgb(rgb, name)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if (w, h) != (int(wh[0]), int(wh[1])):
        raise CameraContractError(f"{name} must be {wh[0]}×{wh[1]}, got {w}×{h}")
    return arr


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb)
    if x.dtype == np.uint8:
        return x
    if float(np.max(x)) <= 1.5:
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return np.clip(x, 0, 255).astype(np.uint8)


def crop_rectify_to_encoder(rgb: np.ndarray, encoder_wh: tuple[int, int] = ENCODER_WH) -> np.ndarray:
    """Center-crop to encoder aspect, then scale. Never stretch 16:9 into 4:3."""
    arr = _as_rgb(rgb, "frame")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    ew, eh = int(encoder_wh[0]), int(encoder_wh[1])
    if ew < 1 or eh < 1:
        raise CameraContractError("encoder size must be positive")
    target_aspect = ew / eh
    src_aspect = w / h
    if src_aspect > target_aspect + 1e-6:
        new_w = int(round(h * target_aspect))
        x0 = max(0, (w - new_w) // 2)
        arr = arr[:, x0 : x0 + new_w]
    elif src_aspect < target_aspect - 1e-6:
        new_h = int(round(w / target_aspect))
        y0 = max(0, (h - new_h) // 2)
        arr = arr[y0 : y0 + new_h, :]
    img = Image.fromarray(_to_u8(arr)).resize((ew, eh), Image.Resampling.BILINEAR)
    out = np.asarray(img, dtype=np.float32)
    if float(out.max()) > 1.5:
        out = out / 255.0
    return out


def _luma01(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32)
    if float(np.max(x)) > 1.5:
        x = x / 255.0
    return 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]


def pad_frac(rgb: np.ndarray) -> float:
    """Fraction of the lower field that looks like the white mat."""
    arr = _as_rgb(rgb, "pad")
    h = int(arr.shape[0])
    lower = arr[h // 3 :, :, :]
    lum = _luma01(lower)
    x = lower.astype(np.float32)
    if float(np.max(x)) > 1.5:
        x = x / 255.0
    sat = x.max(axis=2) - x.min(axis=2)
    mat = (lum > 0.55) & (sat < 0.2)
    return float(mat.mean())


def pad_like(rgb: np.ndarray, *, min_frac: float = 0.12) -> bool:
    return pad_frac(rgb) >= float(min_frac)


def pad_roi_encoder(rgb: np.ndarray, encoder_wh: tuple[int, int] = ENCODER_WH) -> np.ndarray:
    """Crop the white-mat bbox (ignore upper third) and scale to the encoder."""
    arr = _as_rgb(rgb, "overview")
    h, w = int(arr.shape[0]), int(arr.shape[1])
    lum = _luma01(arr)
    x = arr.astype(np.float32)
    if float(np.max(x)) > 1.5:
        x = x / 255.0
    sat = x.max(axis=2) - x.min(axis=2)
    mat = (lum > 0.55) & (sat < 0.2)
    mat[: h // 3, :] = False
    ys, xs = np.where(mat)
    if ys.size < 100:
        return crop_rectify_to_encoder(arr, encoder_wh)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    roi = arr[y0:y1, x0:x1]
    ew, eh = int(encoder_wh[0]), int(encoder_wh[1])
    img = Image.fromarray(_to_u8(roi)).resize((ew, eh), Image.Resampling.BILINEAR)
    out = np.asarray(img, dtype=np.float32)
    if float(out.max()) > 1.5:
        out = out / 255.0
    return out


def encoder_pair(frames: dict, profile: str) -> list[np.ndarray]:
    """Two encoder-sized RGBs in Front/Gripper (overview/wrist) order."""
    if profile == "lab":
        rgbs = select_named(frames, LAB_CAMERAS)
        return [require_size(rgb, LAB_CAPTURE_WH, name) for rgb, name in zip(rgbs, LAB_CAMERAS)]
    if profile == "hardware":
        rgbs = select_named(frames, HW_ENCODER_ORDER)
        out = []
        for rgb, name in zip(rgbs, HW_ENCODER_ORDER):
            require_size(rgb, HW_CAPTURE_WH, name)
            out.append(crop_rectify_to_encoder(rgb))
        return out
    raise CameraContractError(f"unknown camera profile {profile!r}")


def stamp_frame(
    name: str,
    rgb: np.ndarray,
    *,
    capture_time: float,
    receive_time: float,
    command_time: float | None = None,
) -> TimedFrame:
    return TimedFrame(
        name=name,
        rgb=np.asarray(rgb),
        capture_time=float(capture_time),
        receive_time=float(receive_time),
        command_time=None if command_time is None else float(command_time),
    )
