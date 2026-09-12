from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DROP_STAGES = frozenset({"release", "done", "landed", "spawn"})
VIEW_CAMERAS = ("Front", "Gripper")


def policy_cameras(meta: dict | None = None) -> tuple[str, ...]:
    """Cameras the live loop should capture. Prefer cameras whose JPEGs entered the fit."""
    if meta:
        for key in ("logged_cameras", "cameras"):
            cams = [str(c) for c in (meta.get(key) or []) if c]
            if cams:
                return tuple(cams)
    return VIEW_CAMERAS


def yaw_from_xyzw(q) -> float:
    """Yaw of tool +X in the world XY plane from a xyzw quaternion."""
    x, y, z, w = (float(v) for v in q)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    return math.atan2(r10, r00)


@dataclass
class Sample:
    image: Path
    tcp_mm: np.ndarray
    cube_mm: np.ndarray
    gripper_mm: float
    attached: bool
    action: np.ndarray  # Δx, Δy, Δz, Δgrip in mm
    stage: str = ""
    cube_size_mm: float = 20.0
    yaw_rad: float = 0.0
    camera: str = "Front"
    views: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _delta(a: dict, b: dict) -> np.ndarray:
    return np.array(
        [
            b["x"] - a["x"],
            b["y"] - a["y"],
            b["z"] - a["z"],
            0.0,
        ],
        dtype=np.float64,
    )


def load_teacher_run(run: Path) -> list[Sample]:
    log = run / "log.jsonl"
    if not log.is_file():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    samples = []
    for current, nxt in zip(rows, rows[1:]):
        tcp = current["tcp_mm"]
        nxt_tcp = nxt["tcp_mm"]
        action = np.array(
            [
                nxt_tcp["x"] - tcp["x"],
                nxt_tcp["y"] - tcp["y"],
                nxt_tcp["z"] - tcp["z"],
                float(nxt["gripper_mm"]) - float(current["gripper_mm"]),
            ],
            dtype=np.float64,
        )
        cube = current.get("cube") or {}
        center = cube.get("center_mm") or {}
        rpy = current.get("tcp_rpy_deg") or {}
        photos = dict(_teacher_photos(run, current))
        for camera, image in photos.items():
            samples.append(
                Sample(
                    image=image,
                    tcp_mm=np.array([tcp["x"], tcp["y"], tcp["z"]], dtype=np.float64),
                    cube_mm=np.array([center.get("x", 0), center.get("y", 0), center.get("z", 0)], dtype=np.float64),
                    gripper_mm=float(current["gripper_mm"]),
                    attached=bool(cube.get("attached")),
                    action=action,
                    stage=str(current.get("step") or ""),
                    cube_size_mm=float(cube.get("size_mm") or 40.0),
                    yaw_rad=math.radians(float(rpy.get("yaw") or 0.0)),
                    camera=camera,
                    views=dict(photos),
                )
            )
    return samples


def _teacher_photos(run: Path, row: dict) -> list[tuple[str, Path]]:
    photos = dict(row.get("photos") or {})
    if not photos and row.get("photo"):
        photos["Front"] = row["photo"]
    found: list[tuple[str, Path]] = []
    for camera in VIEW_CAMERAS:
        raw = photos.get(camera)
        if not raw:
            continue
        image = Path(raw)
        if not image.is_file():
            image = run / "frames" / Path(raw).name
        if image.is_file():
            found.append((camera, image))
    return found


def load_episode_v1(episode: Path) -> list[Sample]:
    log = episode / "steps.jsonl"
    if not log.is_file():
        return []
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    samples = []
    for current, nxt in zip(rows, rows[1:]):
        obs = current["observation"]
        nxt_obs = nxt["observation"]
        named = {item.get("name"): item for item in obs.get("images", []) if item.get("name")}
        views = [(n, named[n]) for n in VIEW_CAMERAS if n in named]
        if not views:
            continue
        tcp = obs["tool_pose"]["position_mm"]
        nxt_tcp = nxt_obs["tool_pose"]["position_mm"]
        action = np.array(
            [
                nxt_tcp[0] - tcp[0],
                nxt_tcp[1] - tcp[1],
                nxt_tcp[2] - tcp[2],
                float(nxt_obs["gripper_mm"]) - float(obs["gripper_mm"]),
            ],
            dtype=np.float64,
        )
        cube_pos = obs.get("cube_pose", {}).get("position_mm") or [0, 0, 0]
        evaluation = current.get("evaluation") or {}
        stage = str(current.get("stage") or obs.get("phase") or "")
        held = bool(evaluation.get("held")) or stage in {"lift", "lift_low", "lift_middle", "hold"}
        quat = (obs.get("tool_pose") or {}).get("quaternion_xyzw")
        size = evaluation.get("cube_size_mm")
        if size is None:
            size = (obs.get("goal") or {}).get("cube_size_mm")
        view_paths = {}
        for camera, item in views:
            image = episode / item["file"]
            if image.is_file():
                view_paths[camera] = image
        for camera, image in view_paths.items():
            samples.append(
                Sample(
                    image=image,
                    tcp_mm=np.array(tcp, dtype=np.float64),
                    cube_mm=np.array(cube_pos, dtype=np.float64),
                    gripper_mm=float(obs["gripper_mm"]),
                    attached=held,
                    action=action,
                    stage=stage,
                    cube_size_mm=float(size or 40.0),
                    yaw_rad=yaw_from_xyzw(quat) if quat else 0.0,
                    camera=camera,
                    views=dict(view_paths),
                )
            )
    return samples


def load_dataset(root: Path, split: str | None = None, stride: int = 1, max_episodes: int | None = None) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    if (root / "splits.json").is_file():
        names = json.loads((root / "splits.json").read_text())["splits"].get(split or "train", [])
        if max_episodes is not None:
            names = names[:max_episodes]
        for name in names:
            samples.extend(load_episode_v1(root / name)[:: max(stride, 1)])
        return samples
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "log.jsonl").is_file())
    if max_episodes is not None:
        runs = runs[:max_episodes]
    for run in runs:
        samples.extend(load_teacher_run(run)[:: max(stride, 1)])
    return samples


def for_imitation(samples: list[Sample]) -> list[Sample]:
    """Drop spawn/release frames so open-gripper deltas do not dominate dgrip."""
    kept = []
    for sample in samples:
        if (sample.stage or "").lower() in DROP_STAGES:
            continue
        kept.append(sample)
    return kept


def pair_view_samples(samples: list[Sample]) -> list[Sample]:
    """One training row per timestep, Front+Gripper JPEGs kept together."""
    out: list[Sample] = []
    seen: set[tuple] = set()
    for sample in samples:
        views = dict(sample.views or {})
        if not views:
            views = {sample.camera or "Front": sample.image}
        key = (
            str(sorted((str(k), str(v)) for k, v in views.items())),
            round(float(sample.tcp_mm[0]), 2),
            round(float(sample.tcp_mm[1]), 2),
            round(float(sample.tcp_mm[2]), 2),
            round(float(sample.gripper_mm), 1),
            bool(sample.attached),
            str(sample.stage or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        sample.views = views
        out.append(sample)
    return out
