from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from . import paths
from .prepare import _role_path

SOMA_MAGIC = b"MLC2"
ACTIVITY_MAGIC = b"MLCN"


def _region(kind: str, superclass: str) -> int:
    text = f"{kind} {superclass}".lower()
    if any(key in text for key in ("r1-r6", "photoreceptor", "optic", "lamina", "medulla", "lobula", "visual", "ocellar")) or kind.startswith(("R7", "R8")):
        return 1
    if any(key in text for key in ("kenyon", "mbon", "mushroom", "ppl101", "dan", "kcg", "kca")) or kind.startswith("KC"):
        return 2
    if any(key in text for key in ("descend", "motor", "ventral", "nerve cord", "ascending")) or kind.startswith("DN"):
        return 3
    return 0


def prepare_somas(release: str = "v1.0", force: bool = False) -> Path:
    dest = paths.somas_path(release)
    if dest.exists() and not force and dest.read_bytes()[:4] == SOMA_MAGIC:
        return dest
    graph = np.load(paths.graph_npz(release), allow_pickle=False)
    body_id = graph["body_id"]
    n = int(body_id.size)
    ann = feather.read_table(_role_path(release, "annotations"), columns=["bodyId", "somaLocation", "type", "superclass"])
    ann_id = np.array(ann.column("bodyId").to_pylist(), dtype=np.int64)
    locs = ann.column("somaLocation").to_pylist()
    types = [t or "" for t in ann.column("type").fill_null("").to_pylist()]
    supers = [s or "" for s in ann.column("superclass").fill_null("").to_pylist()]
    order = np.argsort(ann_id)
    sorted_id = ann_id[order]
    found = np.searchsorted(sorted_id, body_id)
    ok = found < sorted_id.size
    ok &= sorted_id[np.minimum(found, max(sorted_id.size - 1, 0))] == body_id
    xyz, index, region = [], [], []
    for i, hit in enumerate(ok):
        if not hit:
            continue
        original = int(order[found[i]])
        loc = locs[original]
        if loc is None or len(loc) < 3:
            continue
        xyz.append((float(loc[0]), float(loc[1]), float(loc[2])))
        index.append(i)
        region.append(_region(types[original], supers[original]))
    pts = np.asarray(xyz, dtype=np.float64)
    idx = np.asarray(index, dtype=np.uint32)
    reg = np.asarray(region, dtype=np.uint8)
    center = pts.mean(axis=0)
    local = pts - center
    scale = float(np.abs(local).max()) or 1.0
    # Same display axes as the MaleCNS T-silhouette: x, -z, y.
    display = np.column_stack((local[:, 0], -local[:, 2], local[:, 1])) / scale
    display = display.astype(np.float32)
    payload = SOMA_MAGIC + struct.pack("<II", display.shape[0], n) + display.tobytes(order="C") + idx.tobytes() + reg.tobytes()
    dest.write_bytes(payload)
    return dest


def _write_activity_file(dest: Path, raw: np.ndarray, generation: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(ACTIVITY_MAGIC + struct.pack("<IQ", raw.size, int(generation)) + raw.tobytes())
    tmp.replace(dest)


def ipc_activity_path() -> Path:
    override = __import__("os").environ.get("REBOT_CONTROL_DIRECTORY")
    if override:
        return Path(override) / "activity.bin"
    tmp = Path(__import__("tempfile").gettempdir())
    uid = __import__("os").getuid()
    return tmp / f"rebot-motionlab-grok-{uid}" / "activity.bin"


def write_activity(rates: np.ndarray, generation: int) -> Path:
    raw = np.asarray(rates, dtype=np.float32).ravel()
    dest = paths.activity_path()
    _write_activity_file(dest, raw, generation)
    try:
        _write_activity_file(ipc_activity_path(), raw, generation)
    except OSError:
        pass
    return dest
