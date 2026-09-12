from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from . import paths
from .prepare import prepare_graph


@dataclass
class Connectome:
    weights: csr_matrix
    body_id: np.ndarray
    sign: np.ndarray
    groups: dict[str, list[int]]
    meta: dict
    path: Path

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    def indices(self, name: str) -> np.ndarray:
        return np.asarray(self.groups.get(name, []), dtype=np.int32)


def load_graph(release: str = "v1.0", prepare: bool = True) -> Connectome:
    npz_path = paths.graph_npz(release)
    meta_path = paths.graph_meta(release)
    if prepare and (not npz_path.exists() or not meta_path.exists()):
        prepare_graph(release)
    blob = np.load(npz_path, allow_pickle=False)
    weights = csr_matrix(
        (blob["data"], blob["indices"], blob["indptr"]),
        shape=tuple(int(x) for x in blob["shape"]),
    )
    meta = json.loads(meta_path.read_text())
    return Connectome(
        weights=weights,
        body_id=blob["body_id"],
        sign=blob["sign"],
        groups={k: list(v) for k, v in meta["groups"].items()},
        meta=meta,
        path=npz_path,
    )


def write_stub_graph(folder: Path, n: int = 8) -> tuple[Path, Path]:
    """Tiny chain used by tests. Does not touch the 1.1 GB cache."""
    folder.mkdir(parents=True, exist_ok=True)
    data = np.array([1.0, 1.0, 1.0, 0.5], dtype=np.float32)
    indices = np.array([0, 1, 2, 2], dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    # post 1 <- 0, post 2 <- 1, post 3 <- 2, post 4 (MBON11) <- 2 (KC)
    indptr[2] = 1
    indptr[3] = 2
    indptr[4] = 3
    indptr[5] = 4
    indptr[6:] = 4
    body_id = np.arange(n, dtype=np.int64) + 1000
    sign = np.ones(n, dtype=np.float32)
    npz = folder / "malecns-stub-graph.npz"
    meta_path = folder / "malecns-stub-meta.json"
    np.savez(npz, data=data, indices=indices, indptr=indptr, shape=np.array([n, n], dtype=np.int64), body_id=body_id, sign=sign)
    groups = {
        "photoreceptors_r1r6": [0, 1],
        "photoreceptors_r8": [1],
        "DNp20_L": [n - 2],
        "DNp20_R": [n - 1],
        "DNp20": [n - 2, n - 1],
        "DNpe017": [n - 1],
        "PPL101": [3],
        "MBON11": [4],
        "KC": [2, 3],
        "DN": [n - 2, n - 1],
        "vnc_motor": [n - 2, n - 1],
        "leg_mn_front": [n - 1],
        "Ti_flexor": [n - 2],
        "Ti_extensor": [n - 1],
        "Ti_flexor_L": [n - 2],
        "Ti_extensor_R": [n - 1],
    }
    meta_path.write_text(json.dumps({"release": "stub", "n": n, "nnz": 3, "groups": groups}) + "\n")
    return npz, meta_path


def load_stub(folder: Path) -> Connectome:
    npz = folder / "malecns-stub-graph.npz"
    meta_path = folder / "malecns-stub-meta.json"
    if not npz.exists() or not meta_path.exists():
        write_stub_graph(folder)
    blob = np.load(npz, allow_pickle=False)
    weights = csr_matrix((blob["data"], blob["indices"], blob["indptr"]), shape=tuple(int(x) for x in blob["shape"]))
    meta = json.loads(meta_path.read_text())
    return Connectome(weights=weights, body_id=blob["body_id"], sign=blob["sign"], groups=meta["groups"], meta=meta, path=npz)
