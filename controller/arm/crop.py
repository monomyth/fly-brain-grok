"""Visuo-motor CSR crop: photoreceptors, lamina/medulla/T4-T5/LC, scored DNs, T1 log, KC→MBON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from malecns_cache.graph import Connectome, load_graph, load_stub, write_stub_graph
from malecns_cache.paths import project_data

CROP_VERSION = "v1-optic-rich-mancType-DNfl-DNxl"
DNFL_SOURCE = "mancType startswith DNfl"
DNXL_SOURCE = "mancType startswith DNxl"
EXACT_DN_TYPES = ("DNa01", "DNa02", "DNp01", "MDN", "DNp07", "DNp10")
EXCLUDE_TYPES = {"DNp15", "DNp20", "DNp22", "DNg13"}
OPTIC_EXACT = {
    "R1-R6",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "C2",
    "C3",
    "T1",
    "Mi1",
    "Mi4",
    "Mi9",
    "Tm1",
    "Tm2",
    "Tm3",
    "Tm4",
    "Tm9",
    "Tm20",
    "T2",
    "T2a",
    "T3",
    "T4a",
    "T4b",
    "T4c",
    "T4d",
    "T5a",
    "T5b",
    "T5c",
    "T5d",
    "LC10a",
    "LC10c",
    "LC10c-1",
    "LC10c-2",
    "LC10d",
    "LC10e",
    "LPLC1",
    "LPLC2",
    "HSN",
    "HSE",
    "HSS",
    "VS",
    "H2",
    "MBON11",
    "PPL101",
}
GAIN_CLASSES = (
    "R1",
    "R8",
    "L1",
    "L2",
    "L3",
    "Mi1",
    "Tm3",
    "T4",
    "T5",
    "LC10",
    "LPLC",
    "LPTC",
    "DNfl",
    "DNxl",
    "DNa01",
    "DNa02",
    "DNp01",
    "MDN",
    "DNp07",
    "DNp10",
    "KC",
    "MBON",
    "other",
)


def crop_path(release: str = "v1.0") -> Path:
    folder = project_data() / "prepared"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"malecns-{release}-crop-{CROP_VERSION}.npz"


def crop_meta_path(release: str = "v1.0") -> Path:
    return crop_path(release).with_suffix(".json")


def collapse_class(kind: str, manc: str) -> str:
    t, m = str(kind), str(manc)
    if t == "R1-R6":
        return "R1"
    if t.startswith("R8"):
        return "R8"
    if t in {"L1", "L2", "L3"}:
        return t
    if t in {"Mi1", "Tm3"}:
        return t
    if t.startswith("T4"):
        return "T4"
    if t.startswith("T5"):
        return "T5"
    if t.startswith("LC10"):
        return "LC10"
    if t.startswith("LPLC"):
        return "LPLC"
    if t in {"HSN", "HSE", "HSS", "VS", "H2"}:
        return "LPTC"
    if m.startswith("DNfl"):
        return "DNfl"
    if m.startswith("DNxl"):
        return "DNxl"
    if t in EXACT_DN_TYPES:
        return t
    if t.startswith("KC") or t == "Kenyon_Cell":
        return "KC"
    if t == "MBON11":
        return "MBON"
    return "other"


def _side_letter(root: str, instance: str, soma: str) -> str:
    for raw in (root, soma):
        if raw in {"L", "R"}:
            return raw
    inst = str(instance)
    if inst.endswith("_L"):
        return "L"
    if inst.endswith("_R"):
        return "R"
    return ""


def _align_annotations(connectome: Connectome):
    import pyarrow.feather as feather

    from malecns_cache.prepare import _role_path

    table = feather.read_table(_role_path("v1.0", "annotations"))
    ann_id = np.array(table.column("bodyId").to_pylist(), dtype=np.int64)
    order = np.argsort(ann_id, kind="stable")
    ann_id = ann_id[order]
    found = np.searchsorted(ann_id, connectome.body_id)
    ok = found < ann_id.size
    ok &= ann_id[np.clip(found, 0, max(ann_id.size - 1, 0))] == connectome.body_id
    if not bool(np.all(ok)):
        raise RuntimeError("prepared body_id does not match annotations")
    idx = found

    def col(name: str, empty=""):
        return np.array(table.column(name).fill_null(empty).to_pylist(), dtype=object)[order][idx]

    types = col("type")
    manc = col("mancType")
    subclass = col("subclass")
    soma = col("somaSide")
    root = col("rootSide")
    instance = col("instance")
    hex1 = np.array(table.column("assignedOlHex1").fill_null(-1).to_pylist(), dtype=np.int32)[order][idx]
    hex2 = np.array(table.column("assignedOlHex2").fill_null(-1).to_pylist(), dtype=np.int32)[order][idx]
    return types, manc, subclass, soma, root, instance, hex1, hex2


def _scored_dn_mask(types: np.ndarray, manc: np.ndarray) -> np.ndarray:
    keep = np.zeros(types.size, dtype=bool)
    for i, (kind, m) in enumerate(zip(types, manc)):
        kind, m = str(kind), str(m)
        if m.startswith("DNg02") or kind.startswith("DNg02") or kind in EXCLUDE_TYPES:
            continue
        if m.startswith("DNfl") or m.startswith("DNxl") or kind in EXACT_DN_TYPES:
            keep[i] = True
    return keep


def _optic_mask(types: np.ndarray) -> np.ndarray:
    keep = np.zeros(types.size, dtype=bool)
    for i, kind in enumerate(types):
        t = str(kind)
        if t in OPTIC_EXACT or t.startswith("R8") or t.startswith("KC") or t == "Kenyon_Cell":
            keep[i] = True
    return keep


def _r1_hex(weights: csr_matrix, r1: np.ndarray, lamina: np.ndarray, hex1: np.ndarray, hex2: np.ndarray):
    if r1.size == 0 or lamina.size == 0:
        return np.full(r1.size, -1, np.int32), np.full(r1.size, -1, np.int32)
    block = np.abs(weights[lamina][:, r1].toarray())
    best = block.argmax(axis=0)
    w = block.max(axis=0)
    h1 = np.full(r1.size, -1, dtype=np.int32)
    h2 = np.full(r1.size, -1, dtype=np.int32)
    ok = w > 0
    h1[ok] = hex1[lamina][best[ok]]
    h2[ok] = hex2[lamina][best[ok]]
    return h1, h2


@dataclass
class Crop:
    weights: csr_matrix
    w0: np.ndarray
    body_id: np.ndarray
    types: np.ndarray
    manc: np.ndarray
    groups: dict[str, np.ndarray]
    hex1: np.ndarray
    hex2: np.ndarray
    pre_class: np.ndarray
    class_names: tuple[str, ...]
    parent_index: np.ndarray
    meta: dict
    path: Path

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    def indices(self, name: str) -> np.ndarray:
        return np.asarray(self.groups.get(name, []), dtype=np.int32)

    def class_id(self, name: str) -> int:
        try:
            return self.class_names.index(name)
        except ValueError:
            return self.class_names.index("other")


def _groups_from_types(
    types: np.ndarray,
    manc: np.ndarray,
    soma: np.ndarray,
    scored: np.ndarray,
    t1: np.ndarray,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {k: [] for k in (
        "photoreceptors_r1r6",
        "photoreceptors_r8",
        "L1",
        "L2",
        "Mi1",
        "Tm3",
        "T4",
        "T5",
        "LC10",
        "LPLC",
        "LPTC",
        "DNfl",
        "DNxl",
        "DNa01",
        "DNa02",
        "DNa02_L",
        "DNa02_R",
        "DNp01",
        "MDN",
        "DNp07",
        "DNp10",
        "scored_dn",
        "leg_mn_front",
        "KC",
        "MBON11",
        "PPL101",
    )}
    for i, (kind, m, side) in enumerate(zip(types, manc, soma)):
        kind, m, side = str(kind), str(m), str(side)
        if kind == "R1-R6":
            groups["photoreceptors_r1r6"].append(i)
        elif kind.startswith("R8"):
            groups["photoreceptors_r8"].append(i)
        elif kind == "L1":
            groups["L1"].append(i)
        elif kind == "L2":
            groups["L2"].append(i)
        elif kind == "Mi1":
            groups["Mi1"].append(i)
        elif kind == "Tm3":
            groups["Tm3"].append(i)
        elif kind.startswith("T4"):
            groups["T4"].append(i)
        elif kind.startswith("T5"):
            groups["T5"].append(i)
        elif kind.startswith("LC10a") or kind.startswith("LC10c") or kind.startswith("LC10d") or kind.startswith("LC10e"):
            groups["LC10"].append(i)
        elif kind.startswith("LPLC"):
            groups["LPLC"].append(i)
        elif kind in {"HSN", "HSE", "HSS", "VS", "H2"}:
            groups["LPTC"].append(i)
        elif kind.startswith("KC") or kind == "Kenyon_Cell":
            groups["KC"].append(i)
        elif kind == "MBON11":
            groups["MBON11"].append(i)
        elif kind == "PPL101":
            groups["PPL101"].append(i)
        if m.startswith("DNfl"):
            groups["DNfl"].append(i)
        if m.startswith("DNxl"):
            groups["DNxl"].append(i)
        if kind == "DNa01":
            groups["DNa01"].append(i)
        if kind == "DNa02":
            groups["DNa02"].append(i)
            groups.setdefault(f"DNa02_{side or 'U'}", []).append(i)
        if kind == "DNp01":
            groups["DNp01"].append(i)
        if kind == "MDN":
            groups["MDN"].append(i)
        if kind == "DNp07":
            groups["DNp07"].append(i)
        if kind == "DNp10":
            groups["DNp10"].append(i)
    groups["scored_dn"] = list(np.where(scored)[0].tolist())
    groups["leg_mn_front"] = list(np.where(t1)[0].tolist())
    return groups


def _finish(weights: csr_matrix, body_id, types, manc, soma, hex1, hex2, scored, t1, parent, meta, path) -> Crop:
    groups = {k: np.asarray(v, dtype=np.int32) for k, v in _groups_from_types(types, manc, soma, scored, t1).items()}
    class_names = GAIN_CLASSES
    pre_class = np.array([class_names.index(collapse_class(t, m)) if collapse_class(t, m) in class_names else class_names.index("other") for t, m in zip(types, manc)], dtype=np.int32)
    return Crop(
        weights=weights,
        w0=np.array(weights.data, copy=True),
        body_id=np.asarray(body_id, dtype=np.int64),
        types=np.asarray(types, dtype=object),
        manc=np.asarray(manc, dtype=object),
        groups=groups,
        hex1=np.asarray(hex1, dtype=np.int32),
        hex2=np.asarray(hex2, dtype=np.int32),
        pre_class=pre_class,
        class_names=class_names,
        parent_index=np.asarray(parent, dtype=np.int32),
        meta=meta,
        path=path,
    )


def build_crop(release: str = "v1.0", force: bool = False) -> Crop:
    dest = crop_path(release)
    meta_path = crop_meta_path(release)
    if dest.exists() and meta_path.exists() and not force:
        return load_crop(release)
    if dest.exists() and not meta_path.exists() and not force:
        raise FileNotFoundError(
            f"crop npz at {dest} but missing sidecar {meta_path.name}; will not rebuild from full MaleCNS"
        )
    connectome = load_graph(release)
    types, manc, _subclass, soma, root, instance, hex1, hex2 = _align_annotations(connectome)
    scored = _scored_dn_mask(types, manc)
    keep = _optic_mask(types) | scored
    t1_full = np.zeros(connectome.n, dtype=bool)
    t1_full[connectome.indices("leg_mn_front")] = True
    keep |= t1_full
    r1 = np.where(types == "R1-R6")[0]
    lamina = np.where(types == "L1")[0]
    r1_h1, r1_h2 = _r1_hex(connectome.weights, r1, lamina, hex1, hex2)
    hex1 = hex1.copy()
    hex2 = hex2.copy()
    hex1[r1] = r1_h1
    hex2[r1] = r1_h2
    sides = np.array([_side_letter(str(a), str(b), str(c)) for a, b, c in zip(root, instance, soma)], dtype=object)
    ix = np.where(keep)[0].astype(np.int32)
    sub = connectome.weights[ix][:, ix].tocsr()
    sub.sum_duplicates()
    types_c = types[ix]
    manc_c = manc[ix]
    soma_c = sides[ix]
    scored_c = scored[ix]
    t1_c = t1_full[ix]
    hex1_c = hex1[ix]
    hex2_c = hex2[ix]
    r1_c = np.where(types_c == "R1-R6")[0]
    front = r1_c[(sides[ix][r1_c] == "L") & (hex1_c[r1_c] >= 0)]
    grip = r1_c[(sides[ix][r1_c] == "R") & (hex1_c[r1_c] >= 0)]
    if front.size == 0 or grip.size == 0:
        mid = int(np.median(hex1_c[r1_c[hex1_c[r1_c] >= 0]])) if np.any(hex1_c[r1_c] >= 0) else 18
        front = r1_c[hex1_c[r1_c] <= mid]
        grip = r1_c[hex1_c[r1_c] > mid]
    groups_tmp = _groups_from_types(types_c, manc_c, soma_c, scored_c, t1_c)
    groups_tmp["front_r1"] = front.tolist()
    groups_tmp["grip_r1"] = grip.tolist()
    meta = {
        "release": release,
        "crop_version": CROP_VERSION,
        "n": int(sub.shape[0]),
        "nnz": int(sub.nnz),
        "dnfl_source": DNFL_SOURCE,
        "dnxl_source": DNXL_SOURCE,
        "exact_dn_types": list(EXACT_DN_TYPES),
        "excluded": sorted(EXCLUDE_TYPES) + ["DNg02"],
        "n_scored_dn": int(scored_c.sum()),
        "n_front_r1": int(front.size),
        "n_grip_r1": int(grip.size),
        "n_t1": int(t1_c.sum()),
        "groups": {k: [int(x) for x in v] for k, v in groups_tmp.items()},
        "parent_index": ix.tolist(),
    }
    np.savez(
        dest,
        data=sub.data,
        indices=sub.indices,
        indptr=sub.indptr,
        shape=np.array(sub.shape, dtype=np.int64),
        body_id=connectome.body_id[ix],
        types=np.asarray(types_c, dtype=object),
        manc=np.asarray(manc_c, dtype=object),
        soma=np.asarray(soma_c, dtype=object),
        hex1=hex1_c,
        hex2=hex2_c,
        scored=scored_c.astype(np.uint8),
        t1=t1_c.astype(np.uint8),
        parent_index=ix,
        pre_class=np.array(
            [GAIN_CLASSES.index(collapse_class(t, m)) if collapse_class(t, m) in GAIN_CLASSES else GAIN_CLASSES.index("other") for t, m in zip(types_c, manc_c)],
            dtype=np.int32,
        ),
    )
    meta_path.write_text(json.dumps({k: v for k, v in meta.items() if k != "parent_index"}) + "\n")
    return load_crop(release)


def load_crop(release: str = "v1.0") -> Crop:
    dest = crop_path(release)
    meta_path = crop_meta_path(release)
    if not dest.exists() or not meta_path.exists():
        return build_crop(release, force=True)
    blob = np.load(dest, allow_pickle=True)
    weights = csr_matrix(
        (blob["data"], blob["indices"], blob["indptr"]),
        shape=tuple(int(x) for x in blob["shape"]),
    )
    meta = json.loads(meta_path.read_text())
    types = blob["types"]
    manc = blob["manc"]
    soma = blob["soma"] if "soma" in blob.files else np.array([""] * weights.shape[0], dtype=object)
    scored = blob["scored"].astype(bool)
    t1 = blob["t1"].astype(bool)
    hex1 = blob["hex1"]
    hex2 = blob["hex2"]
    parent = blob["parent_index"]
    crop = _finish(
        weights,
        blob["body_id"],
        types,
        manc,
        soma,
        hex1,
        hex2,
        scored,
        t1,
        parent,
        meta,
        dest,
    )
    stored = meta.get("groups") or {}
    for key in ("front_r1", "grip_r1"):
        if key in stored:
            crop.groups[key] = np.asarray(stored[key], dtype=np.int32)
    return crop


def write_stub_crop(folder: Path) -> Crop:
    """Tiny excitatory chain for tests. Does not touch MaleCNS feathers."""
    folder.mkdir(parents=True, exist_ok=True)
    n = 24
    # 0,1 front R1; 2,3 grip R1; 4 L1; 5 Mi1; 6 T4a; 7 LC10a; 8 DNfl; 9 DNxl;
    # 10 DNa01; 11 DNa02_L; 12 DNa02_R; 13 DNp01; 14 MDN; 15 T1; 16 KC; 17 MBON
    # DNa02: left-hex R1 → L, right-hex R1 → R.
    pairs = [
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (5, 4),
        (6, 5),
        (7, 6),
        (8, 7),
        (9, 7),
        (10, 7),
        (11, 0),
        (11, 2),
        (12, 1),
        (12, 3),
        (13, 7),
        (14, 7),
        (17, 16),
    ]
    data = np.full(len(pairs), 8.0, dtype=np.float32)
    # One R1 spike must move DNa02; leak*W*gain ≈ 1 needs W≈80 at gain 2.5.
    data[10:14] = 80.0
    post = np.array([p[0] for p in pairs], dtype=np.int32)
    pre = np.array([p[1] for p in pairs], dtype=np.int32)
    weights = csr_matrix((data, (post, pre)), shape=(n, n), dtype=np.float32)
    types = np.array(
        ["R1-R6"] * 4
        + ["L1", "Mi1", "T4a", "LC10a", "DNfl001", "DNxl001", "DNa01", "DNa02", "DNa02", "DNp01", "MDN", "Ti flexor MN", "KC", "MBON11", "PPL101"]
        + ["other"] * (n - 19),
        dtype=object,
    )
    manc = np.array([""] * n, dtype=object)
    manc[8] = "DNfl001"
    manc[9] = "DNxl001"
    soma = np.array(["L", "L", "R", "R"] + [""] * (n - 4), dtype=object)
    soma[11] = "L"
    soma[12] = "R"
    hex1 = np.full(n, -1, dtype=np.int32)
    hex2 = np.full(n, -1, dtype=np.int32)
    # Left/right hexes sit on cube edges so ON+luma exceeds LIF threshold in 150 steps.
    hex1[:4] = [12, 26, 12, 26]
    hex2[:4] = [20, 20, 20, 20]
    scored = np.zeros(n, dtype=bool)
    scored[8:15] = True
    t1 = np.zeros(n, dtype=bool)
    t1[15] = True
    types[8] = "DNfl_stub"
    path = folder / "stub-crop.npz"
    meta = {
        "release": "stub",
        "crop_version": CROP_VERSION,
        "n": n,
        "nnz": int(weights.nnz),
        "dnfl_source": DNFL_SOURCE,
        "dnxl_source": DNXL_SOURCE,
        "n_scored_dn": int(scored.sum()),
        "groups": {},
        "hex_world": [[1.0, 36.0], [1.0, 39.0]],
    }
    crop = _finish(weights, np.arange(n, dtype=np.int64) + 10, types, manc, soma, hex1, hex2, scored, t1, np.arange(n, dtype=np.int32), meta, path)
    crop.groups["front_r1"] = np.array([0, 1], dtype=np.int32)
    crop.groups["grip_r1"] = np.array([2, 3], dtype=np.int32)
    crop.groups["DNfl"] = np.array([8], dtype=np.int32)
    crop.groups["DNxl"] = np.array([9], dtype=np.int32)
    return crop


def as_connectome(crop: Crop) -> Connectome:
    return Connectome(
        weights=crop.weights,
        body_id=crop.body_id,
        sign=np.ones(crop.n, dtype=np.float32),
        groups={k: [int(x) for x in v] for k, v in crop.groups.items()},
        meta=crop.meta,
        path=crop.path,
    )


# keep import of load_stub used by tests that share the cache helpers
_ = (load_stub, write_stub_graph)
