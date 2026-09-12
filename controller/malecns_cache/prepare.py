from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix

from . import catalog, paths
from .lockfile import verify_release


def _role_path(release: str, role: str) -> Path:
    folder = paths.release_dir(release)
    for name, spec in catalog.FILES[release].items():
        if spec["role"] == role:
            return folder / name
    raise KeyError(role)


LEG_SUBCLASS = {"fl": "leg_mn_front", "ml": "leg_mn_mid", "hl": "leg_mn_hind"}
LEG_NEUROMERE = {"T1": "leg_mn_front", "T2": "leg_mn_mid", "T3": "leg_mn_hind"}
LEG_POOLS = (
    ("Ti_flexor", ("Ti flexor MN", "Acc. ti flexor MN")),
    ("Ti_extensor", ("Ti extensor MN",)),
    ("Tr_flexor", ("Tr flexor MN", "Acc. tr flexor MN")),
    ("Tr_extensor", ("Tr extensor MN",)),
    ("Fe_reductor", ("Fe reductor MN",)),
    ("Sternotrochanter", ("Sternotrochanter MN",)),
    ("Ta_depressor", ("Ta depressor MN",)),
    ("Ta_levator", ("Ta levator MN",)),
)


def group_cells(
    types: list[str],
    sides: list[str],
    superclasses: list[str] | None = None,
    subclasses: list[str] | None = None,
    neuromeres: list[str] | None = None,
) -> dict[str, list[int]]:
    """Index photoreceptors, DNs, mushroom body, and VNC leg motor neurons."""
    n = len(types)
    supers = superclasses if superclasses is not None else [""] * n
    subs = subclasses if subclasses is not None else [""] * n
    nms = neuromeres if neuromeres is not None else [""] * n
    groups: dict[str, list[int]] = defaultdict(list)
    pool_lookup = {name: key for key, names in LEG_POOLS for name in names}
    for i, (kind, side, super_c, sub, nm) in enumerate(zip(types, sides, supers, subs, nms)):
        if kind == "R1-R6":
            groups["photoreceptors_r1r6"].append(i)
        elif kind.startswith("R8"):
            groups["photoreceptors_r8"].append(i)
        elif kind.startswith("R7"):
            groups["photoreceptors_r7"].append(i)
        elif kind == "DNp20":
            groups[f"DNp20_{side or 'U'}"].append(i)
            groups["DNp20"].append(i)
        elif kind == "DNpe017":
            groups["DNpe017"].append(i)
        elif kind == "DNa02":
            groups[f"DNa02_{side or 'U'}"].append(i)
            groups["DNa02"].append(i)
        elif kind == "PPL101":
            groups["PPL101"].append(i)
        elif kind == "MBON11":
            groups["MBON11"].append(i)
        elif kind.startswith("KC") or kind == "Kenyon_Cell":
            groups["KC"].append(i)
        if kind.startswith("DN"):
            groups["DN"].append(i)
        if super_c == "vnc_motor":
            groups["vnc_motor"].append(i)
            leg = LEG_SUBCLASS.get(sub) or LEG_NEUROMERE.get(nm)
            if leg:
                groups[leg].append(i)
            pool = pool_lookup.get(kind)
            if pool:
                groups[pool].append(i)
                groups[f"{pool}_{side or 'U'}"].append(i)
    return {key: value for key, value in groups.items()}


def _group_indices(types: list[str], sides: list[str]) -> dict[str, list[int]]:
    return group_cells(types, sides)


def prepare_graph(release: str = "v1.0", force: bool = False) -> tuple[Path, Path]:
    verify_release(release)
    npz_path = paths.graph_npz(release)
    meta_path = paths.graph_meta(release)
    if npz_path.exists() and meta_path.exists() and not force:
        return npz_path, meta_path

    annotations = feather.read_table(_role_path(release, "annotations"))
    superclass = [s or "" for s in annotations.column("superclass").fill_null("").to_pylist()]
    keep = np.array([s != "" for s in superclass], dtype=bool)
    body_id = np.array(annotations.column("bodyId").to_pylist(), dtype=np.int64)[keep]
    types = np.array(annotations.column("type").fill_null("").to_pylist(), dtype=object)[keep]
    sides = np.array(annotations.column("somaSide").fill_null("").to_pylist(), dtype=object)[keep]
    supers = np.array(superclass, dtype=object)[keep]
    subs = np.array(annotations.column("subclass").fill_null("").to_pylist(), dtype=object)[keep]
    nms = np.array(annotations.column("somaNeuromere").fill_null("").to_pylist(), dtype=object)[keep]
    order = np.argsort(body_id, kind="stable")
    body_id = body_id[order]
    types = types[order]
    sides = sides[order]
    supers = supers[order]
    subs = subs[order]
    nms = nms[order]
    n = int(body_id.size)

    nt_table = feather.read_table(
        _role_path(release, "neurotransmitters"),
        columns=["body", "consensus_nt", "predicted_nt"],
    )
    nt_body = np.array(nt_table.column("body").to_pylist(), dtype=np.int64)
    consensus = np.array(nt_table.column("consensus_nt").fill_null("").to_pylist(), dtype=object)
    predicted = np.array(nt_table.column("predicted_nt").fill_null("").to_pylist(), dtype=object)
    nt_order = np.argsort(nt_body)
    nt_body = nt_body[nt_order]
    nt = np.where(consensus[nt_order] != "", consensus[nt_order], predicted[nt_order])
    found = np.searchsorted(nt_body, body_id)
    in_nt = found < nt_body.size
    in_nt &= nt_body[np.clip(found, 0, max(nt_body.size - 1, 0))] == body_id
    transmitter = np.full(n, "unclear", dtype=object)
    transmitter[in_nt] = nt[found[in_nt]]
    sign = np.array([-1.0 if str(name).lower() in catalog.INHIBITORY else 1.0 for name in transmitter], dtype=np.float32)

    edges = feather.read_table(_role_path(release, "edges"), memory_map=True)
    pre = edges.column("body_pre").to_numpy()
    post = edges.column("body_post").to_numpy()
    weight = edges.column("weight").to_numpy().astype(np.float32, copy=False)

    def map_ids(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(body_id, arr)
        ok = idx < n
        ok &= body_id[np.minimum(idx, n - 1)] == arr
        return idx.astype(np.int32, copy=False), ok

    pre_i, pre_ok = map_ids(pre)
    post_i, post_ok = map_ids(post)
    keep_e = pre_ok & post_ok
    pre_i = pre_i[keep_e]
    post_i = post_i[keep_e]
    signed = weight[keep_e] * sign[pre_i]
    graph = csr_matrix((signed, (post_i, pre_i)), shape=(n, n), dtype=np.float32)
    graph.sum_duplicates()

    groups = group_cells(
        [str(x) for x in types],
        [str(x) for x in sides],
        [str(x) for x in supers],
        [str(x) for x in subs],
        [str(x) for x in nms],
    )
    np.savez(
        npz_path,
        data=graph.data,
        indices=graph.indices,
        indptr=graph.indptr,
        shape=np.array(graph.shape, dtype=np.int64),
        body_id=body_id,
        sign=sign,
    )
    meta = {
        "release": release,
        "n": n,
        "nnz": int(graph.nnz),
        "node_policy": "assigned superclass; glia have none in v1.0",
        "edge_policy": "all published weights between retained nodes; signed by consensus/predicted NT",
        "inhibitory": sorted(catalog.INHIBITORY),
        "groups": groups,
        "source_lock": str(paths.lock_path(release)),
    }
    meta_path.write_text(json.dumps(meta) + "\n")
    from .somas import prepare_somas

    prepare_somas(release, force=True)
    return npz_path, meta_path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args
    groups_only = "--groups" in args
    release = next((a for a in args if not a.startswith("-")), "v1.0")
    if groups_only:
        counts = refresh_groups(release)
        print("refreshed groups")
        for key in ("vnc_motor", "leg_mn_front", "leg_mn_mid", "leg_mn_hind", "Ti_flexor", "Ti_extensor"):
            print(f"{key} {counts.get(key, 0)}")
        return 0
    npz_path, meta_path = prepare_graph(release, force=force)
    meta = json.loads(meta_path.read_text())
    print(f"wrote {npz_path}")
    print(f"neurons {meta['n']}  edges {meta['nnz']}")
    print(f"R1-R6 {len(meta['groups'].get('photoreceptors_r1r6', []))}  DNp20 {len(meta['groups'].get('DNp20', []))}")
    print(
        f"vnc_motor {len(meta['groups'].get('vnc_motor', []))}  "
        f"leg_mn_front {len(meta['groups'].get('leg_mn_front', []))}"
    )
    return 0


def refresh_groups(release: str = "v1.0") -> dict:
    """Rewrite meta groups from annotations without rebuilding the CSR."""
    from .graph import load_graph

    connectome = load_graph(release, prepare=True)
    annotations = feather.read_table(_role_path(release, "annotations"))
    ann_id = np.array(annotations.column("bodyId").to_pylist(), dtype=np.int64)
    types = np.array(annotations.column("type").fill_null("").to_pylist(), dtype=object)
    sides = np.array(annotations.column("somaSide").fill_null("").to_pylist(), dtype=object)
    supers = np.array(annotations.column("superclass").fill_null("").to_pylist(), dtype=object)
    subs = np.array(annotations.column("subclass").fill_null("").to_pylist(), dtype=object)
    nms = np.array(annotations.column("somaNeuromere").fill_null("").to_pylist(), dtype=object)
    order = np.argsort(ann_id, kind="stable")
    ann_id = ann_id[order]
    found = np.searchsorted(ann_id, connectome.body_id)
    ok = found < ann_id.size
    ok &= ann_id[np.clip(found, 0, max(ann_id.size - 1, 0))] == connectome.body_id
    if not bool(np.all(ok)):
        raise RuntimeError("prepared body_id does not match annotations")
    idx = found
    groups = group_cells(
        [str(x) for x in types[order][idx]],
        [str(x) for x in sides[order][idx]],
        [str(x) for x in supers[order][idx]],
        [str(x) for x in subs[order][idx]],
        [str(x) for x in nms[order][idx]],
    )
    meta = dict(connectome.meta)
    meta["groups"] = groups
    paths.graph_meta(release).write_text(json.dumps(meta) + "\n")
    return {key: len(value) for key, value in sorted(groups.items())}


if __name__ == "__main__":
    raise SystemExit(main())
