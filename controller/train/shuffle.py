from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix

from malecns_cache.graph import Connectome


def shuffle_csr_edges(weights: csr_matrix, seed: int = 0) -> csr_matrix:
    """Rewire existing edges: permute presynaptic indices, keep stored weights."""
    shuffled = weights.tocsr(copy=True)
    if shuffled.indices.size == 0:
        return shuffled
    rng = np.random.default_rng(seed)
    shuffled.indices = rng.permutation(np.asarray(shuffled.indices, dtype=np.int32))
    shuffled.sort_indices()
    return shuffled


def shuffle_edges(connectome: Connectome, seed: int = 0) -> Connectome:
    """Same neurons and weight multiset, scrambled partners. Not a new download."""
    weights = shuffle_csr_edges(connectome.weights, seed=seed)
    meta = dict(connectome.meta)
    meta["shuffled_edges"] = True
    meta["shuffle_seed"] = int(seed)
    return replace(connectome, weights=weights, meta=meta)
