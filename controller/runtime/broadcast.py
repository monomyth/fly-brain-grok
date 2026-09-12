from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from malecns_cache.somas import write_activity
from runtime.lif import LIFNetwork

_generation = 0
_abs_weights: csr_matrix | None = None


def publish_activity(brain: LIFNetwork) -> None:
    """Map activity onto CNS somata.

    Photoreceptors often have no soma in MaleCNS, so a 0-hop current map
    is invisible. One |W| hop lights lamina/medulla cells that do have somata.
    """
    global _generation, _abs_weights
    _generation += 1
    if _abs_weights is None or _abs_weights.shape != brain.weights.shape:
        _abs_weights = brain.weights.tocsr(copy=True)
        _abs_weights.data = np.abs(_abs_weights.data)
    drive = np.maximum(np.abs(brain.i_ext), brain.rate_hz)
    hop1 = _abs_weights.dot(drive)
    hop2 = _abs_weights.dot(hop1)
    raw = drive + hop1 + 0.45 * hop2
    active = raw[raw > 0]
    scale = float(np.percentile(active, 88)) if active.size else 1.0
    activity = np.clip(raw / max(scale, 1e-6), 0, 1)
    np.sqrt(activity, out=activity)
    write_activity(activity, _generation)
