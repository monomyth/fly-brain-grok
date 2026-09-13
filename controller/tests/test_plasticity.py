import numpy as np
import pytest
from scipy.sparse import csr_matrix

from malecns_cache.graph import Connectome, load_stub
from runtime.lif import LIFNetwork
from runtime.plasticity import KCToMBON
from train.features import drive_kenyon_from_image


def test_reward_changes_existing_kc_mbon_edge(tmp_path):
    graph = load_stub(tmp_path)
    brain = LIFNetwork(graph.weights)
    assert brain.weights is graph.weights
    memory = KCToMBON(graph, brain)
    assert memory.slots.size >= 1
    before = brain.weights.data[memory.slots].copy()
    brain.duty[graph.indices("KC")] = 1.0
    brain.duty[graph.indices("MBON11")] = 1.0
    brain.rate_hz[graph.indices("PPL101")] = 40
    n = memory.update(1.0, eta=1e-3)
    after = brain.weights.data[memory.slots]
    assert n == memory.slots.size
    assert float(np.max(np.abs(after - before))) > 0


def test_spatial_kc_drive_is_not_uniform():
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[20:30, 20:30] = 1.0

    class B:
        def __init__(self):
            self.n = 64
            self.i_ext = np.zeros(64, np.float32)

    b = B()
    kc = np.arange(64, dtype=np.int32)
    drive_kenyon_from_image(b, kc, rgb)
    assert b.i_ext.std() > 10.0
    assert float((b.i_ext > 50).mean()) < 0.4
    assert float(b.i_ext.max()) > 50


def test_two_blob_positions_drive_different_kenyon_cells():
    a = np.zeros((64, 64, 3), dtype=np.float32)
    b = np.zeros((64, 64, 3), dtype=np.float32)
    a[4:14, 4:14] = 1.0
    b[48:62, 48:62] = 1.0

    class Brain:
        def __init__(self):
            self.n = 64
            self.i_ext = np.zeros(64, np.float32)

    kc = np.arange(64, dtype=np.int32)
    left, right = Brain(), Brain()
    drive_kenyon_from_image(left, kc, a)
    drive_kenyon_from_image(right, kc, b)
    na, nb = np.linalg.norm(left.i_ext), np.linalg.norm(right.i_ext)
    cos = float(left.i_ext @ right.i_ext / (na * nb)) if na and nb else 1.0
    assert cos < 0.5


def _sheet(n_kc: int = 8) -> tuple[Connectome, LIFNetwork, KCToMBON]:
    n = n_kc + 4
    mbon = n_kc
    data = np.ones(n_kc, dtype=np.float32)
    indices = np.arange(n_kc, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[mbon + 1 :] = n_kc
    weights = csr_matrix((data, indices, indptr), shape=(n, n))
    graph = Connectome(
        weights=weights,
        body_id=np.arange(n, dtype=np.int64),
        sign=np.ones(n, dtype=np.float32),
        groups={"KC": list(range(n_kc)), "MBON11": [mbon], "PPL101": [n_kc + 1]},
        meta={"n": n},
        path=__import__("pathlib").Path("."),
    )
    brain = LIFNetwork(weights)
    return graph, brain, KCToMBON(graph, brain)


def test_negative_reward_decreases_weight_even_if_ppl_rate_is_huge(tmp_path):
    graph = load_stub(tmp_path)
    brain = LIFNetwork(graph.weights)
    memory = KCToMBON(graph, brain)
    before = brain.weights.data[memory.slots].copy()
    brain.duty[graph.indices("KC")] = 1.0
    brain.duty[graph.indices("MBON11")] = 1.0
    brain.duty[graph.indices("PPL101")] = 1.0
    brain.rate_hz[graph.indices("PPL101")] = 400
    memory.update(-1.0, eta=1e-2)
    after = brain.weights.data[memory.slots]
    assert float(np.min(after - before)) < 0


def test_shuffled_phase_rewards_change_delta_direction():
    def run(signs: tuple[float, float]) -> np.ndarray:
        graph, brain, memory = _sheet(8)
        rest = memory.rest.copy()
        brain.duty[:] = 0
        brain.duty[0:4] = 1.0
        brain.duty[graph.indices("MBON11")] = 1.0
        memory.update(signs[0], eta=1e-2)
        brain.duty[:] = 0
        brain.duty[4:8] = 1.0
        brain.duty[graph.indices("MBON11")] = 1.0
        memory.update(signs[1], eta=1e-2)
        return brain.weights.data[memory.slots] - rest

    task = run((1.0, 1.0))
    shuffle = run((1.0, -1.0))
    na, nb = float(np.linalg.norm(task)), float(np.linalg.norm(shuffle))
    cos = float(task @ shuffle / (na * nb)) if na and nb else 1.0
    assert na > 0 and nb > 0
    assert cos < 0.5


def test_update_uses_kenyon_pre_trace_not_silent_lif():
    graph, brain, memory = _sheet(8)
    brain.duty[:] = 0
    brain.rate_hz[:] = 0
    drive = np.zeros(brain.n, dtype=np.float32)
    drive[0:4] = 4.0
    before = brain.weights.data[memory.slots].copy()
    memory.accumulate_from_pre(drive)
    memory.update(1.0, eta=1e-2)
    delta = float(np.max(np.abs(brain.weights.data[memory.slots] - before)))
    assert delta == pytest.approx(0.0035, rel=1e-3)


def test_flat_nonblack_image_still_drives_kenyon():
    rgb = np.full((32, 32, 3), 0.4, dtype=np.float32)

    class B:
        def __init__(self):
            self.n = 64
            self.i_ext = np.zeros(64, np.float32)

    b = B()
    kc = np.arange(64, dtype=np.int32)
    drive_kenyon_from_image(b, kc, rgb)
    assert float(b.i_ext.min()) > 1.0
    b.i_ext[:] = 0
    drive_kenyon_from_image(b, kc, np.zeros((32, 32, 3), dtype=np.float32))
    assert float(b.i_ext.max()) == 0.0
