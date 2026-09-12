from pathlib import Path

from malecns_cache.catalog import FILES
from malecns_cache.graph import load_stub, write_stub_graph
from malecns_cache.lockfile import sha256_file, verify_release
from malecns_cache.paths import WORKSPACE, checkpoints_dir, datasets_dir, home, live_dir, project_data, release_dir
from malecns_cache.prepare import group_cells


def test_home_defaults_to_shared_cache():
    assert home() == Path("/Users/monomyth/code/data/malecns")


def test_generated_artifacts_live_in_this_workspace(monkeypatch, tmp_path):
    monkeypatch.delenv("FLYBRAIN_DATA", raising=False)
    root = project_data()
    assert root == WORKSPACE / "data"
    assert str(checkpoints_dir("rebot-pickup")).startswith(str(root))
    assert str(datasets_dir("rebot-teacher")).startswith(str(root))
    assert str(live_dir()).startswith(str(root))
    assert "/code/data/malecns" not in str(checkpoints_dir("rebot-pickup"))


def test_flybrain_data_env_overrides_project_root(monkeypatch, tmp_path):
    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    assert project_data() == tmp_path.resolve()
    assert checkpoints_dir("rebot-pickup") == tmp_path.resolve() / "checkpoints" / "rebot-pickup"


def test_catalog_has_three_v1_roles():
    roles = {spec["role"] for spec in FILES["v1.0"].values()}
    assert roles == {"annotations", "neurotransmitters", "edges"}


def test_stub_graph_does_not_touch_feathers(tmp_path):
    write_stub_graph(tmp_path, n=8)
    graph = load_stub(tmp_path)
    assert graph.n == 8
    assert graph.weights.nnz == 4
    assert graph.indices("photoreceptors_r1r6").tolist() == [0, 1]
    assert graph.indices("DNp20_R").tolist() == [7]
    assert graph.indices("vnc_motor").tolist() == [6, 7]
    assert graph.indices("leg_mn_front").tolist() == [7]


def test_group_cells_indexes_vnc_leg_motor_neurons():
    groups = group_cells(
        types=["R1-R6", "Ti flexor MN", "Ti extensor MN", "DNp20"],
        sides=["L", "L", "R", "R"],
        superclasses=["ol_sensory", "vnc_motor", "vnc_motor", "descending_neuron"],
        subclasses=["", "fl", "fl", ""],
        neuromeres=["", "T1", "T1", ""],
    )
    assert groups["photoreceptors_r1r6"] == [0]
    assert groups["vnc_motor"] == [1, 2]
    assert groups["leg_mn_front"] == [1, 2]
    assert groups["Ti_flexor"] == [1]
    assert groups["Ti_extensor_R"] == [2]
    assert groups["DN"] == [3]


def test_downloaded_v1_hashes_if_present():
    folder = release_dir("v1.0")
    required = FILES["v1.0"]
    if not all((folder / name).exists() for name in required):
        return
    verify_release("v1.0")
    for name, spec in required.items():
        assert sha256_file(folder / name) == spec["sha256"]
