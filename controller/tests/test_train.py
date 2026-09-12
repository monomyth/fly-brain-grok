import math
from pathlib import Path

import numpy as np
from PIL import Image

from malecns_cache.graph import load_stub
from rebot_adapter.pick import pick_success, score_pick, servo_stalled, spawn_pad_overlay
from runtime.decoder import ArmCommand
from runtime.lif import LIFNetwork
from train.dataset import DROP_STAGES, VIEW_CAMERAS, Sample, for_imitation, load_episode_v1, load_teacher_run, policy_cameras
from train.expert import expert_action, imitation_actions
from train.features import FEATURE_NAMES, features_from_rgbs, neural_features
from train.readout import absorb_constant_columns, apply_linear, fit_linear, random_readout
from train.shuffle import shuffle_csr_edges, shuffle_edges


def test_teacher_loader_reads_consecutive_deltas(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (32, 32), (10, 10, 10)).save(frames / "a.jpg")
    Image.new("RGB", (32, 32), (20, 20, 20)).save(frames / "b.jpg")
    log = tmp_path / "log.jsonl"
    rows = [
        {
            "photo": str(frames / "a.jpg"),
            "tcp_mm": {"x": 100, "y": 0, "z": 40},
            "gripper_mm": 90,
            "cube": {"attached": False},
        },
        {
            "photo": str(frames / "b.jpg"),
            "tcp_mm": {"x": 110, "y": 5, "z": 50},
            "gripper_mm": 80,
            "cube": {"attached": True},
        },
    ]
    log.write_text("".join(json_line(row) for row in rows))
    samples = load_teacher_run(tmp_path)
    assert len(samples) == 1
    assert np.allclose(samples[0].action, [10, 5, 10, -10])
    assert samples[0].yaw_rad == 0.0
    assert samples[0].camera == "Front"


def test_teacher_loader_emits_front_and_gripper(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (32, 32), (10, 10, 10)).save(frames / "front.jpg")
    Image.new("RGB", (32, 32), (11, 11, 11)).save(frames / "grip.jpg")
    Image.new("RGB", (32, 32), (20, 20, 20)).save(frames / "front2.jpg")
    Image.new("RGB", (32, 32), (21, 21, 21)).save(frames / "grip2.jpg")
    (tmp_path / "log.jsonl").write_text(
        json_line(
            {
                "photos": {"Front": str(frames / "front.jpg"), "Gripper": str(frames / "grip.jpg")},
                "tcp_mm": {"x": 100, "y": 0, "z": 40},
                "gripper_mm": 90,
                "cube": {"attached": False},
            }
        )
        + json_line(
            {
                "photos": {"Front": str(frames / "front2.jpg"), "Gripper": str(frames / "grip2.jpg")},
                "tcp_mm": {"x": 110, "y": 0, "z": 40},
                "gripper_mm": 90,
                "cube": {"attached": False},
            }
        )
    )
    samples = load_teacher_run(tmp_path)
    assert [s.camera for s in samples] == ["Front", "Gripper"]
    assert samples[0].image.name == "front.jpg"
    assert samples[1].image.name == "grip.jpg"
    assert "Top" not in VIEW_CAMERAS
    assert VIEW_CAMERAS == ("Front", "Gripper")


def test_teacher_loader_skips_top(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (32, 32), (10, 10, 10)).save(frames / "front.jpg")
    Image.new("RGB", (32, 32), (11, 11, 11)).save(frames / "grip.jpg")
    Image.new("RGB", (32, 32), (12, 12, 12)).save(frames / "top.jpg")
    Image.new("RGB", (32, 32), (20, 20, 20)).save(frames / "front2.jpg")
    Image.new("RGB", (32, 32), (21, 21, 21)).save(frames / "grip2.jpg")
    Image.new("RGB", (32, 32), (22, 22, 22)).save(frames / "top2.jpg")
    (tmp_path / "log.jsonl").write_text(
        json_line(
            {
                "photos": {
                    "Front": str(frames / "front.jpg"),
                    "Gripper": str(frames / "grip.jpg"),
                    "Top": str(frames / "top.jpg"),
                },
                "tcp_mm": {"x": 100, "y": 0, "z": 40},
                "gripper_mm": 90,
                "cube": {"attached": False},
            }
        )
        + json_line(
            {
                "photos": {
                    "Front": str(frames / "front2.jpg"),
                    "Gripper": str(frames / "grip2.jpg"),
                    "Top": str(frames / "top2.jpg"),
                },
                "tcp_mm": {"x": 110, "y": 0, "z": 40},
                "gripper_mm": 90,
                "cube": {"attached": False},
            }
        )
    )
    samples = load_teacher_run(tmp_path)
    assert [s.camera for s in samples] == ["Front", "Gripper"]
    assert all(s.image.name != "top.jpg" for s in samples)


def test_episode_loader_emits_front_and_gripper_skips_top(tmp_path):
    episode = tmp_path / "ep"
    images = episode / "images"
    images.mkdir(parents=True)
    for name in ("f1.jpg", "g1.jpg", "t1.jpg", "f2.jpg", "g2.jpg", "t2.jpg"):
        Image.new("RGB", (16, 16), (8, 8, 8)).save(images / name)

    def row(front, grip, top, x):
        return {
            "observation": {
                "images": [
                    {"name": "Front", "file": f"images/{front}"},
                    {"name": "Gripper", "file": f"images/{grip}"},
                    {"name": "Top", "file": f"images/{top}"},
                ],
                "tool_pose": {"position_mm": [x, 0.0, 48.0]},
                "gripper_mm": 90,
                "cube_pose": {"position_mm": [280.0, 0.0, 19.0]},
                "phase": "approach",
            },
            "stage": "approach",
        }

    (episode / "steps.jsonl").write_text(
        json_line(row("f1.jpg", "g1.jpg", "t1.jpg", 100.0)) + json_line(row("f2.jpg", "g2.jpg", "t2.jpg", 110.0))
    )
    samples = load_episode_v1(episode)
    assert [s.camera for s in samples] == ["Front", "Gripper"]
    assert {s.image.name for s in samples} == {"f1.jpg", "g1.jpg"}


def json_line(row: dict) -> str:
    import json

    return json.dumps(row) + "\n"


def test_teacher_log_reads_yaw(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (32, 32), (10, 10, 10)).save(frames / "a.jpg")
    Image.new("RGB", (32, 32), (20, 20, 20)).save(frames / "b.jpg")
    (tmp_path / "log.jsonl").write_text(
        json_line(
            {
                "photo": str(frames / "a.jpg"),
                "tcp_mm": {"x": 303, "y": 26, "z": 48},
                "tcp_rpy_deg": {"roll": 0, "pitch": 0, "yaw": 49.0},
                "gripper_mm": 90,
                "cube": {"attached": False, "center_mm": {"x": 280, "y": 0, "z": 19}, "size_mm": 40},
            }
        )
        + json_line(
            {
                "photo": str(frames / "b.jpg"),
                "tcp_mm": {"x": 303, "y": 26, "z": 48},
                "tcp_rpy_deg": {"yaw": 49.0},
                "gripper_mm": 42,
                "cube": {"attached": True, "center_mm": {"x": 280, "y": 0, "z": 19}, "size_mm": 40},
            }
        )
    )
    samples = load_teacher_run(tmp_path)
    assert abs(samples[0].yaw_rad - math.radians(49.0)) < 1e-6
    act = expert_action(samples[0])
    gx = 280.0 + 35.0 * math.cos(math.radians(49.0))
    gy = 0.0 + 35.0 * math.sin(math.radians(49.0))
    assert abs(gx - 303) < 8 or abs(act[0]) <= 8


def test_stub_features_and_readout(tmp_path):
    connectome = load_stub(tmp_path)
    brain = LIFNetwork(connectome.weights)
    image = tmp_path / "top.jpg"
    Image.new("RGB", (64, 48), (180, 80, 30)).save(image)
    sample = Sample(
        image=image,
        tcp_mm=np.array([280.0, 0.0, 80.0]),
        cube_mm=np.array([280.0, 0.0, 19.0]),
        gripper_mm=60.0,
        attached=False,
        action=np.array([1.0, 0.0, 2.0, -1.0]),
    )
    feat = neural_features(connectome, brain, sample, hops=2, proprio=True)
    assert feat.shape[0] >= 12
    from train.features import BIAS_INDEX

    assert feat[BIAS_INDEX] == 1.0
    x = np.vstack([feat, feat + 0.01])
    y = np.vstack([sample.action, sample.action])
    weights = fit_linear(x, y, l2=1e-3)
    pred = apply_linear(weights, feat[None, :])[0]
    assert pred.shape == (4,)


def test_load_readout_roundtrip(tmp_path):
    from train.readout import apply_kernel, load_kernel_tables, load_readout, save_readout

    weights = np.ones((3, 4), dtype=np.float32)
    path = tmp_path / "w.npz"
    save_readout(path, weights, {"hops": 3})
    loaded, meta = load_readout(path)
    assert loaded.shape == (3, 4)
    assert meta["hops"] == 3
    x = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    y = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    save_readout(path, weights, {"hops": 3, "kernel": True}, kernel_x=x, kernel_y=y, kernel_w=np.ones(3))
    tables = load_kernel_tables(path)
    assert tables is not None
    kx, ky, kw = tables
    pred = apply_kernel(x[1], kx, ky, sample_weight=kw, tau=0.4)
    assert pred[1] > pred[0]
    assert pred[1] > pred[2]
    far = apply_kernel(np.array([8.0, 8.0, 8.0]), kx, ky, sample_weight=kw, tau=None, k_neighbors=2)
    assert float(np.max(np.abs(far))) > 1e-6
    tiny = np.array([[0.001, 0.0, 1.0], [0.002, 0.0, 1.0], [0.0015, 0.1, 1.0]])
    ty = np.array([[0.0, 0.0, -8.0, 0.0], [0.0, 0.0, -8.0, 0.0], [0.0, 0.0, 8.0, 0.0]])
    live = apply_kernel(np.array([0.02, 0.0, 1.0]), tiny, ty, tau=None, min_std=0.05, k_neighbors=2)
    assert live[2] < 0.0
    from train.readout import apply_local_linear

    rng = np.random.default_rng(0)
    n = 40
    xs = rng.uniform(250.0, 330.0, n)
    feat = np.zeros((n, 35), dtype=np.float64)
    feat[:, 27] = xs / 400.0
    feat[:, 28] = 0.0
    feat[:, 29] = 48.0 / 200.0
    feat[:, 33] = 1.0
    ys = np.zeros((n, 4), dtype=np.float64)
    ys[:, 0] = np.clip(315.0 - xs, -8.0, 8.0)
    left = feat[0].copy()
    left[27] = 259.0 / 400.0
    pred = apply_local_linear(left, feat, ys, k_neighbors=24)
    assert pred[0] > 2.0
    right = feat[0].copy()
    right[27] = 330.0 / 400.0
    pred_r = apply_local_linear(right, feat, ys, k_neighbors=24)
    assert pred_r[0] < -2.0


def _sample(**kwargs) -> Sample:
    image = kwargs.pop("image", Path("missing.jpg"))
    base = dict(
        image=image,
        tcp_mm=np.array([315.0, 0.0, 48.0]),
        cube_mm=np.array([280.0, 0.0, 20.0]),
        gripper_mm=90.0,
        attached=False,
        action=np.array([1.0, 0.0, 0.0, 5.0]),
        stage="approach",
        cube_size_mm=40.0,
        yaw_rad=0.0,
    )
    base.update(kwargs)
    return Sample(**base)


def test_teacher_close_is_negative_dgrip():
    sample = _sample(stage="close", gripper_mm=90.0, action=np.array([0.0, 0.0, 0.0, -8.0]))
    act = expert_action(sample)
    assert act[3] < 0
    assert act[3] == -8.0
    grok = _sample(
        stage="approach",
        tcp_mm=np.array([302.8, 26.4, 48.0]),
        cube_mm=np.array([280.0, 0.0, 19.0]),
        gripper_mm=90.0,
        yaw_rad=0.0,
    )
    assert expert_action(grok)[3] < 0


def test_hold_bonus_zeros_dgrip_and_lifts():
    sample = _sample(attached=True, tcp_mm=np.array([315.0, 0.0, 80.0]), gripper_mm=42.0, stage="hold")
    act = expert_action(sample)
    assert act[3] == 0.0
    assert act[2] > 0
    ys, w = imitation_actions([sample])
    assert ys[0, 3] == 0.0
    assert w[0] >= 6.0


def test_imitation_keeps_logged_open_dgrip_unless_attached():
    opened = _sample(attached=False, action=np.array([1.0, 0.0, 0.0, 5.0]))
    ys, _ = imitation_actions([opened], hold_bonus=True)
    assert ys[0, 3] == 5.0
    raw, _ = imitation_actions([opened], hold_bonus=False)
    assert raw[0, 3] == 5.0
    held = _sample(attached=True, action=np.array([0.0, 0.0, 0.0, 8.0]), tcp_mm=np.array([315.0, 0.0, 80.0]), gripper_mm=42.0)
    bonus, w = imitation_actions([held], hold_bonus=True)
    assert bonus[0, 3] == 0.0
    assert w[0] == 6.0


def test_for_imitation_drops_release_and_spawn():
    samples = [
        _sample(stage="spawn"),
        _sample(stage="approach"),
        _sample(stage="release"),
        _sample(stage="done"),
        _sample(stage="close"),
    ]
    kept = for_imitation(samples)
    assert [s.stage for s in kept] == ["approach", "close"]
    assert "release" in DROP_STAGES


def test_fit_linear_sample_weight_changes_dgrip():
    x = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([[0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, -8.0], [0.0, 0.0, 0.0, -8.0]])
    unweighted = fit_linear(x, y, l2=1e-6)
    weighted = fit_linear(x, y, l2=1e-6, sample_weight=np.array([0.1, 8.0, 8.0]))
    pred_w = apply_linear(weighted, np.array([[1.0, 1.0]]))[0, 3]
    pred_u = apply_linear(unweighted, np.array([[1.0, 1.0]]))[0, 3]
    assert pred_w < pred_u
    assert pred_w < 0


def test_shuffle_edges_keeps_nnz_changes_partners(tmp_path):
    graph = load_stub(tmp_path)
    orig_idx = np.array(graph.weights.indices, copy=True)
    orig_data = np.array(graph.weights.data, copy=True)
    shuffled = None
    seed = 0
    for seed in range(8):
        candidate = shuffle_edges(graph, seed=seed)
        if not np.array_equal(candidate.weights.indices, orig_idx):
            shuffled = candidate
            break
    assert shuffled is not None
    assert np.array_equal(graph.weights.indices, orig_idx)
    assert np.allclose(np.sort(shuffled.weights.data), np.sort(orig_data))
    assert shuffled.weights.nnz == graph.weights.nnz
    assert shuffled.weights.shape == graph.weights.shape
    again = shuffle_csr_edges(graph.weights, seed=seed)
    assert np.array_equal(again.indices, shuffled.weights.indices)


def test_pick_success_requires_hold_level_and_height():
    assert not pick_success(attached=True, cube_z_mm=131, tcp_level=True, hold_s=1.0)
    assert not pick_success(attached=True, cube_z_mm=80, tcp_level=True, hold_s=2.0)
    assert not pick_success(attached=True, cube_z_mm=131, tcp_level=False, hold_s=2.0)
    assert not pick_success(attached=False, cube_z_mm=131, tcp_level=True, hold_s=2.0)
    assert pick_success(attached=True, cube_z_mm=131, tcp_level=True, hold_s=2.0)


def test_score_pick_uses_live_z_not_peak():
    state = {"tcp_level": True, "objects": {"cube": {"attached": True, "center_mm": {"x": 280, "y": 0, "z": 80}}}}
    assert not score_pick(state, hold_s=2.0)
    state["objects"]["cube"]["center_mm"]["z"] = 131
    assert score_pick(state, hold_s=2.0)


def test_servo_stalled_compares_command_to_tcp():
    cmd = ArmCommand(8.0, 0.0, 8.0, 0.0)
    before = {"x": 100.0, "y": 0.0, "z": 100.0}
    assert servo_stalled(cmd, before, {"x": 100.2, "y": 0.0, "z": 100.1})
    assert not servo_stalled(cmd, before, {"x": 107.0, "y": 0.0, "z": 107.0})


def test_overlay_keeps_gripper_open_until_pad_then_closes():
    far = spawn_pad_overlay({"x": 543.0, "y": 0.0, "z": 409.0}, 90.0, False, 0.0)
    assert far.phase == "approach"
    assert far.command.dgrip_mm == 0.0
    assert far.command.dx_mm < 0
    assert far.command.dy_mm == 0.0
    assert far.command.dz_mm < 0
    mid = spawn_pad_overlay({"x": 299.0, "y": 0.0, "z": 59.0}, 90.0, False, 0.0)
    assert mid.phase in {"align", "close"}
    assert mid.command.dx_mm > 0
    if mid.phase == "close":
        assert mid.command.dgrip_mm < 0
    else:
        assert mid.command.dgrip_mm == 0.0
    close = spawn_pad_overlay({"x": 315.0, "y": 0.0, "z": 48.0}, 90.0, False, 0.0)
    assert close.phase == "close"
    assert close.command.dgrip_mm < 0
    lift = spawn_pad_overlay({"x": 315.0, "y": 0.0, "z": 60.0}, 42.0, True, 0.0)
    assert lift.phase == "lift"
    assert lift.command.dgrip_mm == 0.0
    assert lift.command.dz_mm > 0


def test_overlay_pad_follows_yaw():
    hovered = spawn_pad_overlay({"x": 280.0, "y": 0.0, "z": 48.0}, 90.0, False, math.pi / 2)
    assert hovered.phase == "align"
    assert hovered.command.dy_mm > 0
    assert hovered.pad_xyz[1] > 20


def test_random_readout_shape():
    w = random_readout(16, 4, seed=0)
    assert w.shape == (16, 4)


def test_neural_fit_uses_jpeg_pixels(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys

    root = tmp_path / "runs"
    ep = root / "ep"
    frames = ep / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (64, 48), (16, 16, 16)).save(frames / "a-front.jpg")
    Image.new("RGB", (64, 48), (200, 80, 30)).save(frames / "a-grip.jpg")
    Image.new("RGB", (64, 48), (24, 24, 24)).save(frames / "b-front.jpg")
    Image.new("RGB", (64, 48), (210, 90, 40)).save(frames / "b-grip.jpg")
    Image.new("RGB", (64, 48), (30, 30, 30)).save(frames / "c-front.jpg")
    Image.new("RGB", (64, 48), (180, 70, 20)).save(frames / "c-grip.jpg")
    rows = []
    for name, x, grip, attached in (("a", 100, 90, False), ("b", 110, 80, False), ("c", 120, 42, True)):
        rows.append(
            {
                "step": "approach" if not attached else "hold",
                "photos": {
                    "Front": str(frames / f"{name}-front.jpg"),
                    "Gripper": str(frames / f"{name}-grip.jpg"),
                },
                "tcp_mm": {"x": x, "y": 0, "z": 48},
                "gripper_mm": grip,
                "cube": {"attached": attached, "center_mm": {"x": 280, "y": 0, "z": 19}, "size_mm": 40},
            }
        )
    (ep / "log.jsonl").write_text("".join(json_line(r) for r in rows))
    out = tmp_path / "data"
    env = {**os.environ, "FLYBRAIN_DATA": str(out), "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_readout.py"
    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--stub",
            "--extra",
            str(root),
            "--no-overlay-traces",
            "--neural",
            "--tag",
            "vision-test",
        ],
        cwd=str(script.parent.parent),
        env=env,
    )
    meta = json.loads((out / "checkpoints" / "rebot-pickup" / "vision-test.json").read_text())
    assert meta["neural"] is True
    assert meta["jpeg_in_fit"] is True
    assert meta["cameras"] == ["Front", "Gripper"]
    assert any("r1_mean" in name for name in meta["visual_columns_kept"])
    assert meta["r1_std_logged"] > 1e-4
    from train.readout import load_readout

    weights, stored = load_readout(out / "checkpoints" / "rebot-pickup" / "vision-test.npz")
    assert stored["jpeg_in_fit"] is True
    assert not np.allclose(weights[0], 0.0)


def test_absorb_constant_columns_keeps_varying_visuals():
    x = np.array([[0.2, 0.0, 1.0], [0.8, 0.0, 1.0], [0.5, 0.0, 1.0]])
    y = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1.5, 0.0, 0.0, 0.0]])
    raw = np.zeros((3, 4))
    raw[0, 0] = 2.0
    raw[1, 0] = 3.0
    raw[2, 0] = 1.0
    w, kept, absorbed = absorb_constant_columns(raw, x, columns=(0, 1), bias_row=2)
    assert kept == [0]
    assert absorbed == [1]
    assert np.allclose(w[0], raw[0])
    assert np.allclose(w[1], 0.0)


def test_policy_cameras_prefers_logged():
    assert policy_cameras(None) == VIEW_CAMERAS
    assert policy_cameras({"cameras": [], "logged_cameras": ["Front", "Gripper"]}) == ("Front", "Gripper")
    assert policy_cameras({"cameras": ["Front"]}) == ("Front",)


def test_jpeg_means_differ_across_views(tmp_path):
    connectome = load_stub(tmp_path)
    brain = LIFNetwork(connectome.weights)
    dark = tmp_path / "dark.jpg"
    bright = tmp_path / "bright.jpg"
    Image.new("RGB", (64, 48), (10, 10, 10)).save(dark)
    Image.new("RGB", (64, 48), (220, 180, 40)).save(bright)
    tcp = np.array([280.0, 0.0, 80.0])
    dark_s = Sample(image=dark, tcp_mm=tcp, cube_mm=tcp, gripper_mm=90.0, attached=False, action=np.zeros(4), camera="Gripper")
    bright_s = Sample(image=bright, tcp_mm=tcp, cube_mm=tcp, gripper_mm=90.0, attached=False, action=np.zeros(4), camera="Front")
    d = neural_features(connectome, brain, dark_s, hops=2, proprio=True)
    b = neural_features(connectome, brain, bright_s, hops=2, proprio=True)
    assert d[0] != b[0] or d[FEATURE_NAMES.index("grip_r1_mean")] != b[FEATURE_NAMES.index("grip_r1_mean")]
    paired = features_from_rgbs(
        connectome,
        brain,
        [np.asarray(Image.open(bright).convert("RGB"), dtype=np.float32) / 255.0,
         np.asarray(Image.open(dark).convert("RGB"), dtype=np.float32) / 255.0],
        gripper_mm=90.0,
        tcp_mm=tcp,
        attached=False,
        hops=2,
        cameras=("Front", "Gripper"),
    )
    assert paired.shape == d.shape
    assert paired[0] == b[0] or abs(paired[0] - b[0]) < 0.2
    assert FEATURE_NAMES[0] == "front_r1_mean"


def test_clip_spawn_dy_blocks_y_drift():
    from rebot_adapter.pick import clip_spawn_dy

    ready = clip_spawn_dy(ArmCommand(8.0, -8.0, -8.0, 0.0), {"x": 543.0, "y": 0.0, "z": 409.0}, cube_y=0.0)
    assert ready.dy_mm == -8.0
    assert ready.dx_mm == 8.0
    edge = clip_spawn_dy(ArmCommand(0.0, -8.0, 0.0, 0.0), {"x": 400.0, "y": -40.0, "z": 100.0}, cube_y=0.0)
    assert edge.dy_mm == 0.0
    drifted = clip_spawn_dy(ArmCommand(0.0, -8.0, 0.0, 0.0), {"x": 469.0, "y": -363.0, "z": 87.0}, cube_y=0.0)
    assert drifted.dy_mm > 0
    pad = clip_spawn_dy(ArmCommand(0.0, 8.0, 0.0, 0.0), {"x": 280.0, "y": 20.0, "z": 48.0}, cube_y=0.0)
    assert pad.dy_mm == 8.0
    off = clip_spawn_dy(ArmCommand(0.0, -8.0, 0.0, 0.0), {"x": 280.0, "y": 0.0, "z": 48.0}, cube_y=40.0)
    assert off.dy_mm == -8.0
    overshoot = clip_spawn_dy(ArmCommand(-8.0, 0.0, -8.0, -8.0), {"x": 280.0, "y": 0.0, "z": 90.0}, cube_y=0.0)
    assert overshoot.dx_mm == -8.0
    assert overshoot.dgrip_mm == -8.0


def test_overlay_relabel_and_rollout():
    from train.expert import overlay_action, overlay_rollout, overlay_training_rows

    ready = _sample(tcp_mm=np.array([543.0, 0.0, 409.0]), gripper_mm=90.0, attached=False)
    act = overlay_action(ready)
    assert act[0] < 0
    assert abs(act[1]) < 1e-6
    assert act[2] < 0
    assert act[3] == 0.0
    pad = _sample(tcp_mm=np.array([315.0, 0.0, 48.0]), gripper_mm=90.0, attached=False, yaw_rad=0.0)
    close = overlay_action(pad)
    assert close[3] < 0
    trace = overlay_rollout(start_tcp=(543.0, 0.0, 409.0), yaw_rad=0.0, max_steps=80)
    phases = [row["phase"] for row in trace]
    assert "approach" in phases
    assert "close" in phases
    assert "lift" in phases or "hold" in phases
    assert max(abs(row["tcp"][1]) for row in trace) < 45.0
    train_rows, val_rows = overlay_training_rows()
    assert len(train_rows) > 100
    assert len(val_rows) > 10
    assert any(row["phase"] == "approach" for row in val_rows)


def test_overlay_trace_readout_stays_on_y0():
    from train.expert import overlay_training_rows
    from train.features import proprio_feature_row

    train_rows, val_rows = overlay_training_rows()
    rows = train_rows + val_rows
    x = np.vstack(
        [proprio_feature_row(r["grip"], r["tcp"], r["attached"], yaw_rad=float(r.get("yaw_rad") or 0.0)) for r in rows]
    )
    y = np.vstack([r["action"] for r in rows])
    w = np.array([r["weight"] for r in rows])
    hat = fit_linear(x, y, l2=1e-2, sample_weight=w)
    ready = proprio_feature_row(90.0, np.array([543.0, 0.0, 409.0]), False, yaw_rad=0.0)
    pred = apply_linear(hat, ready[None, :])[0]
    assert pred[0] < 0
    assert abs(pred[1]) <= 8.0
    assert pred[2] < 0
    pad = proprio_feature_row(90.0, np.array([303.0, 26.0, 48.0]), False, yaw_rad=math.radians(57.0))
    close = apply_linear(hat, pad[None, :])[0]
    assert close[3] < 0
    held = proprio_feature_row(42.0, np.array([303.0, 26.0, 80.0]), True, yaw_rad=math.radians(57.0))
    lift = apply_linear(hat, held[None, :])[0]
    assert lift[2] > 0
    assert abs(lift[3]) < 3.0


def test_episode_outcome_flags():
    from rebot_adapter.pick import episode_outcome

    quintic = episode_outcome(
        overlay=False, readout=True, quintic_ran=True, attached=True, cube_z_mm=130.0, tcp_level=True, hold_s=2.0
    )
    assert quintic["picked"] is False
    assert quintic["readout_picked"] is False
    assert quintic["teacher_ok"] is True
    assert quintic["quintic_ran"] is True
    assert quintic["controller"] == "quintic-teacher"
    overlay = episode_outcome(
        overlay=True, readout=False, quintic_ran=False, attached=True, cube_z_mm=122.0, tcp_level=True, hold_s=2.0
    )
    assert overlay["controller"] == "overlay-servo"
    assert overlay["readout"] is False
    assert overlay["picked"] is True
    assert overlay["readout_picked"] is False
    assert overlay["lab_picked"] is True
    assert overlay["fly_picked"] is False
    readout = episode_outcome(
        overlay=False, readout=True, quintic_ran=False, attached=False, cube_z_mm=19.0, tcp_level=True, hold_s=0.0
    )
    assert readout["controller"] == "readout"
    assert readout["overlay"] is False
    assert readout["quintic_ran"] is False
    assert readout["picked"] is False
    both = episode_outcome(
        overlay=True, readout=True, quintic_ran=False, attached=True, cube_z_mm=122.0, tcp_level=True, hold_s=2.0
    )
    assert both["controller"] == "overlay-servo"
    assert both["readout"] is False
    assert both["picked"] is True
    assert both["readout_picked"] is False
    assert both["fly_picked"] is False
    fly = episode_outcome(
        overlay=False, readout=True, quintic_ran=False, attached=True, cube_z_mm=122.0, tcp_level=True, hold_s=2.0
    )
    assert fly["controller"] == "readout"
    assert fly["fly_picked"] is False
    assert fly["lab_picked"] is True
    assert fly["readout_picked"] is True
    assert fly["overlay"] is False
    assert fly["quintic_ran"] is False
    assert fly["acting_map"] == "knn"
    assert quintic["fly_picked"] is False
    assert overlay["fly_picked"] is False
    gated_lift = episode_outcome(
        overlay=False,
        readout=True,
        quintic_ran=False,
        attached=True,
        cube_z_mm=122.0,
        tcp_level=True,
        hold_s=2.0,
        xyz_diverged=True,
    )
    assert gated_lift["controller"] == "readout"
    assert gated_lift["picked"] is False
    assert gated_lift["fly_picked"] is False
    assert gated_lift["readout_picked"] is False
    assert gated_lift["xyz_diverged"] is True


def test_eval_live_commands_cannot_score_teacher_or_overlay_as_fly():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py").read_text()
    assert '"--no-overlay"' in src or "'--no-overlay'" in src
    assert "frozen" in src
    assert "--readout" in src
    assert "run_teacher.py" in src
    trained_line = [line for line in src.splitlines() if "eval-trained" in line]
    assert trained_line, "live trained eval must exist"
    joined = "\n".join(trained_line)
    assert "--overlay" not in joined.replace("--no-overlay", "")
    pick_src = (Path(__file__).resolve().parents[1] / "rebot_adapter" / "pick.py").read_text()
    assert 'controller == "readout"' in pick_src
    assert "fly_picked" in pick_src
    assert "score_episode" in pick_src
    assert "quintic-teacher" in pick_src
    assert "xyz_diverged" in pick_src
    assert "readout_xyz_diverged" in pick_src
    score_src = (Path(__file__).resolve().parents[1] / "arm" / "score.py").read_text()
    assert "ActingMap" in score_src
    assert "dn_bus" in score_src
    assert "def fly_picked" in score_src
    assert "def lab_picked" in score_src
    assert "def da_learned" in score_src
    assert "6.0 * math.cos" not in pick_src
    assert "tcp_z < 158" not in pick_src
    arm_src = (Path(__file__).resolve().parents[1] / "scripts" / "run_dn_arm.py").read_text()
    assert "readout_xyz_diverged" in arm_src
    assert "dx_raw" in arm_src and "dz_raw" in arm_src and "dgrip_raw" in arm_src
    assert "clip_spawn_xy" not in arm_src
    assert "kx[:, att_i]" not in arm_src
    assert "col_scale[att_i]" not in arm_src
    assert "-np.sign(y)" not in pick_src
    assert "column_scale" not in arm_src
    assert "apply_linear" in arm_src
    assert "attached=False" in arm_src
    assert "overlay_attached_lift_policy" in arm_src
    assert "acting_map_changed" in arm_src
    assert "clipped_plus8_hold_policy" in arm_src
    assert "gripper_mm > 72" not in arm_src
    assert "3.0 * cmd.dgrip" not in arm_src
    assert "kx[:, att_i]" not in arm_src
    assert "col_scale[att_i]" not in arm_src
    assert "max(32.0" in arm_src
    assert "max(46.0" not in arm_src
    assert "z_floor_bound" in arm_src
    assert "<= 46.0" not in arm_src
    train_src = (Path(__file__).resolve().parents[1] / "scripts" / "train_readout.py").read_text()
    assert "np.repeat(x_logged[rare]" not in train_src
    assert "overlay_matrix(overlay_kernel_rows)" not in train_src
    expert_src = (Path(__file__).resolve().parents[1] / "train" / "expert.py").read_text()
    assert "clip(hold_z - z" not in expert_src
    readout_src = (Path(__file__).resolve().parents[1] / "train" / "readout.py").read_text()
    assert "std[-3]" not in readout_src


def test_overlay_attached_lift_policy_blocks_fly_picked():
    from rebot_adapter.pick import episode_outcome, overlay_attached_lift_policy

    rows = [
        {"attached": True, "cube_z": 30.0 + 8.0 * i, "dx": 0.0, "dy": 0.0, "dz": 8.0, "dgrip": 0.0}
        for i in range(12)
    ]
    assert overlay_attached_lift_policy(rows) is True
    locked = episode_outcome(
        overlay=False,
        readout=True,
        quintic_ran=False,
        attached=True,
        cube_z_mm=122.0,
        tcp_level=True,
        hold_s=2.0,
        overlay_lift_policy=True,
    )
    assert locked["controller"] == "readout"
    assert locked["fly_picked"] is False
    assert locked["readout_picked"] is False
    assert locked["overlay_lift_policy"] is True
    varied = [
        {"attached": True, "cube_z": 40.0 + 7.0 * i, "dx": 0.5, "dy": -0.3, "dz": 7.5 - 0.4 * i, "dgrip": 0.0}
        for i in range(12)
    ]
    assert overlay_attached_lift_policy(varied) is False
    ok = episode_outcome(
        overlay=False,
        readout=True,
        quintic_ran=False,
        attached=True,
        cube_z_mm=122.0,
        tcp_level=True,
        hold_s=2.0,
        overlay_lift_policy=False,
    )
    assert ok["fly_picked"] is False
    assert ok["readout_picked"] is True
    assert ok["lab_picked"] is True
    from rebot_adapter.pick import acting_map_changed, clipped_plus8_hold_policy

    swapped = [{"map": "kernel", "attached": False, "cube_z": 19, "dz": -8}] * 8
    swapped += [
        {"map": "linear", "attached": True, "cube_z": 30.0 + 8.0 * i, "dx": 0.3, "dy": 0.2, "dz": 8.0, "dgrip": -1.0}
        for i in range(8)
    ]
    assert acting_map_changed(swapped) is True
    assert clipped_plus8_hold_policy(swapped) is True
    switched = episode_outcome(
        overlay=False,
        readout=True,
        quintic_ran=False,
        attached=True,
        cube_z_mm=122.0,
        tcp_level=True,
        hold_s=2.0,
        map_changed=True,
        plus8_hold_policy=True,
    )
    assert switched["fly_picked"] is False


def test_pair_view_samples_keeps_front_and_gripper(tmp_path):
    from train.dataset import Sample, pair_view_samples

    front = tmp_path / "f.jpg"
    grip = tmp_path / "g.jpg"
    front.write_bytes(b"x")
    grip.write_bytes(b"y")
    tcp = np.array([280.0, 0.0, 48.0])
    rows = [
        _sample(image=front, tcp_mm=tcp, camera="Front", views={"Front": front, "Gripper": grip}),
        _sample(image=grip, tcp_mm=tcp, camera="Gripper", views={"Front": front, "Gripper": grip}),
    ]
    paired = pair_view_samples(rows)
    assert len(paired) == 1
    assert set(paired[0].views) == {"Front", "Gripper"}


def test_optical_moments_find_orange_blob():
    from runtime.encoder import optical_vector

    rgb = np.zeros((40, 60, 3), dtype=np.float32)
    rgb[30:38, 40:55, 0] = 1.0
    rgb[30:38, 40:55, 1] = 0.2
    vec, r1, r8 = optical_vector(rgb, n_r1r6=64, n_r8=64)
    assert vec[4] > 0.0
    assert vec[2] > 0.5
    assert vec[3] > 0.5
    dark = np.zeros((40, 60, 3), dtype=np.float32)
    blank, _, _ = optical_vector(dark, n_r1r6=64, n_r8=64)
    assert blank[4] == 0.0


def test_optical_close_gate_blocks_early_close():
    from rebot_adapter.pick import optical_close_gate, readout_xyz_diverged
    from runtime.decoder import ArmCommand
    from train.features import FEATURE_NAMES, pack_features

    feat_far = pack_features(
        np.array([0.4, 0.0, 0.5, 0.5, 0.0, 0, 0, 0, 0]),
        np.array([0.2, 0.0, 0.5, 0.5, 0.0, 0, 0, 0, 0]),
        np.zeros(8),
        90.0,
        np.array([500.0, 0.0, 300.0]),
        False,
    )
    raw_far = ArmCommand(-3.0, 2.0, -8.0, -8.0)
    closed = optical_close_gate(raw_far, feat_far, FEATURE_NAMES, tcp_z=300.0)
    assert closed.dgrip_mm == 0.0
    assert closed.dx_mm == raw_far.dx_mm
    assert closed.dy_mm == raw_far.dy_mm
    assert closed.dz_mm == raw_far.dz_mm
    assert not readout_xyz_diverged(raw_far, closed)
    feat_near = pack_features(
        np.array([0.4, 0.1, 0.5, 0.6, 0.02, 0, 0, 0, 0.02]),
        np.array([0.5, 0.2, 0.5, 0.8, 0.04, 0, 0, 0, 0.04]),
        np.zeros(8),
        90.0,
        np.array([315.0, 0.0, 48.0]),
        False,
    )
    raw_near = ArmCommand(1.0, -2.0, -4.0, -8.0)
    pinch = optical_close_gate(raw_near, feat_near, FEATURE_NAMES, tcp_z=48.0)
    assert pinch.dgrip_mm < 0
    pad_band = optical_close_gate(raw_near, feat_near, FEATURE_NAMES, tcp_z=55.0)
    assert pad_band.dgrip_mm < 0
    assert pinch.dx_mm == raw_near.dx_mm
    assert pinch.dy_mm == raw_near.dy_mm
    assert pinch.dz_mm == raw_near.dz_mm
    held_raw = ArmCommand(1.0, 1.0, 1.0, -4.0)
    held = optical_close_gate(held_raw, feat_near, FEATURE_NAMES, tcp_z=80.0, attached=True)
    assert held.dx_mm == 1.0
    assert held.dy_mm == 1.0
    assert held.dz_mm == 1.0
    assert held.dgrip_mm == 0.0
    assert not readout_xyz_diverged(held_raw, held)
    injected = ArmCommand(0.0, 0.0, 6.0, 0.0)
    assert readout_xyz_diverged(held_raw, injected)
