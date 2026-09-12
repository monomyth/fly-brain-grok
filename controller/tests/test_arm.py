from pathlib import Path

import numpy as np

from arm.bus import BusRates, pool_rates
from arm.crop import GAIN_CLASSES, write_stub_crop
from arm.eye import camera_hex_spans, cube_chroma_frac, drive_camera, local_contrast, luma, orange_chroma, phase_scramble
from arm.hop_probe import (
    DY_FLIP_MIN,
    _pick_pair,
    _synthetic_cube,
    load_optical_fg_pairs,
    live_replay_probe,
    passed,
    run as hop_run,
    train_gate,
    vision_ok,
    vision_probe,
)
from arm.lif_crop import CropLIF
from arm.log import g_hash
from arm.score import (
    ActingMap,
    bus_commanded_plus_z,
    fly_picked,
    lift_without_plus_z,
    open_gripper_tick0_attach,
    overlay_plus8_lift,
    score_episode,
    tcp_rose_while_dz_negative,
    trace_blocks_fly,
)
from arm.teacher import teacher_target
from arm.unpack import U0, UParams, command_is_abort, command_saturated, grip_target_mm, unpack
from rebot_adapter.pick import (
    NEAR_PAD_MM,
    episode_outcome,
    hold_pad_dx,
    hold_pad_dy,
    in_jaw_box,
    jaw_z_floor_mm,
    optical_close_gate,
    pad_aim_x_mm,
    pad_y_tol_mm,
    pick_success,
    tip_grasp_z_mm,
)
from runtime.decoder import ArmCommand


def test_acting_map_enum_values():
    assert {m.value for m in ActingMap} == {"overlay", "knn", "teacher_distill", "dn_bus", "leg_mn", "idle"}


def test_overlay_pick_is_lab_not_fly():
    scored = score_episode(
        acting_map=ActingMap.overlay,
        kinematic_success=True,
        teacher_in_path=True,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        g_trained=True,
    )
    assert scored.lab_picked is True
    assert scored.fly_picked is False
    assert scored.da_learned is False


def test_knn_pick_is_lab_not_fly():
    scored = score_episode(
        acting_map="kernel",
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=12.0,
        black_dn_hz=0.0,
        g_trained=True,
    )
    assert scored.acting_map is ActingMap.knn
    assert scored.lab_picked is True
    assert scored.fly_picked is False


def test_dn_bus_silent_is_not_fly():
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=0.0,
        black_dn_hz=0.0,
        g_trained=True,
    )
    assert scored.fly_picked is False
    assert scored.lab_picked is False


def test_teacher_distill_pick_is_not_fly():
    scored = score_episode(
        acting_map=ActingMap.teacher_distill,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        g_trained=True,
    )
    assert scored.lab_picked is True
    assert scored.fly_picked is False


def test_untrained_dn_bus_pick_is_not_fly():
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        g_trained=False,
    )
    assert scored.fly_picked is False


def test_trained_dn_bus_with_live_rates_is_fly():
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        dn_l2=20.0,
        g_trained=True,
    )
    assert scored.fly_picked is True
    assert scored.lab_picked is False
    assert scored.da_learned is False


def test_da_learned_only_from_ablation():
    ok = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        g_trained=True,
        ablation_changed=True,
    )
    no = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        g_trained=True,
        ablation_changed=False,
    )
    assert ok.da_learned is True
    assert no.da_learned is False


def test_episode_outcome_dn_bus_flags():
    row = episode_outcome(
        overlay=False,
        readout=False,
        quintic_ran=False,
        attached=True,
        cube_z_mm=120.0,
        tcp_level=True,
        hold_s=2.0,
        acting_map="dn_bus",
        mean_dn_hz=15.0,
        black_dn_hz=0.0,
        g_trained=True,
    )
    assert row["acting_map"] == "dn_bus"
    assert row["fly_picked"] is True
    assert row["lab_picked"] is False
    overlay = episode_outcome(
        overlay=True,
        readout=False,
        quintic_ran=False,
        attached=True,
        cube_z_mm=120.0,
        tcp_level=True,
        hold_s=2.0,
    )
    assert overlay["lab_picked"] is True
    assert overlay["fly_picked"] is False


def test_unpack_zeros_when_silent(tmp_path):
    crop = write_stub_crop(tmp_path)
    hz = np.zeros(crop.n, dtype=np.float32)
    rates = pool_rates(hz, crop.groups)
    cmd = unpack(rates, U0)
    assert cmd.dx_mm == 0 and cmd.dy_mm == 0 and cmd.dz_mm == 0 and cmd.dgrip_mm == 0


def test_unpack_does_not_use_dnp20(tmp_path):
    src = Path(__file__).resolve().parents[1] / "arm" / "unpack.py"
    text = src.read_text()
    assert "DNp20" not in text.split('"""', 2)[-1]
    assert "DNg02" not in text


def test_eye_split_not_averaged():
    rgb = _synthetic_cube()
    contrast = local_contrast(rgb)
    assert contrast.shape == rgb.shape[:2]
    # hex that lands on the cube edge (DoG is near zero in the interior)
    front = drive_camera(rgb, np.array([12, 18], dtype=np.int32), np.array([14, 16], dtype=np.int32))
    grip = drive_camera(np.zeros_like(rgb), np.array([10, 28], dtype=np.int32), np.array([14, 22], dtype=np.int32))
    assert front.shape == (2,)
    assert float(np.max(np.abs(grip))) == 0.0
    assert float(np.max(front)) > 0.0
    src = (Path(__file__).resolve().parents[1] / "arm" / "eye.py").read_text()
    assert "np.mean" not in src or "cameras" not in src
    assert "concat" not in src.lower()


def test_camera_hex_spans_cover_full_jpeg():
    hex1 = np.array([6.0, 21.0, 36.0], dtype=np.float32)
    hex2 = np.array([6.0, 20.0, 39.0], dtype=np.float32)
    s1, s2 = camera_hex_spans(hex1, hex2)
    assert s1 == (6.0, 36.0)
    assert s2 == (6.0, 39.0)
    rgb = np.zeros((120, 160, 3), dtype=np.float32)
    rgb[:, :20, 0] = 0.85
    rgb[:, :20, 1] = 0.35
    rgb[:, :20, 2] = 0.08
    left = drive_camera(rgb, hex1[:1], hex2[:1], hex1_span=s1, hex2_span=s2)
    rgb[:] = 0
    rgb[:, -20:, 0] = 0.85
    rgb[:, -20:, 1] = 0.35
    rgb[:, -20:, 2] = 0.08
    right = drive_camera(rgb, hex1[-1:], hex2[-1:], hex1_span=s1, hex2_span=s2)
    assert float(left[0]) > 0.5
    assert float(right[0]) > 0.5
    world = camera_hex_spans(hex1, hex2, world=((1.0, 36.0), (1.0, 39.0)))
    assert world[0] == (1.0, 36.0)


def test_phase_scramble_keeps_energy():
    rgb = _synthetic_cube()
    scr = phase_scramble(rgb, np.random.default_rng(0))
    assert scr.shape == rgb.shape
    g, s = luma(rgb), luma(scr)
    assert abs(float(g.mean()) - float(s.mean())) < 0.02
    assert abs(float(g.std()) - float(s.std())) / max(float(g.std()), 1e-6) < 0.15
    assert float(np.corrcoef(g.ravel(), s.ravel())[0, 1]) < 0.6


def test_teacher_sets_overlay_map():
    cmd = teacher_target({"x": 543.0, "y": 0.0, "z": 409.0}, 90.0, False, 0.0)
    assert cmd.acting_map is ActingMap.overlay
    assert cmd.phase == "approach"


def test_stub_crop_lif_no_per_frame_reset(tmp_path):
    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, synaptic_gain=2.5, nsteps=100)
    rgb = _synthetic_cube()
    resets: list[int] = []
    real = lif.reset_episode

    def wrapped() -> None:
        resets.append(1)
        real()

    lif.reset_episode = wrapped  # type: ignore[method-assign]
    lif.brain.v.fill(0.4)
    lif.step_vision(rgb, rgb)
    assert resets == []
    assert lif.nsteps >= 100
    black = lif.black_baseline()
    cube_rates = lif.rates()
    assert black.scored_mean_hz >= 0.0
    cmd0 = unpack(cube_rates.silenced(), U0)
    assert cmd0.dx_mm == 0.0


def test_g_hash_changes_when_trained(tmp_path):
    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, synaptic_gain=2.0, nsteps=100)
    init = g_hash(lif.g)
    g = lif.g.copy()
    g[0] = 1.4
    lif.set_g(g)
    assert g_hash(lif.g) != init
    assert lif.g_trained is True


def test_hop_probe_stub(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    report = hop_run(stub=True, nsteps=100, dest=tmp_path / "hop.json")
    assert "hops" in report
    assert report["n_scored_dn"] > 0
    assert report["scored_types"]["DNp01"] >= 1
    assert any(row["layer"] == "scored_dn" for row in report["hops"])
    # stub chain should light DNs under direct injection
    dn_row = next(row for row in report["hops"] if row["layer"] == "DNfl")
    assert dn_row["self_hz"] > 0.0


def test_pick_success_still_requires_hold():
    assert not pick_success(attached=True, cube_z_mm=120, tcp_level=True, hold_s=1.0)
    assert pick_success(attached=True, cube_z_mm=120, tcp_level=True, hold_s=2.0)


def test_skip_checkpoint_retires_flood_npz(tmp_path):
    import json as json_lib

    from arm.log import write_skip_checkpoint

    npz = tmp_path / "g-distill.npz"
    np.savez(npz, g=np.ones(3, dtype=np.float32))
    js = tmp_path / "g-distill.json"
    js.write_text(json_lib.dumps({"ok": True, "skipped": False, "g_hash": "5163abe549abd9fa"}) + "\n")
    out = write_skip_checkpoint(json_path=js, npz=npz, payload={"reason": "hop_probe not green"})
    assert out["ok"] is False
    assert out["skipped"] is True
    assert out["fly_picked"] is False
    assert not npz.is_file()
    assert (tmp_path / "g-distill.agc-flood.npz").is_file()
    saved = json_lib.loads(js.read_text())
    assert saved["ok"] is False and saved["skipped"] is True
    assert saved["fly_picked"] is False


def test_live_rewards_keeps_child_only_after_its_own_R(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import GAIN_CLASSES
    from arm.hop_probe import _synthetic_cube
    from arm.train_operant import operant

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    rgb = _synthetic_cube()
    n = len(GAIN_CLASSES)
    g_bad = np.full(n, 0.5, dtype=np.float32)
    g_good = np.full(n, 1.5, dtype=np.float32)
    out = operant(
        stub=True,
        live_rewards=[
            {"R": 1.0},
            {"R": -1.0, "g": g_bad, "front_rgb": rgb, "grip_rgb": rgb},
            {"R": 1.0, "g": g_good, "front_rgb": rgb, "grip_rgb": rgb},
        ],
    )
    assert out["skipped"] is False
    assert out["history"][0]["kept"] is False
    assert out["history"][1]["kept"] is False
    assert out["history"][2]["kept"] is True
    assert np.allclose(np.asarray(out["g"], dtype=np.float32), g_good)


def test_g_init_freezes_loaded_gains(tmp_path):
    from arm.run_dn_bus import apply_controller_g

    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, nsteps=100)
    g = lif.g.copy()
    g[0] = 1.7
    path = tmp_path / "g.npz"
    np.savez(path, g=g)
    apply_controller_g(lif, path, freeze_init=False)
    assert abs(float(lif.g[0]) - 1.7) < 1e-5
    apply_controller_g(lif, path, freeze_init=True)
    assert np.allclose(lif.g, 1.0)
    assert not lif.g_trained


def test_freeze_g_of_distilled_gains_reaborts_on_pad():
    import pytest
    from runtime.encoder import load_image
    from arm.crop import build_crop
    from arm.run_dn_bus import apply_controller_g, load_u

    repo = Path(__file__).resolve().parents[2]
    gains = repo / "data" / "checkpoints" / "rebot-pickup" / "g-distill.npz"
    front_p = repo / "data" / "datasets" / "rebot-optical-fg" / "ep20-pad-z48" / "frames" / "0000-front.jpg"
    grip_p = repo / "data" / "datasets" / "rebot-optical-fg" / "ep20-pad-z48" / "frames" / "0000-gripper.jpg"
    if not (gains.is_file() and front_p.is_file() and grip_p.is_file()):
        pytest.skip("need g-distill.npz and ep20-pad-z48 Front+Gripper JPEGs")
    assert U0.abort_hz == 2000.0
    u = load_u(gains)
    assert abs(float(u.abort_hz) - 2000.0) < 1e-9
    crop = build_crop("v1.0")
    front = load_image(front_p)
    grip = load_image(grip_p)
    lif = CropLIF(crop, synaptic_gain=2.5, nsteps=150)

    def abort_ticks() -> list[bool]:
        lif.reset_episode()
        out = []
        for _ in range(3):
            lif.step_vision(front, grip)
            rates = lif.rates()
            cmd = unpack(rates, u)
            out.append(bool(rates.abort or command_is_abort(cmd)))
        return out

    apply_controller_g(lif, gains, freeze_init=False)
    assert lif.g_trained
    distilled = abort_ticks()
    apply_controller_g(lif, gains, freeze_init=True)
    assert not lif.g_trained
    frozen = abort_ticks()
    assert U0.abort_hz == 2000.0
    assert not any(distilled)
    assert any(frozen)


def test_apply_gains_preserves_da_delta(tmp_path):
    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, nsteps=100)
    lif.crop.weights.data[0] = lif.crop.w0[0] + 0.5
    lif.record_da_delta()
    g = np.full(lif.g.shape, 1.2, dtype=np.float32)
    lif.set_g(g)
    expected = float(lif.crop.w0[0] * 1.2 + 0.5)
    assert abs(float(lif.crop.weights.data[0]) - expected) < 1e-5


def test_offline_tick_log_is_dn_bus_and_nonempty(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.log import write_jsonl
    from arm.run_dn_bus import run_offline_ticks

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, nsteps=100)
    summary, log = run_offline_ticks(lif, n=3)
    assert len(log) == 3
    assert all(row["acting_map"] == "dn_bus" for row in log)
    assert "command" in log[0] and "bus" in log[0]
    dest = tmp_path / "trace.jsonl"
    write_jsonl(dest, log)
    lines = [ln for ln in dest.read_text().splitlines() if ln.strip()]
    assert len(lines) == 3
    assert summary["g_hash_init"] == g_hash(lif.g_init)
    assert summary["g_hash_init"] != "init"


def _vision_ok_base(**extra):
    base = {
        "cube_dn_mean": 10.0,
        "scramble_dn_mean": 4.0,
        "black_dn_mean": 0.0,
        "l2_cube_black": 10.0,
        "l2_cube_scramble": 8.0,
        "l2_cube_shift": 5.0,
        "cmd_l1_cube_shift": 2.0,
        "cos_cube_scramble": 0.5,
        "cube_is_abort": False,
        "gf_hz_cube": 10.0,
        "cube_r1": 5.0,
        "scramble_r1": 4.5,
        "dy_left": -0.8,
        "dy_right": 0.9,
        "laterality_dy_flips": True,
        "split_dy_flips": True,
        "cube_command_saturated": False,
        "silence_command_zero": True,
    }
    base.update(extra)
    return base


def test_hop_vision_ok_rejects_flood_and_abort():
    base = _vision_ok_base()
    assert vision_ok(base)
    copy = dict(base, cos_cube_scramble=0.95)
    assert not vision_ok(copy)
    abort = dict(base, cube_is_abort=True)
    assert not vision_ok(abort)
    gf = dict(base, gf_hz_cube=9000.0)
    assert not vision_ok(gf)


def test_vision_ok_rejects_energy_starved_scramble_and_clipped_shift():
    vis = {
        "cube_dn_mean": 95.4,
        "black_dn_mean": 0.0,
        "scramble_dn_mean": 0.0,
        "l2_cube_black": 4876.0,
        "l2_cube_scramble": 4876.0,
        "l2_cube_shift": 149.0,
        "cmd_l1_cube_shift": 0.002,
        "cos_cube_black": 0.0,
        "cos_cube_scramble": 0.0,
        "gf_hz_cube": 1400.0,
        "cube_is_abort": False,
        "cube_r1": 11.9,
        "scramble_r1": 3.57,
        "cube_command": {"dx_mm": 2.59, "dy_mm": 3.999, "dz_mm": -3.999, "dgrip_mm": 3.96},
        "cube_command_saturated": True,
        "silence_command_zero": True,
        "dy_left": 3.999,
        "dy_right": 3.999,
        "laterality_dy_flips": False,
        "split_dy_flips": False,
    }
    assert not vision_ok(vis)
    starved = _vision_ok_base(scramble_dn_mean=0.0, scramble_r1=1.0, cube_r1=11.9)
    assert vision_ok(starved)
    rail = _vision_ok_base(
        cmd_l1_cube_shift=2.0,
        cube_command_saturated=True,
        cube_command={"dx_mm": 3.9, "dy_mm": 3.9, "dz_mm": -3.9, "dgrip_mm": 3.9},
    )
    assert not vision_ok(rail)
    tiny_shift = _vision_ok_base(cmd_l1_cube_shift=0.002, cube_command_saturated=False)
    assert not vision_ok(tiny_shift)


def test_vision_ok_does_not_require_dna02_laterality():
    """DNa02 is two cells with no eyemap; laterality is logged, not a hop gate."""
    zero = _vision_ok_base(dy_left=0.0, dy_right=0.0, laterality_dy_flips=False, split_dy_flips=False)
    assert vision_ok(zero)
    assert passed(zero)
    missing = _vision_ok_base()
    missing.pop("dy_left")
    missing.pop("dy_right")
    missing["laterality_dy_flips"] = False
    assert vision_ok(missing)
    assert vision_ok(_vision_ok_base())


def test_vision_ok_rejects_gf_near_abort_without_raising_abort():
    assert U0.abort_hz == 2000.0
    near = _vision_ok_base(gf_hz_cube=1400.0)
    assert not vision_ok(near)
    ok = _vision_ok_base(gf_hz_cube=200.0)
    assert vision_ok(ok)


def test_unpack_laterality_flips_dy():
    def rates(a02: float) -> BusRates:
        vec = np.array([10.0, 10.0, 0.0, a02, 50.0, 10.0, 0.0, 10.0], dtype=np.float64)
        return BusRates(
            vec=vec,
            pools={"DNp01": 50.0},
            t1_mn_hz=0.0,
            abort=False,
            scored_mean_hz=10.0,
            scored=np.array([10.0]),
        )

    left = unpack(rates(-80.0), U0)
    right = unpack(rates(80.0), U0)
    assert left.dy_mm < 0.0
    assert right.dy_mm > 0.0
    sat = unpack(rates(200.0), U0)
    assert abs(sat.dy_mm) < 0.95 * U0.dy_scale
    assert not command_saturated(sat)


def test_hold_pad_dy_stops_laterality_drift():
    """DNa02 is R-biased; envelope keeps pads on a 20 mm cube at y=0."""
    tol = pad_y_tol_mm(20.0)
    assert 1.0 <= tol <= 2.0
    pad = {"x": 245.0, "y": 0.0, "z": 48.0}
    inside = hold_pad_dy(ArmCommand(1.5, 0.5, -4.0, -8.0), pad, cube_y=0.0, size_mm=20.0)
    assert inside.dy_mm == 0.0
    assert inside.dx_mm == 1.5
    assert inside.dz_mm == -4.0
    assert inside.dgrip_mm == -8.0
    rail = hold_pad_dy(ArmCommand(1.5, 2.0, -4.0, -8.0), {"x": 245.0, "y": tol, "z": 48.0}, cube_y=0.0, size_mm=20.0)
    assert rail.dy_mm < 0.0
    drifted = hold_pad_dy(
        ArmCommand(1.55, 1.87, -4.23, -7.89),
        {"x": 274.3, "y": 45.9, "z": 44.0},
        cube_y=0.0,
        size_mm=20.0,
    )
    assert drifted.dy_mm < 0.0
    assert drifted.dx_mm == 1.55
    toward = hold_pad_dy(ArmCommand(0.0, -8.0, 0.0, 0.0), pad, cube_y=10.0, size_mm=20.0)
    assert toward.dy_mm > 0.0


def test_dn_bus_live_applies_hold_pad_dy():
    src = (Path(__file__).resolve().parents[1] / "arm" / "run_dn_bus.py").read_text()
    assert "hold_pad_dy" in src
    assert "hold_pad_dx" in src
    assert "optical_close_gate" in src
    assert "jaw_z_floor_mm" in src
    assert "tip_grasp_z_mm" in src
    assert "pad_aim_x_mm" in src
    assert "rebot_move_to_pose" in src
    assert "rebot_move_to_position" not in src
    assert '"fingers_down"' in src
    assert "pitch_tips" not in src
    assert '"yaw_deg": 0' in src
    assert '"keep_level": True' in src
    assert "abs(dz)" not in src
    assert "attached=contact" in src
    assert "ticks=log" in src
    assert "dz = abs" not in src


def test_hold_pad_dx_keeps_hanging_jaws_on_pad():
    pad = {"x": 245.0, "y": 0.0, "z": 32.0}
    inside = hold_pad_dx(ArmCommand(1.5, 0.5, -4.0, -8.0), pad, pad_x=245.0)
    assert inside.dx_mm == 1.5
    assert inside.dy_mm == 0.5
    rail = hold_pad_dx(
        ArmCommand(1.5, 0.5, -4.0, -8.0),
        {"x": 245.0 + NEAR_PAD_MM, "y": 0.0, "z": 32.0},
        pad_x=245.0,
        cube_x=280.0,
        size_mm=40.0,
    )
    assert rail.dx_mm == 0.0
    reach = hold_pad_dx(
        ArmCommand(8.0, 0.0, 0.0, -8.0),
        {"x": 263.0, "y": 0.0, "z": 48.0},
        pad_x=245.0,
        cube_x=280.0,
        size_mm=20.0,
    )
    assert reach.dx_mm > 0.0
    tip_rail = hold_pad_dx(
        ArmCommand(8.0, 0.0, 0.0, -8.0),
        {"x": 282.0, "y": 0.0, "z": 20.0},
        pad_x=280.0,
        cube_x=280.0,
        size_mm=20.0,
    )
    assert tip_rail.dx_mm <= 0.0
    off_cube = hold_pad_dx(
        ArmCommand(1.55, 0.0, -4.0, -8.0),
        {"x": 298.0, "y": 0.0, "z": 20.0},
        pad_x=280.0,
        cube_x=280.0,
        size_mm=20.0,
    )
    assert off_cube.dx_mm < 0.0
    drifted = hold_pad_dx(
        ArmCommand(1.55, 1.87, -4.23, -7.89),
        {"x": 274.3, "y": 0.0, "z": 32.0},
        pad_x=245.0,
    )
    assert drifted.dx_mm < 0.0
    assert drifted.dy_mm == 1.87
    assert drifted.dz_mm == -4.23


def test_jaw_z_floor_lets_20mm_enter_in_jaws():
    assert tip_grasp_z_mm(20.0) == 8.0
    assert tip_grasp_z_mm(40.0) == 48.0
    assert pad_aim_x_mm(280.0, 20.0) == 280.0
    assert pad_aim_x_mm(280.0, 40.0) == 245.0
    assert jaw_z_floor_mm(20.0, attached=False) == 8.0
    assert jaw_z_floor_mm(40.0, attached=False) == 44.0
    assert jaw_z_floor_mm(20.0, attached=True) == 8.0
    tip = {"x": 280.0, "y": 0.0, "z": 8.0}
    assert in_jaw_box(tip, (280.0, 0.0), 20.0, cube_z_mm=9.0, pad_x=280.0)
    pad = {"x": 245.0, "y": 0.0, "z": 48.0}
    assert in_jaw_box(pad, (280.0, 0.0), 20.0, cube_z_mm=9.0, pad_x=245.0)
    low = {"x": 245.0, "y": 0.0, "z": 32.0}
    assert in_jaw_box(low, (280.0, 0.0), 20.0, cube_z_mm=9.0, pad_x=245.0)
    drifted = {"x": 245.0, "y": 46.0, "z": 32.0}
    assert not in_jaw_box(drifted, (280.0, 0.0), 20.0, cube_z_mm=9.0, pad_x=245.0)
    raw = ArmCommand(1.5, 0.5, -4.0, -8.0)
    gated = optical_close_gate(
        raw,
        np.zeros(1),
        tcp_z=20.0,
        tcp=tip,
        cube_xy=(280.0, 0.0),
        size_mm=20.0,
        cube_z_mm=9.0,
        pad_x=280.0,
    )
    assert gated.dgrip_mm == -8.0
    assert gated.dz_mm == -4.0
    high = optical_close_gate(
        raw,
        np.zeros(1),
        tcp_z=80.0,
        tcp={"x": 280.0, "y": 0.0, "z": 80.0},
        cube_xy=(280.0, 0.0),
        size_mm=20.0,
        cube_z_mm=9.0,
        pad_x=280.0,
    )
    assert high.dgrip_mm == 0.0
    pinch = optical_close_gate(
        raw,
        np.zeros(1),
        tcp_z=32.0,
        tcp=low,
        cube_xy=(280.0, 0.0),
        size_mm=20.0,
        cube_z_mm=9.0,
        pad_x=245.0,
    )
    assert pinch.dgrip_mm == -8.0


def test_stub_laterality_flips_dy(tmp_path):
    from arm.hop_probe import _vision_run

    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, synaptic_gain=2.5, nsteps=150)
    # Stub hexes sit on these bar edges; 150-step LIF needs that extra DoG.
    left = _synthetic_cube(y0=0, x0=0, bh=120, bw=51)
    right = _synthetic_cube(y0=0, x0=110, bh=120, bw=50)
    dy_l = float(_vision_run(lif, left, left, 150)["cmd"]["dy_mm"])
    dy_r = float(_vision_run(lif, right, right, 150)["cmd"]["dy_mm"])
    assert dy_l * dy_r < 0.0
    assert abs(dy_l - dy_r) >= DY_FLIP_MIN
    src = (Path(__file__).resolve().parents[1] / "arm" / "hop_probe.py").read_text()
    assert "run(cube2, cube)" not in src
    assert "_vision_run(lif, shift, shift, nsteps)" in src


def test_drive_camera_orange_beats_gray_table():
    orange = np.zeros((32, 32, 3), dtype=np.float32)
    orange[8:24, 8:24, 0] = 0.85
    orange[8:24, 8:24, 1] = 0.35
    orange[8:24, 8:24, 2] = 0.08
    gray = np.full((32, 32, 3), 0.22, dtype=np.float32)
    hex1 = np.array([12, 18], dtype=np.int32)
    hex2 = np.array([14, 16], dtype=np.int32)
    o = drive_camera(orange, hex1, hex2)
    g = drive_camera(gray, hex1, hex2)
    assert float(np.max(o)) > 0.5
    assert float(np.max(g)) < 0.2
    assert float(orange_chroma(gray).max()) < 0.01


def test_cube_chroma_frac_flags_orange_not_table():
    orange = np.zeros((8, 8, 3), dtype=np.float32)
    orange[..., 0] = 0.85
    orange[..., 1] = 0.35
    orange[..., 2] = 0.08
    gray = np.full((8, 8, 3), 0.22, dtype=np.float32)
    assert cube_chroma_frac(orange) > 0.9
    assert cube_chroma_frac(gray) < 0.01


def test_load_optical_prefers_gripper_cube_frame(tmp_path):
    import json as json_lib

    from PIL import Image

    ep = tmp_path / "ep01-x280-y0"
    frames = ep / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (32, 24), (30, 30, 30)).save(frames / "0000-front.jpg")
    Image.new("RGB", (32, 24), (30, 30, 30)).save(frames / "0000-gripper.jpg")
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0048-front.jpg")
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0048-gripper.jpg")
    recs = []
    for tick, name in ((0, "0000"), (48, "0048")):
        recs.append(
            json_lib.dumps(
                {
                    "tick": tick,
                    "cube": {"center_mm": {"x": 280, "y": 0, "z": 19}, "present": True},
                    "photos": {
                        "Front": str(frames / f"{name}-front.jpg"),
                        "Gripper": str(frames / f"{name}-gripper.jpg"),
                    },
                }
            )
        )
    (ep / "log.jsonl").write_text("\n".join(recs) + "\n")
    pairs = load_optical_fg_pairs(tmp_path, per_episode=1)
    assert len(pairs) == 1
    assert pairs[0]["grip"].name == "0048-gripper.jpg"
    assert pairs[0]["grip_chroma"] > 0.5


def test_load_optical_pairs_keeps_front_gripper_split(tmp_path):
    import json as json_lib

    from PIL import Image

    ep = tmp_path / "ep01-x280-y0"
    frames = ep / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0000-front.jpg")
    Image.new("RGB", (32, 24), (10, 12, 14)).save(frames / "0000-gripper.jpg")
    rec = {
        "tick": 0,
        "cube": {"center_mm": {"x": 280, "y": 0, "z": 19}, "present": True},
        "photos": {"Front": str(frames / "0000-front.jpg"), "Gripper": str(frames / "0000-gripper.jpg")},
    }
    (ep / "log.jsonl").write_text(json_lib.dumps(rec) + "\n")
    pairs = load_optical_fg_pairs(tmp_path)
    assert len(pairs) == 1
    assert pairs[0]["front"].resolve() != pairs[0]["grip"].resolve()
    assert pairs[0]["front"].name.endswith("front.jpg")
    assert pairs[0]["grip"].name.endswith("gripper.jpg")


def test_live_replay_stub_uses_split_cameras(tmp_path):
    import json as json_lib

    from PIL import Image

    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=2.5, nsteps=100)
    for name, y, front_rgb, grip_rgb in (
        ("ep01-x280-y0", 0, (200, 80, 20), (30, 30, 30)),
        ("ep02-x270-y10", 10, (180, 70, 15), (40, 20, 10)),
        ("ep03-x290-y-10", -10, (160, 60, 10), (20, 40, 50)),
    ):
        ep = tmp_path / name
        frames = ep / "frames"
        frames.mkdir(parents=True)
        Image.new("RGB", (32, 24), front_rgb).save(frames / "0000-front.jpg")
        Image.new("RGB", (32, 24), grip_rgb).save(frames / "0000-gripper.jpg")
        rec = {
            "tick": 0,
            "cube": {"center_mm": {"x": 280, "y": y, "z": 19}, "present": True},
            "photos": {"Front": str(frames / "0000-front.jpg"), "Gripper": str(frames / "0000-gripper.jpg")},
        }
        (ep / "log.jsonl").write_text(json_lib.dumps(rec) + "\n")
    out = live_replay_probe(lif, tmp_path, nsteps=100)
    assert out["split_cameras"] is True
    assert Path(out["front_path"]).name.endswith("front.jpg")
    assert Path(out["grip_path"]).name.endswith("gripper.jpg")
    assert out["front_path"] != out["grip_path"]
    assert out.get("empty") is None
    assert "empty-table" in (out.get("empty_table_note") or "")


def test_unpack_abort_identity():

    abort_vec = np.array([0.0, 0.0, 0.0, 0.0, 5000.0, 0.0, 0.0, 0.0])
    rates = BusRates(
        vec=abort_vec,
        pools={"DNp01": 5000.0},
        t1_mn_hz=0.0,
        abort=True,
        scored_mean_hz=10.0,
        scored=np.array([10.0]),
    )
    cmd = unpack(rates, U0)
    assert command_is_abort(cmd)
    silent = unpack(rates.silenced(), U0)
    assert not command_is_abort(silent)


def test_score_py_is_only_fly_picked_true_assignment():
    root = Path(__file__).resolve().parents[1]
    hits = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts or path.name.endswith(".pyc"):
            continue
        text = path.read_text()
        if "fly_picked = True" in text or "fly_picked=True" in text:
            hits.append(str(path))
    assert hits == [], hits


def test_overlay_knn_aliases_cannot_set_fly_picked():
    for amap in ("overlay", "knn", "kernel", "linear", "readout", "quintic", "teacher_distill"):
        scored = score_episode(
            acting_map=amap,
            kinematic_success=True,
            teacher_in_path=False,
            mean_dn_hz=40.0,
            black_dn_hz=0.0,
            g_trained=True,
        )
        assert scored.fly_picked is False, amap


def test_hop_probe_laterality_uses_translated_cubes_not_jpeg_x():
    hop = (Path(__file__).resolve().parents[1] / "arm" / "hop_probe.py").read_text()
    bus = (Path(__file__).resolve().parents[1] / "arm" / "bus.py").read_text()
    unpack_src = (Path(__file__).resolve().parents[1] / "arm" / "unpack.py").read_text()
    assert "left = _synthetic_cube(y0=40, x0=0, bh=50, bw=60)" in hop
    assert "right = _synthetic_cube(y0=40, x0=100, bh=50, bw=60)" in hop
    assert "bh=120, bw=51" not in hop
    assert 'dy_left = float(left_r["cmd"]["dy_mm"])' in hop
    assert 'dy_right = float(right_r["cmd"]["dy_mm"])' in hop
    assert 'a02 = pools["DNa02_R"] - pools["DNa02_L"]' in bus
    assert "jpeg" not in unpack_src.lower()
    assert "x0" not in unpack_src


def test_train_gate_requires_live_go():
    assert train_gate({"ok": True, "live_ok": True}) is True
    assert train_gate({"ok": True, "live_ok": False}) is False
    assert train_gate({"ok": False, "live_ok": True}) is False
    assert train_gate({"ok": True}) is False
    assert train_gate({}) is False
    assert train_gate(None) is False


def test_distill_and_operant_closed_until_live_go(tmp_path, monkeypatch):
    import json as json_lib

    from malecns_cache import paths as cache_paths
    from arm.train_distill import distill
    from arm.train_operant import operant

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    hop = tmp_path / "checkpoints" / "rebot-pickup" / "hop-probe.json"
    hop.parent.mkdir(parents=True)
    hop.write_text(json_lib.dumps({"ok": True, "live_ok": False, "synaptic_gain": 2.5}) + "\n")
    d = distill(stub=False)
    o = operant(stub=False, live_rewards=[{"R": 1.0}])
    assert d["skipped"] is True and d["fly_picked"] is False
    assert o["skipped"] is True and o.get("da_learned") is False


def test_load_optical_keeps_empty_table_when_chroma_ranks_cube(tmp_path):
    import json as json_lib

    from PIL import Image

    ep = tmp_path / "ep01-x280-y0"
    frames = ep / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (32, 24), (30, 30, 30)).save(frames / "0000-front.jpg")
    Image.new("RGB", (32, 24), (30, 30, 30)).save(frames / "0000-gripper.jpg")
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0048-front.jpg")
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0048-gripper.jpg")
    recs = [
        json_lib.dumps(
            {
                "tick": 0,
                "present": False,
                "cube": {"center_mm": {"x": 280, "y": 0, "z": 19}, "present": False},
                "photos": {
                    "Front": str(frames / "0000-front.jpg"),
                    "Gripper": str(frames / "0000-gripper.jpg"),
                },
            }
        ),
        json_lib.dumps(
            {
                "tick": 48,
                "cube": {"center_mm": {"x": 280, "y": 0, "z": 19}, "present": True},
                "photos": {
                    "Front": str(frames / "0048-front.jpg"),
                    "Gripper": str(frames / "0048-gripper.jpg"),
                },
            }
        ),
    ]
    (ep / "log.jsonl").write_text("\n".join(recs) + "\n")
    empty = tmp_path / "empty-table"
    eframes = empty / "frames"
    eframes.mkdir(parents=True)
    Image.new("RGB", (32, 24), (40, 40, 40)).save(eframes / "0000-front.jpg")
    Image.new("RGB", (32, 24), (40, 40, 40)).save(eframes / "0000-gripper.jpg")
    (empty / "log.jsonl").write_text(
        json_lib.dumps(
            {
                "tick": 0,
                "present": False,
                "cube": {"center_mm": {"x": 280, "y": 0, "z": 19}, "present": False},
                "photos": {
                    "Front": str(eframes / "0000-front.jpg"),
                    "Gripper": str(eframes / "0000-gripper.jpg"),
                },
            }
        )
        + "\n"
    )
    pairs = load_optical_fg_pairs(tmp_path, per_episode=1)
    empties = [p for p in pairs if p.get("present") is False]
    assert any(p["grip"].name == "0048-gripper.jpg" for p in pairs)
    assert any(p["episode"] == "empty-table" and p.get("present") is False for p in empties)
    assert any(p["episode"] == "ep01-x280-y0" and p.get("present") is False for p in empties)


def test_live_replay_uses_empty_table_present_false(tmp_path):
    import json as json_lib

    from PIL import Image

    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=2.5, nsteps=100)
    for name, y, present, front_rgb, grip_rgb in (
        ("ep01-x280-y0", 0, True, (200, 80, 20), (30, 30, 30)),
        ("ep02-x270-y10", 10, True, (180, 70, 15), (40, 20, 10)),
        ("ep03-x290-y-10", -10, True, (160, 60, 10), (20, 40, 50)),
        ("empty-table", 0, False, (40, 40, 40), (45, 45, 48)),
    ):
        ep = tmp_path / name
        frames = ep / "frames"
        frames.mkdir(parents=True)
        Image.new("RGB", (32, 24), front_rgb).save(frames / "0000-front.jpg")
        Image.new("RGB", (32, 24), grip_rgb).save(frames / "0000-gripper.jpg")
        rec = {
            "tick": 0,
            "present": present,
            "cube": {"center_mm": {"x": 280, "y": y, "z": 19}, "present": present},
            "photos": {"Front": str(frames / "0000-front.jpg"), "Gripper": str(frames / "0000-gripper.jpg")},
        }
        (ep / "log.jsonl").write_text(json_lib.dumps(rec) + "\n")
    out = live_replay_probe(lif, tmp_path, nsteps=100)
    assert out.get("empty") is not None
    assert out.get("empty_table_note") is None
    assert out["cube_episode"] != "empty-table"


def test_no_overlay_overrides_teacher(tmp_path, monkeypatch):
    import json as json_lib

    from malecns_cache import paths as cache_paths
    from arm import run_dn_bus as mod
    from arm.score import ActingMap, score_episode

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    seen: dict = {}

    def fake_live(args):
        seen["teacher"] = bool(args.teacher)
        scored = score_episode(
            acting_map=ActingMap.overlay if args.teacher else ActingMap.dn_bus,
            kinematic_success=True,
            teacher_in_path=bool(args.teacher),
            mean_dn_hz=20.0,
            black_dn_hz=0.0,
            g_trained=False,
        )
        return (
            {
                "fly_picked": scored.fly_picked,
                "lab_picked": scored.lab_picked,
                "acting_map": scored.acting_map.value,
            },
            [],
        )

    monkeypatch.setattr(mod, "run_live", fake_live)
    hop = tmp_path / "checkpoints" / "rebot-pickup" / "hop-probe.json"
    hop.parent.mkdir(parents=True, exist_ok=True)
    hop.write_text(json_lib.dumps({"ok": True, "live_ok": True, "synaptic_gain": 2.5}) + "\n")
    rc = mod.main(["--no-overlay", "--teacher", "--ticks", "1", "--steps", "100", "--tag", "no-ov"])
    assert rc == 0
    assert seen["teacher"] is False
    saved = json_lib.loads((tmp_path / "checkpoints" / "rebot-pickup" / "no-ov.json").read_text())
    assert saved["fly_picked"] is False
    assert saved["acting_map"] == "dn_bus"


def test_agc_flood_gains_are_refused(tmp_path):
    from arm.run_dn_bus import _load_g

    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, nsteps=100)
    path = tmp_path / "g-distill.agc-flood.npz"
    g = lif.g.copy()
    g[0] = 3.0
    np.savez(path, g=g)
    assert _load_g(lif, path) == "refused-agc-flood"
    assert abs(float(lif.g[0]) - 1.0) < 1e-6


def test_excitatory_only_does_not_set_hop_ok(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm import hop_probe as hp

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    signed = _vision_ok_base(dy_left=3.0, dy_right=0.0, laterality_dy_flips=False, gf_hz_cube=1733.0)
    exo = _vision_ok_base()
    assert not vision_ok(signed)
    assert vision_ok(exo)

    def fake_gain_pass(crop, nsteps, gains=None):
        vis = dict(signed)
        vis["synaptic_gain"] = 2.5
        vis["luma_cut"] = False
        return 2.5, vis

    def fake_exo(crop, nsteps, gain, luma_scale=None):
        out = dict(exo)
        out["synaptic_gain"] = float(gain)
        out["excitatory_only"] = True
        out["luma_cut"] = False
        return out

    monkeypatch.setattr(hp, "gain_pass", fake_gain_pass)
    monkeypatch.setattr(hp, "excitatory_only", fake_exo)
    monkeypatch.setattr(
        hp,
        "hop_table",
        lambda lif, drive=4.0, nsteps=150: [
            {"layer": "scored_dn", "n": 1, "self_hz": 10.0, "scored_dn_hz": 10.0, "t1_mn_hz": 0.0}
        ],
    )
    report = hop_run(stub=True, nsteps=100, dest=tmp_path / "hop.json")
    assert vision_ok(report["excitatory_only"])
    assert not vision_ok(report["vision"])
    assert report["ok"] is False
    assert report["vision"].get("excitatory_only") is not True


def test_pick_pair_no_laterality_y_fallback():
    pairs = [
        {"cube_y": 0.0, "grip_chroma": 0.9, "front": "c", "grip": "c"},
        {"cube_y": 10.0, "grip_chroma": 0.4, "front": "r", "grip": "r"},
    ]
    assert _pick_pair(pairs, y_sign=-1) is None
    right = _pick_pair(pairs, y_sign=1)
    assert right is not None and float(right["cube_y"]) == 10.0
    center = _pick_pair(pairs, y_sign=0)
    assert center is not None and float(center["cube_y"]) == 0.0


def test_distill_skips_without_teacher_frames(tmp_path, monkeypatch):
    import json as json_lib

    from malecns_cache import paths as cache_paths
    from arm.train_distill import distill

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    hop = tmp_path / "checkpoints" / "rebot-pickup" / "hop-probe.json"
    hop.parent.mkdir(parents=True)
    hop.write_text(json_lib.dumps({"ok": True, "live_ok": True, "synaptic_gain": 2.5}) + "\n")
    out = distill(stub=False)
    assert out["skipped"] is True
    assert out["fly_picked"] is False
    assert "teacher" in str(out.get("reason") or "").lower()


def test_live_dn_bus_refuses_when_hop_red(tmp_path, monkeypatch):
    import json as json_lib

    from malecns_cache import paths as cache_paths
    from arm import run_dn_bus as mod

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    hop = tmp_path / "checkpoints" / "rebot-pickup" / "hop-probe.json"
    hop.parent.mkdir(parents=True)
    hop.write_text(json_lib.dumps({"ok": False, "live_ok": False, "synaptic_gain": 2.5}) + "\n")

    def boom(_args):
        raise AssertionError("run_live must not servo when hop is red")

    monkeypatch.setattr(mod, "run_live", boom)
    rc = mod.main(["--no-overlay", "--ticks", "1", "--steps", "100", "--tag", "refused"])
    assert rc == 2
    saved = json_lib.loads((tmp_path / "checkpoints" / "rebot-pickup" / "refused.json").read_text())
    assert saved["skipped"] is True
    assert saved["fly_picked"] is False
    assert saved["acting_map"] == "dn_bus"


def test_grip_target_floors_at_cube_width():
    assert abs(grip_target_mm(90.0, -8.0, 20.0) - 82.0) < 1e-6
    assert abs(grip_target_mm(24.0, -8.0, 20.0) - 20.0) < 1e-6
    assert abs(grip_target_mm(20.0, -8.0, 20.0) - 20.0) < 1e-6
    assert abs(grip_target_mm(20.0, 8.0, 20.0) - 28.0) < 1e-6
    g = 90.0
    for _ in range(20):
        g = grip_target_mm(g, -8.0, 20.0)
    assert abs(g - 20.0) < 1e-6


def test_overlay_close_at_live_pad_is_size_plus_2():
    cmd = teacher_target(
        {"x": 245.0, "y": 0.0, "z": 48.0},
        90.0,
        False,
        0.0,
        cube_xy=(280.0, 0.0),
        cube_size_mm=20.0,
        pad_lock=(245.0, 0.0, 48.0),
    )
    assert cmd.acting_map is ActingMap.overlay
    assert cmd.phase == "close"
    assert cmd.command.dgrip_mm < -0.5
    assert abs(cmd.command.dgrip_mm - (-8.0)) < 1e-6
    scored = score_episode(
        acting_map=cmd.acting_map,
        kinematic_success=True,
        teacher_in_path=True,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        g_trained=True,
    )
    assert scored.fly_picked is False
    assert scored.lab_picked is True


def test_abort_only_motion_is_not_fly():
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        abort_only=True,
    )
    assert scored.fly_picked is False
    mixed = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        abort_only=False,
    )
    assert mixed.fly_picked is True


def test_dnp01_type_gain_scales_incoming(tmp_path):
    crop = write_stub_crop(tmp_path)
    lif = CropLIF(crop, nsteps=100)
    gf = int(GAIN_CLASSES.index("DNp01"))
    onto = np.where(lif._post_class == gf)[0]
    assert onto.size > 0
    w0 = float(lif.crop.w0[onto[0]])
    g = lif.g.copy()
    g[gf] = 0.5
    lif.set_g(g)
    assert abs(float(lif.crop.weights.data[onto[0]]) - 0.5 * w0) < 1e-5


def test_distill_stub_not_fly_and_abort_hz_frozen(tmp_path, monkeypatch):
    import json as json_lib

    from PIL import Image

    from malecns_cache import paths as cache_paths
    from arm.train_distill import distill

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    ep = tmp_path / "datasets" / "rebot-optical-fg" / "ep20-pad-z48"
    frames = ep / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0000-front.jpg")
    Image.new("RGB", (32, 24), (200, 80, 20)).save(frames / "0000-gripper.jpg")
    rec = {
        "tick": 0,
        "step": "pad",
        "tcp_mm": {"x": 245.0, "y": 0.0, "z": 48.0},
        "gripper_mm": 90.0,
        "cube": {"attached": False, "center_mm": {"x": 280, "y": 0, "z": 9}, "present": True, "size_mm": 20},
        "photos": {"Front": str(frames / "0000-front.jpg"), "Gripper": str(frames / "0000-gripper.jpg")},
    }
    (ep / "log.jsonl").write_text(json_lib.dumps(rec) + "\n")
    out = distill(stub=True, n_gen=1, pop=2, frames=1, seed=0)
    assert out["skipped"] is False
    assert out["fly_picked"] is False
    assert out["u_n_params"] == 12
    assert abs(float(out["u_abort_hz"]) - 2000.0) < 1e-9
    assert out.get("freeze_g_required") is True
    assert out["clip_stats"]["sequential_abort_ticks"] == out["sequential_abort_ticks"]
    assert U0.abort_hz == 2000.0
    assert U0.n_params() == 12
    assert abs(float(U0.w_contact)) < 1e-9


def test_distill_frames_prefers_size20_pad(tmp_path, monkeypatch):
    import json as json_lib

    from PIL import Image

    from malecns_cache import paths as cache_paths
    from arm.train_distill import distill_frames

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)

    def write_ep(name, size, z, step, n_lines=3):
        ep = tmp_path / "datasets" / "rebot-optical-fg" / name
        frames = ep / "frames"
        frames.mkdir(parents=True)
        Image.new("RGB", (16, 12), (200, 80, 20)).save(frames / "0000-front.jpg")
        Image.new("RGB", (16, 12), (200, 80, 20)).save(frames / "0000-gripper.jpg")
        lines = []
        for i in range(n_lines):
            lines.append(
                json_lib.dumps(
                    {
                        "tick": i,
                        "step": step,
                        "tcp_mm": {"x": 245.0, "y": 0.0, "z": z},
                        "gripper_mm": 90.0,
                        "cube": {
                            "attached": False,
                            "center_mm": {"x": 280, "y": 0, "z": 9},
                            "present": True,
                            "size_mm": size,
                        },
                        "photos": {
                            "Front": str(frames / "0000-front.jpg"),
                            "Gripper": str(frames / "0000-gripper.jpg"),
                        },
                    }
                )
            )
        (ep / "log.jsonl").write_text("\n".join(lines) + "\n")

    write_ep("ep01-x280-y0", 40, 400.0, "approach", n_lines=8)
    write_ep("ep20-pad-z48", 20, 48.0, "ep20-pad-z48", n_lines=1)
    ep = tmp_path / "datasets" / "rebot-optical-fg" / "ep-lift"
    frames = ep / "frames"
    frames.mkdir(parents=True)
    Image.new("RGB", (16, 12), (200, 80, 20)).save(frames / "0000-front.jpg")
    Image.new("RGB", (16, 12), (200, 80, 20)).save(frames / "0000-gripper.jpg")
    (ep / "log.jsonl").write_text(
        json_lib.dumps(
            {
                "tick": 0,
                "step": "lift",
                "tcp_mm": {"x": 280.0, "y": 0.0, "z": 56.0},
                "gripper_mm": 42.0,
                "cube": {
                    "attached": True,
                    "center_mm": {"x": 280, "y": 0, "z": 26},
                    "present": True,
                    "size_mm": 40,
                },
                "photos": {
                    "Front": str(frames / "0000-front.jpg"),
                    "Gripper": str(frames / "0000-gripper.jpg"),
                },
            }
        )
        + "\n"
    )
    rows = distill_frames(limit=3)
    assert any("pad" in str(r.get("step") or "") for r in rows)
    assert any(bool((r.get("cube") or {}).get("attached")) for r in rows)
    sizes = [float((r.get("cube") or {}).get("size_mm") or 0) for r in rows]
    assert any(abs(s - 20.0) < 0.5 for s in sizes)


def _rates(*, mdn=400.0, p07=50.0, p10=80.0, scored=400.0, xl=200.0) -> BusRates:
    vec = np.array([80.0, xl, 40.0, 0.0, 10.0, mdn, p07, p10], dtype=np.float64)
    return BusRates(
        vec=vec,
        pools={"DNp01": 10.0, "MDN": mdn, "DNp07": p07, "DNp10": p10},
        t1_mn_hz=0.0,
        abort=False,
        scored_mean_hz=float(scored),
        scored=np.array([scored], dtype=np.float64),
    )


def _tick(tick, *, attached, cube_z, gripper, dz, tcp_z, acting="dn_bus", dx=0.1, dy=0.0, dgrip=-4.0):
    return {
        "tick": tick,
        "acting_map": acting,
        "attached": attached,
        "cube_z_mm": cube_z,
        "gripper_mm": gripper,
        "tcp": {"x": 280.0, "y": 0.0, "z": tcp_z},
        "command": {"dx_mm": dx, "dy_mm": dy, "dz_mm": dz, "dgrip_mm": dgrip},
        "dn_hz": 40.0,
        "abort": False,
    }


def test_unpack_contact_term_lifts_from_scored_rates_not_sign_flip():
    u = UParams()
    u.w_contact = 2.5
    rates = _rates()
    down = unpack(rates, u, attached=False)
    up = unpack(rates, u, attached=True)
    silent = unpack(rates.silenced(), u, attached=True)
    zero_c = unpack(rates, U0, attached=True)
    flooded = _rates(mdn=9000.0, scored=1200.0)
    flood_up = unpack(flooded, u, attached=True)
    flood_down = unpack(flooded, u, attached=False)
    assert down.dz_mm < 0.0
    assert up.dz_mm > 0.5
    assert abs(silent.dz_mm) < 1e-9
    assert abs(zero_c.dz_mm) < 1e-6
    assert flood_down.dz_mm < 0.0
    assert flood_up.dz_mm > 0.5
    assert abs(up.dz_mm - abs(down.dz_mm)) > 0.05


def test_fly_picked_rejects_negative_dz_rewrite_lift():
    ticks = []
    z = 9.0
    tcp = 8.0
    for i in range(20):
        attached = i >= 2
        if attached:
            z = min(120.0, 9.0 + (i - 1) * 8.0)
            tcp = min(140.0, 8.0 + (i - 1) * 8.0)
        ticks.append(_tick(i, attached=attached, cube_z=z, gripper=21.0 if attached else 90.0, dz=-4.2, tcp_z=tcp))
    assert lift_without_plus_z(ticks) is True
    assert tcp_rose_while_dz_negative(ticks) is True
    assert bus_commanded_plus_z(ticks) is False
    assert trace_blocks_fly(ticks) is True
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ticks=ticks,
    )
    assert scored.fly_picked is False
    assert scored.teacher_in_path is True


def test_fly_picked_rejects_tick0_open_gripper_attach():
    ticks = [
        _tick(0, attached=True, cube_z=9.0, gripper=90.0, dz=-3.6, tcp_z=20.0),
        _tick(1, attached=True, cube_z=40.0, gripper=22.0, dz=4.0, tcp_z=50.0),
        _tick(2, attached=True, cube_z=120.0, gripper=22.0, dz=4.0, tcp_z=130.0),
    ]
    assert open_gripper_tick0_attach(ticks) is True
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ticks=ticks,
    )
    assert scored.fly_picked is False


def test_fly_picked_rejects_overlay_plus8_even_if_map_says_dn_bus():
    ticks = [
        _tick(i, attached=True, cube_z=30.0 + 8.0 * i, gripper=22.0, dz=8.0, tcp_z=50.0 + 8.0 * i, dx=0.0, dgrip=0.0)
        for i in range(8)
    ]
    ticks[0]["cube_z_mm"] = 9.0
    assert overlay_plus8_lift(ticks) is True
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ticks=ticks,
    )
    assert scored.fly_picked is False


def test_fly_picked_accepts_bus_plus_z_after_pinch():
    ticks = []
    z = 9.0
    tcp = 8.0
    for i in range(20):
        attached = i >= 3
        if i < 3:
            ticks.append(_tick(i, attached=False, cube_z=9.0, gripper=90.0 - 20.0 * i, dz=-4.0, tcp_z=8.0))
            continue
        z = min(120.0, 9.0 + (i - 2) * 8.0)
        tcp = min(140.0, 8.0 + (i - 2) * 8.0)
        ticks.append(_tick(i, attached=True, cube_z=z, gripper=21.0, dz=4.0, tcp_z=tcp))
    assert open_gripper_tick0_attach(ticks) is False
    assert lift_without_plus_z(ticks) is False
    assert overlay_plus8_lift(ticks) is False
    assert tcp_rose_while_dz_negative(ticks) is False
    assert bus_commanded_plus_z(ticks) is True
    assert trace_blocks_fly(ticks) is False
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ticks=ticks,
    )
    assert scored.fly_picked is True
    assert scored.lab_picked is False
    knn = score_episode(
        acting_map=ActingMap.knn,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ticks=[{**t, "acting_map": "knn"} for t in ticks],
    )
    assert knn.fly_picked is False


def test_fly_picked_gate_uses_ticks_not_source_comment():
    """Removing trace_blocks_fly from fly_picked must fail this (ticks are required)."""
    cheat = [_tick(i, attached=True, cube_z=9.0 + 8.0 * i, gripper=21.0, dz=-4.2, tcp_z=8.0 + 8.0 * i) for i in range(16)]
    ok = fly_picked(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        g_trained=True,
        dn_l2=40.0,
        ticks=cheat,
    )
    assert ok is False
    honest = [_tick(i, attached=i >= 1, cube_z=min(120.0, 9.0 + 8.0 * max(i - 1, 0)), gripper=21.0, dz=4.0 if i else -4.0, tcp_z=min(140.0, 8.0 + 8.0 * max(i - 1, 0))) for i in range(16)]
    honest[0]["attached"] = False
    honest[0]["gripper_mm"] = 90.0
    assert fly_picked(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        g_trained=True,
        dn_l2=40.0,
        ticks=honest,
    ) is True


def test_load_u_pads_short_vector_contact_zero(tmp_path):
    from arm.run_dn_bus import load_u

    path = tmp_path / "g.npz"
    np.savez(path, u=np.ones(11, dtype=np.float64))
    u = load_u(path)
    assert u.n_params() == 12
    assert abs(float(u.w_contact)) < 1e-9
