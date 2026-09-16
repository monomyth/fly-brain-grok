from pathlib import Path

import numpy as np

from arm.bus import BusRates, dead_pools, pick_unpack_report, pool_rates
from arm.crop import GAIN_CLASSES, write_dna01_inhibit_crop, write_stub_crop
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
from arm.lif_crop import CropLIF, pad_type_gains
from arm.log import g_hash
from arm.score import (
    ActingMap,
    bus_commanded_plus_z,
    da_learned,
    fly_picked,
    lift_without_plus_z,
    live_mbon_gate_used,
    open_gripper_tick0_attach,
    overlay_plus8_lift,
    score_episode,
    tcp_rose_while_dz_negative,
    trace_blocks_fly,
)
from arm.teacher import teacher_target
from arm.unpack import U0, UParams, command_is_abort, command_saturated, grip_target_mm, unpack
from rebot_adapter.pick import (
    FINGERS_DOWN_Z_MM,
    NEAR_PAD_MM,
    READY_Z_MM,
    episode_outcome,
    hold_pad_dx,
    hold_pad_dy,
    in_jaw_box,
    jaw_z_floor_mm,
    optical_close_gate,
    pad_aim_x_mm,
    pad_y_tol_mm,
    pick_success,
    scripted_start_waypoints,
    servo_orientation,
    shape_pad_command,
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
    overlay = score_episode(
        acting_map=ActingMap.overlay,
        kinematic_success=True,
        teacher_in_path=True,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        g_trained=True,
        ablation_changed=True,
    )
    knn = score_episode(
        acting_map=ActingMap.knn,
        kinematic_success=True,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        g_trained=True,
        ablation_changed=True,
    )
    assert ok.da_learned is True
    assert no.da_learned is False
    assert overlay.da_learned is False
    assert knn.da_learned is False
    assert overlay.fly_picked is False
    assert knn.fly_picked is False
    assert da_learned(ablation_changed=True) is False
    assert da_learned(ablation_changed=None, acting_map=ActingMap.dn_bus) is False
    assert da_learned(ablation_changed=True, acting_map=ActingMap.dn_bus) is True
    assert da_learned(ablation_changed=True, acting_map=ActingMap.leg_mn) is False
    assert da_learned(ablation_changed=True, acting_map=ActingMap.idle) is False
    assert da_learned(ablation_changed=True, acting_map=ActingMap.dn_bus, teacher_in_path=True) is False


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


def test_unpack_does_not_invent_dx_from_silent_a01():
    mdn = 800.0
    vec = np.array([0.0, 120.0, 0.0, 90.0, 900.0, mdn, 0.0, 0.0], dtype=np.float64)
    pools = {
        "DNfl": 0.0,
        "DNxl": 120.0,
        "DNa01": 0.0,
        "DNa02_L": 0.0,
        "DNa02_R": 90.0,
        "DNp01": 900.0,
        "MDN": mdn,
        "DNp07": 0.0,
        "DNp10": 0.0,
    }
    rates = BusRates(
        vec=vec,
        pools=pools,
        t1_mn_hz=0.0,
        abort=False,
        scored_mean_hz=200.0,
        scored=np.array([200.0]),
    )
    cmd = unpack(rates, U0)
    dead = dead_pools(pools)
    u_big = UParams()
    u_big.w_a01 = 50.0
    cmd_big = unpack(rates, u_big)
    assert "DNa01" in dead
    assert "DNp07" in dead
    assert "DNp10" in dead
    assert abs(cmd.dx_mm) < 1e-9
    assert abs(cmd_big.dx_mm) < 1e-9
    assert abs(cmd.dz_mm) > 0.5
    live_a01 = BusRates(
        vec=np.array([0.0, 120.0, 400.0, 90.0, 900.0, mdn, 0.0, 0.0], dtype=np.float64),
        pools={**pools, "DNa01": 400.0},
        t1_mn_hz=0.0,
        abort=False,
        scored_mean_hz=200.0,
        scored=np.array([200.0]),
    )
    assert abs(unpack(live_a01, U0).dx_mm) > 0.5


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
    out = write_skip_checkpoint(
        json_path=js, npz=npz, payload={"reason": "hop_probe not green", "da_learned": True}
    )
    assert out["ok"] is False
    assert out["skipped"] is True
    assert out["fly_picked"] is False
    assert out["da_learned"] is False
    assert not npz.is_file()
    assert (tmp_path / "g-distill.agc-flood.npz").is_file()
    saved = json_lib.loads(js.read_text())
    assert saved["ok"] is False and saved["skipped"] is True
    assert saved["fly_picked"] is False
    assert saved["da_learned"] is False


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


def test_shape_pad_command_holds_pad_and_keeps_signed_dz():
    cmd = ArmCommand(1.5, 8.0, -4.0, -8.0)
    high = {"x": 280.0, "y": 0.0, "z": 80.0}
    out = shape_pad_command(
        cmd,
        high,
        cube_xy=(280.0, 0.0),
        size_mm=20.0,
        cube_z_mm=9.0,
        pad_x=280.0,
    )
    assert out.dy_mm == 0.0
    assert out.dx_mm == 0.0
    assert out.dz_mm == -4.0
    assert out.dgrip_mm == 0.0
    low = {"x": 280.0, "y": 12.0, "z": 20.0}
    pinched = shape_pad_command(
        ArmCommand(1.5, 8.0, -4.0, -8.0),
        low,
        cube_xy=(280.0, 0.0),
        size_mm=20.0,
        cube_z_mm=9.0,
        pad_x=280.0,
    )
    assert pinched.dy_mm < 0.0
    assert pinched.dgrip_mm == -8.0


def test_scripted_start_20mm_is_farther_than_pad():
    wps = scripted_start_waypoints(280.0, 0.0, 20.0)
    assert len(wps) == 1
    assert wps[0]["z_mm"] == READY_Z_MM
    assert wps[0]["keep_level"] is True
    assert "fingers_down" not in wps[0]
    assert wps[0]["z_mm"] > tip_grasp_z_mm(20.0)
    big = scripted_start_waypoints(280.0, 0.0, 40.0)
    assert len(big) == 2
    assert big[0]["z_mm"] == READY_Z_MM
    assert big[1]["z_mm"] == tip_grasp_z_mm(40.0)
    assert servo_orientation(size_mm=20.0, tcp_z=READY_Z_MM) == {"keep_level": True}
    assert servo_orientation(size_mm=20.0, tcp_z=FINGERS_DOWN_Z_MM - 1.0) == {"fingers_down": True}
    assert servo_orientation(size_mm=40.0, tcp_z=20.0) == {"keep_level": True}


def test_command_path_gate_publishes_then_gates(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm import run_dn_bus as mod
    from arm.crop import as_connectome, write_stub_crop
    from arm.hop_probe import _synthetic_cube
    from arm.lif_crop import CropLIF, window_hz
    from runtime.plasticity import KCToMBON

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=8.0, nsteps=50)
    memory = KCToMBON(as_connectome(crop), lif.brain)
    assert memory.slots.size > 0
    order: list[str] = []
    real_step = lif.step_vision
    real_pub = mod.publish_crop
    real_dg = mod.drive_and_gate

    def step(*a, **k):
        order.append("vision")
        assert k.get("drive_kc") is False
        return real_step(*a, **k)

    def pub(*a, **k):
        order.append("publish")
        return real_pub(*a, **k)

    def dg(*a, **k):
        order.append("gate")
        return real_dg(*a, **k)

    lif.step_vision = step  # type: ignore[method-assign]
    monkeypatch.setattr(mod, "publish_crop", pub)
    monkeypatch.setattr(mod, "drive_and_gate", dg)
    front = _synthetic_cube()
    lif.reset_episode()
    baseline = float(lif.rates().scored_mean_hz)
    rates, gate = mod.command_path_gate(lif, memory, front, front)
    assert order == ["vision", "publish", "gate"]
    assert float(rates.scored_mean_hz) > baseline
    assert abs(gate - 1.0) >= 1e-3
    silent = float(np.mean(window_hz(lif.brain)[memory.kc]))
    assert silent < 1.0


def test_dn_bus_live_shapes_pad_and_gates_on_fake(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from malecns_cache import paths as cache_paths
    from arm import run_dn_bus as mod
    from rebot_adapter.teacher import cube
    from test_mcp_fake import FakeRobot

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    robot = FakeRobot()
    robot._state["objects"]["cube"]["size_mm"] = 20
    robot._state["objects"]["cube"]["center_mm"] = {"x": 280.0, "y": 0.0, "z": 9.0}
    shaped = {"n": 0}
    real_shape = mod.shape_pad_command

    def wrap_shape(*a, **k):
        shaped["n"] += 1
        return real_shape(*a, **k)

    monkeypatch.setattr(mod, "shape_pad_command", wrap_shape)
    args = SimpleNamespace(
        mcp=tmp_path / "missing-mcp",
        ticks=1,
        steps=100,
        x=280.0,
        y=0.0,
        size=20.0,
        hold=2.0,
        teacher=False,
        silence_dn=False,
        g_init=False,
        gains=None,
        stub=True,
        episodes=1,
    )
    summary, log = mod.run_live(args, client=robot)
    names = [name for name, _ in robot.calls]
    assert "rebot_move_to_position" not in names
    assert "rebot_move_to_pose" in names
    pose = next(args for name, args in robot.calls if name == "rebot_move_to_pose")
    assert pose.get("z_mm") == READY_Z_MM
    assert pose.get("keep_level") is True
    servo = next(args for name, args in robot.calls if name == "rebot_servo_tcp")
    assert servo.get("keep_level") is True
    assert "fingers_down" not in servo
    assert "pitch_tips" not in servo
    assert log
    assert log[0]["acting_map"] == "dn_bus"
    assert shaped["n"] >= 1
    assert log[0]["command"]["dy_mm"] == 0.0
    assert abs(float(log[0]["mbon_gate"]) - 1.0) >= 1e-3
    assert live_mbon_gate_used(log) is True
    assert isinstance(summary["da_learned"], bool)
    c = cube(robot.state())
    assert c["attached"] is False
    assert abs(float(c["center_mm"]["x"]) - 280.0) < 1e-9
    assert float(c["center_mm"]["z"]) < 20.0


def test_main_episodes_calls_run_live_n_times(tmp_path, monkeypatch):
    import json as json_lib

    from malecns_cache import paths as cache_paths
    from arm import run_dn_bus as mod

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    n = {"i": 0}

    def fake_live(args, client=None):
        n["i"] += 1
        return (
            {
                "acting_map": "dn_bus",
                "fly_picked": False,
                "da_learned": False,
                "skipped": False,
            },
            [],
        )

    monkeypatch.setattr(mod, "run_live", fake_live)
    rc = mod.main(["--stub", "--ticks", "1", "--steps", "100", "--episodes", "2", "--tag", "ep2"])
    assert rc == 0
    assert n["i"] == 2
    saved = json_lib.loads((tmp_path / "checkpoints" / "rebot-pickup" / "ep2.json").read_text())
    assert saved["episodes"] == 2
    assert saved["da_learned"] is False


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


def test_operant_da_learned_requires_pulse_move(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import GAIN_CLASSES
    from arm.hop_probe import _synthetic_cube
    from arm.train_operant import operant

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    front = _synthetic_cube()
    g = np.ones(len(GAIN_CLASSES), dtype=np.float32)
    rest = operant(
        stub=True,
        live_rewards=[{"g": g, "front_rgb": front, "grip_rgb": front, "R": 0.0}],
    )
    assert rest["skipped"] is False
    assert rest["da_learned"] is False
    assert rest["fly_picked"] is False
    pulsed = operant(
        stub=True,
        live_rewards=[{"g": g, "front_rgb": front, "grip_rgb": front, "R": 1.0}],
    )
    assert pulsed["skipped"] is False
    assert pulsed["fly_picked"] is False
    assert pulsed["da_learned"] is True
    assert abs(float(pulsed["mbon_gate_before"]) - float(pulsed["mbon_gate_after_ablate"])) >= 0.02
    assert live_mbon_gate_used([{"mbon_gate": pulsed["mbon_gate_before"]}]) is True


def test_operant_da_learned_denies_unity_live_gate(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import GAIN_CLASSES
    from arm.hop_probe import _synthetic_cube
    from arm import train_operant as to

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    monkeypatch.setattr(to, "drive_and_gate", lambda *a, **k: 1.0)
    monkeypatch.setattr(to, "terminal_da_changed", lambda *a, **k: True)
    front = _synthetic_cube()
    g = np.ones(len(GAIN_CLASSES), dtype=np.float32)
    out = to.operant(
        stub=True,
        live_rewards=[{"g": g, "front_rgb": front, "grip_rgb": front, "R": 1.0}],
    )
    assert out["skipped"] is False
    assert out["da_learned"] is False
    assert out["fly_picked"] is False


def test_terminal_da_changed_skips_teacher_and_zero_reward(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import as_connectome, write_stub_crop
    from arm.hop_probe import _synthetic_cube
    from arm.lif_crop import CropLIF
    from arm.train_operant import terminal_da_changed
    from arm.unpack import U0, drive_and_gate
    from runtime.plasticity import KCToMBON

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=8.0, nsteps=50)
    memory = KCToMBON(as_connectome(crop), lif.brain)
    assert memory.slots.size > 0
    lif.brain.weights.data[memory.slots] = 80.0
    front = _synthetic_cube()
    lif.step_vision(front, front, drive_kc=False)
    drive_and_gate(lif.brain, memory, front)
    assert terminal_da_changed(lif, memory, front, front, U0, False, teacher=True, reward=1.0) is False
    assert terminal_da_changed(lif, memory, front, front, U0, False, teacher=False, reward=0.0) is False
    assert terminal_da_changed(lif, memory, front, front, U0, False, teacher=False, reward=1.0) is True


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
    assert saved["da_learned"] is False
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
        ablation_changed=True,
        ticks=ticks,
    )
    assert scored.fly_picked is False
    assert scored.teacher_in_path is True
    assert scored.da_learned is False


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
        ablation_changed=True,
        ticks=ticks,
    )
    assert scored.fly_picked is False
    assert scored.da_learned is False


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
        ablation_changed=True,
        ticks=ticks,
    )
    assert scored.fly_picked is False
    assert scored.da_learned is False


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


def test_scene_cube_xy_keeps_dragged_cube():
    from rebot_adapter.pick import scene_cube_xy

    x, y, already = scene_cube_xy(None, 280.0, 0.0)
    assert (x, y, already) == (280.0, 0.0, False)
    state = {"objects": {"cube": {"present": True, "center_mm": {"x": 350.0, "y": 12.0, "z": 19.0}}}}
    x, y, already = scene_cube_xy(state, 280.0, 0.0)
    assert already is True
    assert abs(x - 350.0) < 1e-9 and abs(y - 12.0) < 1e-9
    missing = {"objects": {"cube": {"present": False, "center_mm": {"x": 1.0, "y": 1.0, "z": 1.0}}}}
    x, y, already = scene_cube_xy(missing, 280.0, 0.0)
    assert already is False and abs(x - 280.0) < 1e-9
    wreck = {"objects": {"cube": {"present": True, "attached": False, "center_mm": {"x": 200.0, "y": -54.0, "z": 15.0}, "size_mm": 20}}}
    x, y, already = scene_cube_xy(wreck, 280.0, 0.0)
    assert already is False
    air = {"objects": {"cube": {"present": True, "attached": True, "center_mm": {"x": 280.0, "y": 0.0, "z": 120.0}, "size_mm": 20}}}
    assert scene_cube_xy(air, 280.0, 0.0)[2] is False


def test_publish_crop_scatters_to_parent_index(tmp_path, monkeypatch):
    import struct

    from malecns_cache import paths as cache_paths
    from malecns_cache.somas import ACTIVITY_MAGIC
    from arm.crop import write_stub_crop
    from arm.lif_crop import CropLIF
    from runtime.broadcast import publish_crop

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, nsteps=50)
    lif.inject(np.array([0], dtype=np.int32), 3.0, nsteps=50)
    publish_crop(lif)
    blob = (tmp_path / "live" / "activity.bin").read_bytes()
    assert blob[:4] == ACTIVITY_MAGIC
    n = struct.unpack_from("<I", blob, 4)[0]
    rates = np.frombuffer(blob, dtype=np.float32, offset=16, count=n)
    assert float(rates[crop.parent_index].max()) > 0.0


def test_kc_mbon_gate_ablation_drops_to_one():
    from arm.unpack import kc_mbon_gate

    hz = np.array([0.0, 80.0], dtype=np.float64)
    w = np.array([0.0, 10.0], dtype=np.float64)
    slots = np.array([1], dtype=np.int32)
    pre = np.array([1], dtype=np.int32)
    on = kc_mbon_gate(hz, w, slots, pre)
    off = kc_mbon_gate(hz, np.zeros_like(w), slots, pre)
    assert on > 1.0
    assert abs(off - 1.0) < 1e-6
    assert abs(kc_mbon_gate(hz, w, slots, np.array([99], dtype=np.int32)) - 1.0) < 1e-6
    assert abs(kc_mbon_gate(hz, w, np.array([99], dtype=np.int32), pre) - 1.0) < 1e-6
    assert abs(kc_mbon_gate(hz, w, slots, np.array([1, 1], dtype=np.int32)) - 1.0) < 1e-6


def test_mbon_gate_scales_unpack_command():
    from arm.bus import BusRates
    from arm.unpack import mbon_gate, unpack

    idx = np.array([0], dtype=np.int32)
    silent = mbon_gate(np.array([0.0], dtype=np.float32), idx)
    loud = mbon_gate(np.array([80.0], dtype=np.float32), idx)
    assert abs(silent - 1.0) < 1e-6
    assert loud > silent
    vec = np.array([400.0, 400.0, 400.0, 400.0, 0.0, 400.0, 400.0, 400.0], dtype=np.float32)
    rates = BusRates(
        vec=vec,
        pools={},
        t1_mn_hz=0.0,
        abort=False,
        scored_mean_hz=400.0,
        scored=vec,
    )
    a = unpack(rates, gate=1.0)
    b = unpack(rates, gate=loud)
    assert abs(b.dx_mm) > abs(a.dx_mm)


def test_mbon_ablation_changed_on_stub_crop(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import write_stub_crop
    from arm.lif_crop import CropLIF
    from arm.unpack import mbon_ablation_changed
    from runtime.plasticity import KCToMBON
    from arm.crop import as_connectome

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=8.0, nsteps=50)
    memory = KCToMBON(as_connectome(crop), lif.brain)
    if memory.slots.size:
        lif.brain.weights.data[memory.slots] = 80.0
    from arm.hop_probe import _synthetic_cube

    front = _synthetic_cube()
    grip = front.copy()
    changed = mbon_ablation_changed(lif, front, grip, attached=False, memory=memory)
    assert changed is True
    from arm.lif_crop import window_hz
    from arm.unpack import drive_and_gate, kc_mbon_gate

    drive_and_gate(lif.brain, memory, front, accumulate=False)
    drive = np.array(lif.brain.i_ext, copy=True)
    g0 = kc_mbon_gate(drive, lif.brain.weights.data, memory.slots, memory.pre)
    saved = np.array(lif.brain.weights.data[memory.slots], copy=True)
    lif.brain.weights.data[memory.slots] = 0
    g1 = kc_mbon_gate(drive, lif.brain.weights.data, memory.slots, memory.pre)
    lif.brain.weights.data[memory.slots] = saved
    assert abs(g0 - g1) >= 0.02
    lif.reset_episode()
    lif.step_vision(front, grip, drive_kc=False)
    silent = kc_mbon_gate(window_hz(lif.brain), lif.brain.weights.data, memory.slots, memory.pre)
    assert abs(silent - 1.0) < 1e-3


def test_mbon_ablation_unchanged_when_weights_zero(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import as_connectome, write_stub_crop
    from arm.hop_probe import _synthetic_cube
    from arm.lif_crop import CropLIF
    from arm.unpack import mbon_ablation_changed
    from runtime.plasticity import KCToMBON

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=8.0, nsteps=50)
    memory = KCToMBON(as_connectome(crop), lif.brain)
    if memory.slots.size:
        lif.brain.weights.data[memory.slots] = 0
    front = _synthetic_cube()
    assert mbon_ablation_changed(lif, front, front.copy(), attached=False, memory=memory) is False


def test_drive_and_gate_uses_kenyon_iext_not_silent_kc_hz(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import as_connectome, write_stub_crop
    from arm.hop_probe import _synthetic_cube
    from arm.lif_crop import CropLIF, window_hz
    from arm.unpack import drive_and_gate, kc_mbon_gate
    from runtime.plasticity import KCToMBON

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=8.0, nsteps=50)
    memory = KCToMBON(as_connectome(crop), lif.brain)
    assert memory.slots.size > 0
    lif.brain.weights.data[memory.slots] = 80.0
    front = _synthetic_cube()
    lif.reset_episode()
    lif.step_vision(front, front, drive_kc=False)
    silent = kc_mbon_gate(window_hz(lif.brain), lif.brain.weights.data, memory.slots, memory.pre)
    gated = drive_and_gate(lif.brain, memory, front, accumulate=False)
    assert abs(silent - 1.0) < 1e-3
    assert gated > 1.02


def test_drive_and_gate_cube_not_unity_and_pulse_moves(tmp_path, monkeypatch):
    from malecns_cache import paths as cache_paths
    from arm.crop import as_connectome, write_stub_crop
    from arm.hop_probe import _synthetic_cube
    from arm.lif_crop import CropLIF, window_hz
    from arm.train_operant import pulse_da, terminal_da_changed
    from arm.unpack import U0, drive_and_gate, mbon_ablation_changed, slots_moved
    from runtime.plasticity import KCToMBON

    monkeypatch.setenv("FLYBRAIN_DATA", str(tmp_path))
    monkeypatch.setattr(cache_paths, "project_data", lambda: tmp_path)
    crop = write_stub_crop(tmp_path / "crop")
    lif = CropLIF(crop, synaptic_gain=8.0, nsteps=50)
    memory = KCToMBON(as_connectome(crop), lif.brain)
    assert memory.slots.size > 0
    front = _synthetic_cube()
    black = np.zeros_like(front)
    lif.reset_episode()
    g_black = drive_and_gate(lif.brain, memory, black, accumulate=False)
    g_cube = drive_and_gate(lif.brain, memory, front, accumulate=True)
    assert abs(g_black - 1.0) < 1e-3
    assert abs(g_cube - 1.0) >= 1e-3
    before = np.array(lif.brain.weights.data[memory.slots], copy=True)
    nsyn = pulse_da(memory, 1.0)
    after = np.array(lif.brain.weights.data[memory.slots], copy=True)
    assert nsyn == int(memory.slots.size)
    assert slots_moved(before, after)
    assert mbon_ablation_changed(lif, front, front.copy(), attached=False, memory=memory) is True
    assert terminal_da_changed(lif, memory, front, front, U0, False, teacher=False, reward=1.0) is True
    lif.reset_episode()
    lif.step_vision(front, front, drive_kc=False)
    silent = float(np.mean(window_hz(lif.brain)[memory.kc]))
    assert silent < 1.0


def test_da_learned_denies_unity_live_gate_even_if_ablation_painted():
    ticks = []
    for i in range(16):
        row = _tick(
            i,
            attached=i >= 1,
            cube_z=min(120.0, 9.0 + 8.0 * max(i - 1, 0)),
            gripper=90.0 if i == 0 else 21.0,
            dz=-4.0 if i == 0 else 4.0,
            tcp_z=min(140.0, 8.0 + 8.0 * max(i - 1, 0)),
        )
        if i == 0:
            row["attached"] = False
        row["mbon_gate"] = 1.0
        ticks.append(row)
    unused = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ablation_changed=True,
        ticks=ticks,
    )
    assert live_mbon_gate_used(ticks) is False
    assert unused.fly_picked is True
    assert unused.da_learned is False
    for row in ticks:
        row["mbon_gate"] = 1.12
    used = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=True,
        teacher_in_path=False,
        mean_dn_hz=40.0,
        black_dn_hz=0.0,
        dn_l2=40.0,
        g_trained=True,
        ablation_changed=True,
        ticks=ticks,
    )
    assert live_mbon_gate_used(ticks) is True
    assert used.fly_picked is True
    assert used.da_learned is True


def test_episode_summary_extra_cannot_paint_flags():
    from arm.log import episode_summary

    score = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=False,
        mean_dn_hz=20.0,
        black_dn_hz=0.0,
        g_trained=True,
        ablation_changed=False,
    )
    out = episode_summary(
        score=score,
        ticks=[],
        g_hash_s="a",
        g_hash_init="b",
        extra={
            "da_learned": True,
            "fly_picked": True,
            "lab_picked": True,
            "acting_map": "overlay",
            "hold_s": 2.0,
        },
    )
    assert out["da_learned"] is False
    assert out["fly_picked"] is False
    assert out["lab_picked"] is False
    assert out["acting_map"] == "dn_bus"
    assert out["hold_s"] == 2.0


def test_historical_loop_da2_rescore_denies_unused_da():
    import json as json_lib

    import pytest

    path = Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "rebot-pickup" / "loop-da2.json"
    if not path.is_file():
        pytest.skip("loop-da2.json not in tree")
    raw = path.read_bytes()
    blob = json_lib.loads(raw)
    ticks = []
    for key in ("tick_log", "tick_log_head", "tick_log_tail"):
        ticks.extend(blob.get(key) or [])
    assert ticks
    assert live_mbon_gate_used(ticks) is False
    scored = score_episode(
        acting_map=blob.get("acting_map") or "dn_bus",
        kinematic_success=True,
        mean_dn_hz=float(blob.get("mean_dn_hz") or 0.0),
        black_dn_hz=float(blob.get("black_dn_hz") or 0.0),
        dn_l2=blob.get("dn_l2"),
        g_trained=True,
        ablation_changed=True,
        ticks=ticks,
    )
    assert scored.da_learned is False
    assert path.read_bytes() == raw


def test_pad_type_gains_unmask_dna01_without_gf_abort(tmp_path):
    crop = write_dna01_inhibit_crop(tmp_path)
    r1 = crop.indices("photoreceptors_r1r6")
    lif0 = CropLIF(crop, synaptic_gain=2.1, nsteps=170, luma_scale=4.0, chroma_scale=0.0)
    lif0.reset_episode()
    hz0 = lif0.inject(r1, 4.0, nsteps=170)
    r0 = pool_rates(hz0, crop.groups)
    assert r0.pools["DNa01"] <= 0.05
    assert r0.pools["DNp07"] <= 0.05
    assert r0.pools["DNp10"] <= 0.05
    assert r0.pools["DNp01"] < 2000.0
    assert "DNa01" in dead_pools(r0.pools)
    cmd0 = unpack(r0, U0)

    lif1 = CropLIF(
        crop,
        synaptic_gain=2.1,
        nsteps=170,
        luma_scale=4.0,
        chroma_scale=0.0,
        type_gains=pad_type_gains(),
    )
    lif1.reset_episode()
    hz1 = lif1.inject(r1, 4.0, nsteps=170)
    r1_rates = pool_rates(hz1, crop.groups)
    dead = dead_pools(r1_rates.pools)
    cmd1 = unpack(r1_rates, U0)
    assert r1_rates.pools["DNa01"] > 1.0
    assert r1_rates.pools["DNp01"] < 2000.0
    assert lif1.g_trained is False
    assert "DNp07" in dead
    assert "DNp10" in dead
    assert "DNa01" not in dead
    assert abs(cmd1.dx_mm) > abs(cmd0.dx_mm)


def test_pick_unpack_report_does_not_fit_lying_u():
    n = 8
    X = np.zeros((n, 8), dtype=np.float64)
    X[:, 0] = np.linspace(5.0, 20.0, n)
    X[:, 1] = np.linspace(40.0, 80.0, n)
    X[:, 4] = np.linspace(700.0, 1100.0, n)
    X[:, 5] = np.linspace(50.0, 400.0, n)
    Y = np.zeros((n, 3), dtype=np.float64)
    Y[:, 0] = np.linspace(-10.0, 30.0, n)
    Y[:, 1] = np.linspace(-20.0, 40.0, n)
    Y[:, 2] = np.linspace(40.0, -80.0, n)
    fly_z = -0.01 * X[:, 5]
    silent = pick_unpack_report(X, Y, fly_z)
    assert silent["can_unpack_pick"] is False
    assert "pred" not in silent
    assert "mse" not in silent

    X2 = X.copy()
    X2[:, 2] = np.linspace(0.0, 200.0, n)
    Y2 = Y.copy()
    Y2[:, 2] = np.linspace(0.0, 40.0, n)
    live = pick_unpack_report(X2, Y2, Y2[:, 2])
    assert live["can_unpack_pick"] is True
    assert "pred" in live


def test_score_py_is_only_da_learned_true_assignment():
    root = Path(__file__).resolve().parents[1]
    needles = (
        "da_learned = True",
        "da_learned=True",
        '"da_learned": True',
        '["da_learned"] = True',
        "['da_learned'] = True",
    )
    hits = []
    for sub in ("arm", "rebot_adapter", "runtime", "scripts"):
        base = root / sub
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text()
            if any(n in text for n in needles):
                hits.append(str(path))
    assert hits == [], hits
