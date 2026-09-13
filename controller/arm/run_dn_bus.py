"""Phase 2–4 live/offline dn_bus. Fail closed. Overlay is teacher, never fly_picked."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from malecns_cache.paths import checkpoints_dir
from rebot_adapter.episode import capture_rgbs, default_mcp_binary
from rebot_adapter.mcp import MCPClient
from rebot_adapter.pick import (
    cleanup_episode,
    jaw_z_floor_mm,
    pad_aim_x_mm,
    pick_success,
    qualifying,
    scene_cube_xy,
    scripted_start_waypoints,
    servo_orientation,
    servo_stalled,
    shape_pad_command,
)
from rebot_adapter.teacher import CAPTURE_CAMERAS, cube

from runtime.broadcast import publish_crop
from runtime.plasticity import KCToMBON

from .bus import l2
from .crop import as_connectome, build_crop, write_stub_crop
from .hop_probe import train_gate
from .lif_crop import CropLIF
from .log import episode_summary, g_hash, tick_row, write_json, write_jsonl
from .score import ActingMap, score_episode, trace_blocks_fly
from .teacher import teacher_target
from .train_operant import terminal_da_changed
from .unpack import (
    U0,
    UParams,
    command_dict,
    command_is_abort,
    drive_and_gate,
    grip_target_mm,
    unpack,
)


def _load_g(lif: CropLIF, path: Path | None) -> str:
    if path is None or not path.is_file():
        return "init"
    if "agc-flood" in path.name:
        return "refused-agc-flood"
    blob = np.load(path)
    if "g" not in blob.files or blob["g"].shape != lif.g.shape:
        return "shape-mismatch"
    lif.set_g(blob["g"])
    return str(path)


def load_u(path: Path | None) -> UParams:
    if path is None or not path.is_file() or "agc-flood" in path.name:
        return UParams()
    blob = np.load(path)
    if "u" not in blob.files:
        return UParams()
    u = UParams.from_vector(np.asarray(blob["u"], dtype=np.float64), base=U0)
    u.abort_hz = float(U0.abort_hz)
    u.step = float(U0.step)
    u.rate_div = float(U0.rate_div)
    return u


def apply_controller_g(lif: CropLIF, gains_path: Path | None, freeze_init: bool) -> str:
    loaded = _load_g(lif, gains_path)
    if freeze_init:
        lif.set_g(lif.g_init)
        return "g_init"
    return loaded


def _make_lif(stub: bool, steps: int, gains_path: Path | None, freeze_init: bool) -> tuple[CropLIF, str, UParams]:
    crop = write_stub_crop(checkpoints_dir("rebot-pickup") / "stub-crop") if stub else build_crop("v1.0")
    hop = checkpoints_dir("rebot-pickup") / ("hop-probe-stub.json" if stub else "hop-probe.json")
    gain = 1.5
    if hop.is_file():
        blob = json.loads(hop.read_text())
        live = blob.get("live_replay") or {}
        gain = float(live.get("synaptic_gain") or blob.get("synaptic_gain") or 1.5)
        if gain < 2.5:
            gain = 2.5
    lif = CropLIF(crop, synaptic_gain=gain, nsteps=steps)
    loaded = apply_controller_g(lif, gains_path, freeze_init)
    u = load_u(gains_path)
    return lif, loaded, u


def run_offline_ticks(lif: CropLIF, n: int = 4, u: UParams | None = None) -> tuple[dict, list[dict]]:
    from .hop_probe import _synthetic_cube
    from .eye import phase_scramble

    u = u or U0
    cube = _synthetic_cube()
    black = np.zeros_like(cube)
    black_r = lif.black_baseline(cube.shape)
    log = []
    lif.reset_episode()
    for tick in range(n):
        rgb = cube if tick % 2 == 0 else phase_scramble(cube, np.random.default_rng(tick))
        lif.step_vision(rgb, rgb, drive_kc=False)
        rates = lif.rates()
        cmd = unpack(rates, u)
        log.append(
            tick_row(
                tick=tick,
                acting_map=ActingMap.dn_bus,
                dn_hz=rates.scored_mean_hz,
                t1_mn_hz=rates.t1_mn_hz,
                g_hash_s=lif.g_hash,
                attached=False,
                cube_z_mm=19.0,
                tcp={"x": 280.0, "y": 0.0, "z": 80.0},
                gripper_mm=90.0,
                command=command_dict(cmd),
                abort=rates.abort,
                extra={"bus": rates.vec.tolist()},
            )
        )
    mean_dn = float(np.mean([r["dn_hz"] for r in log])) if log else 0.0
    scored = score_episode(
        acting_map=ActingMap.dn_bus,
        kinematic_success=False,
        teacher_in_path=False,
        mean_dn_hz=mean_dn,
        black_dn_hz=black_r.scored_mean_hz,
        dn_l2=l2(np.array([mean_dn]), np.array([black_r.scored_mean_hz])),
        g_trained=lif.g_trained,
    )
    summary = episode_summary(
        score=scored,
        ticks=log,
        g_hash_s=lif.g_hash,
        g_hash_init=g_hash(lif.g_init),
        extra={"mode": "offline", "tick_log": log, "g_trained": lif.g_trained},
    )
    return summary, log


def _hop_blob(stub: bool) -> dict:
    path = checkpoints_dir("rebot-pickup") / ("hop-probe-stub.json" if stub else "hop-probe.json")
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def command_path_gate(lif, memory, front, grip, *, accumulate: bool = True):
    lif.step_vision(front, grip, drive_kc=False)
    rates = lif.rates()
    publish_crop(lif)
    gate = drive_and_gate(lif.brain, memory, front, accumulate=accumulate)
    return rates, float(gate)


def skip_live(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    amap = ActingMap.overlay if args.teacher else ActingMap.dn_bus
    scored = score_episode(
        acting_map=amap,
        kinematic_success=False,
        teacher_in_path=bool(args.teacher) or amap is ActingMap.overlay,
        mean_dn_hz=0.0,
        black_dn_hz=0.0,
        g_trained=False,
    )
    summary = episode_summary(
        score=scored,
        ticks=[],
        g_hash_s="init",
        g_hash_init="init",
        extra={
            "skipped": True,
            "interrupted": "hop_probe not green or live_go false; live servo refused",
            "controller": "overlay" if amap is ActingMap.overlay else "dn_bus",
            "picked": False,
            "untrained_dn_bus": False,
        },
    )
    return summary, []


def run_live(args: argparse.Namespace, client=None) -> tuple[dict, list[dict]]:
    own_client = client is None
    if own_client:
        client = MCPClient(args.mcp, launch_lab=False)
    log: list[dict] = []
    interrupted = None
    acting = ActingMap.overlay if args.teacher else ActingMap.dn_bus
    hold_t0 = None
    hold_s = 0.0
    try:
        client.initialize("flybrain-dn-bus")
        client.wait_ready(timeout=20)
        client.call("rebot_set_control_mode", {"mode": "scripted"}, retries=2)
        client.apply_preset("Folded")
        try:
            client.wait_stopped(timeout=30)
        except Exception:
            pass
        client.apply_preset("Ready")
        try:
            client.wait_stopped(timeout=30)
        except Exception:
            pass
        try:
            client.call("rebot_playback", {"action": "stop"}, retries=2)
        except Exception:
            pass
        live0 = client.try_state() or {}
        cube_x, cube_y, already = scene_cube_xy(live0, args.x, args.y)
        cube_size = float(args.size)
        if already:
            try:
                cube_size = float(cube(live0).get("size_mm") or args.size)
            except Exception:
                cube_size = float(args.size)
        else:
            cube_x, cube_y = float(args.x), float(args.y)
            client.call(
                "rebot_set_cube",
                {
                    "present": True,
                    "x_mm": cube_x,
                    "y_mm": cube_y,
                    "size_mm": cube_size,
                    "yaw_deg": 0,
                    "attached": False,
                },
                retries=2,
            )
        client.call("rebot_set_gripper", {"opening_mm": 90}, retries=2)
        try:
            client.wait_stopped(timeout=40)
        except Exception:
            try:
                client.call("rebot_playback", {"action": "stop"}, retries=2)
            except Exception:
                pass
        pad_x = pad_aim_x_mm(cube_x, cube_size)
        for pose in scripted_start_waypoints(cube_x, cube_y, cube_size):
            client.call("rebot_move_to_pose", pose, retries=2)
            try:
                client.wait_stopped(timeout=40)
            except Exception:
                try:
                    client.call("rebot_playback", {"action": "stop"}, retries=2)
                except Exception:
                    pass
        client.call("rebot_set_control_mode", {"mode": "servo"}, retries=2)
        # Crop LIF is CPU-heavy; load after scripted setup so the 5s sim socket stays alive.
        lif, loaded, u = _make_lif(args.stub, args.steps, args.gains, args.g_init)
        memory = KCToMBON(as_connectome(lif.crop), lif.brain)
        last_front = last_grip = None
        last_attached = False
        last_gate = 1.0
        lif.reset_episode()
        black = np.zeros((120, 160, 3), dtype=np.float32)
        black_r = lif.black_baseline(black.shape)
        lif.reset_episode()
        for tick in range(args.ticks):
            state = client.try_state()
            if state is None:
                interrupted = "state timeout"
                break
            tcp = state["tcp_mm"]
            c = cube(state)
            contact = bool(c["attached"])
            attached = contact
            rgbs = capture_rgbs(client, CAPTURE_CAMERAS, width=160, height=120)
            front = rgbs[0] if rgbs else black
            grip_rgb = rgbs[1] if len(rgbs) > 1 else black
            last_front, last_grip, last_attached = front, grip_rgb, attached
            rates, last_gate = command_path_gate(lif, memory, front, grip_rgb)
            if args.teacher:
                yaw = float(np.radians((state.get("tcp_rpy_deg") or {}).get("yaw") or 0.0))
                teach = teacher_target(tcp, float(state["gripper_mm"]), attached, yaw, cube_xy=(cube_x, cube_y), cube_size_mm=cube_size)
                cmd = teach.command
                map_now = ActingMap.overlay
            else:
                cmd = unpack(
                    rates.silenced() if args.silence_dn else rates,
                    u,
                    gate=last_gate,
                    attached=contact,
                )
                cmd = shape_pad_command(
                    cmd,
                    tcp,
                    cube_xy=(cube_x, cube_y),
                    size_mm=cube_size,
                    cube_z_mm=float(c["center_mm"]["z"]),
                    pad_x=pad_x,
                )
                map_now = ActingMap.dn_bus
            z_floor = jaw_z_floor_mm(cube_size, attached=attached)
            dz = float(cmd.dz_mm)
            z_sent = max(z_floor, float(tcp["z"] + dz))
            target = {
                "x_mm": tcp["x"] + cmd.dx_mm,
                "y_mm": tcp["y"] + cmd.dy_mm,
                "z_mm": z_sent,
                **servo_orientation(size_mm=cube_size, tcp_z=float(tcp["z"])),
            }
            try:
                client.call("rebot_servo_tcp", target)
            except (RuntimeError, TimeoutError) as exc:
                interrupted = str(exc)
                break
            client.wait_servo_step(target=target, timeout=3.0)
            after = client.try_state() or state
            if (not attached) and servo_stalled(cmd, tcp, after["tcp_mm"]):
                face = float(cube_x) - 0.5 * float(cube_size)
                at_face = float(after["tcp_mm"]["x"]) >= face - 5.0
                hop_z = -4.0 if at_face else 2.0
                flat = {
                    "x_mm": float(after["tcp_mm"]["x"] + cmd.dx_mm),
                    "y_mm": float(after["tcp_mm"]["y"] + cmd.dy_mm),
                    "z_mm": float(after["tcp_mm"]["z"]) + hop_z,
                    **servo_orientation(size_mm=cube_size, tcp_z=float(after["tcp_mm"]["z"])),
                }
                try:
                    client.call("rebot_servo_tcp", flat)
                    client.wait_servo_step(target=flat, timeout=3.0)
                    after = client.try_state() or after
                except (RuntimeError, TimeoutError):
                    pass
            if abs(cmd.dgrip_mm) > 0.1:
                opening = grip_target_mm(float(state["gripper_mm"]), cmd.dgrip_mm, cube_size)
                try:
                    client.call("rebot_servo_joints", {"gripper_mm": opening})
                except (RuntimeError, TimeoutError):
                    pass
            after = client.try_state() or after
            tcp = after["tcp_mm"]
            c = cube(after)
            attached = bool(c["attached"])
            cube_z = float(c["center_mm"]["z"])
            level = bool(after.get("tcp_level"))
            now = time.monotonic()
            if qualifying(attached=attached, cube_z_mm=cube_z, tcp_level=level):
                if hold_t0 is None:
                    hold_t0 = now
                hold_s = now - hold_t0
            else:
                hold_t0 = None
                hold_s = 0.0
            log.append(
                tick_row(
                    tick=tick,
                    acting_map=map_now,
                    dn_hz=rates.scored_mean_hz,
                    t1_mn_hz=rates.t1_mn_hz,
                    g_hash_s=lif.g_hash,
                    attached=attached,
                    cube_z_mm=cube_z,
                    tcp=tcp,
                    gripper_mm=float(after["gripper_mm"]),
                    command=command_dict(cmd),
                    abort=rates.abort,
                    extra={
                        "tcp_level": level,
                        "hold_s": hold_s,
                        "tcp_rpy_deg": after.get("tcp_rpy_deg"),
                        "contact": contact,
                        "bus": rates.vec.tolist(),
                        "mbon_gate": last_gate,
                    },
                )
            )
            if pick_success(attached=attached, cube_z_mm=cube_z, tcp_level=level, hold_s=hold_s, min_hold_s=args.hold):
                break
            time.sleep(0.05)
        live = client.try_state() or state
        c = cube(live)
        success = pick_success(
            attached=bool(c["attached"]),
            cube_z_mm=float(c["center_mm"]["z"]),
            tcp_level=bool(live.get("tcp_level")),
            hold_s=hold_s,
            min_hold_s=args.hold,
        )
        mean_dn = float(np.mean([r["dn_hz"] for r in log])) if log else 0.0
        teacher_in = acting is ActingMap.overlay or args.teacher
        maps = {str(r.get("acting_map")) for r in log}
        if maps - {ActingMap.dn_bus.value}:
            teacher_in = teacher_in or bool(maps & {ActingMap.overlay.value, ActingMap.knn.value})
        teacher_in = teacher_in or trace_blocks_fly(log)
        abort_only = bool(log) and all(
            bool(r.get("abort")) or command_is_abort(r.get("command") or {}) for r in log
        )
        if abort_only:
            success = False
        ablation_changed = terminal_da_changed(
            lif,
            memory,
            last_front,
            last_grip,
            u,
            last_attached,
            teacher=bool(args.teacher),
            reward=1.0 if success else -1.0,
        )
        scored = score_episode(
            acting_map=acting if not args.teacher else ActingMap.overlay,
            kinematic_success=success,
            teacher_in_path=teacher_in,
            mean_dn_hz=mean_dn,
            black_dn_hz=black_r.scored_mean_hz,
            dn_l2=abs(mean_dn - black_r.scored_mean_hz),
            g_trained=lif.g_trained and not args.g_init,
            ablation_changed=ablation_changed,
            abort_only=abort_only,
            ticks=log,
        )
        extra_log = {"tick_log": log} if len(log) <= 8 else {"tick_log_head": log[:3], "tick_log_tail": log[-3:]}
        summary = episode_summary(
            score=scored,
            ticks=log,
            g_hash_s=lif.g_hash,
            g_hash_init=g_hash(lif.g_init),
            extra={
                "controller": "dn_bus" if acting is ActingMap.dn_bus and not args.teacher else "overlay",
                "picked": bool(success),
                "weights": loaded,
                "interrupted": interrupted,
                "hold_s": hold_s,
                "attached": bool(c["attached"]),
                "cube_z_mm": float(c["center_mm"]["z"]),
                "tcp_level": bool(live.get("tcp_level")),
                "untrained_dn_bus": bool(success) and acting is ActingMap.dn_bus and not lif.g_trained,
                "abort_only": abort_only,
                "u_n_params": u.n_params(),
                "mbon_gate": last_gate,
                "da_ablation_changed": bool(ablation_changed),
                **extra_log,
            },
        )
        return summary, log
    finally:
        try:
            cleanup_episode(client, x_mm=args.x, y_mm=args.y, size_mm=args.size)
        except Exception:
            pass
        if own_client:
            try:
                client.close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp", type=Path, default=default_mcp_binary())
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--x", type=float, default=280.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--size", type=float, default=20.0)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--teacher", action="store_true", help="Overlay commands; lab_picked only.")
    parser.add_argument("--no-overlay", action="store_true", help="Teacher off; acting_map=dn_bus.")
    parser.add_argument("--silence-dn", action="store_true")
    parser.add_argument("--g-init", action="store_true", help="Freeze g at init (ablation).")
    parser.add_argument("--gains", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)
    if args.no_overlay:
        args.teacher = False
    if args.steps < 50:
        raise SystemExit("steps must be ≥ 50")
    tag = args.tag or ("dn-bus-offline" if args.offline else "dn-bus")
    dest = checkpoints_dir("rebot-pickup") / f"{tag}.json"
    n_ep = max(1, int(args.episodes))
    if args.offline:
        lif, loaded, u = _make_lif(args.stub, args.steps, args.gains, args.g_init)
        summary, log = run_offline_ticks(lif, n=min(args.ticks, 6), u=u)
        summary["weights"] = loaded
    elif not args.stub and not train_gate(_hop_blob(False)):
        summary, log = skip_live(args)
    else:
        summary, log = {}, []
        for _ in range(n_ep):
            summary, log = run_live(args)
        if n_ep > 1:
            summary = dict(summary)
            summary["episodes"] = n_ep
    write_json(dest, summary)
    trace = dest.with_name(f"{tag}-trace.jsonl")
    write_jsonl(trace, log)
    summary["trace"] = str(trace)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 2 if summary.get("skipped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
