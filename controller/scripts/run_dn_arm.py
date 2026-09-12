#!/usr/bin/env python3
"""Vision → R1–R6 → MaleCNS LIF → engineered DN joystick on the arm.

DN→TCP assignments are engineered (same honesty as doomfly), not proven
motor identity. Optional KC→MBON checkpoint from Learn can ride along
because those synapses sit in the same CSR.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from malecns_cache.graph import load_graph
from malecns_cache.paths import checkpoints_dir
from malecns_cache.somas import prepare_somas
from rebot_adapter.episode import capture_rgbs, default_mcp_binary
from rebot_adapter.mcp import MCPClient
from rebot_adapter.pick import (
    clip_spawn_dy,
    dist_xy,
    episode_outcome,
    optical_close_gate,
    acting_map_changed,
    clipped_plus8_hold_policy,
    overlay_attached_lift_policy,
    pick_success,
    qualifying,
    readout_xyz_diverged,
    score_pick,
    scripted_pad_pick,
    servo_stalled,
    spawn_pad_overlay,
    yaw_rad,
)
from rebot_adapter.teacher import CAPTURE_CAMERAS, cube
from runtime.broadcast import publish_activity
from runtime.decoder import ArmCommand, decode_hop_drive
from runtime.encoder import encode_frame
from runtime.lif import LIFNetwork
from train.dataset import policy_cameras
from train.features import FEATURE_NAMES, drive_kenyon_from_image, features_from_rgbs
from train.readout import apply_kernel, apply_linear, load_kernel_tables, load_readout
from train.shuffle import shuffle_edges


def load_weights(brain: LIFNetwork, path: Path) -> str:
    if not path.is_file():
        return "none"
    blob = np.load(path)
    data = blob["data"]
    if data.shape != brain.weights.data.shape:
        return f"shape-mismatch {data.shape}"
    brain.weights.data[:] = np.asarray(data, dtype=np.float32)
    return str(path)


def _write(dest: Path, summary: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp", type=Path, default=default_mcp_binary())
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--hops", type=int, default=24)
    parser.add_argument("--scale", type=float, default=12.0, help="Hop-contrast joystick step in mm.")
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--frozen", action="store_true", help="Do not load the Learn checkpoint.")
    parser.add_argument("--weights", type=Path, default=checkpoints_dir("rebot-pickup") / "kc-mbon-learn.npz")
    parser.add_argument("--readout", action="store_true", help="Phase-2 linear readout instead of DN hop joystick.")
    parser.add_argument("--readout-path", type=Path, default=checkpoints_dir("rebot-pickup") / "phase2-linear.npz")
    parser.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quintic-handoff", action="store_true", help="Lab teacher grasp. Not a fly pick.")
    parser.add_argument("--shuffle-edges", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--x", type=float, default=280.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--size", type=float, default=40.0)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Write Front+Gripper JPEGs and jsonl (teacher format) each tick.",
    )
    args = parser.parse_args()
    if args.readout:
        args.hops = min(args.hops, 3)
        if args.ticks == 60:
            args.ticks = 220
    if args.overlay and args.ticks == 60:
        args.ticks = 160

    use_brain = bool(args.readout) or not args.overlay
    connectome = None
    brain = None
    r1 = r8 = kc = np.zeros(0, dtype=np.int32)
    loaded = "none"
    if use_brain:
        prepare_somas("v1.0")
        connectome = load_graph("v1.0")
        if args.shuffle_edges:
            connectome = shuffle_edges(connectome, seed=args.shuffle_seed)
        brain = LIFNetwork(connectome.weights)
        loaded = "frozen"
        if not args.frozen:
            loaded = load_weights(brain, args.weights)
        r1 = connectome.indices("photoreceptors_r1r6")
        r8 = connectome.indices("photoreceptors_r8")
        kc = connectome.indices("KC")
    readout_w = None
    readout_meta: dict = {}
    kernel_tables = None
    if args.readout:
        readout_w, readout_meta = load_readout(args.readout_path)
        kernel_tables = load_kernel_tables(args.readout_path)
        print(
            f"readout {args.readout_path} {readout_w.shape} "
            f"kernel={kernel_tables[0].shape[0] if kernel_tables else 0}",
            flush=True,
        )
    live_cameras = policy_cameras(readout_meta) if readout_w is not None else CAPTURE_CAMERAS
    print(f"cameras {list(live_cameras)}", flush=True)
    n_cells = connectome.n if connectome is not None else 0
    print(
        f"dn-arm n={n_cells} weights={loaded} blind={args.blind} "
        f"readout={args.readout} overlay={args.overlay} quintic={args.quintic_handoff} shuffle={args.shuffle_edges}",
        flush=True,
    )

    tag = args.tag or (
        "eval-readout-blind"
        if args.readout and args.blind
        else "eval-readout"
        if args.readout
        else "dn-arm-blind"
        if args.blind
        else "dn-arm-shuffled"
        if args.shuffle_edges
        else "dn-arm"
    )
    dest = checkpoints_dir("rebot-pickup") / f"{tag}.json"
    flags0 = episode_outcome(
        overlay=bool(args.overlay),
        readout=bool(args.readout),
        quintic_ran=False,
        attached=False,
        cube_z_mm=0.0,
        tcp_level=False,
        hold_s=0.0,
    )
    summary: dict = {
        "ok": False,
        "picked": False,
        "blind": bool(args.blind),
        "readout": bool(flags0["readout"]),
        "overlay": bool(flags0["overlay"]),
        "quintic_handoff": bool(args.quintic_handoff),
        "controller": flags0["controller"],
        "readout_picked": False,
        "fly_picked": False,
        "xyz_diverged": False,
        "overlay_lift_policy": False,
        "map_changed": False,
        "plus8_hold_policy": False,
        "z_floor_mm": 32.0,
        "z_floor_bound": False,
        "clip_prior": "none",
        "shuffled_edges": bool(args.shuffle_edges),
        "weights": loaded,
        "ticks": 0,
        "attached": False,
        "peak_cube_z_mm": 0.0,
        "tcp_level": False,
        "hold_s": 0.0,
        "checkpoint_dir": str(checkpoints_dir("rebot-pickup")),
        "interrupted": None,
        "cameras": list(live_cameras),
        "observation_mode": "black-frames"
        if args.blind
        else ("spawn-pad-overlay" if args.overlay else "vision+proprio"),
        "note": "DN→TCP maps are engineered, not proven motor identity.",
    }

    client = MCPClient(args.mcp, launch_lab=False)
    prev_grid = None
    log: list[dict] = []
    log_dir = args.log_dir
    log_frames = None
    log_jsonl = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_frames = log_dir / "frames"
        log_frames.mkdir(parents=True, exist_ok=True)
        log_jsonl = log_dir / "log.jsonl"
    start_tcp = {"x": 0.0, "y": 0.0, "z": 0.0}
    hold_s = 0.0
    closest = 1e9
    try:
        client.initialize("flybrain-dn-arm")
        client.wait_ready(timeout=8)
        print("Folded idle", flush=True)
        client.call("rebot_set_control_mode", {"mode": "scripted"})
        folded = client.apply_preset("Folded")
        ft = folded["tcp_mm"]
        print(f"folded tcp=({ft['x']:.0f},{ft['y']:.0f},{ft['z']:.0f})", flush=True)
        print("Ready", flush=True)
        ready = client.apply_preset("Ready")
        rt = ready["tcp_mm"]
        print(f"ready tcp=({rt['x']:.0f},{rt['y']:.0f},{rt['z']:.0f})", flush=True)
        client.call("rebot_set_cube", {"present": True, "x_mm": args.x, "y_mm": args.y, "size_mm": args.size})
        client.call("rebot_set_gripper", {"opening_mm": 90})
        client.wait_stopped()
        client.call("rebot_set_control_mode", {"mode": "servo"})
        if brain is not None:
            brain.reset_state()
        start = client.state()
        start_tcp = start["tcp_mm"]
        print(f"tcp0=({start_tcp['x']:.0f},{start_tcp['y']:.0f},{start_tcp['z']:.0f})", flush=True)

        hold_t0: float | None = None
        hold_s = 0.0
        closest = 1e9
        interrupted = None
        last_state = start
        stall_count = 0
        quintic_ran = False
        clip_ran = False
        xyz_diverged = False
        z_floor_bound = False
        pad_lock: tuple[float, float, float] | None = None
        for tick in range(args.ticks):
            state = client.try_state()
            if state is None:
                interrupted = "rebot_get_state timeout"
                print("mcp state timeout, retry", flush=True)
                time.sleep(0.3)
                state = client.try_state()
                if state is None:
                    interrupted = "Simulator connection closed or timed out"
                    print("mcp", interrupted, flush=True)
                    break
            last_state = state
            tcp = state["tcp_mm"]
            tcp_before = dict(tcp)
            c = cube(state)
            cube_xy = (float(c["center_mm"]["x"]), float(c["center_mm"]["y"]))
            attached = bool(c["attached"])
            rgbs: list = []
            need_capture = (not args.blind and not args.overlay) or log_dir is not None
            if need_capture:
                try:
                    rgbs = capture_rgbs(
                        client,
                        live_cameras,
                        width=160,
                        height=120,
                        save_dir=log_frames,
                        stem=f"{tick:04d}",
                    )
                except (RuntimeError, TimeoutError) as exc:
                    print("capture", exc, flush=True)
            rgb = rgbs[0] if rgbs else np.zeros((120, 160, 3), dtype=np.float32)

            cmd = ArmCommand(0, 0, 0, 0)
            raw_cmd = cmd
            phase = "idle"
            acting_map = "idle"
            if args.overlay:
                decision = spawn_pad_overlay(
                    tcp,
                    float(state["gripper_mm"]),
                    attached,
                    yaw_rad(state),
                    cube_xy=(args.x, args.y),
                    cube_size_mm=args.size,
                    pad_lock=pad_lock,
                )
                cmd = decision.command
                phase = decision.phase
                acting_map = "overlay"
                if pad_lock is None and decision.phase in {"align", "close"}:
                    pad_lock = decision.pad_xyz
            elif readout_w is not None:
                feat = features_from_rgbs(
                    connectome,
                    brain,
                    rgbs or [rgb],
                    gripper_mm=float(state["gripper_mm"]),
                    tcp_mm=np.array([tcp["x"], tcp["y"], tcp["z"]]),
                    attached=False,
                    hops=args.hops,
                    proprio=True,
                    reset=True,
                    publish=True,
                    yaw_rad=yaw_rad(state),
                    cameras=tuple(live_cameras) if rgbs else ("Front",),
                )
                if feat.shape[0] != readout_w.shape[0]:
                    raise RuntimeError(
                        f"readout feature dim {feat.shape[0]} != weights {readout_w.shape[0]}"
                    )
                if kernel_tables is not None:
                    kx, ky, kw = kernel_tables
                    act = apply_kernel(feat, kx, ky, sample_weight=kw, tau=None, k_neighbors=12)
                    acting_map = "kernel"
                else:
                    act = apply_linear(readout_w, feat[None, :])[0]
                    acting_map = "linear"
                raw_cmd = ArmCommand(float(act[0]), float(act[1]), float(act[2]), float(act[3])).clip(8.0)
                gated = optical_close_gate(raw_cmd, feat, names=FEATURE_NAMES, tcp_z=float(tcp["z"]))
                cmd = clip_spawn_dy(gated, tcp, cube_y=args.y)
                clip_ran = True
                if readout_xyz_diverged(raw_cmd, cmd):
                    xyz_diverged = True
                phase = "readout"
            elif args.blind:
                cmd = ArmCommand(0, 0, 0, 0)
                brain.i_ext.fill(0)
                brain.step(args.hops)
                publish_activity(brain)
            else:
                encoded = [encode_frame(frame, int(r1.size), int(r8.size)) for frame in (rgbs or [rgb])]
                vis = np.mean([e[0] for e in encoded], axis=0)
                chroma = np.mean([e[1] for e in encoded], axis=0) if encoded[0][1].size else encoded[0][1]
                brain.i_ext.fill(0)
                if r1.size:
                    brain.i_ext[r1] = vis
                if r8.size:
                    brain.i_ext[r8] = np.clip(chroma, 0, 4)
                prev_grid = drive_kenyon_from_image(brain, kc, rgb, prev_grid=prev_grid)
                brain.step(args.hops)
                publish_activity(brain)
                cmd = decode_hop_drive(connectome, brain.i_ext, hops=3, step=args.scale)
                mbon = connectome.indices("MBON11")
                lift = 8.0 * float(np.mean(brain.duty[mbon])) if mbon.size else 0.0
                cmd = ArmCommand(
                    dx_mm=cmd.dx_mm,
                    dy_mm=cmd.dy_mm,
                    dz_mm=cmd.dz_mm + lift,
                    dgrip_mm=cmd.dgrip_mm - 2.0 * lift,
                    keep_level=True,
                ).clip(5.0)
                phase = "hop"
                acting_map = "hop"

            z_cmd = float(tcp["z"] + cmd.dz_mm)
            z_sent = max(32.0, z_cmd)
            tick_z_floor = abs(z_sent - z_cmd) > 1e-3
            z_floor_bound = z_floor_bound or tick_z_floor
            target = {
                "x_mm": tcp["x"] + cmd.dx_mm,
                "y_mm": tcp["y"] + cmd.dy_mm,
                "z_mm": z_sent,
                "keep_level": True,
            }
            try:
                servo = client.call("rebot_servo_tcp", target)
                if tick < 3:
                    print("servo_tcp", {k: servo.get(k) for k in ("ik_error_mm", "clamped")}, flush=True)
            except (RuntimeError, TimeoutError) as exc:
                print("servo", exc, flush=True)
                ping = client.try_state()
                if ping is None:
                    interrupted = str(exc)
                    break
            client.wait_servo_step(target=target, timeout=0.8)
            if abs(cmd.dgrip_mm) > 0.1 or phase == "close":
                grip = float(np.clip(state["gripper_mm"] + cmd.dgrip_mm, 0, 90))
                if phase == "close":
                    grip = float(np.clip(args.size + 2.0, 0, 90))
                try:
                    client.call("rebot_servo_joints", {"gripper_mm": grip})
                except (RuntimeError, TimeoutError) as exc:
                    print("grip", exc, flush=True)

            after = client.try_state() or state
            last_state = after
            tcp = after["tcp_mm"]
            c = cube(after)
            attached = bool(c["attached"])
            cube_z = float(c["center_mm"]["z"])
            level = bool(after.get("tcp_level"))
            dxy = dist_xy(tcp, cube_xy)
            closest = min(closest, dxy)
            now = time.monotonic()
            if qualifying(attached=attached, cube_z_mm=cube_z, tcp_level=level):
                if hold_t0 is None:
                    hold_t0 = now
                hold_s = now - hold_t0
            else:
                hold_t0 = None
                hold_s = 0.0
            if servo_stalled(cmd, tcp_before, after["tcp_mm"]):
                stall_count += 1
            else:
                stall_count = 0
            row = {
                "tick": tick,
                "phase": phase,
                "dx": cmd.dx_mm,
                "dy": cmd.dy_mm,
                "dz": cmd.dz_mm,
                "dgrip": cmd.dgrip_mm,
                "dx_raw": raw_cmd.dx_mm if phase == "readout" else cmd.dx_mm,
                "dy_raw": raw_cmd.dy_mm if phase == "readout" else cmd.dy_mm,
                "dz_raw": raw_cmd.dz_mm if phase == "readout" else cmd.dz_mm,
                "dgrip_raw": raw_cmd.dgrip_mm if phase == "readout" else cmd.dgrip_mm,
                "tcp": tcp,
                "attached": attached,
                "cube_z": cube_z,
                "tcp_level": level,
                "grip": float(after["gripper_mm"]),
                "dist_xy": dxy,
                "hold_s": hold_s,
                "z_floor_bound": tick_z_floor,
                "map": acting_map,
            }
            log.append(row)
            if log_jsonl is not None:
                photos = {}
                if log_frames is not None:
                    for name in live_cameras:
                        path = log_frames / f"{tick:04d}-{name.lower()}.jpg"
                        if path.is_file():
                            photos[str(name)] = str(path)
                with log_jsonl.open("a") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "t": time.time(),
                                "step": phase,
                                "tick": tick,
                                "tcp_mm": tcp_before,
                                "tcp_rpy_deg": state.get("tcp_rpy_deg"),
                                "gripper_mm": float(state["gripper_mm"]),
                                "cube": cube(state),
                                "tcp_level": state.get("tcp_level"),
                                "photos": photos,
                                "action": [cmd.dx_mm, cmd.dy_mm, cmd.dz_mm, cmd.dgrip_mm],
                            }
                        )
                        + "\n"
                    )
            if tick % 10 == 0 or tick + 1 == args.ticks or attached or float(tcp["z"]) < 140.0:
                print(
                    f"t={tick:03d} {phase:8} map={acting_map} d=({cmd.dx_mm:+.2f},{cmd.dy_mm:+.2f},{cmd.dz_mm:+.2f},{cmd.dgrip_mm:+.1f}) "
                    f"rawg={raw_cmd.dgrip_mm:+.1f} tcp=({tcp['x']:.0f},{tcp['y']:.0f},{tcp['z']:.0f}) "
                    f"grip={after['gripper_mm']:.0f} dist_xy={dxy:.0f} att={attached} cubez={cube_z:.0f} hold={hold_s:.1f}",
                    flush=True,
                )
            if pick_success(attached=attached, cube_z_mm=cube_z, tcp_level=level, hold_s=hold_s, min_hold_s=args.hold):
                print("pick success", flush=True)
                break
            if args.quintic_handoff and stall_count >= 3:
                print("quintic pad grasp after IK stall", flush=True)
                try:
                    after = scripted_pad_pick(client, args.x, args.y, args.size, hold_s=args.hold)
                except (RuntimeError, TimeoutError) as exc:
                    interrupted = f"scripted pad grasp: {exc}"
                    print("mcp", interrupted, flush=True)
                    break
                quintic_ran = True
                last_state = after
                tcp = after["tcp_mm"]
                c = cube(after)
                attached = bool(c["attached"])
                cube_z = float(c["center_mm"]["z"])
                level = bool(after.get("tcp_level"))
                hold_s = args.hold if score_pick(after, args.hold, min_hold_s=args.hold) else 0.0
                dxy = dist_xy(tcp, (args.x, args.y))
                closest = min(closest, dxy)
                log.append(
                    {
                        "tick": tick,
                        "phase": "quintic",
                        "dx": 0.0,
                        "dy": 0.0,
                        "dz": 0.0,
                        "dgrip": 0.0,
                        "tcp": tcp,
                        "attached": attached,
                        "cube_z": cube_z,
                        "tcp_level": level,
                        "grip": float(after["gripper_mm"]),
                        "dist_xy": dxy,
                        "hold_s": hold_s,
                    }
                )
                print(
                    f"quintic att={attached} z={cube_z:.0f} level={level} grip={after['gripper_mm']:.0f} hold={hold_s:.1f}",
                    flush=True,
                )
                break
            time.sleep(0.08)

        end_tcp = log[-1]["tcp"] if log else start_tcp
        live = client.try_state() or last_state
        end_tcp = live["tcp_mm"]
        c = cube(live)
        attached_now = bool(c["attached"])
        peak_z = float(max([r["cube_z"] for r in log] + [c["center_mm"]["z"]], default=0.0))
        travel = float(np.hypot(end_tcp["x"] - start_tcp["x"], end_tcp["y"] - start_tcp["y"]))
        lift = float(end_tcp["z"] - start_tcp["z"])
        mean_step = float(np.mean([abs(r["dx"]) + abs(r["dy"]) + abs(r["dz"]) for r in log])) if log else 0.0
        level_now = bool(live.get("tcp_level"))
        live_z = float(c["center_mm"]["z"])
        lift_policy = overlay_attached_lift_policy(log)
        map_changed = acting_map_changed(log)
        plus8_hold = clipped_plus8_hold_policy(log)
        outcome = episode_outcome(
            overlay=bool(args.overlay),
            readout=bool(args.readout),
            quintic_ran=quintic_ran,
            attached=attached_now,
            cube_z_mm=live_z,
            tcp_level=level_now,
            hold_s=hold_s,
            min_hold_s=args.hold,
            xyz_diverged=xyz_diverged,
            overlay_lift_policy=lift_policy,
            map_changed=map_changed,
            plus8_hold_policy=plus8_hold,
        )
        dy_raw = [abs(r.get("dy_raw", r["dy"])) for r in log] if log else [0.0]
        dy_clip = [abs(r["dy"]) for r in log] if log else [0.0]
        picked = bool(outcome["picked"])
        teacher_ok = bool(outcome["teacher_ok"])
        moved = travel >= 5.0 or abs(lift) >= 5.0
        summary.update(
            {
                "ok": teacher_ok if quintic_ran else (picked if outcome["controller"] in {"readout", "overlay-servo"} else moved),
                "picked": picked,
                "readout_picked": bool(outcome["readout_picked"]),
                "fly_picked": bool(outcome.get("fly_picked")),
                "xyz_diverged": bool(outcome.get("xyz_diverged")),
                "overlay_lift_policy": bool(outcome.get("overlay_lift_policy")),
                "map_changed": bool(outcome.get("map_changed")),
                "plus8_hold_policy": bool(outcome.get("plus8_hold_policy")),
                "acting_maps": sorted({str(r.get("map")) for r in log if r.get("map")}),
                "z_floor_bound": bool(z_floor_bound),
                "z_floor_mm": 32.0,
                "teacher_ok": teacher_ok,
                "quintic_ran": quintic_ran,
                "controller": outcome["controller"],
                "readout": bool(outcome["readout"]),
                "overlay": bool(outcome["overlay"]),
                "clip_prior": "y-envelope" if clip_ran else "none",
                "readout_path": str(args.readout_path) if args.readout else None,
                "max_abs_dy_raw": float(max(dy_raw)),
                "max_abs_dy_clipped": float(max(dy_clip)),
                "cube_z_mm": live_z,
                "ticks": len(log),
                "travel_xy_mm": travel,
                "delta_z_mm": lift,
                "mean_abs_step_mm": mean_step,
                "attached": attached_now,
                "peak_cube_z_mm": peak_z,
                "tcp_level": level_now,
                "hold_s": hold_s,
                "closest_dist_xy_mm": None if closest > 1e8 else closest,
                "end_tcp": end_tcp,
                "start_tcp": start_tcp,
                "gripper_mm": float(live.get("gripper_mm") or 0.0),
                "interrupted": interrupted,
                "weights": loaded,
            }
        )
        trace_path = dest.with_name(f"{tag}-trace.jsonl")
        trace_path.write_text("".join(json.dumps(r) + "\n" for r in log))
        summary["trace"] = str(trace_path)
        if log_dir is not None:
            summary["log_dir"] = str(log_dir)
        _write(dest, summary)
        return 0 if summary["ok"] else 1
    except Exception as exc:
        summary["interrupted"] = str(exc)
        if log:
            summary["ticks"] = len(log)
            summary["end_tcp"] = log[-1]["tcp"]
            summary["attached"] = bool(log[-1]["attached"])
            summary["peak_cube_z_mm"] = float(max(r["cube_z"] for r in log))
            summary["gripper_mm"] = float(log[-1].get("grip") or 0.0)
            summary["closest_dist_xy_mm"] = None if closest > 1e8 else closest
            summary["hold_s"] = hold_s
        summary["start_tcp"] = start_tcp
        _write(dest, summary)
        print("mcp", exc, flush=True)
        return 1
    finally:
        try:
            client.call("rebot_set_control_mode", {"mode": "scripted"}, retries=1)
            client.apply_preset("Folded")
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
