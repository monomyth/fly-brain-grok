"""Phase 4: terminal ±1 on g, DA pulse on KC→MBON, MBON gates the bus. No dense distance reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from malecns_cache.paths import checkpoints_dir
from runtime.plasticity import KCToMBON

from .crop import as_connectome, build_crop, write_stub_crop
from .hop_probe import train_gate
from .lif_crop import CropLIF
from .log import g_hash, write_json, write_skip_checkpoint
from .score import ActingMap, da_learned as da_learned_flag
from .unpack import U0, drive_and_gate, mbon_ablation_changed, slots_moved, unpack


def terminal_R(*, attached: bool, cube_z_mm: float, tcp_level: bool, hold_s: float, pinch_air: bool, drop: bool, timeout: bool) -> float:
    if attached and cube_z_mm >= 100.0 and tcp_level and hold_s >= 2.0:
        return 1.0
    if pinch_air or drop or timeout:
        return -1.0
    return -1.0


def pulse_da(memory: KCToMBON, reward: float) -> int:
    """Sign from the task event, not PPL Hz."""
    if reward > 0:
        memory.pulse(0.0)
    else:
        memory.pulse(4.0)
    return memory.update(float(np.clip(reward, -1.0, 1.0)), eta=2e-4)


def terminal_da_changed(
    lif,
    memory: KCToMBON,
    front,
    grip,
    u,
    attached: bool,
    *,
    teacher: bool,
    reward: float,
) -> bool:
    """True when a terminal pulse moves KC→MBON slots and zeroing them changes the gate."""
    if teacher or front is None or grip is None or memory.slots.size == 0:
        return False
    drive_and_gate(lif.brain, memory, front)
    before = np.array(lif.brain.weights.data[memory.slots], copy=True)
    pulse_da(memory, reward)
    record = getattr(lif, "record_da_delta", None)
    if callable(record):
        record()
    if not slots_moved(before, lif.brain.weights.data[memory.slots]):
        return False
    return mbon_ablation_changed(lif, front, grip, u, attached, memory=memory)


def operant(
    *,
    stub: bool = False,
    distill_path: Path | None = None,
    n_gen: int = 4,
    seed: int = 0,
    live_rewards: list[dict] | None = None,
) -> dict:
    del n_gen
    dest = checkpoints_dir("rebot-pickup") / ("g-operant-stub.npz" if stub else "g-operant.npz")
    distill_path = distill_path or checkpoints_dir("rebot-pickup") / ("g-distill-stub.npz" if stub else "g-distill.npz")
    hop = checkpoints_dir("rebot-pickup") / ("hop-probe-stub.json" if stub else "hop-probe.json")
    hop_blob = json.loads(hop.read_text()) if hop.is_file() else {}
    skip_payload = {
        "acting_map": ActingMap.dn_bus.value,
        "da_learned": False,
        "teacher_off": True,
    }
    if not stub and not train_gate(hop_blob):
        skip_payload["reason"] = "hop_probe not green or live_go false; Phase 4 not opened"
        return write_skip_checkpoint(json_path=dest.with_suffix(".json"), npz=dest, payload=skip_payload)
    if not live_rewards:
        skip_payload["reason"] = (
            "no live terminal R (attach∧z≥100∧level∧hold vs pinch-air/drop/timeout); cube-vs-black is not R"
        )
        return write_skip_checkpoint(json_path=dest.with_suffix(".json"), npz=dest, payload=skip_payload)
    crop = write_stub_crop(checkpoints_dir("rebot-pickup") / "stub-crop") if stub else build_crop("v1.0")
    gain = float(hop_blob.get("synaptic_gain") or 1.5)
    lif = CropLIF(crop, synaptic_gain=gain, nsteps=100 if stub else 120)
    if distill_path.is_file() and "agc-flood" not in distill_path.name:
        blob = np.load(distill_path)
        if "g" in blob.files and blob["g"].shape == lif.g.shape:
            lif.set_g(blob["g"])
    g0 = lif.g.copy()
    memory = KCToMBON(as_connectome(crop), lif.brain)

    def episode(gvec: np.ndarray, front: np.ndarray, grip: np.ndarray, reward: float, ablate: bool = False) -> dict:
        lif.set_g(gvec)
        if ablate and memory.slots.size:
            lif.brain.weights.data[memory.slots] = 0
        lif.reset_episode()
        lif.step_vision(front, grip, drive_kc=False)
        rates = lif.rates()
        gate = drive_and_gate(lif.brain, memory, front)
        cmd = unpack(rates, U0, gate=gate)
        nsyn = 0 if ablate else pulse_da(memory, reward)
        if not ablate and nsyn:
            lif.record_da_delta()
        return {"reward": reward, "dn_hz": rates.scored_mean_hz, "gate": gate, "nsyn": nsyn, "cmd": cmd.__dict__}

    parent = g0.copy()
    parent_R = 0.0
    history: list[dict] = []
    gate_ticks: list[dict] = []
    last_front = last_grip = None
    last_R = 0.0
    for rec in live_rewards:
        child = rec.get("g")
        front = rec.get("front_rgb")
        grip = rec.get("grip_rgb")
        r = float(rec.get("R") or 0.0)
        if child is None or front is None or grip is None:
            history.append({"kept": False, "reason": "missing g or live frames; R not assigned to unevaluated noise"})
            continue
        child = np.asarray(child, dtype=np.float32).reshape(-1)
        if child.shape != parent.shape:
            history.append({"kept": False, "reason": "g shape mismatch"})
            continue
        ev = episode(child, front, grip, r)
        last_front, last_grip, last_R = front, grip, r
        gate_ticks.append({"mbon_gate": float(ev["gate"]), "acting_map": ActingMap.dn_bus.value})
        kept = r > parent_R
        if kept:
            parent = child.copy()
            parent_R = r
        history.append({"R": r, "kept": kept, "g_hash": g_hash(child)})

    if last_front is None:
        skip_payload["reason"] = "live_rewards had no evaluable child (need g + front_rgb + grip_rgb + R)"
        return write_skip_checkpoint(json_path=dest.with_suffix(".json"), npz=dest, payload=skip_payload)

    ablation_changed = terminal_da_changed(
        lif,
        memory,
        last_front,
        last_grip,
        U0,
        False,
        teacher=False,
        reward=last_R,
    )
    gate_on = drive_and_gate(lif.brain, memory, last_front, accumulate=False)
    saved = np.array(lif.brain.weights.data[memory.slots], copy=True) if memory.slots.size else None
    if memory.slots.size:
        lif.brain.weights.data[memory.slots] = 0
    gate_off = drive_and_gate(lif.brain, memory, last_front, accumulate=False)
    if saved is not None:
        lif.brain.weights.data[memory.slots] = saved
    lif.apply_gains()
    out = {
        "ok": g_hash(parent) != g_hash(g0),
        "skipped": False,
        "reason": None,
        "acting_map": ActingMap.dn_bus.value,
        "teacher_off": True,
        "g_hash": g_hash(parent),
        "g_hash_init": g_hash(g0),
        "g": parent.tolist(),
        "history": history,
        "kc_mbon_slots": int(memory.slots.size),
        "mbon_gate_before": float(gate_on),
        "mbon_gate_after_ablate": float(gate_off),
        "da_learned": bool(
            da_learned_flag(
                ablation_changed=ablation_changed,
                acting_map=ActingMap.dn_bus,
                ticks=gate_ticks,
            )
        ),
        "fly_picked": False,
        "note": "child g kept only if that rollout's terminal R beats the parent; DA uses the same live frames",
    }
    np.savez(dest, g=parent, g_init=g0)
    write_json(dest.with_suffix(".json"), out)
    out["path"] = str(dest)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true")
    parser.add_argument("--gens", type=int, default=4)
    args = parser.parse_args(argv)
    result = operant(stub=args.stub, n_gen=args.gens)
    print(json.dumps({k: result[k] for k in result if k not in {"g", "history"}}, indent=2))
    return 0 if not result.get("skipped") else 2


if __name__ == "__main__":
    raise SystemExit(main())
