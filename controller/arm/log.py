from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .score import ActingMap, EpisodeScore

_SEALED_SUMMARY = frozenset({"acting_map", "fly_picked", "lab_picked", "da_learned"})


def g_hash(g: np.ndarray) -> str:
    raw = np.asarray(g, dtype=np.float32).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def tick_row(
    *,
    tick: int,
    acting_map: ActingMap | str,
    dn_hz: float,
    t1_mn_hz: float,
    g_hash_s: str,
    attached: bool,
    cube_z_mm: float,
    tcp: dict,
    gripper_mm: float,
    command: dict,
    abort: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tick": int(tick),
        "acting_map": str(getattr(acting_map, "value", acting_map)),
        "dn_hz": float(dn_hz),
        "t1_mn_hz": float(t1_mn_hz),
        "g_hash": g_hash_s,
        "attached": bool(attached),
        "cube_z_mm": float(cube_z_mm),
        "tcp": tcp,
        "gripper_mm": float(gripper_mm),
        "command": command,
        "abort": bool(abort),
    }
    if extra:
        row.update(extra)
    return row


def episode_summary(
    *,
    score: EpisodeScore,
    ticks: list[dict],
    g_hash_s: str,
    g_hash_init: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    acting = sorted({str(r.get("acting_map")) for r in ticks if r.get("acting_map")})
    dn = [float(r.get("dn_hz") or 0.0) for r in ticks]
    t1 = [float(r.get("t1_mn_hz") or 0.0) for r in ticks]
    out: dict[str, Any] = {
        "acting_map": str(score.acting_map.value if hasattr(score.acting_map, "value") else score.acting_map),
        "acting_maps": acting,
        "fly_picked": bool(score.fly_picked),
        "lab_picked": bool(score.lab_picked),
        "da_learned": bool(score.da_learned),
        "g_trained": bool(score.g_trained),
        "g_hash": g_hash_s,
        "g_hash_init": g_hash_init,
        "mean_dn_hz": float(score.mean_dn_hz),
        "black_dn_hz": float(score.black_dn_hz),
        "dn_l2": float(score.dn_l2),
        "t1_mn_hz_mean": float(np.mean(t1)) if t1 else 0.0,
        "ticks": len(ticks),
        "tick_dn_hz_mean": float(np.mean(dn)) if dn else 0.0,
    }
    if extra:
        out.update({k: v for k, v in extra.items() if k not in _SEALED_SUMMARY})
    return out


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def retire_weights(npz: Path, tag: str = "agc-flood") -> Path | None:
    """Move a flood-era npz aside so it is not the unmarked latest."""
    if not npz.is_file():
        return None
    dest = npz.with_name(f"{npz.stem}.{tag}{npz.suffix}")
    if dest.exists():
        npz.unlink()
        return dest
    npz.rename(dest)
    return dest


def write_skip_checkpoint(*, json_path: Path, npz: Path, payload: dict) -> dict:
    retired = retire_weights(npz)
    out = dict(payload)
    out["ok"] = False
    out["skipped"] = True
    out["fly_picked"] = False
    out["da_learned"] = False
    if retired is not None:
        out["retired_weights"] = str(retired)
    write_json(json_path, out)
    out["path"] = str(json_path)
    return out
