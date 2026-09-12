"""Does retina current reach front-leg motor neurons in this LIF?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from malecns_cache.graph import load_graph
from malecns_cache.paths import checkpoints_dir
from runtime.lif import LIFNetwork


def _mean(rate: np.ndarray, idx: np.ndarray) -> float:
    if idx.size == 0:
        return 0.0
    return float(np.mean(rate[idx]))


def run(steps: tuple[int, ...] = (3, 16, 50, 100, 200, 500), drive: float = 4.0) -> dict:
    graph = load_graph("v1.0")
    r1 = graph.indices("photoreceptors_r1r6")
    r8 = graph.indices("photoreceptors_r8")
    front = graph.indices("leg_mn_front")
    vnc = graph.indices("vnc_motor")
    dn = graph.indices("DN")
    if r1.size == 0 or front.size == 0:
        raise RuntimeError("missing R1-R6 or leg_mn_front groups; run prepare.py --groups")
    rows = []
    for nsteps in steps:
        for reset in (True, False):
            brain = LIFNetwork(graph.weights)
            if reset:
                brain.reset_state()
            brain.i_ext.fill(0)
            brain.i_ext[r1] = np.float32(drive)
            if r8.size:
                brain.i_ext[r8] = np.float32(drive * 0.5)
            brain.step(int(nsteps))
            rows.append(
                {
                    "steps": int(nsteps),
                    "reset_before": bool(reset),
                    "r1_hz": _mean(brain.rate_hz, r1),
                    "dn_hz": _mean(brain.rate_hz, dn),
                    "vnc_motor_hz": _mean(brain.rate_hz, vnc),
                    "leg_mn_front_hz": _mean(brain.rate_hz, front),
                    "leg_mn_front_duty": _mean(brain.duty, front),
                }
            )
    out = {
        "n": graph.n,
        "n_r1": int(r1.size),
        "n_leg_mn_front": int(front.size),
        "n_vnc_motor": int(vnc.size),
        "drive": drive,
        "rows": rows,
        "front_moves": any(r["leg_mn_front_hz"] > 0.01 for r in rows),
    }
    dest = checkpoints_dir("rebot-pickup") / "audit-leg-mn.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> int:
    report = run()
    print(f"wrote {checkpoints_dir('rebot-pickup') / 'audit-leg-mn.json'}")
    print(f"front-leg MNs indexed: {report['n_leg_mn_front']}  vnc_motor: {report['n_vnc_motor']}")
    print(f"{'steps':>6} {'reset':>6} {'R1 Hz':>10} {'DN Hz':>10} {'VNC Hz':>10} {'T1 MN Hz':>10}")
    for row in report["rows"]:
        print(
            f"{row['steps']:6d} {str(row['reset_before']):>6} "
            f"{row['r1_hz']:10.4f} {row['dn_hz']:10.4f} "
            f"{row['vnc_motor_hz']:10.4f} {row['leg_mn_front_hz']:10.4f}"
        )
    print("front_moves", report["front_moves"])
    return 0 if report["front_moves"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
