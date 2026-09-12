#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

from malecns_cache.paths import checkpoints_dir


def main() -> int:
    root = checkpoints_dir("rebot-pickup")
    a = np.load(root / "kc-mbon-learn.npz")
    b = np.load(root / "kc-mbon-learn-shuffle.npz")
    slots = a["slots"]
    rest = a["rest"]
    da = a["data"][slots] - rest
    db = b["data"][slots] - rest
    na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    cos = float(da @ db / (na * nb)) if na and nb else 0.0
    sign_agree = float(np.mean(np.sign(da) == np.sign(db))) if da.size else 0.0
    report = {
        "edges": int(slots.size),
        "task_l2": na,
        "shuffle_l2": nb,
        "task_max": float(np.max(np.abs(da))) if da.size else 0,
        "shuffle_max": float(np.max(np.abs(db))) if db.size else 0,
        "task_std": float(da.std()) if da.size else 0,
        "shuffle_std": float(db.std()) if db.size else 0,
        "cosine": cos,
        "sign_agree": sign_agree,
        "l2_task_minus_shuffle": float(np.linalg.norm(da - db)),
        "pattern_differs": bool(cos < 0.95 and da.std() > 1e-4),
    }
    print(json.dumps(report, indent=2))
    (root / "learn-vs-shuffle.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["pattern_differs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
