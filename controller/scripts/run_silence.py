#!/usr/bin/env python3
"""Load the prepared graph and integrate with zero input."""
from __future__ import annotations

import argparse
import time

from malecns_cache.graph import load_graph
from malecns_cache.lockfile import verify_release
from runtime.lif import LIFNetwork


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=0.1)
    args = parser.parse_args()
    verify_release("v1.0")
    graph = load_graph("v1.0")
    brain = LIFNetwork(graph.weights)
    steps = max(1, int(round(args.seconds / brain.dt)))
    started = time.perf_counter()
    brain.step(steps)
    elapsed = time.perf_counter() - started
    print(
        f"n={graph.n} nnz={graph.weights.nnz} steps={steps} "
        f"elapsed_s={elapsed:.3f} max_|v|={abs(brain.v).max():.4f} "
        f"mean_rate_hz={float(brain.rate_hz.mean()):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
