#!/usr/bin/env python3
"""Copy the fly-brain controller to isengard. Does not open CAN."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO = ROOT.parent
DEFAULT_HOST = "isengard.local"
DEFAULT_REMOTE = "/home/monomyth/fly-brain-grok"


def rsync_argv(host: str, remote: str) -> list[list[str]]:
    dest = f"{host}:{remote}/"
    common = ["rsync", "-az", "--exclude", ".venv", "--exclude", "__pycache__", "--exclude", "*.pyc"]
    crop = REPO / "data" / "prepared" / "malecns-v1.0-crop-v1-optic-rich-mancType-DNfl-DNxl.npz"
    distill = REPO / "data" / "checkpoints" / "rebot-pickup" / "g-distill.npz"
    cmds = [
        [*common, f"{ROOT}/arm/", f"{dest}controller/arm/"],
        [*common, f"{ROOT}/malecns_cache/", f"{dest}controller/malecns_cache/"],
        [*common, f"{ROOT}/rebot_adapter/", f"{dest}controller/rebot_adapter/"],
        [*common, f"{ROOT}/runtime/", f"{dest}controller/runtime/"],
        [*common, f"{ROOT}/train/", f"{dest}controller/train/"],
        [*common, f"{ROOT}/scripts/", f"{dest}controller/scripts/"],
        [*common, str(ROOT / "pyproject.toml"), f"{dest}controller/"],
        [*common, f"{REPO}/data/calibration/", f"{dest}data/calibration/"],
    ]
    if crop.is_file():
        cmds.append([*common, str(crop), f"{dest}data/prepared/"])
    if distill.is_file():
        cmds.append([*common, str(distill), f"{dest}data/checkpoints/rebot-pickup/"])
    return cmds


def ensure_venv(host: str, remote: str) -> None:
    from rebot_adapter.arm_host import ssh_base

    script = (
        f"mkdir -p {remote}/controller {remote}/data/prepared {remote}/data/checkpoints/rebot-pickup {remote}/data/calibration && "
        f"test -x {remote}/.venv/bin/python || python3 -m venv {remote}/.venv && "
        f"{remote}/.venv/bin/pip install -q numpy scipy pillow"
    )
    subprocess.run([*ssh_base(host), "bash", "-lc", script], check=True)


def sync(host: str, remote: str) -> None:
    from rebot_adapter.arm_host import ssh_base

    for cmd in rsync_argv(host, remote):
        subprocess.run(cmd, check=True)
    ensure_venv(host, remote)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rsync fly-brain controller to the arm host")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--remote", default=DEFAULT_REMOTE)
    args = p.parse_args(argv)
    sync(args.host, args.remote)
    print(f"synced {args.host}:{args.remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
