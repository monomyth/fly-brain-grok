#!/usr/bin/env python3
"""Phase 0/1: ownership on the robot host, hop-probe on this machine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO = ROOT.parent
GRAB = Path(__file__).resolve().parent / "hw_orbbec_grab.py"
DEFAULT_HOST = "isengard.local"
ORBBEC_PY = "/home/monomyth/robotics/orbbec/.venv/bin/python"
REMOTE_GRAB = "/tmp/hw_orbbec_grab.py"
REMOTE_OUT = "/tmp/hw-phase1"


def _ssh(host: str) -> list[str]:
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host]


def run_phase0(host: str | None) -> dict:
    from rebot_adapter.hw_phase import collect_ownership_local, collect_ownership_ssh

    if host:
        return collect_ownership_ssh(host)
    return collect_ownership_local()


def capture_ssh(host: str, frames: int, dest: Path) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-o", "BatchMode=yes", str(GRAB), f"{host}:{REMOTE_GRAB}"], check=True)
    cmd = [
        *_ssh(host),
        ORBBEC_PY,
        REMOTE_GRAB,
        "--out",
        REMOTE_OUT,
        "--frames",
        str(frames),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"grab exit {proc.returncode}")
    for name in ("wrist.jpg", "overview.jpg", "capture.json"):
        subprocess.run(["scp", "-o", "BatchMode=yes", f"{host}:{REMOTE_OUT}/{name}", str(dest / name)], check=True)
    return json.loads((dest / "capture.json").read_text())


def run_hop(folder: Path, stub: bool, steps: int, out: Path) -> dict:
    from arm.crop import build_crop, write_stub_crop
    from arm.lif_crop import CropLIF
    from malecns_cache.paths import project_data
    from rebot_adapter.hw_phase import hop_from_dir, write_json

    if stub:
        crop = write_stub_crop(project_data() / "prepared" / "stub-hw-phase1")
    else:
        crop = build_crop()
    lif = CropLIF(crop, nsteps=steps)
    hop = hop_from_dir(lif, folder, nsteps=steps)
    hop["stub"] = stub
    write_json(out, hop)
    return hop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 0 ownership + Phase 1 camera hop")
    p.add_argument("--phase", choices=("0", "1", "01"), default="01")
    p.add_argument("--host", default=None, help="SSH host for USB/SDK work (e.g. isengard.local)")
    p.add_argument("--from-dir", type=Path, default=None, help="Existing wrist.jpg/overview.jpg")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out", type=Path, default=REPO / "reports")
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.phase in ("0", "01"):
        host = args.host
        report = run_phase0(host)
        (out / "hw-phase0.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"phase0 capture_ok={report['capture_ok']} policy_qualify={report['policy_qualify']} wrote {out / 'hw-phase0.json'}")
        if args.phase == "0":
            return 0 if report["capture_ok"] else 2
        if not report["capture_ok"] and args.from_dir is None:
            print("phase1 skipped: capture_ok is false")
            return 2

    if args.phase in ("1", "01"):
        folder = args.from_dir
        if folder is None:
            if not args.host:
                print("phase1 live grab needs --host or --from-dir", file=sys.stderr)
                return 2
            folder = out / "hw-phase1-frames"
            capture_ssh(args.host, args.frames, folder)
        hop = run_hop(folder, stub=args.stub, steps=args.steps, out=out / "hw-cam-hop.json")
        print(
            f"phase1 cube_dn={hop.get('cube_dn_mean')} black={hop.get('empty_dn_mean')} "
            f"ok={hop.get('ok')} fly_picked={hop.get('fly_picked')} wrote {out / 'hw-cam-hop.json'}"
        )
        return 0 if hop.get("ok") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
