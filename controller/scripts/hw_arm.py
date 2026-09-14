#!/usr/bin/env python3
"""Physical arm: run on isengard. Other machines may only --initiate over SSH."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rebot_adapter.arm_host import (
    DEFAULT_ARM_SSH,
    OffHostError,
    initiate_argv,
    on_arm_host,
    open_physical_can,
    require_arm_host,
)


def cmd_status() -> dict:
    return {
        "on_arm_host": on_arm_host(),
        "can_open": False,
        "connect_called": False,
        "note": "arm control process must run on isengard; this Mac only SSHs",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="B601 arm host gate")
    p.add_argument("--initiate", action="store_true", help="SSH from this Mac; do not open CAN here")
    p.add_argument("--host", default=DEFAULT_ARM_SSH)
    p.add_argument("action", nargs="?", default="status", choices=("status", "command"))
    args = p.parse_args(argv)

    if args.initiate:
        if args.action == "command":
            print("command must execute on isengard; this machine cannot drive CAN", file=sys.stderr)
            return 2
        try:
            argv_ssh = initiate_argv(args.host, ["hostname", "-s"])
        except OffHostError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        proc = subprocess.run(argv_ssh, check=False, capture_output=True, text=True)
        print(
            json.dumps(
                {
                    "initiated": True,
                    "host": args.host,
                    "remote_hostname": (proc.stdout or "").strip(),
                    "can_open": False,
                    "connect_called": False,
                }
            )
        )
        return 0 if proc.returncode == 0 else 2

    if args.action == "status":
        print(json.dumps(cmd_status()))
        return 0
    try:
        require_arm_host()
        open_physical_can()
    except OffHostError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
