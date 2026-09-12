#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from malecns_cache.paths import checkpoints_dir
from rebot_adapter.episode import default_mcp_binary
from rebot_adapter.mcp import MCPClient
from rebot_adapter.teacher import record_pick


def main() -> int:
    binary = Path(sys.argv[1]) if len(sys.argv) > 1 else default_mcp_binary()
    print(f"MCP {binary}", file=sys.stderr)
    dest = checkpoints_dir("rebot-pickup") / "teacher-eval.json"
    summary: dict = {"ok": False, "picked": False}
    client = MCPClient(binary, launch_lab=False)
    try:
        client.initialize("flybrain-teacher")
        client.wait_ready(timeout=8)
        summary = record_pick(client)
        dest.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("picked") or summary.get("ok") else 1
    except Exception as exc:
        summary["interrupted"] = str(exc)
        dest.write_text(json.dumps(summary, indent=2) + "\n")
        print("mcp", exc, flush=True)
        print(json.dumps(summary, indent=2))
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
