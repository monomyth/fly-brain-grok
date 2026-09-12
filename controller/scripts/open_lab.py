#!/usr/bin/env python3
"""Launch the packaged simulator only. Does not run the fly brain."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rebot_adapter.episode import default_mcp_binary
from rebot_adapter.mcp import app_bundle_for


def main() -> int:
    mcp = Path(sys.argv[1]) if len(sys.argv) > 1 else default_mcp_binary()
    app = app_bundle_for(mcp)
    if app is None or not app.is_dir():
        print("No .app found. Wrap Debug binaries first:", file=sys.stderr)
        print("  bash ../rebot-motion-lab-grok/scripts/wrap-debug-app.sh", file=sys.stderr)
        return 1
    subprocess.run(["/usr/bin/open", str(app)], check=True)
    print(app)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
