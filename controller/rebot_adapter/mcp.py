from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path


def lab_binary_for(mcp: Path) -> Path:
    return mcp.parent / "ReBotMotionLab"


def app_bundle_for(mcp: Path) -> Path | None:
    mcp = Path(mcp).resolve()
    if mcp.parent.name == "MacOS" and mcp.parent.parent.name == "Contents" and mcp.parent.parent.parent.suffix == ".app":
        return mcp.parent.parent.parent
    for parent in mcp.parents:
        for name in ("ReBot Motion Lab Grok.app", "ReBot Motion Lab.app"):
            candidate = parent / "dist" / name
            if (candidate / "Contents/MacOS/ReBotMCP").is_file():
                return candidate
    return None


def lab_pids(binary: Path) -> list[int]:
    """Only the grok Debug binary, not a different ReBotMotionLab on the machine."""
    needle = str(Path(binary).resolve())
    try:
        out = subprocess.check_output(["pgrep", "-f", needle], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


class MCPClient:
    def __init__(self, binary: Path, env: dict[str, str] | None = None, launch_lab: bool = False):
        self.binary = Path(binary)
        if not self.binary.is_file():
            raise FileNotFoundError(f"ReBotMCP not found: {self.binary}")
        merged = {**os.environ, **(env or {})}
        # Never exec the GUI from this process. Open the .app yourself (Dock icon).
        merged.setdefault("REBOT_MCP_NO_LAUNCH", "1")
        self.process = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=merged,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.index = 0
        self.launch_lab = launch_lab
        self._lab: subprocess.Popen | None = None
        self._pending_id: int | None = None

    def drain_pending(self, timeout: float = 0.5) -> dict | None:
        """Read a late reply. Keep `_pending_id` until a line is consumed."""
        if self._pending_id is None or not self.process.stdout:
            return None
        if not self.selector.select(timeout):
            return None
        line = self.process.stdout.readline()
        pending = self._pending_id
        self._pending_id = None
        if not line:
            raise RuntimeError(f"MCP exited {self.process.poll()}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"bad MCP line {line[:120]!r}") from exc
        reply["_pending_id"] = pending
        return reply

    def _result(self, reply: dict, req_id: int) -> dict:
        if reply.get("id") != req_id:
            raise RuntimeError(reply)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def rpc(self, method: str, params: dict | None = None, timeout: float = 12) -> dict:
        deadline = time.monotonic() + max(timeout, 0.05)
        while self._pending_id is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("pending reply")
            late = self.drain_pending(timeout=remaining)
            if late is None:
                raise TimeoutError("pending reply")
        self.index += 1
        req_id = self.index
        assert self.process.stdin and self.process.stdout
        try:
            self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(f"MCP exited {self.process.poll()}") from exc
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.selector.select(remaining):
            self._pending_id = req_id
            raise TimeoutError(method)
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"MCP exited {self.process.poll()}")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"bad MCP line {line[:120]!r}") from exc
        return self._result(reply, req_id)

    def initialize(self, name: str = "flybrain-controller") -> dict:
        info = self.rpc(
            "initialize",
            {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": name, "version": "0.1"}},
        )
        assert self.process.stdin
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
        self.process.stdin.flush()
        return info

    def call(self, name: str, arguments: dict | None = None, retries: int = 1, timeout: float = 12) -> dict:
        last: Exception | None = None
        attempts = max(int(retries), 1)
        for attempt in range(attempts):
            try:
                result = self.rpc("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)
                if result.get("isError"):
                    text = result["content"][0]["text"] if result.get("content") else "tool error"
                    raise RuntimeError(text)
                return result.get("structuredContent") or json.loads(result["content"][0]["text"])
            except TimeoutError as exc:
                last = exc
                if str(exc) == "pending reply":
                    self.drain_pending(timeout=min(timeout, 8.0))
                    if self.process.poll() is not None or attempt + 1 >= attempts:
                        break
                    continue
                late = self.drain_pending(timeout=min(timeout, 8.0))
                if late is not None and late.get("id") == late.get("_pending_id"):
                    result = self._result(late, int(late["_pending_id"]))
                    if result.get("isError"):
                        text = result["content"][0]["text"] if result.get("content") else "tool error"
                        raise RuntimeError(text)
                    return result.get("structuredContent") or json.loads(result["content"][0]["text"])
                if self.process.poll() is not None or attempt + 1 >= attempts:
                    break
            except BrokenPipeError as exc:
                raise RuntimeError(f"MCP exited {self.process.poll()}") from exc
            except (RuntimeError, json.JSONDecodeError) as exc:
                last = exc
                # Helper SO_RCVTIMEO is 5s. The sim holds ≤4 control slots until MainActor replies;
                # rapid retries fill the slots and every later connect is closed immediately.
                if "check state before retrying" in str(exc) and attempt + 1 < attempts:
                    time.sleep(6.0)
                    continue
                break
        raise last or RuntimeError(name)

    def state(self, retries: int = 1, timeout: float = 12) -> dict:
        return self.call("rebot_get_state", retries=retries, timeout=timeout)

    def wait_servo_step(
        self,
        target: dict | None = None,
        timeout: float = 0.8,
    ) -> dict | None:
        """Wait until TCP L1-error to `target` is ≤ 2.5 mm, or `timeout`."""
        deadline = time.monotonic() + max(timeout, 0.05)
        last = None
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            last = self.try_state(timeout=min(0.2, remaining))
            if last is None:
                continue
            if target is None:
                time.sleep(0.03)
                continue
            t = last["tcp_mm"]
            err = abs(t["x"] - float(target["x_mm"])) + abs(t["y"] - float(target["y_mm"])) + abs(
                t["z"] - float(target["z_mm"])
            )
            if err <= 2.5:
                return last
            time.sleep(0.03)
        return last

    def try_state(self, timeout: float = 8) -> dict | None:
        try:
            return self.state(retries=1, timeout=timeout)
        except (RuntimeError, TimeoutError, BrokenPipeError):
            return None

    def spawn_lab(self) -> Path:
        lab = lab_binary_for(self.binary)
        if not lab.is_file():
            raise FileNotFoundError(
                f"ReBotMotionLab not next to MCP ({lab}). Build the grok lab or pass the ReBotMCP path."
            )
        if lab_pids(lab):
            return lab
        app = app_bundle_for(self.binary)
        log_path = Path(os.environ.get("TMPDIR", "/tmp")) / "rebot-motionlab-grok-launch.log"
        log = log_path.open("ab")
        if app is not None:
            # Launch the .app via LaunchServices so the Dock shows the real icon.
            self._lab = subprocess.Popen(
                ["/usr/bin/open", str(app)],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            log.write(f"\nopened app={app} pid={self._lab.pid}\n".encode())
        else:
            self._lab = subprocess.Popen(
                [str(lab)],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            log.write(f"\nspawned pid={self._lab.pid} path={lab}\n".encode())
        log.flush()
        return lab

    def wait_ready(self, timeout: float = 45) -> dict:
        deadline = time.monotonic() + timeout
        last = "no state yet"
        while time.monotonic() < deadline:
            remaining = max(0.2, deadline - time.monotonic())
            try:
                last = self.state(retries=1, timeout=min(8.0, remaining))
                if last.get("scene_ready"):
                    return last
                last = f"scene_ready={last.get('scene_ready')}"
            except (RuntimeError, TimeoutError) as exc:
                last = str(exc)
            time.sleep(0.4)
        app = app_bundle_for(self.binary)
        raise TimeoutError(
            f"simulator not ready after {timeout:.0f}s. Last error: {last}. "
            f"Open the app first: open \"{app or lab_binary_for(self.binary)}\""
        )

    def wait_stopped(self, timeout: float = 20) -> dict:
        deadline = time.monotonic() + timeout
        state = None
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                state = self.state()
                last_err = None
                if state["playback"] == "stopped" and not state["manual_motion"]:
                    return state
            except (RuntimeError, TimeoutError) as exc:
                last_err = exc
            time.sleep(0.08)
        if last_err is not None:
            raise TimeoutError(str(last_err))
        raise TimeoutError(state and state.get("status"))

    def apply_preset(self, name: str, timeout: float = 40) -> dict:
        """Start a named pose and wait until it actually finishes (or never starts)."""
        self.call("rebot_apply_preset", {"name": name}, retries=2)
        started = time.monotonic()
        saw_move = False
        state = None
        while time.monotonic() - started < timeout:
            remaining = max(0.2, timeout - (time.monotonic() - started))
            try:
                state = self.state(retries=1, timeout=min(8.0, remaining))
            except (RuntimeError, TimeoutError):
                time.sleep(1.0)
                continue
            moving = state.get("playback") != "stopped" or bool(state.get("manual_motion"))
            if moving:
                saw_move = True
            elif saw_move:
                return state
            if not saw_move and time.monotonic() - started > 2.0:
                return state
            time.sleep(0.05)
        if state is not None:
            return state
        return self.state(retries=2)

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except BrokenPipeError:
            pass
        self.process.wait(timeout=5)
        self.selector.close()
