"""Arm control runs on isengard. This Mac may only initiate a remote process."""

from __future__ import annotations

import os
import socket

ARM_HOSTNAMES = frozenset({"isengard"})
ENV_ARM_HOST = "FLYBRAIN_ARM_HOST"
CAN_DEVICE = "/dev/ttyACM0"
DEFAULT_ARM_SSH = "isengard.local"


class OffHostError(RuntimeError):
    """CAN/motorbridge must not open on this machine."""


def short_hostname(name: str | None = None) -> str:
    raw = name if name is not None else socket.gethostname()
    return raw.split(".")[0].lower()


def on_arm_host(*, hostname: str | None = None, env: dict | None = None) -> bool:
    env = os.environ if env is None else env
    force = env.get(ENV_ARM_HOST)
    if force == "0":
        return False
    if force == "1":
        return True
    return short_hostname(hostname) in ARM_HOSTNAMES


def require_arm_host(*, hostname: str | None = None, env: dict | None = None) -> None:
    if not on_arm_host(hostname=hostname, env=env):
        raise OffHostError(
            "arm control runs on isengard; this machine may only initiate (SSH a process there)"
        )


def ssh_base(host: str) -> list[str]:
    if not host or host.startswith("-"):
        raise OffHostError("ssh host is empty or looks like a flag")
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host]


def initiate_argv(host: str, remote_cmd: list[str], *, hostname: str | None = None, env: dict | None = None) -> list[str]:
    """Build SSH argv. Does not open CAN. Refuses if we are already the arm host."""
    if on_arm_host(hostname=hostname, env=env):
        raise OffHostError("already on the arm host; run the command locally")
    if not remote_cmd:
        raise OffHostError("initiate needs a remote command")
    if any(tok.startswith("-") and tok != "--" for tok in remote_cmd[:1]):
        raise OffHostError("remote command looks like an ssh flag")
    return [*ssh_base(host), "--", *remote_cmd]


def open_physical_can(*, hostname: str | None = None, env: dict | None = None) -> None:
    """Only the arm host may touch the Damiao serial device. Not implemented yet."""
    require_arm_host(hostname=hostname, env=env)
    raise OffHostError(f"physical CAN ({CAN_DEVICE}) is not wired; refuse rather than open")
