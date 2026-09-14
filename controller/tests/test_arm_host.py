from __future__ import annotations

import pytest

from rebot_adapter.arm_host import (
    OffHostError,
    initiate_argv,
    on_arm_host,
    open_physical_can,
    require_arm_host,
    ssh_base,
)


def test_on_arm_host_isengard_only():
    assert on_arm_host(hostname="isengard", env={}) is True
    assert on_arm_host(hostname="isengard.local", env={}) is True
    assert on_arm_host(hostname="MacBook-Pro", env={}) is False


def test_env_force_overrides_hostname():
    assert on_arm_host(hostname="isengard", env={"FLYBRAIN_ARM_HOST": "0"}) is False
    assert on_arm_host(hostname="MacBook-Pro", env={"FLYBRAIN_ARM_HOST": "1"}) is True


def test_require_arm_host_fails_closed_off_box():
    with pytest.raises(OffHostError, match="initiate"):
        require_arm_host(hostname="MacBook-Pro", env={})
    require_arm_host(hostname="isengard", env={})


def test_open_physical_can_refuses_mac_and_unwired_host():
    with pytest.raises(OffHostError, match="initiate"):
        open_physical_can(hostname="MacBook-Pro", env={})
    with pytest.raises(OffHostError, match="not wired"):
        open_physical_can(hostname="isengard", env={})


def test_initiate_argv_is_ssh_not_local_can():
    argv = initiate_argv("isengard.local", ["hostname", "-s"], hostname="MacBook-Pro", env={})
    assert argv == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "isengard.local",
        "--",
        "hostname",
        "-s",
    ]
    assert "/dev/ttyACM0" not in argv
    with pytest.raises(OffHostError, match="already on the arm host"):
        initiate_argv("isengard.local", ["hostname"], hostname="isengard", env={})
    with pytest.raises(OffHostError):
        initiate_argv("isengard.local", [], hostname="MacBook-Pro", env={})


def test_ssh_base_rejects_flag_host():
    with pytest.raises(OffHostError):
        ssh_base("-o")
    with pytest.raises(OffHostError):
        ssh_base("")
