import struct

import numpy as np

from malecns_cache.somas import ACTIVITY_MAGIC, SOMA_MAGIC, write_activity


def test_activity_file_roundtrip(tmp_path, monkeypatch):
    from malecns_cache import paths

    monkeypatch.setenv("MALECNS_HOME", str(tmp_path))
    monkeypatch.setenv("REBOT_CONTROL_DIRECTORY", str(tmp_path / "ipc"))
    rates = np.linspace(0, 1, 16, dtype=np.float32)
    path = write_activity(rates, 7)
    data = path.read_bytes()
    assert data[:4] == ACTIVITY_MAGIC
    n, gen = struct.unpack_from("<IQ", data, 4)
    assert n == 16 and gen == 7
    got = np.frombuffer(data[16:], dtype=np.float32)
    assert np.allclose(got, rates)


def test_soma_magic_constant():
    assert SOMA_MAGIC == b"MLC2"
