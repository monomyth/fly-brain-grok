from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]


def default_malecns_home() -> Path:
    """No user or host in the tree. GPU box: /data/malecns if present, else ~/data/malecns."""
    gpu = Path("/data/malecns")
    if gpu.is_dir():
        return gpu
    return Path.home() / "data" / "malecns"


def home() -> Path:
    """Shared MaleCNS download cache (GCS feathers, prepared CSR, somas)."""
    raw = os.environ.get("MALECNS_HOME")
    path = Path(raw).expanduser().resolve() if raw else default_malecns_home()
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_data() -> Path:
    """Artifacts this grok fly-brain run writes: checkpoints, teacher logs, live activity."""
    raw = os.environ.get("FLYBRAIN_DATA")
    path = Path(raw).expanduser().resolve() if raw else (WORKSPACE / "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def release_dir(release: str = "v1.0") -> Path:
    path = home() / release
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepared_dir() -> Path:
    path = home() / "prepared"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(name: str = "rebot-pickup") -> Path:
    path = project_data() / "checkpoints" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir(name: str = "rebot-teacher") -> Path:
    path = project_data() / "datasets" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def graph_npz(release: str = "v1.0") -> Path:
    return prepared_dir() / f"malecns-{release}-graph.npz"


def graph_meta(release: str = "v1.0") -> Path:
    return prepared_dir() / f"malecns-{release}-meta.json"


def somas_path(release: str = "v1.0") -> Path:
    return prepared_dir() / f"malecns-{release}-somas.bin"


def live_dir() -> Path:
    path = project_data() / "live"
    path.mkdir(parents=True, exist_ok=True)
    return path


def activity_path() -> Path:
    return live_dir() / "activity.bin"


def lock_path(release: str = "v1.0") -> Path:
    return release_dir(release) / "source.lock.json"
