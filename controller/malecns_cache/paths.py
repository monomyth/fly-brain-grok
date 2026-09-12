from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME = Path("/Users/monomyth/code/data/malecns")
WORKSPACE = Path(__file__).resolve().parents[2]


def home() -> Path:
    """Shared MaleCNS download cache (GCS feathers, prepared CSR, somas)."""
    raw = os.environ.get("MALECNS_HOME", str(DEFAULT_HOME))
    path = Path(raw).expanduser().resolve()
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
