from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

from . import catalog, paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_release(release: str = "v1.0") -> dict:
    folder = paths.release_dir(release)
    known = catalog.FILES.get(release, {})
    files = {}
    extras = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name in catalog.IGNORE_NAMES:
            continue
        info = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.name in known:
            spec = known[path.name]
            info["role"] = spec["role"]
            info["url"] = spec["url"]
            info["expected_sha256"] = spec["sha256"]
            files[path.name] = info
        else:
            extras[path.name] = info
    return {"release": release, "files": files, "extras": extras}


def write_lock(release: str = "v1.0") -> Path:
    payload = scan_release(release)
    payload["fetched"] = date.today().isoformat()
    dest = paths.lock_path(release)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest


def verify_release(release: str = "v1.0", write: bool = True) -> dict:
    known = catalog.FILES[release]
    folder = paths.release_dir(release)
    missing = [name for name in known if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {release} files in {folder}: {missing}")
    report = scan_release(release)
    errors = []
    for name, spec in known.items():
        got = report["files"][name]
        if got["sha256"] != spec["sha256"]:
            errors.append(f"{name}: sha256 {got['sha256']} != {spec['sha256']}")
        if got["bytes"] != spec["bytes"]:
            errors.append(f"{name}: size {got['bytes']} != {spec['bytes']}")
    if errors:
        raise ValueError("MaleCNS cache failed verification:\n" + "\n".join(errors))
    if write:
        write_lock(release)
    return report


def download_missing(release: str = "v1.0") -> list[Path]:
    folder = paths.release_dir(release)
    written = []
    for name, spec in catalog.FILES[release].items():
        dest = folder / name
        if dest.exists() and sha256_file(dest) == spec["sha256"]:
            continue
        partial = dest.with_suffix(dest.suffix + ".download")
        urllib.request.urlretrieve(spec["url"], partial)
        if sha256_file(partial) != spec["sha256"]:
            partial.unlink(missing_ok=True)
            raise ValueError(f"Download checksum failed: {name}")
        partial.replace(dest)
        written.append(dest)
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    release = args[0] if args else "v1.0"
    report = verify_release(release)
    extras = ", ".join(report["extras"]) or "(none)"
    print(f"verified {release} in {paths.release_dir(release)}")
    print(f"known files: {len(report['files'])}; extras: {extras}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
