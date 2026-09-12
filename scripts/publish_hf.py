#!/usr/bin/env python3
"""Create/update huggingface.co/monomyth/fly-brain-grok from data/hf-fly-brain-grok/.

Needs a Hugging Face write token (huggingface-cli login). A read-only token will 403.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "data" / "hf-fly-brain-grok"
REPO = "monomyth/fly-brain-grok"


def main() -> int:
    if not (STAGE / "g-distill.npz").is_file():
        raise SystemExit(f"missing {STAGE / 'g-distill.npz'}")
    api = HfApi()
    who = api.whoami()
    role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
    print("hf user", who.get("name"), "token_role", role)
    create_repo(REPO, repo_type="model", exist_ok=True, private=False)
    upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model")
    print("https://huggingface.co", REPO, sep="/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
