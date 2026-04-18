#!/usr/bin/env python3
"""
Download Molmo2 weights from Hugging Face into a local directory.

Typical models: allenai/Molmo2-4B (smallest), allenai/Molmo2-8B, allenai/Molmo2-O-7B.
See https://github.com/allenai/molmo2 for native .tar checkpoints and HF conversion.

Usage:
  python download_weights.py --local-dir /data/Molmo2-4B
  python download_weights.py --repo allenai/Molmo2-8B --local-dir ./Molmo2-8B

Then point the server at the directory:
  export MOLMO2_MODEL=/data/Molmo2-4B
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Molmo2 weights from Hugging Face.")
    parser.add_argument(
        "--repo",
        default=os.environ.get("MOLMO2_REPO", "allenai/Molmo2-4B"),
        help="Hugging Face repo id (default: allenai/Molmo2-4B or MOLMO2_REPO).",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Directory to store the snapshot (e.g. /data/Molmo2-4B).",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HF token if needed (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN).",
    )
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo,
        local_dir=args.local_dir,
        token=args.token,
    )
    print(f"Downloaded {args.repo} to {path}")


if __name__ == "__main__":
    main()
