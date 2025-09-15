"""
Recreate the original PNG folders from CV-Bench parquet files.
Default behaviour (no flags) → rebuild both 2D and 3D images beneath ./img/.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import io
import tqdm


def dump_parquet(parquet_path: Path, out_root: Path, prefix: str) -> None:
    """
    Write PNGs extracted from `parquet_path` into  <out_root>/<prefix>/… .
    * Assumes the parquet has a column called 'image' that stores either a
      numpy array or a nested Python list.
    * If an 'id' column exists it is used for naming; else an incremental index.
    """
    df = pd.read_parquet(parquet_path)
    out_dir = out_root / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc=f"writing {prefix}"
    ):
        img_b = row["image"]["bytes"]
        img = Image.open(io.BytesIO(img_b))

        name = row["id"] if "id" in row else idx
        img.save(out_dir / f"{name:06}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=["2D", "3D", "both"],
        default="both",
        help="Which split(s) to rebuild (default: both)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Dataset root path containing the parquet files",
    )
    parser.add_argument(
        "--out",
        default="img",
        help="Directory where the reconstructed images will be written",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    choice = args.subset.lower()

    if choice in {"2d", "both"}:
        dump_parquet(root / "test_2d.parquet", out_root, "2D")
    if choice in {"3d", "both"}:
        dump_parquet(root / "test_3d.parquet", out_root, "3D")


if __name__ == "__main__":
    main()