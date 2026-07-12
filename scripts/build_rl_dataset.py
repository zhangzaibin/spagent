#!/usr/bin/env python3
"""
build_rl_dataset.py — fix image paths in the RL training JSONL.

The raw file uses placeholder paths like:
    /<replace_with_your_own_data_path>/mindcube/data/other_all_image/...

This script replaces the placeholder prefix with the actual local path, then
optionally verifies that every referenced image exists.

Usage
-----
python scripts/build_rl_dataset.py \
    --input  "dataset/crossviewQA_train_rl (1).jsonl" \
    --output  dataset/crossviewQA_train_rl_fixed.jsonl \
    --mindcube-root dataset/mindcube

Optional flags
--------------
--verify       : warn (not fail) for missing images
--skip-missing : drop samples whose images are all missing
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

PLACEHOLDER = "/<replace_with_your_own_data_path>/mindcube/data/"


def fix_paths(raw_path: str, mindcube_root: str) -> str:
    """Replace placeholder prefix with the real local prefix."""
    if raw_path.startswith(PLACEHOLDER):
        rel = raw_path[len(PLACEHOLDER):]          # e.g.  other_all_image/among/shoe_225/front_001.jpg
        return os.path.join(mindcube_root, "data", rel)
    return raw_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True,  help="Raw training JSONL with placeholder paths")
    p.add_argument("--output", required=True,  help="Output JSONL with fixed paths")
    p.add_argument("--mindcube-root", default="dataset/mindcube",
                   help="Path to extracted MindCube dataset dir (default: dataset/mindcube)")
    p.add_argument("--verify",        action="store_true", help="Check image paths exist")
    p.add_argument("--skip-missing",  action="store_true", help="Drop samples with missing images")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    mc_root  = args.mindcube_root

    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = kept = skipped = missing_imgs = 0

    with open(in_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Line {lineno}: JSON error – {e}")
                skipped += 1
                continue

            # Fix image paths in the "images" field
            raw_images = item.get("images") or []
            if isinstance(raw_images, str):
                raw_images = [raw_images]

            fixed_images = [fix_paths(p, mc_root) for p in raw_images]
            item["images"] = fixed_images

            # Verify
            if args.verify or args.skip_missing:
                bad = [p for p in fixed_images if not os.path.exists(p)]
                if bad:
                    missing_imgs += len(bad)
                    if args.skip_missing:
                        print(f"  Line {lineno}: SKIP – missing {len(bad)} images: {bad[:2]}…")
                        skipped += 1
                        continue
                    elif args.verify:
                        print(f"  Line {lineno}: WARN – missing {len(bad)}: {bad[:2]}…")

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\nDone.")
    print(f"  Total lines   : {total}")
    print(f"  Written       : {kept}")
    print(f"  Skipped       : {skipped}")
    if args.verify or args.skip_missing:
        print(f"  Missing images: {missing_imgs}")
    print(f"  Output        : {out_path.resolve()}")


if __name__ == "__main__":
    main()
