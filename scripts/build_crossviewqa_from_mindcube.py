#!/usr/bin/env python3
"""
build_crossviewqa_from_mindcube.py — expand the crossviewQA RL training set
using the already-downloaded MindCube training split.

Background
----------
`dataset/crossviewQA_train_rl (1).jsonl` (~1k examples, from the HF dataset
`jialianjie/Think3DQA`) is the *entire* upstream release — there is no larger
official version to download. However, `dataset/mindcube/data/raw/MindCube_train.jsonl`
(the official MindCube training split, ~10k examples) is already fully
downloaded locally and was the source pool the 1k file was built from
(~91% of its scenes appear in MindCube_train.jsonl). This script draws
additional examples from that same pool, converts them into the RL
`images` / `messages` / `solution` schema used by `train/train_grpo.sh`,
and merges them with the existing fixed file to reach a larger training set.

Safety checks performed:
  - Every sampled example's images must exist on disk (mindcube-root aware).
  - Scenes that appear in MindCube_tinybench.jsonl (the eval split used by
    `scripts/quick_eval.py --datasets MindCube`) are excluded, to avoid
    training/eval leakage. (Note: the existing 1k seed file already has ~30
    scenes overlapping tinybench — this script does not touch that file, it
    only guards the *new* examples it adds.)
  - Category mix (`among` / `around` / `rotation`) is rebalanced: MindCube_train
    is ~80/11/9% by default, which the existing 1k file mirrors. This script
    includes ~all available `around`/`rotation` examples (scarcer) and fills
    the remainder with a random `among` sample, so the combined set is less
    skewed toward `among`.

Usage
-----
python scripts/build_crossviewqa_from_mindcube.py \
    --mindcube-train dataset/mindcube/data/raw/MindCube_train.jsonl \
    --mindcube-tinybench dataset/mindcube/data/raw/MindCube_tinybench.jsonl \
    --mindcube-root dataset/mindcube \
    --existing dataset/crossviewQA_train_rl_fixed.jsonl \
    --num-new 3000 \
    --output dataset/crossviewQA_train_rl_4k.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def scene_of(image_path: str) -> str:
    """e.g. 'other_all_image/among/shoe_225/front_001.jpg' -> 'among/shoe_225'."""
    parts = image_path.split("/")
    return "/".join(parts[1:3]) if len(parts) >= 3 else image_path


def category_of(image_paths: List[str]) -> Optional[str]:
    for tag in ("among", "around", "rotation"):
        if any(f"/{tag}/" in p for p in image_paths):
            return tag
    return None


def to_rl_example(item: Dict[str, Any], mindcube_root: str) -> Dict[str, Any]:
    images = [os.path.join(mindcube_root, "data", p) for p in item["images"]]
    question = item["question"]
    answer = item["gt_answer"]

    user_content = (
        f"Question: {question}\n\n"
        f"{'<image>' * len(images)}\n\n"
        f"Please look at the images and answer the question."
    )

    return {
        "images": images,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "solution": f"<answer> {answer} </answer>",
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mindcube-train", default="dataset/mindcube/data/raw/MindCube_train.jsonl")
    p.add_argument("--mindcube-tinybench", default="dataset/mindcube/data/raw/MindCube_tinybench.jsonl")
    p.add_argument("--mindcube-root", default="dataset/mindcube",
                    help="Root dir containing MindCube's data/ subfolder")
    p.add_argument("--existing", default="dataset/crossviewQA_train_rl_fixed.jsonl",
                    help="Existing RL training file to merge with (schema: images/messages/solution)")
    p.add_argument("--num-new", type=int, default=3000, help="Target number of NEW examples to add")
    p.add_argument("--output", default="dataset/crossviewQA_train_rl_4k.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verify-images", action="store_true", default=True)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading MindCube train pool: {args.mindcube_train}")
    train_items = load_jsonl(Path(args.mindcube_train))
    print(f"  {len(train_items)} raw examples")

    print(f"Loading tinybench (eval) scenes to exclude: {args.mindcube_tinybench}")
    tb_items = load_jsonl(Path(args.mindcube_tinybench))
    tb_scenes = {scene_of(p) for item in tb_items for p in item["images"]}
    print(f"  {len(tb_scenes)} eval scenes to avoid")

    def dedup_key(ex: Dict[str, Any]) -> tuple:
        # Full question text (not just the answer letter) + images, so two
        # different questions about the same scene are never treated as dupes.
        user_content = next((m["content"] for m in ex.get("messages", []) if m.get("role") == "user"), "")
        return (tuple(ex.get("images", [])), user_content, ex.get("solution", ""))

    existing_path = Path(args.existing)
    existing_examples: List[Dict[str, Any]] = []
    existing_keys = set()
    if existing_path.exists():
        existing_examples = load_jsonl(existing_path)
        for ex in existing_examples:
            existing_keys.add(dedup_key(ex))
        print(f"Loaded {len(existing_examples)} existing examples from {existing_path}")
    else:
        print(f"WARNING: existing file not found ({existing_path}); output will contain only new examples")

    # Filter candidate pool: drop tinybench-leaking scenes and missing images.
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    dropped_leak = dropped_missing = 0
    for item in train_items:
        scenes = {scene_of(p) for p in item["images"]}
        if scenes & tb_scenes:
            dropped_leak += 1
            continue

        if args.verify_images:
            missing = [
                p for p in item["images"]
                if not os.path.exists(os.path.join(args.mindcube_root, "data", p))
            ]
            if missing:
                dropped_missing += 1
                continue

        cat = category_of(item["images"]) or "unknown"
        by_category[cat].append(item)

    print(f"Filtered pool: dropped {dropped_leak} (eval leakage), {dropped_missing} (missing images)")
    for cat, items in by_category.items():
        print(f"  candidates in '{cat}': {len(items)}")

    # Rebalance: take (almost) all of the scarcer categories, fill the rest with 'among'.
    scarce_cats = [c for c in by_category if c != "among"]
    new_examples_raw: List[Dict[str, Any]] = []
    for cat in scarce_cats:
        pool = by_category[cat][:]
        rng.shuffle(pool)
        new_examples_raw.extend(pool)

    remaining_budget = max(args.num_new - len(new_examples_raw), 0)
    among_pool = by_category.get("among", [])[:]
    rng.shuffle(among_pool)
    new_examples_raw.extend(among_pool[:remaining_budget])

    if len(new_examples_raw) > args.num_new:
        rng.shuffle(new_examples_raw)
        new_examples_raw = new_examples_raw[:args.num_new]

    # Convert + final de-dup (against existing file and within the new batch itself),
    # keyed on images + full question text + answer so different questions about
    # the same scene are never conflated.
    new_examples: List[Dict[str, Any]] = []
    seen_keys = set(existing_keys)
    skipped_dupe = 0
    for item in new_examples_raw:
        rl_ex = to_rl_example(item, args.mindcube_root)
        key = dedup_key(rl_ex)
        if key in seen_keys:
            skipped_dupe += 1
            continue
        seen_keys.add(key)
        new_examples.append(rl_ex)

    print(f"\nConverted {len(new_examples)} new examples (skipped {skipped_dupe} exact duplicates)")

    cat_counter = Counter(category_of(ex["images"]) or "unknown" for ex in new_examples)
    letter_counter = Counter(ex["solution"].split()[1] for ex in new_examples)
    print(f"  New-examples category mix: {dict(cat_counter)}")
    print(f"  New-examples answer-letter mix: {dict(letter_counter)}")

    combined = existing_examples + new_examples
    rng.shuffle(combined)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in combined:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nDone. Combined dataset: {len(combined)} examples ({len(existing_examples)} existing + {len(new_examples)} new)")
    print(f"Output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
