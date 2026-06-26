"""Generate MindCube_tiny.jsonl – 20 samples per task category (seed=42)."""
import json
from collections import defaultdict
import random

random.seed(42)
PER_CAT = 20
src = "dataset/MindCube_data.jsonl"
dst = "dataset/MindCube_tiny.jsonl"

buckets = defaultdict(list)
with open(src) as f:
    for line in f:
        item = json.loads(line)
        buckets[item["task"]].append(item)

selected = []
for task in sorted(buckets):
    items = buckets[task]
    chosen = items[:PER_CAT]
    selected.extend(chosen)
    print(f"  {task}: {len(chosen)} samples")

with open(dst, "w") as f:
    for item in selected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nTotal: {len(selected)} samples written to {dst}")
