#!/usr/bin/env python3
"""
Smoke test for the Pi3X offline rendering pipeline.
Creates a synthetic .npz cache and exercises Pi3XOfflineTool.call().
No Pi3X model or network access required.
"""
import sys, os, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pathlib import Path

from spagent.tools.pi3x_tool import extract_scene_id

test_image = "assets/example.png"
scene_id = extract_scene_id(test_image)
print(f"scene_id = {scene_id!r}")

cache_dir = Path("dataset/pi3x_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N = 50_000
points = rng.standard_normal((N, 3)).astype(np.float32)
colors = np.clip(rng.random((N, 3)), 0, 1).astype(np.float32)
camera_poses = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
for i in range(4):
    camera_poses[i, 0, 3] = float(i) * 0.5

npz_path = cache_dir / f"{scene_id}.npz"
np.savez_compressed(
    npz_path,
    points=points, colors=colors,
    camera_poses=camera_poses,
    image_paths=np.array([test_image], dtype=object),
)
print(f"Saved synthetic cache: {npz_path}")

os.environ["PI3X_CACHE_DIR"] = str(cache_dir)
from spagent.tools.pi3x_offline_tool import Pi3XOfflineTool

tool = Pi3XOfflineTool(cache_dir=str(cache_dir))
print(f"Tool name: {tool.name}")

# Test 1: azimuth=45
result = tool.call(image_path=[test_image], azimuth_angle=45, elevation_angle=0)
assert result.get('success'), f"Test 1 failed: {result.get('error')}"
assert result['points_count'] == N
views = result['result']['camera_views']
assert len(views) == 1
raw = base64.b64decode(views[0]['image'])
png_magic = b'\x89PNG'
assert raw[:4] == png_magic, f"Not a PNG! Magic bytes: {raw[:4]}"
print(f"Test 1 PASSED: azimuth=45 -> image {len(raw)} bytes (PNG)")

# Test 2: elevation=60 (top view)
result2 = tool.call(image_path=[test_image], azimuth_angle=0, elevation_angle=60)
assert result2.get('success'), f"Test 2 failed: {result2.get('error')}"
print(f"Test 2 PASSED: elevation=60 (top view)")

# Test 3: missing cache
os.environ["PI3X_CACHE_DIR"] = "/nonexistent_dir"
tool_bad = Pi3XOfflineTool(cache_dir="/nonexistent_dir")
result3 = tool_bad.call(image_path=[test_image], azimuth_angle=0, elevation_angle=0)
assert not result3.get('success'), "Test 3 should fail with missing cache"
assert 'No cached point cloud' in result3['error']
print(f"Test 3 PASSED: missing cache returns proper error")

print("\nAll smoke tests PASSED.")
