"""
Mock D4RT service for testing without GPU.

Returns plausible dummy outputs for depth, camera, and tracking tasks.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional


class MockD4RTService:
    """Mock D4RT service returning deterministic dummy results."""

    def reconstruct(
        self,
        frame_dir: str,
        task: str = "full_4d",
        query_points: Optional[List[List[int]]] = None,
        output_dir: str = "outputs/d4rt",
        max_frames: int = -1,
    ) -> dict:
        """Return mock reconstruction results.

        Args:
            frame_dir: Path to frame directory (frame count is read if accessible).
            task: One of 'depth_and_camera', 'tracking', 'full_4d'.
            query_points: List of [x, y] pixel coordinates to track.
            output_dir: Directory for output files.
            max_frames: Maximum frames to process (-1 for all).

        Returns:
            Dict with mock reconstruction results.
        """
        try:
            frames = sorted(Path(frame_dir).glob("*.jpg")) + \
                     sorted(Path(frame_dir).glob("*.png"))
            n = len(sorted(frames))
        except Exception:
            n = 8
        if max_frames > 0:
            n = min(n, max_frames)
        n = max(n, 1)

        result = {"num_frames": n, "output_dir": output_dir}

        if task in ("depth_and_camera", "full_4d"):
            result["depth_paths"] = [
                f"{output_dir}/depth_{i:04d}.npy" for i in range(n)
            ]
            result["camera_poses"] = [
                {
                    "frame": i,
                    "extrinsic": [
                        [1, 0, 0, i * 0.05],
                        [0, 1, 0, 0.0],
                        [0, 0, 1, 0.0],
                        [0, 0, 0, 1.0],
                    ],
                    "intrinsic": {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0},
                }
                for i in range(n)
            ]

        if task in ("tracking", "full_4d"):
            pts = query_points or [[100, 200]]
            result["trajectories"] = [
                {
                    "query_point": pt,
                    "track_3d": [
                        [
                            pt[0] * 0.001 + i * 0.002,
                            pt[1] * 0.001,
                            1.5 + random.uniform(-0.05, 0.05),
                        ]
                        for i in range(n)
                    ],
                }
                for pt in pts
            ]

        return result
