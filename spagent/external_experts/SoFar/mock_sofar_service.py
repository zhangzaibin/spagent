"""
Mock SoFar service for testing without GPU.

Returns plausible dummy outputs for 6-DoF pose estimation.
"""

from __future__ import annotations

import random
from typing import Optional


class MockSoFarService:
    """Deterministic (seeded by instruction) mock for offline testing."""

    def infer(
        self,
        image_path: str,
        instruction: str,
        camera_intrinsics: Optional[dict] = None,
    ) -> dict:
        """Return mock pose estimation results.

        Args:
            image_path: Path to the scene image (not read in mock mode).
            instruction: Manipulation instruction (used as RNG seed).
            camera_intrinsics: Optional camera intrinsics (ignored in mock).

        Returns:
            Dict with mock pose estimation results.
        """
        rng = random.Random(hash(instruction) & 0xFFFFFFFF)
        z = rng.uniform(0.3, 1.2)
        return {
            "bbox": [
                rng.randint(100, 200), rng.randint(100, 200),
                rng.randint(300, 500), rng.randint(300, 500),
            ],
            "position": {
                "x": round(rng.uniform(-0.3, 0.3), 4),
                "y": round(rng.uniform(-0.2, 0.2), 4),
                "z": round(z, 4),
            },
            "quaternion": {
                "w": round(rng.uniform(0.7, 1.0), 4),
                "x": round(rng.uniform(-0.3, 0.3), 4),
                "y": round(rng.uniform(-0.3, 0.3), 4),
                "z": round(rng.uniform(-0.3, 0.3), 4),
            },
            "approach_vector": [
                round(rng.uniform(-0.1, 0.1), 3),
                round(rng.uniform(-0.1, 0.1), 3),
                -1.0,
            ],
            "spatial_description": (
                f"Target object detected at approximately {z:.2f}m from camera. "
                f"Orientation suggests top-down approach is optimal."
            ),
            "confidence": round(rng.uniform(0.72, 0.96), 3),
        }
