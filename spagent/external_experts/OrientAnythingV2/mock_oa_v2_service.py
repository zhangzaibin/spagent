"""
Mock service for Orient Anything V2.

Returns plausible dummy outputs - no GPU or model weights required.
Use --use_mock during development and CI.
"""

from __future__ import annotations

import math
import random
from typing import Optional


class MockOrientAnythingV2Service:
    """Deterministic (seeded by object_category) mock for offline testing."""

    SYMMETRY_TYPES = ["none", "bilateral", "rotational"]

    def infer(
        self,
        image_path: str,
        object_category: str,
        task: str = "orientation",
        image_path2: Optional[str] = None,
    ) -> dict:
        """Return mock inference results.

        Args:
            image_path: Path to the input image (not read in mock mode).
            object_category: Object category string (used as RNG seed).
            task: One of 'orientation', 'symmetry', 'relative_rotation'.
            image_path2: Path to second image (not read in mock mode).

        Returns:
            Dict with task-specific result keys.
        """
        rng = random.Random(hash(object_category) & 0xFFFFFFFF)

        if task == "orientation":
            return self._mock_orientation(rng)
        elif task == "symmetry":
            return self._mock_symmetry(rng)
        elif task == "relative_rotation":
            return self._mock_relative_rotation(rng)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _mock_orientation(self, rng: random.Random) -> dict:
        yaw = rng.uniform(-180.0, 180.0)
        pitch = rng.uniform(-30.0, 30.0)
        roll = rng.uniform(-15.0, 15.0)
        yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)
        front = [
            round(math.cos(pitch_r) * math.sin(yaw_r), 4),
            round(-math.sin(pitch_r), 4),
            round(math.cos(pitch_r) * math.cos(yaw_r), 4),
        ]
        return {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
            "confidence": round(rng.uniform(0.75, 0.98), 3),
            "front_vector": front,
        }

    def _mock_symmetry(self, rng: random.Random) -> dict:
        sym_type = rng.choice(self.SYMMETRY_TYPES)
        axis = None
        if sym_type == "bilateral":
            axis = [round(rng.uniform(-1, 1), 3) for _ in range(3)]
        elif sym_type == "rotational":
            axis = [0.0, 1.0, 0.0]
        return {
            "symmetry_type": sym_type,
            "axis": axis,
            "confidence": round(rng.uniform(0.70, 0.96), 3),
        }

    def _mock_relative_rotation(self, rng: random.Random) -> dict:
        angle = rng.uniform(0.0, 180.0)
        angle_r = math.radians(angle)
        c, s = math.cos(angle_r), math.sin(angle_r)
        mat = [
            [round(c, 4), 0.0, round(s, 4)],
            [0.0, 1.0, 0.0],
            [round(-s, 4), 0.0, round(c, 4)],
        ]
        return {
            "rotation_matrix": mat,
            "euler_angles": [0.0, round(angle, 2), 0.0],
            "angular_distance_deg": round(angle, 2),
        }
