from __future__ import annotations

import os
from pathlib import Path

from spagent.tools.map_anything_tool import MapAnythingTool


def test_map_anything_tool_real() -> None:
    server_url = os.environ.get("MAPANYTHING_SERVER_URL", "http://127.0.0.1:20033")
    image1 = os.environ.get("MAPANYTHING_TEST_IMAGE_1")
    image2 = os.environ.get("MAPANYTHING_TEST_IMAGE_2")

    assert image1 is not None, "Please set MAPANYTHING_TEST_IMAGE_1"
    assert image2 is not None, "Please set MAPANYTHING_TEST_IMAGE_2"
    assert Path(image1).exists(), f"Image not found: {image1}"
    assert Path(image2).exists(), f"Image not found: {image2}"

    tool = MapAnythingTool(
        use_mock=False,
        server_url=server_url,
        timeout=1800,
    )

    result = tool.call(
        image_paths=[image1, image2],
        memory_efficient_inference=True,
        minibatch_size=None,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
        save_outputs=True,
    )

    assert result["success"] is True
    assert "output_dir" in result
    assert Path(result["output_dir"]).exists()
    assert result["num_views"] >= 1

    print("Tool output_dir:", result["output_dir"])
    print("Tool num_views:", result["num_views"])
    print("Tool summaries:", result["summaries"])