import os
from pathlib import Path

import pytest

from spagent.tools import OrientAnythingTool


def test_orient_anything_tool():
    """
    Real test for OrientAnythingTool.

    Run with:
    RUN_REAL_ORIENT_ANYTHING_TEST=1 ORIENT_ANYTHING_DEVICE=cuda:0 \
    python -m pytest test/orient_anything/test_orient_anything_tool.py -s
    """
    if os.getenv("RUN_REAL_ORIENT_ANYTHING_TEST") != "1":
        pytest.skip("Skipping real Orient Anything test. Set RUN_REAL_ORIENT_ANYTHING_TEST=1 to enable.")

    image_path = os.getenv(
        "ORIENT_ANYTHING_TEST_IMAGE",
        "assets/example.png"
    )
    repo_root = os.getenv(
        "ORIENT_ANYTHING_REPO_ROOT",
        "./Orient-Anything"
    )
    device = os.getenv("ORIENT_ANYTHING_DEVICE", "cuda:0")
    model_size = os.getenv("ORIENT_ANYTHING_MODEL_SIZE", "large")

    image_path = str(Path(image_path).resolve())
    repo_root = str(Path(repo_root).resolve())

    assert Path(image_path).exists(), f"Test image not found: {image_path}"
    assert Path(repo_root).exists(), f"Orient-Anything repo not found: {repo_root}"

    tool = OrientAnythingTool(
        use_mock=False,
        repo_root=repo_root,
        device=device,
    )

    result = tool.call(
        image_path=image_path,
        model_size=model_size,
        use_tta=False,
        remove_background=True,
        device=device,
    )

    print("\n=== OrientAnythingTool result ===")
    print(result)

    assert isinstance(result, dict)
    assert "success" in result

    if not result["success"]:
        raise AssertionError(f"Tool call failed: {result}")

    assert "result" in result
    assert "summary" in result

    pred = result["result"]
    assert "azimuth" in pred
    assert "polar" in pred
    assert "rotation" in pred
    assert "confidence" in pred

    assert isinstance(pred["azimuth"], (int, float))
    assert isinstance(pred["polar"], (int, float))
    assert isinstance(pred["rotation"], (int, float))
    assert isinstance(pred["confidence"], (int, float))