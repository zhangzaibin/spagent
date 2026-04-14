import os
import sys
from pathlib import Path

import pytest


def _find_repo_root() -> Path:
    """
    Current file:
        spagent/test/yolo26/test_yolo26_tool_real.py

    Repo root:
        spagent/
    """
    return Path(__file__).resolve().parents[2]


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_YOLO26_TEST", "0") != "1",
    reason="Real YOLO26 integration test is disabled by default."
)
def test_yolo26_tool_real_detection():
    pytest.importorskip("ultralytics")
    pytest.importorskip("cv2")

    repo_root = _find_repo_root()
    sys.path.insert(0, str(repo_root))

    image_path = repo_root / "assets" / "example.png"
    output_dir = repo_root / "test" / "yolo26" / "outputs" / "yolo26"

    default_model_path = repo_root / "weights" / "yolo26n.pt"
    model_path = Path(os.environ.get("YOLO26_MODEL_PATH", str(default_model_path)))
    device = os.environ.get("YOLO26_DEVICE", "cpu")

    assert image_path.exists(), f"Test image not found: {image_path}"
    assert model_path.exists(), (
        f"YOLO26 weights not found: {model_path}\n"
        f"Please place weights at: {default_model_path}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    from spagent.tools.yolo26_tool import YOLO26Tool

    tool = YOLO26Tool(
        model_path=str(model_path),
        device=device,
        conf=0.25,
        iou=0.45,
        max_det=100,
        save_annotated=True,
        output_dir=str(output_dir),
    )

    result = tool.call(
        image_path=str(image_path),
        conf=0.25,
        save_annotated=True,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert result.get("success") is True, f"Tool failed: {result}"

    assert "result" in result, "Missing 'result' field"
    inner = result["result"]
    assert isinstance(inner, dict), f"Expected dict in result, got {type(inner)}"

    assert inner.get("image_path") == str(image_path)
    assert "num_detections" in inner
    assert "detections" in inner

    assert isinstance(inner["num_detections"], int)
    assert isinstance(inner["detections"], list)

    assert inner["num_detections"] > 0, f"No detections found: {result}"

    for det in inner["detections"]:
        assert isinstance(det, dict)
        assert "bbox_xyxy" in det
        assert "class_id" in det
        assert "class_name" in det
        assert "confidence" in det

        bbox = det["bbox_xyxy"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(x, (int, float)) for x in bbox)

        assert isinstance(det["class_id"], int)
        assert isinstance(det["class_name"], str)
        assert isinstance(det["confidence"], (int, float))

    class_names = [det["class_name"] for det in inner["detections"]]
    assert any(name in class_names for name in ["bus", "person", "car", "truck"]), class_names

    output_path = result.get("output_path")
    assert output_path is not None, f"Expected output_path, got: {result}"
    assert Path(output_path).exists(), f"Annotated output image not found: {output_path}"

    assert "summary" in result
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0