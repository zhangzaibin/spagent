"""
Tests for WildDet3DTool.

Mock tests run without a WildDet3D server. Set WILDDET3D_REAL_TEST=1 to run the
optional live-server smoke test.
"""

import os
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "spagent"))

from spagent.tools import WildDet3DTool
from core.tool_result import ToolResult, validate_payload


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "wilddet3d_sample.jpg"
    Image.new("RGB", (160, 120), color=(80, 120, 170)).save(path, format="JPEG")
    return str(path)


def test_wilddet3d_tool_is_exported():
    assert WildDet3DTool is not None


def test_wilddet3d_is_in_tool_selection_guide():
    from core.prompts import build_tool_selection_guide

    guide = build_tool_selection_guide({"wilddet3d_tool"})
    assert "## 3D & Spatial" in guide
    assert "wilddet3d_tool" in guide


def test_wilddet3d_schema_contains_prompt_inputs():
    tool = WildDet3DTool(use_mock=True)
    params = tool.parameters

    assert "image_path" in params["required"]
    assert "text_prompt" in params["properties"]
    assert "boxes" in params["properties"]
    assert "points" in params["properties"]
    assert "anyOf" in params
    assert tool.name == "wilddet3d_tool"
    assert params["properties"]["boxes"]["minItems"] == 1
    assert params["properties"]["boxes"]["maxItems"] == 20
    assert params["properties"]["points"]["maxItems"] == 100


def test_wilddet3d_mock_text_prompt(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="chair",
        score_threshold=0.3,
        save_visualization=True,
    )

    assert result["success"] is True
    assert isinstance(result, ToolResult)
    assert validate_payload(result, "detection")[0]
    assert result["boxes_2d"]
    assert result["boxes_3d"]
    assert result["scores"]
    assert result["class_names"] == ["chair"]
    assert result["boxes2d"] == result["boxes_2d"]
    assert result["boxes3d"] == result["boxes_3d"]
    assert result["num_detections"] == len(result["boxes_2d"])
    assert result["result"]["boxes3d"] == result["boxes_3d"]
    assert result["output_path"] is not None
    assert os.path.exists(result["output_path"])
    assert result["depth_path"] is not None
    assert os.path.exists(result["depth_path"])


def test_wilddet3d_mock_box_prompt(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        boxes=[[10, 15, 80, 95]],
        save_visualization=True,
    )

    assert result["success"] is True
    assert result["boxes_2d"] == [[10.0, 15.0, 80.0, 95.0]]
    assert len(result["boxes_3d"][0]) == 10
    assert result["class_names"] == ["object"]


def test_wilddet3d_mock_point_prompt(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        points=[[60, 50, 1]],
        save_visualization=False,
    )

    assert result["success"] is True
    assert result["boxes_3d"]
    assert result["output_path"] is None


def test_wilddet3d_prompt_precedence_matches_legacy_api(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="chair",
        boxes=[[10, 15, 80, 95]],
        points=[[60, 50, 1]],
    )

    assert result["success"] is True
    assert result["boxes_2d"] == [[10.0, 15.0, 80.0, 95.0]]
    assert result["class_names"] == ["object"]


def test_wilddet3d_mock_outputs_are_unique(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    first = tool.call(image_path=sample_image_path, text_prompt="chair")
    second = tool.call(image_path=sample_image_path, text_prompt="chair")

    assert first["output_path"] != second["output_path"]
    assert first["depth_path"] != second["depth_path"]


def test_wilddet3d_mock_text_category_parsing_matches_server(tmp_path, sample_image_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=sample_image_path, text_prompt="chair. table, lamp")

    assert result["class_names"] == ["chair", "table", "lamp"]
    assert all(
        0 <= x1 < x2 <= 160 and 0 <= y1 < y2 <= 120
        for x1, y1, x2, y2 in result["boxes_2d"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score_threshold": -0.1}, "score_threshold"),
        ({"boxes": [[20, 20, 10, 40]]}, "non-empty pixel xyxy"),
        ({"points": [[20, 20, 0]]}, "foreground label 1"),
        ({"points": [[20, 20, 2]]}, "labels must be 0"),
    ],
)
def test_wilddet3d_rejects_invalid_numeric_inputs(sample_image_path, kwargs, message):
    tool = WildDet3DTool(use_mock=True)
    call_kwargs = {"image_path": sample_image_path, "text_prompt": "chair"}
    call_kwargs.update(kwargs)

    result = tool.call(**call_kwargs)

    assert result["success"] is False
    assert message in result["error"]


def test_wilddet3d_local_backend_is_lazy(monkeypatch):
    monkeypatch.delenv("WILDDET3D_SERVER_URL", raising=False)
    monkeypatch.delenv("WILDDET3D_CHECKPOINT", raising=False)

    tool = WildDet3DTool(use_mock=False, server_url=None)

    assert tool._backend_kind == "local"
    assert tool._client is None


def test_wilddet3d_rejects_missing_image(tmp_path):
    tool = WildDet3DTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=str(tmp_path / "missing.jpg"), text_prompt="chair")

    assert result["success"] is False
    assert "Image file not found" in result["error"]


def test_wilddet3d_rejects_empty_prompt(sample_image_path):
    tool = WildDet3DTool(use_mock=True)

    result = tool.call(image_path=sample_image_path, text_prompt=" ")

    assert result["success"] is False
    assert "Provide at least one prompt" in result["error"]


def test_wilddet3d_server_filters_on_combined_score_and_keeps_3d_scores_aligned():
    from spagent.external_experts.WildDet3D.wilddet3d_server import _filter_by_score

    filtered = _filter_by_score(
        boxes_2d=np.array([[1, 2, 10, 12], [3, 4, 20, 24]], dtype=np.float32),
        boxes_3d=np.arange(20, dtype=np.float32).reshape(2, 10),
        scores=np.array([0.2, 0.8], dtype=np.float32),
        scores_2d=np.array([0.9, 0.9], dtype=np.float32),
        scores_3d=np.array([0.7, 0.6], dtype=np.float32),
        class_ids=np.array([0, 1]),
        threshold=0.3,
    )

    boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids = filtered
    assert boxes_2d.shape == (1, 4)
    assert boxes_3d.shape == (1, 10)
    assert scores.tolist() == pytest.approx([0.8])
    assert scores_2d.tolist() == pytest.approx([0.9])
    assert scores_3d.tolist() == pytest.approx([0.6])
    assert class_ids.tolist() == [1]


def test_wilddet3d_server_maps_class_ids_to_detection_labels():
    from spagent.external_experts.WildDet3D.wilddet3d_server import _detection_class_names

    labels = _detection_class_names(["chair", "table"], np.array([1, 0, 1]), 3)
    assert labels == ["table", "chair", "table"]


def test_wilddet3d_server_aggregates_multiple_box_prompts(monkeypatch):
    torch = pytest.importorskip("torch")
    from spagent.external_experts.WildDet3D import wilddet3d_server as server

    calls = []

    class FakeModel:
        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def __call__(self, **kwargs):
            box = kwargs["input_boxes"][0]
            calls.append(box)
            boxes_2d = [torch.tensor([box], dtype=torch.float32)]
            boxes_3d = [torch.tensor([[0, 0, 2, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)]
            scores = [torch.tensor([0.9], dtype=torch.float32)]
            return (
                boxes_2d,
                boxes_3d,
                scores,
                scores,
                scores,
                [torch.tensor([0])],
                [torch.ones((1, 8, 8), dtype=torch.float32)],
            )

    monkeypatch.setattr(server, "model", FakeModel())
    monkeypatch.setattr(
        server,
        "preprocess_fn",
        lambda _image: {
            "images": torch.zeros((1, 3, 8, 8)),
            "intrinsics": torch.eye(3),
            "input_hw": (8, 8),
            "original_hw": (8, 8),
            "padding": (0, 0, 0, 0),
            "original_intrinsics": torch.eye(3),
        },
    )
    boxes = [[0, 0, 4, 4], [4, 4, 8, 8]]

    outputs = server._run_wilddet3d(
        image=Image.new("RGB", (8, 8)),
        text_prompt=None,
        boxes=boxes,
        points=None,
        score_threshold=0.3,
    )

    assert calls == boxes
    assert outputs[0].shape == (2, 4)
    assert outputs[1].shape == (2, 10)
    assert outputs[2].tolist() == pytest.approx([0.9, 0.9])


def test_wilddet3d_server_validates_direct_http_prompts():
    from spagent.external_experts.WildDet3D.wilddet3d_server import (
        _validate_boxes,
        _validate_points,
        _validate_threshold,
    )

    with pytest.raises(ValueError, match="inside"):
        _validate_boxes([[0, 0, 200, 20]], (160, 120))
    with pytest.raises(ValueError, match="foreground"):
        _validate_points([[10, 10, 0]], (160, 120))
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _validate_threshold(float("nan"))


def test_wilddet3d_server_serializes_bfloat16_tensors():
    torch = pytest.importorskip("torch")
    from spagent.external_experts.WildDet3D.wilddet3d_server import _to_numpy

    converted = _to_numpy(torch.tensor([0.25, 0.5], dtype=torch.bfloat16))

    assert converted.dtype == np.float32
    assert converted.tolist() == pytest.approx([0.25, 0.5])


@pytest.mark.skipif(
    os.environ.get("WILDDET3D_REAL_TEST") != "1",
    reason="Set WILDDET3D_REAL_TEST=1 to run against a live WildDet3D server.",
)
def test_wilddet3d_real_service(sample_image_path):
    server_url = os.environ.get("WILDDET3D_SERVER_URL", "http://127.0.0.1:20027")
    tool = WildDet3DTool(use_mock=False, server_url=server_url)

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="chair",
        score_threshold=0.3,
        save_visualization=True,
    )

    assert result["success"] is True
    assert "boxes_3d" in result
