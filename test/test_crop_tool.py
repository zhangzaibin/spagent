"""Tests for CropTool."""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spagent.tools import CropTool


def _sample_image(path: Path, size=(100, 80)) -> str:
    image = Image.new("RGB", size, color=(120, 140, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 15, 70, 60], fill=(220, 60, 60))
    image.save(path)
    return str(path)


def test_crop_tool_is_exported():
    assert CropTool is not None


def test_crop_schema_contains_inputs():
    tool = CropTool()
    params = tool.parameters

    assert "image_path" in params["required"]
    for key in ["box", "boxes", "mask_path", "polygon", "padding", "relative_coords"]:
        assert key in params["properties"]
    assert "anyOf" in params


def test_crop_single_box(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg")
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, box=[10, 20, 50, 60])

    assert result["success"] is True
    assert result["box"] == [10, 20, 50, 60]
    assert result["crop_size"] == [40, 40]
    assert os.path.exists(result["output_path"])


def test_crop_relative_box(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg", size=(200, 100))
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, box=[0.25, 0.2, 0.75, 0.8], relative_coords=True)

    assert result["success"] is True
    assert result["box"] == [50, 20, 150, 80]
    assert result["crop_size"] == [100, 60]


def test_crop_padding_clamps_to_image(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg", size=(100, 80))
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, box=[5, 5, 25, 25], padding=10)

    assert result["success"] is True
    assert result["box"] == [0, 0, 35, 35]
    assert result["crop_size"] == [35, 35]


def test_crop_multiple_boxes(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg")
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, boxes=[[0, 0, 20, 20], [20, 10, 60, 50]])

    assert result["success"] is True
    assert result["mode"] == "boxes"
    assert len(result["crops"]) == 2
    assert result["crops"][0]["crop_size"] == [20, 20]
    assert result["crops"][1]["crop_size"] == [40, 40]
    assert all(os.path.exists(path) for path in result["output_paths"])


def test_crop_mask(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg")
    mask = Image.new("L", (100, 80), 0)
    ImageDraw.Draw(mask).rectangle([30, 20, 70, 50], fill=255)
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path)
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, mask_path=str(mask_path))

    assert result["success"] is True
    assert result["mode"] == "mask"
    assert result["box"] == [30, 20, 71, 51]
    assert os.path.exists(result["output_path"])
    with Image.open(result["output_path"]) as crop:
        assert crop.mode == "RGBA"


def test_crop_polygon(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg")
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, polygon=[[20, 10], [80, 20], [40, 60]])

    assert result["success"] is True
    assert result["mode"] == "polygon"
    assert result["box"] == [20, 10, 81, 61]
    assert os.path.exists(result["output_path"])
    with Image.open(result["output_path"]) as crop:
        assert crop.mode == "RGBA"


def test_crop_rejects_missing_image(tmp_path):
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=str(tmp_path / "missing.jpg"), box=[0, 0, 10, 10])

    assert result["success"] is False
    assert "Image file not found" in result["error"]


def test_crop_rejects_invalid_box(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg")
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path, box=[50, 50, 20, 20])

    assert result["success"] is False
    assert "x2 > x1" in result["error"]


def test_crop_requires_exactly_one_crop_input(tmp_path):
    image_path = _sample_image(tmp_path / "sample.jpg")
    tool = CropTool(output_dir=str(tmp_path))

    result = tool.call(image_path=image_path)

    assert result["success"] is False
    assert "exactly one crop input" in result["error"]
