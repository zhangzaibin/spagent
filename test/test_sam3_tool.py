"""
Tests for SAM3Tool.

Mock tests run without a SAM3 server. Set SAM3_REAL_TEST=1 to run the
optional live-service smoke test against a running SAM3 server.
"""

import os
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spagent.tools import SAM3Tool


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "sam3_sample.jpg"
    image = Image.new("RGB", (160, 120), color=(80, 100, 130))
    image.save(path)
    return str(path)


def test_sam3_tool_is_exported():
    assert SAM3Tool is not None


def test_sam3_schema_contains_expected_parameters():
    tool = SAM3Tool(use_mock=True)
    schema = tool.parameters

    assert schema["required"] == ["image_path", "text_prompt"]
    assert "image_path" in schema["properties"]
    assert "text_prompt" in schema["properties"]
    assert "task" in schema["properties"]


def test_sam3_mock_image_segmentation(sample_image_path):
    tool = SAM3Tool(use_mock=True)

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="blue object",
        task="image",
        score_threshold=0.4,
        max_instances=2,
    )

    assert result["success"] is True
    assert result["task"] == "image"
    assert result["output_path"] is not None
    assert os.path.exists(result["output_path"])
    assert result["masks"]
    assert result["boxes"]
    assert result["scores"]


def test_sam3_rejects_empty_prompt(sample_image_path):
    tool = SAM3Tool(use_mock=True)

    result = tool.call(image_path=sample_image_path, text_prompt="   ")

    assert result["success"] is False
    assert "text_prompt must be a non-empty string" in result["error"]


def test_sam3_rejects_missing_input():
    tool = SAM3Tool(use_mock=True)

    result = tool.call(image_path="/tmp/does_not_exist_sam3.jpg", text_prompt="object")

    assert result["success"] is False
    assert "Input file not found" in result["error"]


def test_sam3_mock_video_segmentation(tmp_path):
    cv2 = pytest.importorskip("cv2")
    video_path = tmp_path / "sam3_sample.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer is not available in this environment.")
    for idx in range(4):
        frame = (idx * 30) * np.ones((48, 64, 3), dtype="uint8")
        writer.write(frame)
    writer.release()

    tool = SAM3Tool(use_mock=True)
    result = tool.call(
        image_path=str(video_path),
        text_prompt="moving object",
        task="video",
        frame_index=0,
    )

    assert result["success"] is True
    assert result["task"] == "video"
    assert result["video_path"] is not None
    assert os.path.exists(result["video_path"])
    assert result["frames"] == 4


@pytest.mark.skipif(
    os.environ.get("SAM3_REAL_TEST") != "1",
    reason="Set SAM3_REAL_TEST=1 to run against a live SAM3 server.",
)
def test_sam3_real_service(sample_image_path):
    server_url = os.environ.get("SAM3_SERVER_URL", "http://127.0.0.1:20035")
    tool = SAM3Tool(use_mock=False, server_url=server_url)

    result = tool.call(
        image_path=sample_image_path,
        text_prompt="object",
        task="image",
        score_threshold=0.1,
        max_instances=3,
    )

    assert result["success"] is True
    assert result["boxes"] is not None
