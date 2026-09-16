"""
Tests for SAM3Tool.

Mock tests run without a SAM3 server. Set SAM3_REAL_TEST=1 to run the
optional live-service smoke test against a running SAM3 server.
"""

import os
import base64
import io
import sys
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "spagent"))

from spagent.tools import SAM3Tool
from core.tool_result import ToolResult, validate_payload


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


def test_sam3_is_in_tool_selection_guide():
    from core.prompts import build_tool_selection_guide

    guide = build_tool_selection_guide({"sam3_concept_segmentation_tool"})
    assert "## 2D Perception" in guide
    assert "sam3_concept_segmentation_tool" in guide


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
    assert isinstance(result, ToolResult)
    assert validate_payload(result, "segmentation")[0]
    assert result["task"] == "image"
    assert result["output_path"] is not None
    assert os.path.exists(result["output_path"])
    assert result["masks"]
    assert "boxes" not in result
    assert result["result"]["boxes"]
    assert result["scores"]


def test_sam3_mock_image_outputs_are_unique(sample_image_path):
    tool = SAM3Tool(use_mock=True)

    first = tool.call(image_path=sample_image_path, text_prompt="object")
    second = tool.call(image_path=sample_image_path, text_prompt="object")

    assert first["mask_path"] != second["mask_path"]
    assert first["output_path"] != second["output_path"]
    assert Path(first["mask_path"]).is_file()
    assert Path(second["mask_path"]).is_file()


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
    assert isinstance(result, ToolResult)
    assert validate_payload(result, "segmentation")[0]
    assert result["video_path"] is not None
    assert os.path.exists(result["video_path"])
    assert result["frames"] == 4
    assert "boxes" not in result
    assert len(result["masks"]) == 4
    assert [record["frame_index"] for record in result["masks"]] == [0, 1, 2, 3]
    assert all(record["mask_paths"] for record in result["masks"])
    assert all(
        Path(mask_path).is_file()
        for record in result["masks"]
        for mask_path in record["mask_paths"]
    )


def test_sam3_mock_jpeg_frame_directory(tmp_path):
    pytest.importorskip("cv2")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(3):
        Image.new("RGB", (64, 48), color=(index * 40, 80, 120)).save(
            frame_dir / f"{index:05d}.jpg"
        )

    result = SAM3Tool(use_mock=True).call(
        image_path=str(frame_dir),
        text_prompt="object",
        task="auto",
    )

    assert result["success"] is True
    assert result["task"] == "video"
    assert result["frames"] == 3
    assert Path(result["output_path"]).is_file()
    assert len(result["masks"]) == 3


def test_sam3_mock_video_without_overlay_creates_only_masks(tmp_path):
    pytest.importorskip("cv2")
    from spagent.external_experts.SAM3.mock_sam3_service import MockSAM3Service

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (32, 24), color=(index * 40, 80, 120)).save(frame_dir / f"{index}.jpg")
    output_dir = tmp_path / "out"

    result = MockSAM3Service(output_dir=str(output_dir)).infer_video(
        video_path=str(frame_dir),
        text_prompt="object",
        save_overlay=False,
    )

    assert result["success"] is True
    assert result["output_path"] is None
    assert result["frame_mask_paths"]
    assert list(output_dir.glob("*.mp4")) == []


def test_sam3_mock_rejects_out_of_range_frame_index(tmp_path):
    pytest.importorskip("cv2")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    Image.new("RGB", (32, 24), color="red").save(frame_dir / "0.jpg")

    result = SAM3Tool(use_mock=True).call(
        image_path=str(frame_dir),
        text_prompt="object",
        frame_index=1,
    )

    assert result["success"] is False
    assert "outside 1 frames" in result["error"]


def test_sam3_client_encodes_frame_directory_in_numeric_order(tmp_path):
    pytest.importorskip("cv2")
    from spagent.external_experts.SAM3.sam3_client import SAM3Client

    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    Image.new("RGB", (16, 12), color="red").save(frame_dir / "10.jpg")
    Image.new("RGB", (16, 12), color="blue").save(frame_dir / "2.jpg")

    encoded = SAM3Client._encode_frame_directory(frame_dir)
    assert len(encoded) == 2
    assert all(base64.b64decode(frame) for frame in encoded)


def test_sam3_client_rejects_invalid_mask(tmp_path):
    pytest.importorskip("cv2")
    from spagent.external_experts.SAM3.sam3_client import SAM3Client

    source = np.zeros((12, 16, 3), dtype=np.uint8)
    client = SAM3Client(output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="invalid PNG mask"):
        client._save_image_outputs(
            image_path="sample.jpg",
            image=source,
            result={"masks": [{"mask": base64.b64encode(b"not png").decode("ascii")}]},
            save_overlay=True,
        )


@pytest.mark.parametrize("video", [None, base64.b64encode(b"not mp4").decode("ascii")])
def test_sam3_client_rejects_missing_or_invalid_overlay_video(tmp_path, video):
    pytest.importorskip("cv2")
    from spagent.external_experts.SAM3.sam3_client import SAM3Client

    client = SAM3Client(output_dir=str(tmp_path / "out"))
    with pytest.raises(ValueError, match="overlay"):
        client._save_video_outputs(
            video_path="sample.mp4",
            result={"success": True, "video": video, "frame_masks": []},
            save_overlay=True,
        )


def test_sam3_server_accepts_uploaded_jpeg_frames(monkeypatch):
    pytest.importorskip("cv2")
    from spagent.external_experts.SAM3 import sam3_server

    observed = {}

    class FakeVideoPredictor:
        def handle_request(self, request):
            if request["type"] == "start_session":
                source = Path(request["resource_path"])
                observed["source"] = str(source)
                observed["frames"] = sorted(path.name for path in source.glob("*.jpg"))
                return {"session_id": "test-session"}
            if request["type"] == "add_prompt":
                return {
                    "frame_index": 0,
                    "outputs": {"out_binary_masks": np.ones((1, 24, 32), dtype=bool)},
                }
            if request["type"] == "close_session":
                return {"is_success": True}
            raise AssertionError(request)

        def handle_stream_request(self, request):
            assert request["type"] == "propagate_in_video"
            yield {
                "frame_index": 1,
                "outputs": {"out_binary_masks": np.ones((1, 24, 32), dtype=bool)},
            }

    monkeypatch.setattr(sam3_server, "video_predictor", FakeVideoPredictor())
    encoded_frames = []
    for color in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24), color=color).save(buffer, format="JPEG")
        encoded_frames.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    response = sam3_server.app.test_client().post(
        "/infer_video",
        json={
            "frames": encoded_frames,
            "text_prompt": "object",
            "frame_index": 0,
            "score_threshold": 0.1,
            "max_instances": 2,
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["frames"] == 2
    assert base64.b64decode(payload["video"])
    assert observed["frames"] == ["000000.jpg", "000001.jpg"]
    assert not Path(observed["source"]).exists()

    no_overlay_response = sam3_server.app.test_client().post(
        "/infer_video",
        json={
            "frames": encoded_frames,
            "text_prompt": "object",
            "frame_index": 0,
            "score_threshold": 0.1,
            "max_instances": 2,
            "save_overlay": False,
        },
    )
    no_overlay_payload = no_overlay_response.get_json()
    assert no_overlay_response.status_code == 200
    assert no_overlay_payload["success"] is True
    assert no_overlay_payload["video"] is None
    assert no_overlay_payload["frame_masks"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score_threshold": 1.1}, "score_threshold"),
        ({"max_instances": 0}, "max_instances"),
        ({"task": "video", "frame_index": -1}, "frame_index"),
    ],
)
def test_sam3_rejects_invalid_numeric_parameters(sample_image_path, kwargs, message):
    result = SAM3Tool(use_mock=True).call(
        image_path=sample_image_path,
        text_prompt="object",
        **kwargs,
    )
    assert result["success"] is False
    assert message in result["error"]


def test_sam3_server_extracts_official_video_mask_key():
    pytest.importorskip("cv2")
    from spagent.external_experts.SAM3.sam3_server import _extract_video_masks

    official_output = {
        "out_binary_masks": np.array(
            [[[False, True], [True, False]]],
            dtype=bool,
        )
    }
    masks = _extract_video_masks(official_output)

    assert len(masks) == 1
    assert masks[0].dtype == np.uint8
    assert masks[0].tolist() == [[0, 255], [255, 0]]


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
    assert result["result"]["boxes"] is not None
