"""
Tests for InfiniDepthTool.

Mock tests run without an InfiniDepth server. Set INFINIDEPTH_REAL_TEST=1 to run
the optional live-server smoke test.
"""

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "spagent"))

from spagent.tools import InfiniDepthTool
from core.tool_result import ToolResult, validate_payload


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "infinidepth_sample.jpg"
    Image.new("RGB", (128, 96), color=(90, 130, 170)).save(path, format="JPEG")
    return str(path)


def test_infinidepth_tool_is_exported():
    assert InfiniDepthTool is not None


def test_infinidepth_is_in_tool_selection_guide():
    from core.prompts import build_tool_selection_guide

    guide = build_tool_selection_guide({"infinidepth_tool"})
    assert "## 2D Perception" in guide
    assert "infinidepth_tool" in guide


def test_infinidepth_schema_contains_inputs():
    tool = InfiniDepthTool(use_mock=True)
    params = tool.parameters

    assert "image_path" in params["required"]
    for key in ["task", "save_pcd", "upsample_ratio", "output_resolution_mode", "output_dir"]:
        assert key in params["properties"]
    assert params["properties"]["output_resolution_mode"]["default"] == "original"
    assert params["properties"]["upsample_ratio"]["maximum"] == 4


def test_infinidepth_mock_depth(tmp_path, sample_image_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=sample_image_path, upsample_ratio=2)

    assert result["success"] is True
    assert isinstance(result, ToolResult)
    assert validate_payload(result, "depth")[0]
    assert result["depth_path"] is not None
    assert result["colored_depth_path"] is not None
    assert os.path.exists(result["depth_path"])
    assert os.path.exists(result["colored_depth_path"])
    assert result["shape"] == [96, 128]
    assert result["source_shape"] == [96, 128]
    assert result["output_resolution_mode"] == "original"


def test_infinidepth_mock_upsample_reports_actual_shape(tmp_path, sample_image_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(
        image_path=sample_image_path,
        output_resolution_mode="upsample",
        upsample_ratio=2,
    )

    assert result["success"] is True
    assert result["shape"] == [192, 256]
    assert result["source_shape"] == [96, 128]
    with Image.open(result["depth_path"]) as depth:
        assert depth.size == (256, 192)


def test_infinidepth_mock_outputs_are_unique(tmp_path, sample_image_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))

    first = tool.call(image_path=sample_image_path, save_pcd=True)
    second = tool.call(image_path=sample_image_path, save_pcd=True)

    assert first["depth_path"] != second["depth_path"]
    assert first["colored_depth_path"] != second["colored_depth_path"]
    assert first["point_cloud_path"] != second["point_cloud_path"]


def test_infinidepth_mock_pcd(tmp_path, sample_image_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=sample_image_path, save_pcd=True)

    assert result["success"] is True
    assert result["point_cloud_path"] is not None
    assert os.path.exists(result["point_cloud_path"])
    assert Path(result["point_cloud_path"]).read_text().startswith("ply")


def test_infinidepth_rejects_missing_image(tmp_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=str(tmp_path / "missing.jpg"))

    assert result["success"] is False
    assert "Image file not found" in result["error"]


def test_infinidepth_rejects_unsupported_task(sample_image_path):
    tool = InfiniDepthTool(use_mock=True)

    result = tool.call(image_path=sample_image_path, task="3dgs")

    assert result["success"] is False
    assert "only supports task='depth'" in result["error"]


def test_infinidepth_rejects_invalid_upsample(sample_image_path):
    tool = InfiniDepthTool(use_mock=True)

    result = tool.call(image_path=sample_image_path, upsample_ratio=0)

    assert result["success"] is False
    assert "upsample_ratio must be an integer in [1, 4]" in result["error"]

    fractional = tool.call(image_path=sample_image_path, upsample_ratio=1.5)
    assert fractional["success"] is False
    assert "upsample_ratio must be an integer in [1, 4]" in fractional["error"]

    too_large = tool.call(image_path=sample_image_path, upsample_ratio=5)
    assert too_large["success"] is False
    assert "integer in [1, 4]" in too_large["error"]


def test_infinidepth_rejects_invalid_resolution_mode(sample_image_path):
    tool = InfiniDepthTool(use_mock=True)

    result = tool.call(image_path=sample_image_path, output_resolution_mode="specific")

    assert result["success"] is False
    assert "original' or 'upsample" in result["error"]


def test_infinidepth_server_subprocess_path(tmp_path, sample_image_path):
    from spagent.external_experts.InfiniDepth import infinidepth_server as server

    repo = tmp_path / "InfiniDepth"
    repo.mkdir()
    checkpoint = tmp_path / "infinidepth.ckpt"
    checkpoint.write_text("fake checkpoint")
    script = repo / "inference_depth.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "from PIL import Image",
                "arg = next(v for v in sys.argv if v.startswith('--depth_output_dir='))",
                "out = Path(arg.split('=', 1)[1])",
                "out.mkdir(parents=True, exist_ok=True)",
                "Image.new('L', (16, 12), color=128).save(out / 'fake_depth.png')",
            ]
        )
        + "\n"
    )

    server.configure(repo_path=str(repo), depth_model_path=str(checkpoint), python_bin=sys.executable)
    result = server._run_infinidepth(
        input_path=Path(sample_image_path),
        run_dir=tmp_path / "run",
        save_pcd=False,
        upsample_ratio=1,
        output_resolution_mode="original",
        source_shape=[96, 128],
    )

    assert result["success"] is True
    assert result["colored_depth_image"]
    assert result["shape"] == [12, 16]
    assert result["depth_shape"] == [12, 16]
    assert result["source_shape"] == [96, 128]
    assert "--output_resolution_mode=original" in result["command"]


def test_infinidepth_client_saves_unique_valid_artifacts(tmp_path):
    import base64
    import io

    from spagent.external_experts.InfiniDepth.infinidepth_client import InfiniDepthClient

    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), color="red").save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode()
    client = InfiniDepthClient(output_dir=str(tmp_path))

    first = client._save_outputs({"success": True, "colored_depth_image": payload}, "sample", None)
    second = client._save_outputs({"success": True, "colored_depth_image": payload}, "sample", None)

    assert first["depth_path"] == first["colored_depth_path"]
    assert first["colored_depth_path"] != second["colored_depth_path"]
    with Image.open(first["depth_path"]) as image:
        assert image.size == (8, 6)


def test_infinidepth_requires_requested_point_cloud(tmp_path, sample_image_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))
    original_infer = tool._client.infer

    def without_pcd(**kwargs):
        result = original_infer(**{**kwargs, "save_pcd": False})
        result["point_cloud_path"] = None
        return result

    tool._client.infer = without_pcd
    result = tool.call(image_path=sample_image_path, save_pcd=True)

    assert result["success"] is False
    assert "requested point cloud" in result["error"]


@pytest.mark.skipif(
    os.environ.get("INFINIDEPTH_REAL_TEST") != "1",
    reason="Set INFINIDEPTH_REAL_TEST=1 to run against a live InfiniDepth server.",
)
def test_infinidepth_real_service(sample_image_path):
    server_url = os.environ.get("INFINIDEPTH_SERVER_URL", "http://127.0.0.1:20039")
    tool = InfiniDepthTool(use_mock=False, server_url=server_url)

    result = tool.call(image_path=sample_image_path, upsample_ratio=1)

    assert result["success"] is True
    assert result["depth_path"] is not None
    assert os.path.exists(result["depth_path"])
