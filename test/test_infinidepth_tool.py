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

from spagent.tools import InfiniDepthTool


@pytest.fixture
def sample_image_path(tmp_path):
    path = tmp_path / "infinidepth_sample.jpg"
    Image.new("RGB", (128, 96), color=(90, 130, 170)).save(path, format="JPEG")
    return str(path)


def test_infinidepth_tool_is_exported():
    assert InfiniDepthTool is not None


def test_infinidepth_schema_contains_inputs():
    tool = InfiniDepthTool(use_mock=True)
    params = tool.parameters

    assert "image_path" in params["required"]
    for key in ["task", "save_pcd", "upsample_ratio", "output_dir"]:
        assert key in params["properties"]


def test_infinidepth_mock_depth(tmp_path, sample_image_path):
    tool = InfiniDepthTool(use_mock=True, output_dir=str(tmp_path))

    result = tool.call(image_path=sample_image_path, upsample_ratio=2)

    assert result["success"] is True
    assert result["depth_path"] is not None
    assert result["colored_depth_path"] is not None
    assert os.path.exists(result["depth_path"])
    assert os.path.exists(result["colored_depth_path"])
    assert result["shape"] == [96, 128]


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
    assert "upsample_ratio must be positive" in result["error"]


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
                "from PIL import Image",
                "out = Path('example_data/pred_depth')",
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
    )

    assert result["success"] is True
    assert result["depth_image"]
    assert result["colored_depth_image"]


@pytest.mark.skipif(
    os.environ.get("INFINIDEPTH_REAL_TEST") != "1",
    reason="Set INFINIDEPTH_REAL_TEST=1 to run against a live InfiniDepth server.",
)
def test_infinidepth_real_service(sample_image_path):
    server_url = os.environ.get("INFINIDEPTH_SERVER_URL", "http://127.0.0.1:20037")
    tool = InfiniDepthTool(use_mock=False, server_url=server_url)

    result = tool.call(image_path=sample_image_path, upsample_ratio=1)

    assert result["success"] is True
    assert result["depth_path"] is not None
    assert os.path.exists(result["depth_path"])
