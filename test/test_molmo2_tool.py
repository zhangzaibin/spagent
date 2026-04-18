"""
Tests for Molmo2Tool (mock and optional live server).

Mock (no GPU / no server):
  conda activate /home/zzb/anaconda3/envs/spagent
  cd /home/zzb/projects/spagent
  pytest test/test_molmo2_tool.py -v

Live server (Molmo2 must be running, e.g. port 20025):
  export MOLMO2_LIVE_TEST=1
  export MOLMO2_SERVER_URL=http://127.0.0.1:20025
  pytest test/test_molmo2_tool.py -v -k live

CLI (see test/test_tool.py):
  python test/test_tool.py --tool molmo2 --image path/to.jpg --task qa --prompt "Describe" \\
    --server_url http://127.0.0.1:20025
"""

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from spagent.tools import Molmo2Tool

ASSET_IMAGE = project_root / "assets" / "dog.jpeg"


@pytest.fixture
def sample_image_path(tmp_path):
    """Prefer repo asset; otherwise a tiny JPEG in tmp_path."""
    if ASSET_IMAGE.is_file():
        return str(ASSET_IMAGE)
    p = tmp_path / "molmo2_sample.jpg"
    Image.new("RGB", (128, 128), color=(90, 120, 60)).save(p, format="JPEG")
    return str(p)


def test_molmo2_mock_qa(sample_image_path):
    tool = Molmo2Tool(use_mock=True)
    result = tool.call(
        image_path=sample_image_path,
        task="qa",
        prompt="What is shown in this image?",
    )

    assert result["success"] is True
    assert "generated_text" in result["result"]
    assert result["result"]["task"] == "qa"


def test_molmo2_mock_point(tmp_path, sample_image_path):
    tool = Molmo2Tool(use_mock=True, output_dir=str(tmp_path))
    result = tool.call(
        image_path=sample_image_path,
        task="point",
        prompt="Point to the dog.",
        save_annotated=True,
    )

    assert result["success"] is True
    assert result["result"]["task"] == "point"
    assert result["result"]["num_points"] >= 1
    assert result["output_path"] is not None


@pytest.mark.skipif(not os.environ.get("MOLMO2_LIVE_TEST"), reason="Set MOLMO2_LIVE_TEST=1 to run")
def test_molmo2_live_qa(sample_image_path):
    url = os.environ.get("MOLMO2_SERVER_URL", "http://127.0.0.1:20025")
    tool = Molmo2Tool(use_mock=False, server_url=url)
    result = tool.call(
        image_path=sample_image_path,
        task="qa",
        prompt="What is in this image? One short sentence.",
        max_new_tokens=64,
    )
    assert result["success"] is True
    assert result["result"]["generated_text"].strip()
