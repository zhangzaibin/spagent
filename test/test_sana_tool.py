"""
Minimal tests for SanaTool.

Default behavior:
- Runs mock-mode tests only, so no Sana server is required.

Optional real-service test:
- Start Sana server first
- Run with: SANA_REAL_TEST=1 pytest -q test/test_sana_tool.py
"""

import os
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.tools import SanaTool


def test_sana_tool_is_exported():
    """SanaTool should be importable from spagent.tools."""
    assert SanaTool is not None


def test_sana_tool_mock_generation():
    """Mock mode should generate a local placeholder image successfully."""
    tool = SanaTool(use_mock=True)

    result = tool.call(
        prompt="a compact home robot organizing books on a wooden shelf",
        size="512x512",
        num_inference_steps=10,
        guidance_scale=4.0,
        seed=123,
    )

    assert result["success"] is True
    assert "output_path" in result
    assert result["output_path"] is not None
    assert os.path.exists(result["output_path"])
    assert result["output_path"].endswith(".png")
    assert result["file_size_bytes"] > 0
    assert result["image_paths"]


def test_sana_tool_mock_rejects_empty_prompt():
    """Empty prompts should fail fast with a clear error."""
    tool = SanaTool(use_mock=True)

    result = tool.call(prompt="   ")

    assert result["success"] is False
    assert "Prompt must be a non-empty string" in result["error"]


@pytest.mark.skipif(
    os.environ.get("SANA_REAL_TEST") != "1",
    reason="Set SANA_REAL_TEST=1 to run against a live Sana server.",
)
def test_sana_tool_real_service():
    """Optional smoke test for a live Sana server."""
    server_url = os.environ.get("SANA_SERVER_URL", "http://127.0.0.1:30000")
    tool = SanaTool(use_mock=False, server_url=server_url)

    result = tool.call(
        prompt="a mobile robot navigating a bright modern office hallway",
        size="512x512",
        num_inference_steps=10,
        guidance_scale=4.0,
        seed=7,
    )

    assert result["success"] is True
    assert os.path.exists(result["output_path"])
