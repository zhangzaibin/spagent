"""
Depth Anything 3 Tool 的测试（本目录仅做测试，不提供可被 agent 调用的入口）。

Tool ：Server/Client 架构，真实推理需先启动 depth_anything3_server。
- test_depth_anything3_tool_mock_on_bus_png: mock 模式，用于 CI/快速验证。
- test_depth_anything3_tool_real: 真实模型，需先启动 server，可选设置 DEPTH_ANYTHING3_SERVER_URL（默认 http://localhost:20032）。
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from spagent.tools.depth_anything3_tool import DepthAnything3Tool


def test_depth_anything3_tool_mock_on_bus_png():
    """Mock 模式：不调用真实模型，输出为竖直渐变假深度，仅用于流程验证。"""
    image_path = Path("assets/example.png")
    output_dir = Path("test/depth_anything3/outputs")

    assert image_path.exists(), f"Test image not found: {image_path}"

    tool = DepthAnything3Tool(
        use_mock=True,
        save_dir=str(output_dir),
    )

    result = tool.call(
        image_path=str(image_path),
        output_format="both",
        colormap="inferno",
        normalize=True,
    )

    assert result["success"] is True
    assert "result" in result

    depth_png_path = result["result"]["depth_png_path"]
    depth_npy_path = result["result"]["depth_npy_path"]

    assert depth_png_path is not None
    assert depth_npy_path is not None
    assert Path(depth_png_path).exists(), f"Depth PNG not found: {depth_png_path}"
    assert Path(depth_npy_path).exists(), f"Depth NPY not found: {depth_npy_path}"


def test_depth_anything3_tool_real():
    """
    真实模型测试（Server/Client 模式）。需先启动 depth_anything3_server，
    可选设置 DEPTH_ANYTHING3_SERVER_URL（默认 http://localhost:20032）。若 server 不可达则 skip。
    """
    import pytest
    import requests

    server_url = os.environ.get("DEPTH_ANYTHING3_SERVER_URL", "http://localhost:20032").rstrip("/")
    try:
        r = requests.get(f"{server_url}/health", timeout=3)
        if r.status_code != 200:
            pytest.skip(f"Depth Anything 3 server returned {r.status_code}; start server first.")
        data = r.json()
        if not data.get("model_loaded"):
            pytest.skip("Depth Anything 3 server not ready (model not loaded).")
    except requests.RequestException as e:
        pytest.skip(f"Depth Anything 3 server not running at {server_url}: {e}. Start with: python -m spagent.external_experts.depth_anything3.depth_anything3_server --checkpoint_path depth-anything/DA3MONO-LARGE --port 20032")

    image_path = Path("assets/example.png")
    output_dir = Path("test/depth_anything3/outputs")
    assert image_path.exists(), f"Test image not found: {image_path}"

    tool = DepthAnything3Tool(
        use_mock=False,
        server_url=server_url,
        save_dir=str(output_dir),
    )

    result = tool.call(
        image_path=str(image_path),
        output_format="both",
        colormap="inferno",
        normalize=True,
    )

    assert result["success"] is True
    assert "result" in result
    assert Path(result["result"]["depth_png_path"]).exists()
    return result


if __name__ == "__main__":
    import sys

    if "--real" in sys.argv:
        try:
            r = test_depth_anything3_tool_real()
            print("Real model test passed. Result:", r)
        except Exception as e:
            if "skip" in type(e).__name__.lower() or "skip" in str(e).lower():
                print("Skipped:", e)
            else:
                raise
    else:
        test_depth_anything3_tool_mock_on_bus_png()
        print("Mock test passed (no server required).")