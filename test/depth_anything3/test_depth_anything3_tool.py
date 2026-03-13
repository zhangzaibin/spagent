"""
Depth Anything 3 Tool 测试。

- test_depth_anything3_tool_mock_on_bus_png: 使用 use_mock=True，不调用真实模型，
  只生成竖直渐变假深度图，用于 CI/快速验证，无真实 API 调用。
- test_depth_anything3_tool_real: 使用真实 Depth Anything V3 模型（需安装 depth_anything_v3、
  下载 .pth checkpoint 并设置 DEPTH_ANYTHING3_CHECKPOINT 或传入 checkpoint_path）。
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
    真实模型测试。需满足其一：
    1. 环境变量 DEPTH_ANYTHING3_CHECKPOINT 为 HuggingFace 模型 id，如 depth-anything/DA3MONO-LARGE
    2. 或指向本地 .pth 文件，或本目录下有 .pth
    若无则 skip。
    """
    import pytest

    ckpt = os.environ.get("DEPTH_ANYTHING3_CHECKPOINT")
    if not ckpt:
        cand = list(Path(__file__).parent.glob("*.pth"))
        ckpt = str(cand[0]) if cand else None
    # HF id 如 "depth-anything/DA3MONO-LARGE" 不需要本地存在
    if not ckpt:
        pytest.skip(
            "Real test skipped: set DEPTH_ANYTHING3_CHECKPOINT (e.g. depth-anything/DA3MONO-LARGE) or put a .pth in test/depth_anything3/"
        )
    if "/" in ckpt and not ckpt.startswith("/") and not Path(ckpt).exists():
        pass  # 视为 HF id，不要求本地存在
    elif not Path(ckpt).exists():
        pytest.skip(f"DEPTH_ANYTHING3_CHECKPOINT path not found: {ckpt}")

    image_path = Path("assets/example.png")
    output_dir = Path("test/depth_anything3/outputs")
    assert image_path.exists(), f"Test image not found: {image_path}"

    tool = DepthAnything3Tool(
        use_mock=False,
        checkpoint_path=ckpt,
        encoder="vitl",
        device="cuda",
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

    # 默认跑 mock；加 --real 则尝试跑真实模型（需 checkpoint）
    if "--real" in sys.argv:
        ckpt = os.environ.get("DEPTH_ANYTHING3_CHECKPOINT")
        if not ckpt:
            cand = list(Path(__file__).parent.glob("*.pth"))
            ckpt = str(cand[0]) if cand else None
        is_hf_id = ckpt and "/" in ckpt and not ckpt.startswith("/") and not Path(ckpt).exists()
        if not ckpt or (not is_hf_id and not Path(ckpt).exists()):
            print("Real test skipped: set DEPTH_ANYTHING3_CHECKPOINT=e.g. depth-anything/DA3MONO-LARGE or put a .pth in test/depth_anything3/")
            sys.exit(0)
        r = test_depth_anything3_tool_real()
        print("Real model test passed. Result:", r)
    else:
        test_depth_anything3_tool_mock_on_bus_png()
        print("Mock test passed (no real API/model called — output is fake vertical gradient).")