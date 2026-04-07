from __future__ import annotations

import json
from pathlib import Path

from spagent.tools import RoboReferTool


def main():
    # 改成你的真实测试图片
    image_path = "assets/example.png"

    # 如果你有 depth 图就填，没有就设为 None
    depth_path = None

    tool = RoboReferTool(
        use_mock=False,                       # 真实 API
        server_url="http://127.0.0.1:25547", # RoboRefer 服务地址
        timeout=120,
        default_enable_depth=1,
        default_output_dir="./test/roborefer/output",
    )

    result = tool.call(
        image_path=image_path,
        prompt="Point to the leftmost cup.",
        depth_path=depth_path,
        enable_depth=1,
        return_visualization=True,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()