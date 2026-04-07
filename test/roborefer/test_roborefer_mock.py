from __future__ import annotations

import json
from pathlib import Path

from spagent.tools import RoboReferTool


def main():
    image_path = "assets/example.png"   # 确保这个图片存在
    depth_path = None

    tool = RoboReferTool(
        use_mock=True,                        # 这里一定是 True
        timeout=120,
        default_enable_depth=0,
        default_output_dir="./test/roborefer/output",
    )

    result = tool.call(
        image_path=image_path,
        prompt="Point to the leftmost cup.",
        depth_path=depth_path,
        enable_depth=0,
        return_visualization=True,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
