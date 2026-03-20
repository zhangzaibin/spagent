from __future__ import annotations

import json
from pathlib import Path

from spagent.tools import D4RTTool


def main():
    video_path = "assets/demo.mp4"

    if not Path(video_path).exists():
        print(f"[WARN] video file not found: {video_path}")
        print("Please replace video_path with a real video.")
        return

    tool = D4RTTool(
        use_mock=False,
        server_url="http://127.0.0.1:20034",
        timeout=1800,
    )

    result = tool.call(
        video_path=video_path,
        query_mode="both",
        num_frames=16,
        save_visualization=True,
        output_dir="outputs/test_d4rt_tool",
        query_points=[[0, 100, 120], [3, 250, 180]],
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()