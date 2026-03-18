import sys
from pathlib import Path
import cv2
from typing import List
import os

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.core.spagent import SPAgent
from spagent.models import GPTModel
from spagent.tools.robotracer_tool import RoboTracerTool


def extract_video_frames(video_path: str, num_frames: int = 10) -> List[str]:
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or original_fps <= 0:
        cap.release()
        raise ValueError(f"无法正确读取视频信息: {video_path}")

    total_duration = total_frames / original_fps
    frame_interval = total_frames / num_frames

    frame_paths = []
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)

    video_filename = Path(video_path).stem

    for i in range(num_frames):
        frame_idx = int(i * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"{video_filename}_frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))

    cap.release()
    print(
        f"从视频中提取了 {len(frame_paths)} 帧 "
        f"(时长: {total_duration:.2f}s, 原始fps: {original_fps:.2f}, 均匀采样 {num_frames} 帧)"
    )
    return frame_paths


def main():
    model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)

    tools = [
        RoboTracerTool(
            use_mock=False,
            server_url="http://10.7.8.94:20030"
        )
    ]

    agent = SPAgent(
        model=model,
        tools=tools,
        max_workers=4
    )

    video_path = "dataset/VLM4D_videos/synthetic_synth_216.mp4"
    num_frames = 10

    frame_paths = extract_video_frames(video_path, num_frames)
    print(f"提取的帧: {frame_paths}")

    try:
        result = agent.solve_problem(
            frame_paths,
            (
                f"Based on these {len(frame_paths)} frames from a video, "
                "please use RoboTracerTool to analyze the motion trajectory of the main object. "
                "Then answer: Which direction did the dog move?\n"
                "Select from the following choices:\n"
                "(A) no dog there\n"
                "(B) not moving\n"
                "(C) left\n"
                "(D) right\n"
            ),
            video_path=video_path
        )

        print(f"答案: {result.get('answer')}")
        print(f"使用的工具: {result.get('used_tools')}")
        print(f"生成的图像: {result.get('additional_images')}")

    finally:
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except Exception:
                pass

        try:
            os.rmdir("temp_frames")
        except Exception:
            pass

        print("临时文件已清理")


if __name__ == "__main__":
    main()
