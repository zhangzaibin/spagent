# 1. 导入必要模块
import sys
from pathlib import Path
import cv2
from typing import List

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.core.spagent import SPAgent
from spagent.models import GPTModel, QwenModel
from spagent.tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool, MoondreamTool, Pi3Tool


def extract_video_frames(video_path: str, num_frames: int = 10) -> List[str]:
    """从视频中均匀采样指定数量的帧
    
    Args:
        video_path: 视频文件路径
        num_frames: 要提取的帧数，默认10帧
        
    Returns:
        提取的帧图像路径列表
    """
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_duration = total_frames / original_fps
    
    # 使用指定的帧数
    frame_interval = total_frames / num_frames
    
    frame_paths = []
    temp_dir = Path("temp_frames")
    temp_dir.mkdir(exist_ok=True)
    
    # 从视频路径中提取文件名（不含扩展名）
    video_filename = Path(video_path).stem
    
    # 均匀提取帧
    for i in range(num_frames):
        frame_idx = int(i * frame_interval)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"{video_filename}_frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
    
    cap.release()
    print(f"从视频中提取了 {len(frame_paths)} 帧 (时长: {total_duration:.2f}s, 原始fps: {original_fps:.2f}, 均匀采样 {num_frames} 帧)")
    return frame_paths

# 2. 选择和配置模型
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)

# 3. 选择和配置工具
tools = [
    Pi3Tool(use_mock=False, server_url="http://10.7.8.94:20030")  # 真实模式
]

# 4. 创建智能体
agent = SPAgent(
    model=model, 
    tools=tools, 
    max_workers=4  # 并行工具数量
)

# 5. 解决问题 - 视频测试
video_path = "dataset/VLM4D_videos/synthetic_synth_216.mp4"
num_frames = 10  # 均匀采样10帧

# 从视频中提取帧
frame_paths = extract_video_frames(video_path, num_frames)
print(f"提取的帧: {frame_paths}")

# 使用提取的帧进行推理
result = agent.solve_problem(
    frame_paths, 
    f"Based on these {len(frame_paths)} frames from a video, please answer: Which direction did the dog move?\nSelect from the following choices:\n(A) no dog there\n(B) not moving\n(C) left\n(D) right\n",
    video_path=video_path,  # 传递视频路径，如果调用pi3工具可以提取更多帧
    pi3_num_frames=50  # pi3工具使用更多帧
)

# 6. 处理结果
print(f"答案: {result['answer']}")
print(f"使用的工具: {result['used_tools']}")
print(f"生成的图像: {result['additional_images']}")

# 7. 清理临时文件
import os
for frame_path in frame_paths:
    try:
        os.remove(frame_path)
    except:
        pass
try:
    os.rmdir("temp_frames")
except:
    pass
print("临时文件已清理")