# 1. 导入必要模块
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from spagent.core.spagent import SPAgent
from spagent.models import GPTModel, QwenModel
from spagent.tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool, MoondreamTool, Pi3Tool

# 2. 选择和配置模型
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)
# 或者: model = QwenModel(model_name="qwen2.5-vl-7b-instruct")

# 3. 选择和配置工具
tools = [
    # DepthEstimationTool(use_mock=False, server_url="http://10.7.8.94:20019"),      # Mock模式用于测试
    SegmentationTool(use_mock=False, server_url="http://0.0.0.0:20020"),         # Mock模式用于测试
    # ObjectDetectionTool(use_mock=False,      # 真实模式
    #                    server_url="http://10.7.8.94:20022"),
    # MoondreamTool(use_mock=False, server_url="http://0.0.0.0:20024")  # 真实模式
    # Pi3Tool(use_mock=False, server_url="http://0.0.0.0:20030")  # 真实模式
]

# 4. 创建智能体
agent = SPAgent(
    model=model, 
    tools=tools, 
    max_workers=4  # 并行工具数量
)

# 5. 解决问题
result = agent.solve_problem(["dataset/BLINK_images/Multi-view_Reasoning_val_000075_img1.jpg"], "use segmentation tool to segment the objects in the image. and how many objects are there?")

# 6. 处理结果
print(f"答案: {result['answer']}")
print(f"使用的工具: {result['used_tools']}")
print(f"生成的图像: {result['additional_images']}")