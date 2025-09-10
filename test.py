# 1. 导入必要模块
from spagent import SPAgent
from spagent.models import GPTModel, QwenModel
from spagent.tools import DepthEstimationTool, SegmentationTool, ObjectDetectionTool, MoondreamTool

# 2. 选择和配置模型
model = GPTModel(model_name="gpt-4o-mini", temperature=0.7)
# 或者: model = QwenModel(model_name="qwen2.5-vl-7b-instruct")

# 3. 选择和配置工具
tools = [
    DepthEstimationTool(use_mock=False, server_url="http://127.0.0.1:20019"),      # Mock模式用于测试
    SegmentationTool(use_mock=False, server_url="http://127.0.0.1:20020"),         # Mock模式用于测试
    ObjectDetectionTool(use_mock=False,      # 真实模式
                       server_url="http://10.7.8.94:20022"),
    MoondreamTool(use_mock=False, server_url="http://0.0.0.0:20024")  # 真实模式
]

# 4. 创建智能体
agent = SPAgent(
    model=model, 
    tools=tools, 
    max_workers=4  # 并行工具数量
)

# 5. 解决问题
result = agent.solve_problem("assets/example.png", "While driving, what are the distances and relative positions of obstacles (cars, pedestrians, bicycles) in the vehicle’s path? note that the image_path is assets/example.png")

# 6. 处理结果
print(f"答案: {result['answer']}")
print(f"使用的工具: {result['used_tools']}")
print(f"生成的图像: {result['additional_images']}")