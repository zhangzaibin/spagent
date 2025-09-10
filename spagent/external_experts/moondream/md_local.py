# 本地部署moondream
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

# Load the model

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    torch_dtype=torch.float32  # 使用float32精度
)

# 手动将模型移动到CUDA设备
if torch.cuda.is_available():
    model = model.to('cuda')

print(f"模型设备: {next(model.parameters()).device}")
print(f"模型数据类型: {next(model.parameters()).dtype}")

# Load your image

image = Image.open("assets/example.png")

# 1. Image Captioning

print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("Detailed caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    print(t, end="", flush=True)

# 2. Visual Question Answering

print("Asking questions about the image:")
print(model.query(image, "How many cars are in the image?")["answer"])

# 3. Object Detection

print("Detecting objects:")
objects = model.detect(image, "car")["objects"]
print(f"Found {len(objects)} car(s)")

# 4. Visual Pointing

print("Locating objects:")
points = model.point(image, "car")["points"]
print(f"Found {len(points)} car(s)")