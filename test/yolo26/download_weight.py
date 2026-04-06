from pathlib import Path
import shutil

from ultralytics import YOLO

weights_dir = Path("weights")
weights_dir.mkdir(parents=True, exist_ok=True)

# 触发下载 / 加载
model = YOLO("yolo26n.pt")

# 找到实际权重文件
model_path = Path(model.ckpt_path if hasattr(model, "ckpt_path") else "yolo26n.pt")

target_path = weights_dir / "yolo26n.pt"

if model_path.exists():
    if model_path.resolve() != target_path.resolve():
        shutil.copy2(model_path, target_path)
    print(f"saved to: {target_path}")
else:
    print("Could not locate downloaded checkpoint automatically.")
    print("Please manually place yolo26n.pt into weights/")