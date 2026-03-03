import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("spagent/external_experts/checkpoints/vggt/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9").to(device)
model.eval()  # 设置为评估模式，确保images被添加到predictions中

# Load and preprocess example images (replace with your own image paths)
image_names = ["dataset/BLINK_images/Art_Style_val_000000_img1.jpg", "dataset/BLINK_images/Art_Style_val_000000_img2.jpg"]  
images = load_and_preprocess_images(image_names).to(device)

print("Running inference...")
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print("Converting pose encoding to extrinsic and intrinsic matrices...")
# 将pose编码转换为相机外参和内参矩阵
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

print("Processing model outputs...")
# 先处理images（如果存在），然后再处理其他tensor
if "images" in predictions:
    pred_images = predictions["images"]  # (B, S, 3, H, W) 或 (S, 3, H, W)
    if isinstance(pred_images, torch.Tensor):
        pred_images = pred_images.cpu().numpy()
    if len(pred_images.shape) == 5:  # (B, S, 3, H, W)
        pred_images = pred_images.squeeze(0)  # 移除batch维度 -> (S, 3, H, W)
    # 从predictions中移除，稍后单独处理
    del predictions["images"]
else:
    # 如果没有images，使用原始输入的images
    print("Warning: 'images' not in predictions, using input images instead")
    pred_images = images.cpu().numpy()  # (S, 3, H, W)

# 将其他tensor转换为numpy，并移除batch维度
for key in predictions.keys():
    if isinstance(predictions[key], torch.Tensor):
        predictions[key] = predictions[key].cpu().numpy().squeeze(0)

# 提取点云数据
print("Extracting point cloud data...")
world_points = predictions["world_points"]  # (S, H, W, 3)
world_points_conf = predictions.get("world_points_conf", np.ones_like(world_points[..., 0]))

# 处理图像格式，确保是 (S, H, W, 3)
if pred_images.ndim == 4 and pred_images.shape[1] == 3:  # (S, 3, H, W)
    pred_images = np.transpose(pred_images, (0, 2, 3, 1))  # 转换为 (S, H, W, 3)

# 将点云和颜色展平
vertices_3d = world_points.reshape(-1, 3)  # (N, 3)
colors_rgb = (pred_images.reshape(-1, 3) * 255).astype(np.uint8)  # (N, 3)
conf = world_points_conf.reshape(-1)  # (N,)

# 过滤低置信度的点（保留前50%的点）
conf_threshold = np.percentile(conf, 50.0)
conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

vertices_3d = vertices_3d[conf_mask]
colors_rgb = colors_rgb[conf_mask]

print(f"Point cloud contains {len(vertices_3d)} points after filtering")

# 保存点云为PLY文件（可选）
print("Saving point cloud to pointcloud.ply...")
import trimesh
point_cloud = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
point_cloud.export("pointcloud.ply")
print("Point cloud saved to pointcloud.ply")

# 渲染点云为PNG图片
print("Rendering point cloud to PNG...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 为了更好的可视化，可以下采样点云（如果点太多）
max_points = 100000
if len(vertices_3d) > max_points:
    indices = np.random.choice(len(vertices_3d), max_points, replace=False)
    vertices_3d_vis = vertices_3d[indices]
    colors_rgb_vis = colors_rgb[indices]
else:
    vertices_3d_vis = vertices_3d
    colors_rgb_vis = colors_rgb

# 归一化颜色到 [0, 1] 范围
colors_normalized = colors_rgb_vis / 255.0

# 绘制点云
ax.scatter(vertices_3d_vis[:, 0], 
           vertices_3d_vis[:, 1], 
           vertices_3d_vis[:, 2],
           c=colors_normalized,
           s=0.5,
           alpha=0.6)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud Visualization')

# 设置相等的坐标轴比例
ax.set_box_aspect([1,1,1])

# 保存为PNG
plt.savefig("pointcloud_rendered.png", dpi=150, bbox_inches='tight')
print("Rendered point cloud saved to pointcloud_rendered.png")
plt.close()