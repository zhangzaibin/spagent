import base64
import cv2
import io
import logging
import numpy as np
import torch
import os
import argparse
from flask import Flask, request, jsonify
from PIL import Image
import traceback
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from sklearn.covariance import EmpiricalCovariance

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型
model = None


def load_model(checkpoint_path=None):
    """
    加载VGGT模型
    
    Args:
        checkpoint_path: 模型权重文件路径或包含权重文件的目录路径（可选，默认使用 HuggingFace 预训练权重）
    """
    global model
    try:
        logger.info("正在加载VGGT模型...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备：{device}")
        
        # 创建模型实例
        model = VGGT()
        
        # 加载模型权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            # 支持目录或文件路径
            actual_file_path = None
            
            if os.path.isfile(checkpoint_path):
                # 如果是文件，直接使用
                actual_file_path = checkpoint_path
                logger.info(f"正在从文件 {checkpoint_path} 加载模型权重...")
            elif os.path.isdir(checkpoint_path):
                # 如果是目录，寻找权重文件
                logger.info(f"正在从目录 {checkpoint_path} 查找模型权重...")
                weight_extensions = ['.safetensors', '.pt', '.bin']
                weight_names = ['model', 'pytorch_model']
                
                # 按优先级搜索
                for name in weight_names:
                    for ext in weight_extensions:
                        potential_file = os.path.join(checkpoint_path, name + ext)
                        if os.path.exists(potential_file):
                            actual_file_path = potential_file
                            logger.info(f"找到权重文件: {actual_file_path}")
                            break
                    if actual_file_path:
                        break
                
                # 如果没找到标准命名的文件，列出目录中的所有权重文件
                if not actual_file_path:
                    weight_files = []
                    for file in os.listdir(checkpoint_path):
                        if any(file.endswith(ext) for ext in weight_extensions):
                            weight_files.append(file)
                    
                    if weight_files:
                        actual_file_path = os.path.join(checkpoint_path, weight_files[0])
                        logger.info(f"使用找到的权重文件: {actual_file_path}")
                    else:
                        logger.error(f"在目录 {checkpoint_path} 中未找到有效的权重文件")
                        return False
            else:
                logger.error(f"路径不存在: {checkpoint_path}")
                return False
            
            # 加载权重文件
            try:
                if actual_file_path.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    weight = load_file(actual_file_path)
                    model.load_state_dict(weight)
                    logger.info("使用 safetensors 格式加载模型权重成功")
                else:
                    weight = torch.load(actual_file_path, map_location=device, weights_only=True)
                    model.load_state_dict(weight)
                    logger.info("使用 torch.load 加载模型权重成功")
            except Exception as e:
                logger.error(f"加载模型权重失败：{e}")
                return False
        else:
            logger.info("从 HuggingFace 加载预训练模型...")
            model = VGGT.from_pretrained("facebook/VGGT-1B")
        
        model = model.to(device).eval()
        logger.info("VGGT模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        logger.error(traceback.format_exc())
        return False


def remove_outliers_mahalanobis(points, colors, threshold_std=3.0):
    """
    使用 Mahalanobis 距离移除点云离群点
    
    Args:
        points: (N,3) numpy 数组
        colors: (N,3) 或 (N,C) numpy 数组
        threshold_std: 阈值标准差倍数，默认3.0
        
    Returns:
        filtered_points, filtered_colors, inlier_mask
    """
    if len(points) == 0:
        return points, colors, np.ones(len(points), dtype=bool)
    
    # 转换为numpy数组
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # 计算 Mahalanobis 距离
    cov = EmpiricalCovariance().fit(points)
    dist = cov.mahalanobis(points)
    
    # 根据阈值过滤离群点
    mean_dist = np.mean(dist)
    std_dist = np.std(dist)
    threshold = mean_dist + threshold_std * std_dist
    
    inlier_mask = dist < threshold
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    removed_count = len(points) - len(filtered_points)
    logger.info(f"离群点移除完成: 原始点数={len(points)}, 移除点数={removed_count}, 保留点数={len(filtered_points)}")
    
    return filtered_points, filtered_colors, inlier_mask


def write_ply(points, colors, filepath):
    """
    保存点云为PLY文件并返回base64编码
    
    Args:
        points: (N, 3) 点云坐标
        colors: (N, 3) 颜色值 (0-1 或 0-255)
        filepath: 保存路径
        
    Returns:
        base64编码的PLY文件内容
    """
    import trimesh
    
    # 转换为numpy
    if torch.is_tensor(points):
        points = points.cpu().numpy()
    if torch.is_tensor(colors):
        colors = colors.cpu().numpy()
    
    # 确保颜色在正确范围
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)
    
    # 创建点云并保存
    point_cloud = trimesh.PointCloud(vertices=points, colors=colors)
    point_cloud.export(filepath)
    
    # 读取并编码为base64
    with open(filepath, 'rb') as f:
        ply_data = f.read()
    
    return base64.b64encode(ply_data).decode('utf-8')


def extract_scene_id(image_path: str) -> str:
    """从图片路径中提取scene ID"""
    if 'vlm-3r' in image_path.lower():
        parts = image_path.split('/')
        for part in parts:
            if part.startswith('scene') and '_' in part:
                return part
        
        path_parts = image_path.split('/')
        for part in reversed(path_parts[:-1]):
            if any(c.isdigit() for c in part) or part.lower() in ['scene', 'view', 'camera']:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                if filename and filename != part:
                    return f"{part}_{filename}"
                return part
    
    return os.path.splitext(os.path.basename(image_path))[0]


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        status = {
            "status": "健康",
            "model_loaded": model is not None,
            "model_type": "VGGT-1B",
            "device": str(next(model.parameters()).device) if model is not None else None
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"健康检查失败：{e}")
        return jsonify({
            "status": "不健康",
            "error": str(e)
        }), 500


@app.route('/test', methods=['GET'])
def test():
    """测试接口"""
    global model
    
    try:
        logger.info("正在创建测试图像序列...")
        test_images = []
        img_size = 518  # VGGT 默认图像尺寸
        for i in range(2):
            test_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            for j in range(img_size):
                test_image[:, j, 0] = (i * 80 + j) % 256
                test_image[:, j, 1] = (i * 60 + j) % 256
                test_image[:, j, 2] = (i * 40 + j) % 256
            test_images.append(test_image)
        
        device = next(model.parameters()).device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 转换为tensor
        imgs_tensor = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 
            for img in test_images
        ]).to(device)
        
        logger.info("正在进行VGGT测试推理...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(imgs_tensor)
        
        logger.info("测试推理成功")
        return jsonify({
            "success": True,
            "message": "VGGT测试推理成功",
            "output_keys": list(predictions.keys()),
            "world_points_shape": list(predictions['world_points'].shape) if 'world_points' in predictions else None
        })
        
    except Exception as e:
        logger.error(f"测试推理失败：{e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/infer', methods=['POST'])
def infer():
    """VGGT 3D重建推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({"error": "缺少图像序列数据"}), 400
        
        # 获取参数 - 与Pi3保持一致
        conf_threshold = data.get('conf_threshold', 50.0)  # 置信度百分位阈值（百分比）
        generate_views = data.get('generate_views', True)
        max_views_per_camera = data.get('max_views_per_camera', 7)
        
        # 离群点移除参数
        remove_outliers_flag = data.get('remove_outliers', True)
        
        # 视角参数
        azimuth_angle = data.get('azimuth_angle', None)
        elevation_angle = data.get('elevation_angle', None)
        rotation_reference_camera = data.get('rotation_reference_camera', 1)
        camera_view = data.get('camera_view', False)
        
        # 获取文件名信息
        image_names = data.get('image_names', [])
        
        # 解码base64图像序列
        try:
            images_rgb = []
            for img_b64 in data['images']:
                image_bytes = base64.b64decode(img_b64)
                image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                # VGGT 需要调整到 518x518 或保持比例
                # 使用VGGT的预处理方式
                h, w = image_rgb.shape[:2]
                
                # 调整图像尺寸确保是patch大小(14)的倍数
                patch_size = 14
                new_h = ((h + patch_size - 1) // patch_size) * patch_size
                new_w = ((w + patch_size - 1) // patch_size) * patch_size
                
                if new_h != h or new_w != w:
                    image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                images_rgb.append(image_rgb)
        except Exception as e:
            logger.error(f"图像处理失败：{e}")
            return jsonify({"error": "图像数据无效"}), 400
        
        device = next(model.parameters()).device
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 转换为tensor (S, 3, H, W)
        imgs_tensor = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 
            for img in images_rgb
        ]).to(device)
        
        # 保存原始图像tensor用于颜色提取
        imgs_rgb_tensor = imgs_tensor.clone()
        
        # 运行VGGT推理
        logger.info("正在进行VGGT 3D重建...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(imgs_tensor)
        
        logger.info("VGGT推理完成，正在处理结果...")
        
        # 转换位姿编码为外参和内参矩阵
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], 
            imgs_tensor.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # 提取点云数据 - 参考 test.py 的处理方式
        # 先将所有 tensor 转换为 numpy 并移除 batch 维度
        world_points = predictions["world_points"]
        if isinstance(world_points, torch.Tensor):
            world_points_np = world_points.cpu().numpy()
        else:
            world_points_np = world_points
        
        # 移除 batch 维度（如果存在）: (B, S, H, W, 3) -> (S, H, W, 3)
        if world_points_np.ndim == 5:
            world_points_np = world_points_np[0]
        
        logger.info(f"world_points shape: {world_points_np.shape}")  # 应该是 (S, H, W, 3)
        
        # 处理 world_points_conf
        # VGGT 输出格式: (B, S, H, W) 或 (S, H, W)，注意没有通道维度
        world_points_conf = predictions.get("world_points_conf", None)
        if world_points_conf is not None:
            if isinstance(world_points_conf, torch.Tensor):
                world_points_conf_np = world_points_conf.cpu().numpy()
            else:
                world_points_conf_np = world_points_conf
            
            logger.info(f"world_points_conf original shape: {world_points_conf_np.shape}")
            
            # 处理不同的维度格式
            if world_points_conf_np.ndim == 5:  # (B, S, H, W, 1) - 带通道维度
                world_points_conf_np = world_points_conf_np[0, ..., 0]  # -> (S, H, W)
            elif world_points_conf_np.ndim == 4:
                # 形状是 (B, S, H, W) 或 (S, H, W, 1)
                if world_points_conf_np.shape[-1] == 1:
                    # (S, H, W, 1) -> (S, H, W)
                    world_points_conf_np = world_points_conf_np[..., 0]
                else:
                    # (B, S, H, W) -> (S, H, W)，取第一个 batch
                    world_points_conf_np = world_points_conf_np[0]
            # 如果是 3 维 (S, H, W)，直接使用
            
            logger.info(f"world_points_conf processed shape: {world_points_conf_np.shape}")
        else:
            # 如果没有置信度，使用全1，形状与 world_points 的前3维一致
            world_points_conf_np = np.ones(world_points_np.shape[:-1])
            logger.info(f"world_points_conf created with shape: {world_points_conf_np.shape}")
        
        # 确保 world_points_conf 与 world_points 维度匹配
        expected_shape = world_points_np.shape[:-1]  # (S, H, W)
        if world_points_conf_np.shape != expected_shape:
            logger.warning(f"world_points_conf shape {world_points_conf_np.shape} != expected {expected_shape}, creating ones")
            world_points_conf_np = np.ones(expected_shape)
        
        # 获取图像用于颜色
        if "images" in predictions:
            pred_images = predictions["images"]
            if isinstance(pred_images, torch.Tensor):
                pred_images = pred_images.cpu().numpy()
            if pred_images.ndim == 5:  # (B, S, 3, H, W)
                pred_images = pred_images[0]  # 取第一个 batch
        else:
            pred_images = imgs_rgb_tensor.cpu().numpy()
        
        # 确保图像格式为 (S, H, W, 3)
        if pred_images.ndim == 4 and pred_images.shape[1] == 3:
            pred_images = np.transpose(pred_images, (0, 2, 3, 1))
        
        logger.info(f"pred_images shape: {pred_images.shape}")  # 应该是 (S, H, W, 3)
        
        # 展平点云和颜色
        vertices_3d = world_points_np.reshape(-1, 3)
        colors_rgb = pred_images.reshape(-1, 3)
        conf = world_points_conf_np.reshape(-1)
        
        logger.info(f"After reshape: vertices_3d={vertices_3d.shape}, colors_rgb={colors_rgb.shape}, conf={conf.shape}")
        
        # 置信度过滤 (使用 percentile)
        conf_percentile_threshold = np.percentile(conf, conf_threshold)
        conf_mask = (conf >= conf_percentile_threshold) & (conf > 1e-5)
        
        points_filtered = vertices_3d[conf_mask]
        colors_filtered = colors_rgb[conf_mask]
        
        logger.info(f"置信度过滤: 原始点数={len(vertices_3d)}, 过滤后={len(points_filtered)}")
        
        # 应用离群点移除
        if remove_outliers_flag and len(points_filtered) > 0:
            logger.info("开始移除离群点...")
            points_filtered, colors_filtered, _ = remove_outliers_mahalanobis(
                points_filtered, 
                colors_filtered,
                threshold_std=3.0
            )
        
        # 提取相机位姿信息
        extrinsic_np = extrinsic.cpu().numpy()
        if extrinsic_np.ndim == 4:  # (B, S, 4, 4)
            extrinsic_np = extrinsic_np[0]  # (S, 4, 4)
        
        # ============ 镜像变换：将逆时针相机排列转换为顺时针 ============
        # VGGT输出的相机排列是逆时针的，通过对相机的X轴镜像转为顺时针（与Pi3一致）
        # 注意：只对相机外参做镜像，不对点云做镜像，这样相机看到的内容不变
        mirror_3x3 = np.diag([-1, 1, 1])  # X轴镜像（3x3）
        for i in range(len(extrinsic_np)):
            # 对旋转矩阵做镜像变换（保持相机朝向正确）
            extrinsic_np[i][:3, :3] = mirror_3x3 @ extrinsic_np[i][:3, :3] @ mirror_3x3
            # 对平移部分（相机位置）做X轴镜像
            extrinsic_np[i][:3, 3] = mirror_3x3 @ extrinsic_np[i][:3, 3]
        logger.info("已应用X轴镜像变换到相机外参，相机排列从逆时针转换为顺时针")
        # ================================================================
        
        # 生成PLY文件（镜像变换后）
        import hashlib
        if image_names and len(image_names) > 0:
            scene_id = extract_scene_id(image_names[0])
            safe_name = "".join(c for c in scene_id if c.isalnum() or c in ('-', '_'))
            ply_filename = f"vggt_result_{safe_name}.ply"
        else:
            first_img_bytes = data['images'][0].encode('utf-8')
            img_hash = hashlib.md5(first_img_bytes).hexdigest()[:8]
            ply_filename = f"vggt_result_{img_hash}.ply"
        
        ply_path = f"outputs/{ply_filename}"
        os.makedirs("outputs", exist_ok=True)
        
        ply_b64 = write_ply(points_filtered, colors_filtered, ply_path)
        
        camera_poses_list = []
        reference_pose = extrinsic_np[0]
        R_ref = reference_pose[:3, :3]
        
        from scipy.spatial.transform import Rotation as R_scipy
        
        for i, pose in enumerate(extrinsic_np):
            R_cw = pose[:3, :3]
            t_cw = pose[:3, 3]
            
            R_relative = R_cw @ R_ref.T
            rotation_relative = R_scipy.from_matrix(R_relative)
            
            try:
                euler_yx = rotation_relative.as_euler('yx', degrees=True)
                azimuth_from_cam1 = euler_yx[0]
                elevation_from_cam1 = euler_yx[1]
            except:
                euler_xyz_rel = rotation_relative.as_euler('xyz', degrees=True)
                azimuth_from_cam1 = euler_xyz_rel[1]
                elevation_from_cam1 = euler_xyz_rel[0]
            
            camera_poses_list.append({
                "camera_id": i + 1,
                "position": t_cw.tolist(),
                "azimuth_angle": float(azimuth_from_cam1),
                "elevation_angle": float(elevation_from_cam1)
            })
        
        response_data = {
            "success": True,
            "ply_file": ply_b64,
            "ply_filename": ply_filename,
            "points_count": len(points_filtered),
            "camera_poses": camera_poses_list,
            "camera_views": []
        }
        
        # 生成多视角图片
        if generate_views:
            try:
                if azimuth_angle is not None and elevation_angle is not None:
                    view_mode = "相机视角" if camera_view else "全局视角"
                    logger.info(f"使用自定义角度生成视角图片: 方位角={azimuth_angle}°, 仰角={elevation_angle}°, 参考相机={rotation_reference_camera}, 模式={view_mode}")
                    view_images = generate_custom_angle_views(
                        points_filtered, colors_filtered, extrinsic_np, imgs_rgb_tensor,
                        azimuth_angle, elevation_angle,
                        rotation_reference_camera=rotation_reference_camera,
                        camera_view=camera_view
                    )
                else:
                    view_mode = "相机视角" if camera_view else "全局视角"
                    logger.info(f"使用默认多视角生成图片，参考相机={rotation_reference_camera}, 模式={view_mode}")
                    view_images = generate_camera_views(
                        points_filtered, colors_filtered, extrinsic_np, imgs_rgb_tensor,
                        max_views_per_camera,
                        rotation_reference_camera=rotation_reference_camera,
                        camera_view=camera_view
                    )
                response_data["camera_views"] = view_images
            except Exception as e:
                logger.warning(f"生成多视角图片失败：{e}")
                logger.warning(traceback.format_exc())
        
        logger.info("VGGT重建完成")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"推理失败：{str(e)}"}), 500


def _prepare_points_and_cameras(points_filtered, colors_filtered, camera_poses):
    """
    准备点云和相机数据
    
    Args:
        points_filtered: 过滤后的点云 (N, 3)
        colors_filtered: 过滤后的颜色 (N, 3)
        camera_poses: 相机位姿 (S, 4, 4)
        
    Returns:
        tuple: (points_sample, colors_sample, camera_centers, camera_poses)
    """
    # 转换为numpy
    if torch.is_tensor(points_filtered):
        points_3d = points_filtered.cpu().numpy()
    else:
        points_3d = np.array(points_filtered)
    
    if torch.is_tensor(colors_filtered):
        colors_3d = colors_filtered.cpu().numpy()
    else:
        colors_3d = np.array(colors_filtered)
    
    # 提取相机中心
    camera_centers = []
    camera_poses_list = []
    for pose in camera_poses:
        camera_centers.append(pose[:3, 3])
        camera_poses_list.append(pose)
    
    camera_centers = np.array(camera_centers)
    camera_poses = np.array(camera_poses_list)
    
    # 子采样点云以提高渲染性能
    max_points_to_visualize = 500000
    if len(points_3d) > max_points_to_visualize:
        total_points = len(points_3d)
        step = total_points // max_points_to_visualize
        indices = np.arange(0, total_points, step)[:max_points_to_visualize]
        points_sample = points_3d[indices]
        colors_sample = colors_3d[indices]
        logger.info(f"点云子采样：从 {len(points_3d)} 个点中采样了 {max_points_to_visualize} 个点")
    else:
        points_sample = points_3d
        colors_sample = colors_3d
    
    return points_sample, colors_sample, camera_centers, camera_poses


def _draw_cameras_visualization(ax, camera_centers, camera_poses, current_view_cam_idx, 
                              view_R_cam, view_t_cam, max_range, show_cameras=True):
    """相机可视化函数"""
    if not show_cameras:
        return None, None, None, None, None, None
    
    axis_length = max_range * 0.12
    
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    
    camera_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    is_identity_transform = (np.allclose(view_R_cam, np.eye(3)) and 
                             np.allclose(view_t_cam, np.zeros(3)))
    
    for cam_idx, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
        cam_color = camera_colors[cam_idx % len(camera_colors)]
        
        if is_identity_transform:
            cam_center_in_view = cam_center
        else:
            flip_transform = np.diag([1, -1, -1])
            cam_center_in_view = (view_R_cam @ cam_center.T).T + view_t_cam
            cam_center_in_view = (flip_transform @ cam_center_in_view.T).T
        
        R_cam2world = cam_pose[:3, :3]
        
        if is_identity_transform:
            R_pose_in_view = R_cam2world
        else:
            flip_transform = np.diag([1, -1, -1])
            R_pose_in_view = view_R_cam @ R_cam2world
            R_pose_in_view = flip_transform @ R_pose_in_view
        
        frustum_length = axis_length * 0.8
        frustum_width = frustum_length * 0.3
        frustum_height = frustum_length * 0.3
        
        forward = -R_pose_in_view[:, 2]
        right = R_pose_in_view[:, 0]
        up = -R_pose_in_view[:, 1]
        
        far_center = cam_center_in_view + forward * frustum_length
        
        corner1 = far_center + right * frustum_width + up * frustum_height
        corner2 = far_center - right * frustum_width + up * frustum_height
        corner3 = far_center - right * frustum_width - up * frustum_height
        corner4 = far_center + right * frustum_width - up * frustum_height
        
        line_width = 1.2 if cam_idx == current_view_cam_idx else 1
        alpha = 0.65 if cam_idx == current_view_cam_idx else 0.5
        
        for corner in [corner1, corner2, corner3, corner4]:
            ax.plot([cam_center_in_view[0], corner[0]],
                   [cam_center_in_view[1], corner[1]],
                   [cam_center_in_view[2], corner[2]], 
                   color=cam_color, linewidth=line_width, alpha=alpha)
        
        corners = [corner1, corner2, corner3, corner4, corner1]
        for i in range(len(corners) - 1):
            ax.plot([corners[i][0], corners[i+1][0]],
                   [corners[i][1], corners[i+1][1]],
                   [corners[i][2], corners[i+1][2]], 
                   color=cam_color, linewidth=line_width, alpha=alpha)
        
        marker_size = 35 if cam_idx == current_view_cam_idx else 20
        ax.scatter(cam_center_in_view[0], cam_center_in_view[1], cam_center_in_view[2], 
                  c=cam_color, s=marker_size, marker='o', alpha=0.8, depthshade=False,
                  edgecolors='black', linewidth=0.8)
        
        label_pos = cam_center_in_view + np.array([axis_length * 0.3, 0, axis_length * 0.2])
        marker = '*' if cam_idx == current_view_cam_idx else ''
        ax.text(label_pos[0], label_pos[1], label_pos[2], 
               f'Cam{cam_idx+1}{marker}', fontsize=5, color='black', weight='bold',
               bbox=dict(boxstyle="round,pad=0.05", facecolor=cam_color, alpha=0.6))
        
        coords_to_check = [cam_center_in_view, corner1, corner2, corner3, corner4, far_center]
        
        for coord in coords_to_check:
            all_x_coords.append(coord[0])
            all_y_coords.append(coord[1])
            all_z_coords.append(coord[2])
    
    if all_x_coords:
        return (min(all_x_coords), max(all_x_coords), 
                min(all_y_coords), max(all_y_coords),
                min(all_z_coords), max(all_z_coords))
    else:
        return None, None, None, None, None, None


def _create_view_image(points_sample, colors_sample, camera_centers, camera_poses, cam_idx, 
                      azim_angle, elev_angle, view_name, show_camera_axes=True, show_all_cameras=True,
                      ref_cam_idx: int = 0, camera_view: bool = False):
    """创建单个视角图片"""
    
    if camera_view:
        safe_ref_idx = max(0, min(ref_cam_idx, len(camera_poses) - 1))
        view_cam_pose = camera_poses[safe_ref_idx]
    else:
        view_cam_pose = camera_poses[cam_idx]
    
    if view_cam_pose.shape == (4, 4):
        R_cw = view_cam_pose[:3, :3]
        t_cw = view_cam_pose[:3, 3]
    else:
        R_cw = view_cam_pose[:3, :3]
        t_cw = view_cam_pose[:3, 3]

    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    points_cam = (R_wc @ points_sample.T).T + t_wc

    flip_transform = np.diag([1, -1, -1])
    points_cam = (flip_transform @ points_cam.T).T
    
    try:
        if abs(azim_angle) > 1e-6 or abs(elev_angle) > 1e-6:
            safe_ref_idx = 0 if camera_poses is None else max(0, min(ref_cam_idx, len(camera_poses) - 1))
            first_cam_world_pos = camera_poses[safe_ref_idx][:3, 3]
            first_cam_R_cw = camera_poses[safe_ref_idx][:3, :3]
            
            first_cam_in_current = (R_wc @ first_cam_world_pos.T).T + t_wc
            first_cam_center = (flip_transform @ first_cam_in_current.T).T
            
            first_cam_R_in_current = R_wc @ first_cam_R_cw
            first_cam_R_in_current = flip_transform @ first_cam_R_in_current
            
            first_cam_x_axis = first_cam_R_in_current[:, 0]
            first_cam_y_axis = first_cam_R_in_current[:, 1]
            
            points_centered = points_cam - first_cam_center
            
            if abs(azim_angle) > 1e-6:
                R_azim = R.from_rotvec(np.radians(azim_angle) * first_cam_y_axis).as_matrix()
                points_centered = (R_azim @ points_centered.T).T
            
            if abs(elev_angle) > 1e-6:
                R_elev = R.from_rotvec(np.radians(elev_angle) * first_cam_x_axis).as_matrix()
                points_centered = (R_elev @ points_centered.T).T
            
            points_cam = points_centered + first_cam_center
            
            R_azim_full = R.from_rotvec(np.radians(azim_angle) * first_cam_y_axis).as_matrix() if abs(azim_angle) > 1e-6 else np.eye(3)
            R_elev_full = R.from_rotvec(np.radians(elev_angle) * first_cam_x_axis).as_matrix() if abs(elev_angle) > 1e-6 else np.eye(3)
            R_rel = R_elev_full @ R_azim_full
            
            view_R_cam = R_rel @ R_wc
            view_t_cam = R_rel @ t_wc
            
            rotated_camera_centers = []
            rotated_camera_poses = []
            for i, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
                cam_center_in_current = (R_wc @ cam_center.T).T + t_wc
                cam_center_flipped = (flip_transform @ cam_center_in_current.T).T
                
                cam_center_centered = cam_center_flipped - first_cam_center
                cam_center_rotated = (R_rel @ cam_center_centered.T).T
                cam_center_final = cam_center_rotated + first_cam_center
                
                cam_R = cam_pose[:3, :3]
                cam_R_in_current = R_wc @ cam_R
                cam_R_flipped = flip_transform @ cam_R_in_current
                cam_R_rotated = R_rel @ cam_R_flipped
                
                rotated_pose = np.eye(4)
                rotated_pose[:3, :3] = cam_R_rotated
                rotated_pose[:3, 3] = cam_center_final
                
                rotated_camera_centers.append(cam_center_final)
                rotated_camera_poses.append(rotated_pose)
            
            camera_centers = np.array(rotated_camera_centers)
            camera_poses = np.array(rotated_camera_poses)
        else:
            view_R_cam = R_wc
            view_t_cam = t_wc
    except Exception:
        view_R_cam = R_wc
        view_t_cam = t_wc
    
    if camera_view:
        point_directions = points_cam / (np.linalg.norm(points_cam, axis=1, keepdims=True) + 1e-8)
        camera_forward = np.array([0, 0, -1])
        cos_angles = point_directions @ camera_forward
        
        fov_angle_threshold = np.cos(np.radians(110))
        fov_mask = cos_angles > fov_angle_threshold
        
        if np.sum(fov_mask) > 0:
            points_cam = points_cam[fov_mask]
            colors_sample = colors_sample[fov_mask]
    
    lower_percentile = np.percentile(points_cam, 7, axis=0)
    upper_percentile = np.percentile(points_cam, 93, axis=0)
    
    x_range = upper_percentile[0] - lower_percentile[0]
    y_range = upper_percentile[1] - lower_percentile[1]
    z_range = upper_percentile[2] - lower_percentile[2]
    max_range = max(x_range, y_range, z_range)
    
    if max_range > 0:
        base_point_size = max(0.03, min(0.15, 40.0 / max_range))
        if camera_view:
            point_size = base_point_size * 2.5
            alpha = 0.9
        else:
            point_size = base_point_size
            alpha = 0.8
    else:
        point_size = 1.0
        alpha = 0.8
        if camera_view:
            point_size = 2.5
            alpha = 0.9
    
    fig = plt.figure(figsize=(12, 10), dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False
    
    try:
        if colors_sample.shape[-1] == 3:
            if colors_sample.max() > 1.0:
                colors_normalized = colors_sample / 255.0
            else:
                colors_normalized = np.clip(colors_sample, 0, 1)
        else:
            colors_normalized = colors_sample
        
        scatter = ax.scatter(
            points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
            c=colors_normalized,
            s=point_size,
            alpha=alpha, 
            edgecolors='none',
            depthshade=True,
            linewidth=0
        )
    except Exception as e:
        logger.warning(f"RGB散点绘制失败，使用单色绘制: {e}")
        scatter = ax.scatter(
            points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
            c='steelblue',
            s=point_size * 2,
            alpha=0.9,
            edgecolors='darkblue',
            linewidth=0.1
        )
    
    show_cameras_in_view = show_all_cameras and not camera_view
    
    if show_cameras_in_view:
        identity_R = np.eye(3)
        identity_t = np.zeros(3)
        x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam = _draw_cameras_visualization(
            ax, camera_centers, camera_poses, cam_idx, identity_R, identity_t, max_range, show_cameras=True
        )
        
        if x_min_cam is not None:
            x_min = min(points_cam[:, 0].min(), x_min_cam)
            x_max = max(points_cam[:, 0].max(), x_max_cam)
            y_min = min(points_cam[:, 1].min(), y_min_cam)
            y_max = max(points_cam[:, 1].max(), y_max_cam)
            z_min = min(points_cam[:, 2].min(), z_min_cam)
            z_max = max(points_cam[:, 2].max(), z_max_cam)
        else:
            x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
            y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
            z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    else:
        x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
        y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
        z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    
    if camera_view:
        ax.view_init(elev=0.0, azim=-90.0)
        ax.dist = 7
    else:
        ax.view_init(elev=0.0, azim=-90.0)
        ax.dist = 10
    
    margin_factor = 0.02
    x_margin = (x_max - x_min) * margin_factor
    y_margin = (y_max - y_min) * margin_factor
    z_margin = (z_max - z_min) * margin_factor
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    ax.set_xlabel('X', fontsize=14, fontweight='bold', color='red')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold', color='green')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold', color='blue')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    ax.set_facecolor('lightgray')
    
    buf = io.BytesIO()
    try:
        try:
            fig.canvas.draw()
        except Exception:
            pass
        
        plt.savefig(buf, format='png', dpi=500, bbox_inches='tight', 
                   pad_inches=0.05, facecolor='white', edgecolor='none',
                   transparent=False)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
    except Exception as e:
        logger.error(f"保存图片时出错: {e}")
        img_b64 = ""
    finally:
        try:
            plt.close(fig)
        except:
            pass
        try:
            buf.close()
        except:
            pass
    
    return img_b64


def generate_camera_views(points_filtered, colors_filtered, camera_poses, imgs_rgb_tensor,
                          max_views_per_camera=15, rotation_reference_camera: int = 1, 
                          camera_view: bool = False):
    """生成多视角图片
    
    Args:
        points_filtered: 过滤后的点云
        colors_filtered: 过滤后的颜色
        camera_poses: 相机位姿
        imgs_rgb_tensor: 原始图像tensor（用于0°/0°时返回原图）
        max_views_per_camera: 每个相机最多生成的视角数
        rotation_reference_camera: 参考相机索引（1-based），用于指定旋转中心
        camera_view: 是否使用相机视角模式
    """
    try:
        points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
            points_filtered, colors_filtered, camera_poses
        )
        
        view_angles = [
            (0, 0, "camera_front"),
            (-45, 0, "camera_left_45"),
            (45, 0, "camera_right_45"),
            (0, -45, "camera_front_down"),
            (0, 45, "camera_front_up"),
        ]
        
        view_images = []
        
        # 使用 rotation_reference_camera 作为参考相机索引
        ref_cam_idx = max(0, min(int(rotation_reference_camera) - 1, len(camera_centers) - 1))
        
        for cam_idx in range(min(4, len(camera_centers))):
            limited_view_angles = view_angles[:max_views_per_camera]
            
            for azim_offset, elev_offset, view_name in limited_view_angles:
                # 如果是0°/0°视角，直接返回原图
                if abs(azim_offset) < 1e-6 and abs(elev_offset) < 1e-6:
                    try:
                        if cam_idx < len(imgs_rgb_tensor):
                            img_ref = imgs_rgb_tensor[cam_idx].detach().cpu().numpy()
                            img_ref = (np.clip(img_ref.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                            buf = io.BytesIO()
                            Image.fromarray(img_ref).save(buf, format='PNG')
                            buf.seek(0)
                            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                            buf.close()
                            view_images.append({
                                "camera": cam_idx + 1,
                                "view": view_name,
                                "image": img_b64
                            })
                            continue
                    except Exception as e:
                        logger.warning(f"返回原始图片失败，回退到点云渲染: {e}")
                
                show_camera_axes = view_name in ["camera_front", "camera_left_30", "camera_right_30"]
                adjusted_elev = elev_offset + 100.0
                
                img_b64 = _create_view_image(
                    points_sample, colors_sample, camera_centers, camera_poses,
                    cam_idx, azim_offset, adjusted_elev, view_name, show_camera_axes, show_all_cameras=True,
                    ref_cam_idx=ref_cam_idx,  # 使用参考相机索引
                    camera_view=camera_view
                )
                
                view_images.append({
                    "camera": cam_idx + 1,
                    "view": view_name,
                    "image": img_b64
                })
        
        return view_images
    
    except Exception as e:
        logger.error(f"生成视角图片失败：{e}")
        logger.error(traceback.format_exc())
        return []


def generate_custom_angle_views(points_filtered, colors_filtered, camera_poses, imgs_rgb_tensor,
                                azimuth_angle, elevation_angle,
                                rotation_reference_camera: int = 1, camera_view: bool = False):
    """根据自定义角度生成视角图片"""
    try:
        points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
            points_filtered, colors_filtered, camera_poses
        )
        
        cam_idx = max(0, min(int(rotation_reference_camera) - 1, len(camera_centers) - 1))
        logger.info(f"使用相机 {cam_idx + 1} 的视角生成图片")
        
        # 如果角度为0°/0°，直接返回原图
        if abs(azimuth_angle) < 1e-6 and abs(elevation_angle) < 1e-6:
            try:
                if cam_idx < len(imgs_rgb_tensor):
                    img_ref = imgs_rgb_tensor[cam_idx].detach().cpu().numpy()
                    img_ref = (np.clip(img_ref.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
                    buf = io.BytesIO()
                    Image.fromarray(img_ref).save(buf, format='PNG')
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    return [{
                        "camera": cam_idx + 1,
                        "view": f"custom_azim_0_elev_0_cam{cam_idx + 1}",
                        "azimuth_angle": azimuth_angle,
                        "elevation_angle": elevation_angle,
                        "image": img_b64
                    }]
            except Exception as e:
                logger.warning(f"返回原始图片失败，回退到点云渲染: {e}")
        
        view_images = []
        view_name = f"custom_azim_{azimuth_angle}_elev_{elevation_angle}"
        
        adjusted_elevation = elevation_angle + 100.0
        
        img_b64 = _create_view_image(
            points_sample, colors_sample, camera_centers, camera_poses,
            cam_idx, azimuth_angle, adjusted_elevation, view_name, show_camera_axes=False, show_all_cameras=True,
            ref_cam_idx=cam_idx,
            camera_view=camera_view
        )
        
        view_images.append({
            "camera": cam_idx + 1,
            "view": view_name,
            "azimuth_angle": azimuth_angle,
            "elevation_angle": elevation_angle,
            "image": img_b64
        })
        
        return view_images
        
    except Exception as e:
        logger.error(f"生成自定义角度视角图片失败：{e}")
        logger.error(traceback.format_exc())
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VGGT 3D Reconstruction Server')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to VGGT model weights (optional, uses HuggingFace by default)')
    parser.add_argument('--port', type=int, default=20022,
                        help='Port to run the server on (default: 20022)')
    
    args = parser.parse_args()
    
    logger.info("正在启动VGGT服务器...")
    logger.info(f"模型路径: {args.checkpoint_path if args.checkpoint_path else 'HuggingFace (facebook/VGGT-1B)'}")
    logger.info(f"服务端口: {args.port}")
    
    if not load_model(args.checkpoint_path):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    app.run(host='0.0.0.0', port=args.port, debug=False)
