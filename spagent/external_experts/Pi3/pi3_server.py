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
# 设置matplotlib兼容模式以避免3D投影问题
plt.rcParams['figure.max_open_warning'] = 0
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from safetensors.torch import load_file
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型
model = None

def extract_scene_id(image_path: str) -> str:
    """
    从图片路径中提取scene ID，适配多种数据集格式
    
    Args:
        image_path: 图片路径，支持多种格式:
            - VLM-3R/scannet: "VLM-3R/scannet_frames_25k/scene0296_01/color/000000.jpg" -> "scene0296_01"
            - VLM-3R/arkitscenes: "VLM-3R/scannet_frames_25k/arkitscenes_47333899/frame_0.jpg" -> "arkitscenes_47333899_frame_0"
            - 其他数据集: "dataset/images/file.jpg" -> "file" (仅文件名)
        
    Returns:
        scene ID字符串，VLM-3R返回scene ID，其他数据集返回文件名
    """
    # For VLM-3R datasets only, extract scene ID
    if 'vlm-3r' in image_path.lower():
        # 1. Try to extract scene ID for scannet format first
        parts = image_path.split('/')
        for part in parts:
            if part.startswith('scene') and '_' in part:
                return part
        
        # 2. For arkitscenes or other VLM-3R subdatasets (not scannet)
        path_parts = image_path.split('/')
        for part in reversed(path_parts[:-1]):  # From back to front, skip filename
            if any(c.isdigit() for c in part) or part.lower() in ['scene', 'view', 'camera']:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                if filename and filename != part:
                    return f"{part}_{filename}"
                return part
    
    # For other datasets, just return the filename (original logic)
    return os.path.splitext(os.path.basename(image_path))[0]

def load_model(checkpoint_path=None):
    """
    加载Pi3模型
    
    Args:
        checkpoint_path: 模型权重文件路径或包含权重文件的目录路径
    """
    global model
    try:
        logger.info("正在加载Pi3模型...")
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"使用设备：{device}")
        
        # 创建模型实例
        model = Pi3().to(device).eval()
        
        # 加载检查点
        if checkpoint_path:
            # 支持目录或文件路径
            actual_file_path = None
            
            if os.path.isfile(checkpoint_path):
                # 如果是文件，直接使用
                actual_file_path = checkpoint_path
                logger.info(f"正在从文件 {checkpoint_path} 加载模型权重...")
            elif os.path.isdir(checkpoint_path):
                # 如果是目录，寻找权重文件
                logger.info(f"正在从目录 {checkpoint_path} 查找模型权重...")
                weight_extensions = ['.safetensors', '.bin']
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
                        # 如果有权重文件，选择第一个
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
                weight = load_file(actual_file_path)
                model.load_state_dict(weight)
                logger.info("模型权重加载成功")
            except Exception as e:
                logger.error(f"加载模型权重失败：{e}")
                return False
        
        logger.info("模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        return False

def remove_outliers(points, colors, k_neighbors=20, std_threshold=2.0):
    """
    使用统计离群点移除方法过滤点云中的离群点
    
    Args:
        points: 点云坐标数组 (N, 3)
        colors: 点云颜色数组 (N, 3) 或 (N, C)
        k_neighbors: 用于计算的最近邻数量，默认20
        std_threshold: 标准差阈值，默认2.0（越大保留越多点）
        
    Returns:
        filtered_points: 过滤后的点云坐标
        filtered_colors: 过滤后的点云颜色
        inlier_mask: 内点的布尔掩码
    """
    try:
        if len(points) < k_neighbors:
            logger.warning(f"点云数量({len(points)})小于k_neighbors({k_neighbors})，跳过离群点移除")
            return points, colors, np.ones(len(points), dtype=bool)
        
        # 转换为numpy数组（如果是tensor）
        if torch.is_tensor(points):
            points_np = points.cpu().numpy()
        else:
            points_np = np.array(points)
            
        if torch.is_tensor(colors):
            colors_np = colors.cpu().numpy()
        else:
            colors_np = np.array(colors)
        
        # 使用KNN计算每个点到其k个最近邻的平均距离
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', n_jobs=-1).fit(points_np)
        distances, indices = nbrs.kneighbors(points_np)
        
        # 排除自身（第一个邻居是点自己），计算到其他k个邻居的平均距离
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # 计算全局均值和标准差
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        # 确定离群点阈值
        threshold = global_mean + std_threshold * global_std
        
        # 创建内点掩码（保留距离小于阈值的点）
        inlier_mask = mean_distances < threshold
        
        # 过滤点云
        filtered_points = points_np[inlier_mask]
        filtered_colors = colors_np[inlier_mask]
        
        removed_count = len(points_np) - len(filtered_points)
        removed_percentage = (removed_count / len(points_np)) * 100
        
        logger.info(f"离群点移除完成: 原始点数={len(points_np)}, "
                   f"移除点数={removed_count} ({removed_percentage:.2f}%), "
                   f"保留点数={len(filtered_points)}")
        
        return filtered_points, filtered_colors, inlier_mask
        
    except Exception as e:
        logger.error(f"离群点移除失败: {e}")
        # 失败时返回原始数据
        return points, colors, np.ones(len(points), dtype=bool)

def remove_outliers_iforest(points, colors, contamination=0.01):
    """
    使用 Isolation Forest 移除点云离群点
    Args:
        points: (N,3) numpy 数组
        colors: (N,3) numpy 数组
        contamination: 离群点比例，默认1%
    Returns:
        filtered_points, filtered_colors, inlier_mask
    """
    if len(points) == 0:
        return points, colors, np.ones(len(points), dtype=bool)
    
    clf = IsolationForest(contamination=contamination, n_jobs=-1)
    y_pred = clf.fit_predict(points)  # 1 -> 内点, -1 -> 离群点
    inlier_mask = y_pred == 1
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    return filtered_points, filtered_colors, inlier_mask

def remove_outliers_mahalanobis(points, colors, threshold_std=3.0):
    """
    使用 Mahalanobis 距离移除点云离群点（CPU 版本，接口类似 Isolation Forest 函数）
    
    Args:
        points: (N,3) numpy 数组
        colors: (N,3) 或 (N,C) numpy 数组
        threshold_std: 阈值标准差倍数，默认3.0
        
    Returns:
        filtered_points, filtered_colors, inlier_mask
    """
    if len(points) == 0:
        return points, colors, np.ones(len(points), dtype=bool)
    
    # 计算 Mahalanobis 距离
    cov = EmpiricalCovariance().fit(points)
    dist = cov.mahalanobis(points)  # shape=(N,)
    
    # 根据阈值过滤离群点
    mean_dist = np.mean(dist)
    std_dist = np.std(dist)
    threshold = mean_dist + threshold_std * std_dist
    
    inlier_mask = dist < threshold
    filtered_points = points[inlier_mask]
    filtered_colors = colors[inlier_mask]
    
    return filtered_points, filtered_colors, inlier_mask

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        status = {
            "status": "健康",
            "model_loaded": model is not None,
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
        # 创建测试图像序列
        logger.info("正在创建测试图像序列...")
        test_images = []
        # 使用280x280，因为280 = 14 * 20，是14的倍数
        img_size = 280
        for i in range(3):  # 创建3张测试图像
            test_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            for j in range(img_size):
                # 创建RGB格式的渐变图像
                test_image[:, j, 0] = (i * 80 + j) % 256  # Red channel
                test_image[:, j, 1] = (i * 60 + j) % 256  # Green channel  
                test_image[:, j, 2] = (i * 40 + j) % 256  # Blue channel
            test_images.append(test_image)
        
        # 转换为tensor
        device = next(model.parameters()).device
        imgs_tensor = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 
            for img in test_images
        ]).to(device)
        
        # 运行推理
        logger.info("正在进行测试推理...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                results = model(imgs_tensor[None])
        
        logger.info(f"推理成功")
        return jsonify({
            "success": True,
            "message": "测试推理成功",
            "output_keys": list(results.keys()),
            "points_shape": list(results['points'][0].shape) if 'points' in results else None
        })
        
    except Exception as e:
        logger.error(f"测试推理失败：{e}")
        logger.error(f"错误追踪：{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/infer', methods=['POST'])
def infer():
    """Pi3 3D重建推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'images' not in data:
            return jsonify({"error": "缺少图像序列数据"}), 400
        
        # 获取可选参数 - 优化后的默认值以提升质量
        conf_threshold = data.get('conf_threshold', 0.08)  # 置信度阈值 - 提高到25%获得更高质量
        rtol = data.get('rtol', 0.02)  # 深度边缘检测阈值 - 更严格的边缘过滤
        generate_views = data.get('generate_views', True)  # 是否生成多视角图片
        max_views_per_camera = data.get('max_views_per_camera', 7)  # 减少默认视角数量以提高性能
        
        # 离群点移除参数
        remove_outliers_flag = data.get('remove_outliers', True)  # 是否移除离群点，默认开启
        k_neighbors = data.get('k_neighbors', 50)  # KNN邻居数量
        std_threshold = data.get('std_threshold', 2.0)  # 标准差阈值
        
        # 新增：支持自定义角度参数
        azimuth_angle = data.get('azimuth_angle', None)  # 左右旋转角度（方位角）
        elevation_angle = data.get('elevation_angle', None)  # 上下旋转角度（仰角）
        # 新增：选择用于旋转参考的相机索引（1-based，默认1）
        rotation_reference_camera = data.get('rotation_reference_camera', 1)
        # 新增：是否使用相机视角模式（而非全局视角）
        camera_view = data.get('camera_view', False)

        # 获取文件名信息（可选）
        image_names = data.get('image_names', [])  # 图片文件名列表
        
        # 解码base64图像序列
        try:
            images = []
            images_rgb = []  # 用于PLY和显示的RGB版本
            for img_b64 in data['images']:
                image_bytes = base64.b64decode(img_b64)
                image_bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                # 调整图像尺寸确保是patch大小(14)的倍数
                h, w = image_bgr.shape[:2]
                patch_size = 14
                
                # 计算最接近的有效尺寸
                new_h = ((h + patch_size - 1) // patch_size) * patch_size
                new_w = ((w + patch_size - 1) // patch_size) * patch_size
                
                # 优化：使用高质量插值方法调整尺寸
                if new_h != h or new_w != w:
                    # INTER_LANCZOS4 提供最高质量的插值
                    image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # 转换BGR到RGB用于显示
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                
                images.append(image_bgr)  # 保持原始BGR用于模型推理
                images_rgb.append(image_rgb)  # RGB版本用于PLY和显示
        except Exception as e:
            logger.error(f"图像处理失败：{e}")
            return jsonify({"error": "图像数据无效"}), 400
        
        # 转换为tensor（使用BGR数据用于模型推理以保持一致性）
        device = next(model.parameters()).device
        imgs_tensor = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 
            for img in images
        ]).to(device)
        
        # RGB版本用于PLY和显示
        imgs_rgb_tensor = torch.stack([
            torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0 
            for img in images_rgb
        ]).to(device)
        
        # 运行推理
        logger.info("正在进行Pi3 3D重建...")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                results = model(imgs_tensor[None])
        
        logger.info("重建完成！")
        
        # 处理结果
        masks = torch.sigmoid(results['conf'][..., 0]) > conf_threshold
        non_edge = ~depth_edge(results['local_points'][..., 2], rtol=rtol)
        masks = torch.logical_and(masks, non_edge)[0]
        
        # 获取初步过滤后的点云和颜色
        points_filtered = results['points'][0][masks].cpu()
        colors_filtered = imgs_rgb_tensor.permute(0, 2, 3, 1)[masks].cpu()
        
        # 应用离群点移除（如果启用）
        if remove_outliers_flag:
            logger.info(f"开始移除离群点 (k_neighbors={k_neighbors}, std_threshold={std_threshold})...")
            # points_filtered, colors_filtered, _ = remove_outliers(
            #     points_filtered, 
            #     colors_filtered, 
            #     k_neighbors=k_neighbors, 
            #     std_threshold=std_threshold
            # )
            # points_filtered, colors_filtered, _ = remove_outliers_iforest(
            #     points_filtered, 
            #     colors_filtered
            # )
            points_filtered, colors_filtered, _ = remove_outliers_mahalanobis(
                points_filtered, 
                colors_filtered,
                threshold_std=3.0
            )
            
            # 转换回tensor以便后续处理
            points_filtered = torch.from_numpy(points_filtered) if isinstance(points_filtered, np.ndarray) else points_filtered
            colors_filtered = torch.from_numpy(colors_filtered) if isinstance(colors_filtered, np.ndarray) else colors_filtered
        
        # 生成基于图片名称或内容的文件名
        import hashlib
        if image_names and len(image_names) > 0:
            # 使用extract_scene_id提取scene ID
            scene_id = extract_scene_id(image_names[0])
            # 清理文件名，移除非法字符
            safe_name = "".join(c for c in scene_id if c.isalnum() or c in ('-', '_'))
            ply_filename = f"result_{safe_name}.ply"
        else:
            # 回退到使用哈希值
            first_img_bytes = data['images'][0].encode('utf-8')
            img_hash = hashlib.md5(first_img_bytes).hexdigest()[:8]
            ply_filename = f"result_{img_hash}.ply"
        
        ply_path = f"outputs/{ply_filename}"
        os.makedirs("outputs", exist_ok=True)
        
        ply_b64 = write_ply(
            points_filtered, 
            colors_filtered,  # 使用过滤后的颜色数据
            ply_path
        )
        
        # # 编码PLY文件为base64
        # with open(ply_path, 'rb') as f:
        #     ply_b64 = base64.b64encode(f.read()).decode('utf-8')

        # 提取原始相机位姿信息（未经旋转）
        original_camera_poses = results['camera_poses'][0].cpu().numpy()
        camera_poses_list = []
        
        # 获取第一个相机作为参考
        reference_pose = original_camera_poses[0]
        R_ref = reference_pose[:3, :3]  # 参考相机的旋转矩阵
        
        from scipy.spatial.transform import Rotation as R_scipy
        
        for i, pose in enumerate(original_camera_poses):
            R_cw = pose[:3, :3]  # 旋转矩阵
            t_cw = pose[:3, 3]   # 相机位置（世界坐标系）
            
            # 计算相对于第一个相机的旋转角度（用于 azimuth_angle 和 elevation_angle）
            R_relative = R_cw @ R_ref.T  # 相对旋转矩阵
            rotation_relative = R_scipy.from_matrix(R_relative)
            
            # 使用 YX 欧拉角分解得到方位角和仰角
            try:
                euler_yx = rotation_relative.as_euler('yx', degrees=True)
                azimuth_from_cam1 = euler_yx[0]   # 方位角（左右）
                elevation_from_cam1 = euler_yx[1]  # 仰角（上下）
            except:
                # 如果YX分解失败，使用近似方法
                euler_xyz_rel = rotation_relative.as_euler('xyz', degrees=True)
                azimuth_from_cam1 = euler_xyz_rel[1]   # Y轴旋转
                elevation_from_cam1 = euler_xyz_rel[0]  # X轴旋转
            
            camera_poses_list.append({
                "camera_id": i + 1,
                "position": t_cw.tolist(),  # 相机中心位置 [x, y, z]
                "azimuth_angle": float(azimuth_from_cam1),    # 方位角（相对于相机1），可直接用于 API
                "elevation_angle": float(elevation_from_cam1)  # 仰角（相对于相机1），可直接用于 API
            })
        
        response_data = {
            "success": True,
            "ply_file": ply_b64,
            "ply_filename": ply_filename,
            "points_count": len(points_filtered) if isinstance(points_filtered, np.ndarray) else points_filtered.shape[0],
            "camera_poses": camera_poses_list,  # 添加原始相机位姿信息
            "camera_views": []
        }
        
        # 生成多视角图片（可选）
        if generate_views:
            try:
                if azimuth_angle is not None and elevation_angle is not None:
                    # 使用自定义角度生成图片
                    view_mode = "相机视角" if camera_view else "全局视角"
                    logger.info(f"使用自定义角度生成视角图片: 方位角={azimuth_angle}°, 仰角={elevation_angle}°, 参考相机={rotation_reference_camera}, 模式={view_mode}")
                    view_images = generate_custom_angle_views(
                        results, masks, imgs_rgb_tensor, azimuth_angle, elevation_angle,
                        points_filtered, colors_filtered, rotation_reference_camera=rotation_reference_camera,
                        camera_view=camera_view
                    )
                else:
                    # 使用默认的多视角生成
                    view_mode = "相机视角" if camera_view else "全局视角"
                    logger.info(f"使用默认多视角生成图片，参考相机={rotation_reference_camera}, 模式={view_mode}")
                    view_images = generate_camera_views(
                        results, masks, imgs_rgb_tensor, max_views_per_camera,
                        points_filtered, colors_filtered, rotation_reference_camera=rotation_reference_camera,
                        camera_view=camera_view
                    )  # 传递视角限制参数和过滤后的点云
                response_data["camera_views"] = view_images
            except Exception as e:
                logger.warning(f"生成多视角图片失败：{e}")
        
        logger.info("Pi3重建完成")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

def _prepare_points_and_cameras(results, masks, imgs_rgb_tensor, points_filtered=None, colors_filtered=None):
    """
    准备点云和相机数据的共同函数
    
    Args:
        results: Pi3模型的推理结果
        masks: 点云掩码
        imgs_rgb_tensor: RGB图像张量
        points_filtered: 可选的预过滤点云（已移除离群点）
        colors_filtered: 可选的预过滤颜色（已移除离群点）
        
    Returns:
        tuple: (points_sample, colors_sample, camera_centers, camera_poses)
    """
    # 获取点云和相机位置
    if points_filtered is not None and colors_filtered is not None:
        # 使用预过滤的点云（已经移除离群点）
        if isinstance(points_filtered, np.ndarray):
            points_3d = points_filtered
        else:
            points_3d = points_filtered.cpu().numpy() if hasattr(points_filtered, 'cpu') else np.array(points_filtered)
        
        if isinstance(colors_filtered, np.ndarray):
            colors_3d = colors_filtered
        else:
            colors_3d = colors_filtered.cpu().numpy() if hasattr(colors_filtered, 'cpu') else np.array(colors_filtered)
    else:
        # 使用原始的点云提取方式
        points_3d = results['points'][0][masks].cpu().numpy()
        colors_3d = imgs_rgb_tensor.permute(0, 2, 3, 1)[masks].cpu().numpy()  # 使用RGB颜色数据
    
    original_camera_poses = results['camera_poses'][0].cpu().numpy()  # 原始的camera-to-world矩阵
    
    # 不应用官方旋转 - 直接使用原始相机视角
    # 这样 (0,0) 角度就对应第一张输入图的真实视角
    camera_centers = []
    camera_poses_list = []
    for pose in original_camera_poses:
        camera_centers.append(pose[:3, 3])
        camera_poses_list.append(pose)

    camera_centers = np.array(camera_centers)
    camera_poses = np.array(camera_poses_list)  # 使用原始的camera-to-world矩阵
    
    # 子采样点云以提高渲染性能
    # 优化：增加到50万个点以提升细节（原30万）
    max_points_to_visualize = 500000
    if len(points_3d) > max_points_to_visualize:
        # 使用确定性的采样方法，基于点云数据本身而不是随机数
        # 这确保相同的点云总是产生相同的子集
        total_points = len(points_3d)
        step = total_points // max_points_to_visualize
        indices = np.arange(0, total_points, step)[:max_points_to_visualize]
        points_sample = points_3d[indices]
        colors_sample = colors_3d[indices]
        logger.info(f"点云子采样：从 {len(points_3d)} 个点中采样了 {max_points_to_visualize} 个点用于可视化")
    else:
        points_sample = points_3d
        colors_sample = colors_3d
        logger.info(f"使用全部 {len(points_3d)} 个点进行可视化")
    
    # 添加调试信息
    logger.info(f"点云处理完成: {len(points_sample)} 个点用于可视化")
    
    return points_sample, colors_sample, camera_centers, camera_poses


def _draw_cameras_visualization(ax, camera_centers, camera_poses, current_view_cam_idx, 
                              view_R_cam, view_t_cam, max_range, show_cameras=True):
    """
    专门的相机可视化函数，支持绘制多个相机的位置和位姿
    
    Args:
        ax: matplotlib 3D轴对象
        camera_centers: 所有相机的中心位置数组 (N, 3) - 已应用官方旋转
        camera_poses: 所有相机的姿态矩阵数组 (N, 4, 4) - camera-to-world矩阵
        current_view_cam_idx: 当前视角相机的索引
        view_R_cam: 当前视角的旋转矩阵 (world-to-camera)
        view_t_cam: 当前视角的平移向量
        max_range: 场景的最大范围，用于计算坐标轴长度
        show_cameras: 是否显示相机可视化
        
    Returns:
        tuple: (x_min, x_max, y_min, y_max, z_min, z_max) 包含相机后的边界范围
    """
    if not show_cameras:
        return None, None, None, None, None, None
    
    # 计算坐标轴的长度
    axis_length = max_range * 0.12
    
    # 用于边界计算的列表
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    
    # 为不同相机定义颜色
    camera_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 检查是否是恒等变换（相机坐标已经在正确的坐标系中）
    is_identity_transform = (np.allclose(view_R_cam, np.eye(3)) and 
                             np.allclose(view_t_cam, np.zeros(3)))
    
    # 遍历所有相机
    for cam_idx, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
        # 选择相机颜色
        cam_color = camera_colors[cam_idx % len(camera_colors)]
        
        # 将相机中心转换到当前视图坐标系
        if is_identity_transform:
            # 如果是恒等变换，说明相机坐标已经变换好了，直接使用
            cam_center_in_view = cam_center
        else:
            # 否则需要应用坐标变换和Y/Z翻转
            flip_transform = np.diag([1, -1, -1])  # 与点云保持一致的翻转
            cam_center_in_view = (view_R_cam @ cam_center.T).T + view_t_cam
            cam_center_in_view = (flip_transform @ cam_center_in_view.T).T
        
        # 相机姿态：camera-to-world旋转部分
        R_cam2world = cam_pose[:3, :3]
        
        # 转换到当前视图坐标系
        if is_identity_transform:
            # 如果是恒等变换，说明相机姿态已经变换好了，直接使用
            R_pose_in_view = R_cam2world
        else:
            # 否则需要应用坐标变换和翻转
            flip_transform = np.diag([1, -1, -1])
            R_pose_in_view = view_R_cam @ R_cam2world
            R_pose_in_view = flip_transform @ R_pose_in_view
        
        # 绘制相机锥形（视锥）
        frustum_length = axis_length * 0.8  # 锥形长度
        frustum_width = frustum_length * 0.3   # 锥形底面宽度的一半
        frustum_height = frustum_length * 0.3  # 锥形底面高度的一半
        
        # 相机的朝向（-Z方向，因为相机看向负Z轴）
        forward = -R_pose_in_view[:, 2]  # 相机朝向
        right = R_pose_in_view[:, 0]     # 相机右方向
        up = -R_pose_in_view[:, 1]       # 相机上方向（注意OpenCV坐标系Y向下）
        
        # 计算锥形的四个角点（在锥形底面，远离相机）
        far_center = cam_center_in_view + forward * frustum_length
        
        # 锥形底面的四个角点（大面在远处）
        corner1 = far_center + right * frustum_width + up * frustum_height     # 右上
        corner2 = far_center - right * frustum_width + up * frustum_height     # 左上  
        corner3 = far_center - right * frustum_width - up * frustum_height     # 左下
        corner4 = far_center + right * frustum_width - up * frustum_height     # 右下
        
        # 线条粗细和透明度
        line_width = 1.2 if cam_idx == current_view_cam_idx else 1
        alpha = 0.65 if cam_idx == current_view_cam_idx else 0.5
        
        # 绘制从相机中心到四个角点的线条（锥形边缘）
        for corner in [corner1, corner2, corner3, corner4]:
            ax.plot([cam_center_in_view[0], corner[0]],
                   [cam_center_in_view[1], corner[1]],
                   [cam_center_in_view[2], corner[2]], 
                   color=cam_color, linewidth=line_width, alpha=alpha)
        
        # 绘制锥形底面的矩形框架
        corners = [corner1, corner2, corner3, corner4, corner1]  # 闭合矩形
        for i in range(len(corners) - 1):
            ax.plot([corners[i][0], corners[i+1][0]],
                   [corners[i][1], corners[i+1][1]],
                   [corners[i][2], corners[i+1][2]], 
                   color=cam_color, linewidth=line_width, alpha=alpha)
        
        # 绘制相机中心点
        marker_size = 35 if cam_idx == current_view_cam_idx else 20
        ax.scatter(cam_center_in_view[0], cam_center_in_view[1], cam_center_in_view[2], 
                  c=cam_color, s=marker_size, marker='o', alpha=0.8, depthshade=False,
                  edgecolors='black', linewidth=0.8)
        
        # 添加相机编号标签
        label_pos = cam_center_in_view + np.array([axis_length * 0.3, 0, axis_length * 0.2])
        marker = '*' if cam_idx == current_view_cam_idx else ''  # 当前视角相机加星号标记
        ax.text(label_pos[0], label_pos[1], label_pos[2], 
               f'Cam{cam_idx+1}{marker}', fontsize=5, color='black', weight='bold',
               bbox=dict(boxstyle="round,pad=0.05", facecolor=cam_color, alpha=0.6))
        
        # 收集边界坐标（包括锥形的所有角点）
        coords_to_check = [
            cam_center_in_view,
            corner1, corner2, corner3, corner4,
            far_center
        ]
        
        for coord in coords_to_check:
            all_x_coords.append(coord[0])
            all_y_coords.append(coord[1])
            all_z_coords.append(coord[2])
    
    # 计算包含所有相机的边界
    if all_x_coords:
        x_min_cam = min(all_x_coords)
        x_max_cam = max(all_x_coords)
        y_min_cam = min(all_y_coords)
        y_max_cam = max(all_y_coords)
        z_min_cam = min(all_z_coords)
        z_max_cam = max(all_z_coords)
        
        return x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam
    else:
        return None, None, None, None, None, None

def _create_view_image(points_sample, colors_sample, camera_centers, camera_poses, cam_idx, 
                      azim_angle, elev_angle, view_name, show_camera_axes=True, show_all_cameras=True,
                      ref_cam_idx: int = 0, camera_view: bool = False):
    """
    创建单个视角图片的共同函数
    
    Args:
        points_sample: 采样后的点云
        colors_sample: 采样后的颜色
        camera_centers: 相机中心位置
        camera_poses: 相机姿态
        cam_idx: 相机索引
        azim_angle: 方位角
        elev_angle: 仰角
        view_name: 视角名称
        show_camera_axes: 是否显示相机坐标轴（仅显示当前相机）
        show_all_cameras: 是否显示所有相机的位置和位姿
        ref_cam_idx: 参考相机索引（用于旋转中心）
        camera_view: 是否使用相机视角模式（True=从相机位置观察，False=全局视角）
        
    Returns:
        str: base64编码的图片数据
    """
    # 根据 camera_view 参数选择使用哪个相机进行坐标变换
    if camera_view:
        # 相机视角模式：使用参考相机（ref_cam_idx）的位置和姿态
        safe_ref_idx = max(0, min(ref_cam_idx, len(camera_poses) - 1))
        view_cam_pose = camera_poses[safe_ref_idx]
        logger.info(f"使用相机视角模式：从相机 {safe_ref_idx + 1} 的位置观察场景")
    else:
        # 全局视角模式：使用当前相机（cam_idx）
        view_cam_pose = camera_poses[cam_idx]
    
    # 提取视角相机的旋转矩阵和平移（camera-to-world）
    if view_cam_pose.shape == (4, 4):
        R_cw = view_cam_pose[:3, :3]
        t_cw = view_cam_pose[:3, 3]
    else:
        R_cw = view_cam_pose[:3, :3]
        t_cw = view_cam_pose[:3, 3]

    # 计算world-to-camera变换，将点云从世界坐标系转换到视角相机坐标系
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    points_cam = (R_wc @ points_sample.T).T + t_wc

    # OpenCV相机坐标系: X右, Y下, Z前
    # 为了正确显示，需要翻转Y和Z轴以适配标准的右手坐标系
    # 这样可以将OpenCV坐标系 (X右,Y下,Z前) 转换为 (X右,Y上,Z后)
    flip_transform = np.diag([1, -1, -1])  # 翻转Y和Z轴
    points_cam = (flip_transform @ points_cam.T).T
    
    # 注意：camera_view 模式下的点云过滤将在旋转之后进行，以确保正确过滤
    
    # 使方位角/仰角作为相对于第一个相机坐标轴的旋转
    # 水平旋转（azimuth）：绕第一个相机的垂直轴（Y轴）
    # 竖直旋转（elevation）：绕第一个相机的水平轴（X轴）
    # 关键：旋转围绕第一帧相机的中心点，并使用第一个相机的坐标轴系统
    # 同时旋转点云和相机，保持它们的相对位置关系
    try:
        if abs(azim_angle) > 1e-6 or abs(elev_angle) > 1e-6:
            # 获取参考相机（可配置，默认第一个相机）的世界坐标位置和姿态
            safe_ref_idx = 0 if camera_poses is None else max(0, min(ref_cam_idx, len(camera_poses) - 1))
            first_cam_world_pos = camera_poses[safe_ref_idx][:3, 3]
            first_cam_R_cw = camera_poses[safe_ref_idx][:3, :3]  # 参考相机的旋转矩阵（camera-to-world）
            
            # 将参考相机的位置转换到当前相机坐标系
            first_cam_in_current = (R_wc @ first_cam_world_pos.T).T + t_wc
            
            # 应用Y/Z翻转变换（与点云保持一致）
            first_cam_center = (flip_transform @ first_cam_in_current.T).T
            
            # 获取参考相机的坐标轴（在当前视图坐标系中）
            # 将参考相机的旋转矩阵转换到当前相机坐标系
            first_cam_R_in_current = R_wc @ first_cam_R_cw
            first_cam_R_in_current = flip_transform @ first_cam_R_in_current  # 应用翻转
            
            # 提取第一个相机的坐标轴
            first_cam_x_axis = first_cam_R_in_current[:, 0]  # 第一个相机的X轴（水平轴）
            first_cam_y_axis = first_cam_R_in_current[:, 1]  # 第一个相机的Y轴（垂直轴）
            first_cam_z_axis = first_cam_R_in_current[:, 2]  # 第一个相机的Z轴（朝向）
            
            # 将点云移到原点（以第一个相机中心为基准）
            points_centered = points_cam - first_cam_center
            
            # 按照参考相机的坐标轴进行旋转
            # 1. 先绕参考相机的Y轴（垂直轴）旋转 - 水平旋转（方位角）
            if abs(azim_angle) > 1e-6:
                R_azim = R.from_rotvec(np.radians(azim_angle) * first_cam_y_axis).as_matrix()
                points_centered = (R_azim @ points_centered.T).T
            
            # 2. 再绕参考相机的X轴（水平轴）旋转 - 竖直旋转（仰角）
            if abs(elev_angle) > 1e-6:
                R_elev = R.from_rotvec(np.radians(elev_angle) * first_cam_x_axis).as_matrix()
                points_centered = (R_elev @ points_centered.T).T
            
            # 将点云移回（加上第一个相机中心）
            points_cam = points_centered + first_cam_center
            
            # 组合旋转矩阵（用于相机和视图变换）
            R_azim_full = R.from_rotvec(np.radians(azim_angle) * first_cam_y_axis).as_matrix() if abs(azim_angle) > 1e-6 else np.eye(3)
            R_elev_full = R.from_rotvec(np.radians(elev_angle) * first_cam_x_axis).as_matrix() if abs(elev_angle) > 1e-6 else np.eye(3)
            R_rel = R_elev_full @ R_azim_full
            
            # 更新视图变换
            view_R_cam = R_rel @ R_wc
            view_t_cam = R_rel @ t_wc
            
            # 旋转所有相机的位置和姿态（围绕参考相机中心）
            rotated_camera_centers = []
            rotated_camera_poses = []
            for i, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
                # 将相机中心转换到当前相机坐标系并应用翻转
                cam_center_in_current = (R_wc @ cam_center.T).T + t_wc
                cam_center_flipped = (flip_transform @ cam_center_in_current.T).T
                
                # 围绕参考相机中心旋转
                cam_center_centered = cam_center_flipped - first_cam_center
                cam_center_rotated = (R_rel @ cam_center_centered.T).T
                cam_center_final = cam_center_rotated + first_cam_center
                
                # 旋转相机姿态矩阵
                cam_R = cam_pose[:3, :3]
                cam_R_in_current = R_wc @ cam_R
                cam_R_flipped = flip_transform @ cam_R_in_current
                cam_R_rotated = R_rel @ cam_R_flipped
                
                # 构建旋转后的相机姿态矩阵
                rotated_pose = np.eye(4)
                rotated_pose[:3, :3] = cam_R_rotated
                rotated_pose[:3, 3] = cam_center_final
                
                rotated_camera_centers.append(cam_center_final)
                rotated_camera_poses.append(rotated_pose)
            
            # 用旋转后的相机数据替换原始数据
            camera_centers = np.array(rotated_camera_centers)
            camera_poses = np.array(rotated_camera_poses)
        else:
            view_R_cam = R_wc
            view_t_cam = t_wc
    except Exception:
        # 若旋转计算失败，回退为不旋转
        view_R_cam = R_wc
        view_t_cam = t_wc
    
    # 在相机视角模式下，使用宽视野过滤（在所有旋转完成之后）
    if camera_view:
        # 注意：由于我们应用了 flip_transform (翻转Y和Z)，坐标系从 OpenCV (X右,Y下,Z前) 变成了 (X右,Y上,Z后)
        # 因此，在翻转后的坐标系中，Z < 0 才是相机前方（朝向 -Z 方向）
        
        # 使用宽视野过滤：保留前方 200° 视野范围内的点（左右各 100°，上下各 100°）
        # 计算每个点相对于相机朝向（-Z 方向）的角度
        # 点的方向向量
        point_directions = points_cam / (np.linalg.norm(points_cam, axis=1, keepdims=True) + 1e-8)
        # 相机朝向（-Z 方向）
        camera_forward = np.array([0, 0, -1])
        # 计算点与相机朝向的夹角余弦值
        cos_angles = point_directions @ camera_forward
        
        # 保留前方 200° 视野内的点（从中心轴左右各 100°，上下各 100°）
        # cos(100°) ≈ -0.174，所以保留 cos > -0.2 的点（比 100° 稍宽一点）
        fov_angle_threshold = np.cos(np.radians(110))  # 100° 的余弦值 ≈ -0.174
        
        # 创建视锥体掩码：点与相机朝向夹角 < 100° (即 cos > -0.174)
        fov_mask = cos_angles > fov_angle_threshold
        
        num_total = len(points_cam)
        if np.sum(fov_mask) > 0:
            points_cam = points_cam[fov_mask]
            colors_sample = colors_sample[fov_mask]
            logger.info(f"相机视角模式：使用200°宽视野过滤，保留 {np.sum(fov_mask)}/{num_total} 个点")
        else:
            logger.warning(f"相机视角模式：视野内没有点云（共{num_total}个点），保留所有点")
    
    # 计算点云的实际范围，用于自适应缩放
    # 使用百分位数去除离群点影响(参考官方demo_gradio.py)
    # 优化：使用5%/95%（官方配置）获得更准确的场景范围
    lower_percentile = np.percentile(points_cam, 7, axis=0)
    upper_percentile = np.percentile(points_cam, 93, axis=0)
    
    x_range = upper_percentile[0] - lower_percentile[0]
    y_range = upper_percentile[1] - lower_percentile[1]
    z_range = upper_percentile[2] - lower_percentile[2]
    max_range = max(x_range, y_range, z_range)
    
    # 计算点的大小 - 优化：更细腻的点云显示
    if max_range > 0:
        # 增加点的密度范围，使点云更细腻
        base_point_size = max(0.03, min(0.15, 40.0 / max_range))
        # 在 camera_view 模式下，增大点的大小以便更好地观察
        if camera_view:
            point_size = base_point_size * 2.5  # 增大2倍
            alpha = 0.9
        else:
            point_size = base_point_size
            alpha = 0.8
    else:
        point_size = 1.0
        alpha = 0.8
        if camera_view:
            point_size = 2.5  # camera_view模式下使用更大的默认值
            alpha = 0.9
    
    # 创建图形 - 配置兼容模式，使用更大的尺寸和更高的DPI
    fig = plt.figure(figsize=(12, 10), dpi=500)
    ax = fig.add_subplot(111, projection='3d')
    
    # 禁用可能导致问题的3D效果，确保matplotlib兼容性
    ax.computed_zorder = False  # 禁用自动Z序计算
    
    # 绘制点云 - 修复matplotlib兼容性问题
    try:
        # 确保颜色数据格式正确
        if colors_sample.shape[-1] == 3:  # RGB
            # 确保颜色值在正确范围内
            colors_normalized = np.clip(colors_sample, 0, 1)
        else:
            colors_normalized = colors_sample
        
        scatter = ax.scatter(
            points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
            c=colors_normalized,
            s=point_size,
            alpha=alpha, 
            edgecolors='none',
            depthshade=True,  # 重新启用深度阴影
            linewidth=0
        )
        
        # 尝试设置散点的剪切属性（兼容性处理）
        try:
            scatter.set_clip_on(False)
        except (AttributeError, TypeError):
            # 忽略不支持的属性
            pass
            
    except Exception as e:
        logger.warning(f"RGB散点绘制失败，尝试单色绘制，原因: {e}")
        # 降级到基本绘制方法，使用单色
        try:
            scatter = ax.scatter(
                points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
                c='steelblue',  # 使用明显的蓝色
                s=point_size * 2,  # 增大点的大小以确保可见
                alpha=0.9,
                edgecolors='darkblue',
                linewidth=0.1
            )
        except Exception as e2:
            logger.error(f"单色散点绘制也失败: {e2}")
            # 最后的降级：使用plot3D绘制线条
            ax.plot3D(points_cam[:, 0], points_cam[:, 1], points_cam[:, 2], 
                     'bo', markersize=1, alpha=0.5)
    
    # 使用新的相机可视化函数
    # 注意：相机数据已经在上面旋转好了，这里使用恒等变换直接绘制
    # 在相机视角模式下，不显示相机标记（因为我们就在相机位置观察）
    show_cameras_in_view = show_all_cameras and not camera_view
    
    if show_cameras_in_view:
        # 显示所有相机（相机已经旋转过，使用恒等变换直接绘制）
        identity_R = np.eye(3)
        identity_t = np.zeros(3)
        x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam = _draw_cameras_visualization(
            ax, camera_centers, camera_poses, cam_idx, identity_R, identity_t, max_range, show_cameras=True
        )
        
        # 计算包含点云和相机的边界
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
    elif show_camera_axes and not camera_view:
        # 只显示当前相机（保持原有逻辑兼容性，相机已经旋转过）
        single_camera_centers = np.array([camera_centers[cam_idx]])
        single_camera_poses = np.array([camera_poses[cam_idx]])
        
        identity_R = np.eye(3)
        identity_t = np.zeros(3)
        x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam = _draw_cameras_visualization(
            ax, single_camera_centers, single_camera_poses, 0, identity_R, identity_t, max_range, show_cameras=True
        )
        
        # 计算包含点云和相机的边界
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
        # 不显示相机坐标轴时的边界（包括相机视角模式）
        x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
        y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
        z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    
    # 设置matplotlib观察方向
    if camera_view:
        # 相机视角模式：设置为第一人称视角
        # elev=0 表示水平观察，azim=-90 表示沿着 -Z 方向（相机朝向）
        ax.view_init(elev=0.0, azim=-90.0)
        # 调整视角距离，稍微往后退一点以看到更多场景
        ax.dist = 7  # 增大距离，视角往后退（原来是8，现在是11）
    else:
        # 全局视角模式：使用默认的俯瞰视角
        ax.view_init(elev=0.0, azim=-90.0)
        ax.dist = 10  # 默认距离
    
    # 计算边界框
    margin_factor = 0.02
    x_margin = (x_max - x_min) * margin_factor
    y_margin = (y_max - y_min) * margin_factor
    z_margin = (z_max - z_min) * margin_factor
    
    # 设置坐标轴范围
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    # 设置坐标轴标签和颜色
    ax.set_xlabel('X', fontsize=14, fontweight='bold', color='red')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold', color='green')
    ax.set_zlabel('Z', fontsize=14, fontweight='bold', color='blue')
    
    # 设置坐标轴颜色
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green') 
    ax.zaxis.label.set_color('blue')
    
    # 去除坐标轴和刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    # 设置背景为白色，但确保点云可见
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    
    # 设置背景颜色为浅灰色，以便更容易看到点云
    ax.set_facecolor('lightgray')
    
    # 确保3D轴比例相等
    # ax.set_box_aspect([x_max-x_min, y_max-y_min, z_max-z_min])  # 可能在某些matplotlib版本中不可用

    # 保存为字节流，使用更高的DPI和质量
    buf = io.BytesIO()
    try:
        # 强制渲染并保存，避免matplotlib兼容性问题
        try:
            fig.canvas.draw()  # 强制渲染
        except Exception as render_e:
            pass  # 忽略渲染警告
        
        plt.savefig(buf, format='png', dpi=500, bbox_inches='tight', 
                   pad_inches=0.05, facecolor='white', edgecolor='none',
                   transparent=False)
        buf.seek(0)
        
        # 编码为base64
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
    except Exception as e:
        logger.error(f"保存图片时出错: {e}")
        # 创建一个简单的错误图片，使用更大的尺寸
        try:
            temp_fig = plt.figure(figsize=(8, 6))
            temp_ax = temp_fig.add_subplot(111)
            temp_ax.text(0.5, 0.5, f"生成失败:\n{str(e)}", ha='center', va='center', transform=temp_ax.transAxes)
            temp_ax.set_xticks([])
            temp_ax.set_yticks([])
            
            temp_buf = io.BytesIO()
            temp_fig.canvas.draw()  # 强制渲染
            plt.savefig(temp_buf, format='png', dpi=300, bbox_inches='tight')
            temp_buf.seek(0)
            img_b64 = base64.b64encode(temp_buf.read()).decode('utf-8')
            
            plt.close(temp_fig)
            temp_buf.close()
        except Exception as fallback_e:
            logger.error(f"备用图片生成也失败: {fallback_e}")
            # 最终降级：返回一个简单的base64字符串表示错误
            error_msg = f"Image generation failed: {str(e)}"
            img_b64 = base64.b64encode(error_msg.encode('utf-8')).decode('utf-8')
            
    finally:
        # 确保释放资源
        try:
            plt.close(fig)  # 重要：关闭图形以释放内存和避免兼容性问题
        except:
            pass
        try:
            buf.close()
        except:
            pass
    
    return img_b64


def generate_camera_views(results, masks, imgs_rgb_tensor, max_views_per_camera=15, points_filtered=None, colors_filtered=None,
                          rotation_reference_camera: int = 1, camera_view: bool = False):
    """生成多视角图片"""
    try:
        # 准备点云和相机数据
        points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
            results, masks, imgs_rgb_tensor, points_filtered, colors_filtered
        )
        
        # 生成关键视角
        view_angles = [
            (0, 0, "camera_front"),           # 正面
            (-45, 0, "camera_left_45"),       # 左45度
            (45, 0, "camera_right_45"),       # 右45度
            (0, -45, "camera_front_down"),    # 正面向下45度
            (0, 45, "camera_front_up"),    # 正面向上45度

        ]
        
        view_images = []
        
        for cam_idx in range(min(4, len(camera_centers))):  # 只处理前2个相机
            # 根据max_views_per_camera限制视角数量
            limited_view_angles = view_angles[:max_views_per_camera]
            
            for azim_offset, elev_offset, view_name in limited_view_angles:
                # 判断是否显示相机坐标轴（只在几个关键视角显示）
                show_camera_axes = view_name in ["camera_front", "camera_left_30", "camera_right_30"]
                
                # 补偿90度：由于坐标系翻转，需要在仰角上加90度使(0,0)对应正面
                adjusted_elev = elev_offset + 100.0
                
                # 创建视角图片（显示所有相机）
                # 每个相机围绕自己旋转，所以 ref_cam_idx = cam_idx
                img_b64 = _create_view_image(
                    points_sample, colors_sample, camera_centers, camera_poses,
                    cam_idx, azim_offset, adjusted_elev, view_name, show_camera_axes, show_all_cameras=True,
                    ref_cam_idx=cam_idx,  # 每个相机围绕自己旋转
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
        return []


def generate_custom_angle_views(results, masks, imgs_rgb_tensor, azimuth_angle, elevation_angle, points_filtered=None, colors_filtered=None,
                                rotation_reference_camera: int = 1, camera_view: bool = False):
    """
    根据自定义角度生成视角图片
    
    Args:
        results: Pi3模型的推理结果
        masks: 点云掩码
        imgs_rgb_tensor: RGB图像张量
        azimuth_angle: 方位角（左右旋转），单位：度
        elevation_angle: 仰角（上下旋转），单位：度
        points_filtered: 可选的预过滤点云（已移除离群点）
        colors_filtered: 可选的预过滤颜色（已移除离群点）
        rotation_reference_camera: 参考相机索引（1-based），控制使用哪个相机的视角
        camera_view: 是否使用相机视角模式
        
    Returns:
        生成的视角图片列表
    """
    try:
        # 准备点云和相机数据（复用共同逻辑）
        points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
            results, masks, imgs_rgb_tensor, points_filtered, colors_filtered
        )
        
        # 根据 rotation_reference_camera 选择相机索引（1-based 转 0-based）
        cam_idx = max(0, min(int(rotation_reference_camera) - 1, len(camera_centers) - 1))
        logger.info(f"使用相机 {cam_idx + 1} 的视角生成图片（rotation_reference_camera={rotation_reference_camera}）")
        
        # 如果角度为0°/0°，直接返回对应相机的输入图片
        if abs(azimuth_angle) < 1e-6 and abs(elevation_angle) < 1e-6:
            try:
                # 返回指定相机索引的图片
                if cam_idx < len(imgs_rgb_tensor):
                    img_ref = imgs_rgb_tensor[cam_idx].detach().cpu().numpy()  # C,H,W in [0,1]
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
            except Exception as e_img:
                logger.warning(f"返回原始图片失败，回退到点云渲染: {e_img}")
        
        view_images = []
        view_name = f"custom_azim_{azimuth_angle}_elev_{elevation_angle}"

        # 补偿100度：由于坐标系翻转，需要在仰角上加100度使(0,0)对应正面
        adjusted_elevation = elevation_angle + 100.0
        
        # 创建自定义角度视角图片
        # cam_idx 和 ref_cam_idx 都使用同一个相机，这样既控制观察视角，也控制旋转中心
        img_b64 = _create_view_image(
            points_sample, colors_sample, camera_centers, camera_poses,
            cam_idx, azimuth_angle, adjusted_elevation, view_name, show_camera_axes=False, show_all_cameras=True,
            ref_cam_idx=cam_idx,  # 使用同一个相机作为旋转参考
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
        return []

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Pi3 3D Reconstruction Server')
    parser.add_argument('--checkpoint_path', type=str, default='spagent/external_experts/checkpoints/pi3',
                        help='Path to Pi3 model checkpoint directory or file (default: checkpoints/pi3/)')
    parser.add_argument('--port', type=int, default=20021,
                        help='Port to run the server on (default: 20021)')
    
    args = parser.parse_args()
    
    logger.info("正在启动Pi3服务器...")
    logger.info(f"模型路径: {args.checkpoint_path}")
    logger.info(f"服务端口: {args.port}")
    
    # 检查模型文件或目录是否存在
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"找不到模型文件或目录：{args.checkpoint_path}")
        exit(1)
    
    # 加载指定模型
    if not load_model(args.checkpoint_path):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=args.port, debug=False) 