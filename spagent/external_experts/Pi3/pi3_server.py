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
        
        # 新增：支持自定义角度参数
        azimuth_angle = data.get('azimuth_angle', None)  # 左右旋转角度（方位角）
        elevation_angle = data.get('elevation_angle', None)  # 上下旋转角度（仰角）
        
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
        
        # 生成基于图片名称或内容的文件名
        import hashlib
        if image_names and len(image_names) > 0:
            # 使用第一张图片的文件名（去掉扩展名）
            first_name = os.path.splitext(image_names[0])[0]
            # 清理文件名，移除非法字符
            safe_name = "".join(c for c in first_name if c.isalnum() or c in ('-', '_'))
            ply_filename = f"result_{safe_name}.ply"
        else:
            # 回退到使用哈希值
            first_img_bytes = data['images'][0].encode('utf-8')
            img_hash = hashlib.md5(first_img_bytes).hexdigest()[:8]
            ply_filename = f"result_{img_hash}.ply"
        
        ply_path = f"outputs/{ply_filename}"
        os.makedirs("outputs", exist_ok=True)
        
        write_ply(
            results['points'][0][masks].cpu(), 
            imgs_rgb_tensor.permute(0, 2, 3, 1)[masks].cpu(),  # 使用RGB版本的颜色数据
            ply_path
        )
        
        # 编码PLY文件为base64
        with open(ply_path, 'rb') as f:
            ply_b64 = base64.b64encode(f.read()).decode('utf-8')

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
            "points_count": masks.sum().item(),
            "camera_poses": camera_poses_list,  # 添加原始相机位姿信息
            "camera_views": []
        }
        
        # 生成多视角图片（可选）
        if generate_views:
            try:
                if azimuth_angle is not None and elevation_angle is not None:
                    # 使用自定义角度生成图片
                    logger.info(f"使用自定义角度生成视角图片: 方位角={azimuth_angle}°, 仰角={elevation_angle}°")
                    view_images = generate_custom_angle_views(results, masks, imgs_rgb_tensor, azimuth_angle, elevation_angle)
                else:
                    # 使用默认的多视角生成
                    logger.info("使用默认多视角生成图片")
                    view_images = generate_camera_views(results, masks, imgs_rgb_tensor, max_views_per_camera)  # 传递视角限制参数
                response_data["camera_views"] = view_images
            except Exception as e:
                logger.warning(f"生成多视角图片失败：{e}")
        
        logger.info("Pi3重建完成")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

def _prepare_points_and_cameras(results, masks, imgs_rgb_tensor):
    """
    准备点云和相机数据的共同函数
    
    Args:
        results: Pi3模型的推理结果
        masks: 点云掩码
        imgs_rgb_tensor: RGB图像张量
        
    Returns:
        tuple: (points_sample, colors_sample, camera_centers, camera_poses)
    """
    # 获取点云和相机位置
    points_3d = results['points'][0][masks].cpu().numpy()
    original_camera_poses = results['camera_poses'][0].cpu().numpy()  # 这是camera-to-world矩阵
    colors_3d = imgs_rgb_tensor.permute(0, 2, 3, 1)[masks].cpu().numpy()  # 使用RGB颜色数据
    
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
    
    # 应用官方旋转
    r_y = R.from_euler('y', 100, degrees=True)
    r_x = R.from_euler('x', 155, degrees=True)
    official_rotation = r_x * r_y
    
    # 计算坐标轴的长度
    axis_length = max_range * 0.12
    
    # 用于边界计算的列表
    all_x_coords = []
    all_y_coords = []
    all_z_coords = []
    
    # 为不同相机定义颜色
    camera_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # 遍历所有相机
    for cam_idx, (cam_center, cam_pose) in enumerate(zip(camera_centers, camera_poses)):
        # 选择相机颜色
        cam_color = camera_colors[cam_idx % len(camera_colors)]
        
        # 将相机中心转换到当前视图坐标系，并应用Y/Z翻转
        flip_transform = np.diag([1, -1, -1])  # 与点云保持一致的翻转
        cam_center_in_view = (view_R_cam @ cam_center.T).T + view_t_cam
        cam_center_in_view = (flip_transform @ cam_center_in_view.T).T
        
        # Pi3输出的是camera-to-world矩阵，提取旋转部分
        R_cam2world = cam_pose[:3, :3]
        
        # 转换到当前视图坐标系，并应用翻转
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
                      azim_angle, elev_angle, view_name, show_camera_axes=True, show_all_cameras=True):
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
        
    Returns:
        str: base64编码的图片数据
    """
    cam_center = camera_centers[cam_idx]
    cam_pose = camera_poses[cam_idx]
    
    # 提取相机的旋转矩阵和平移
    if cam_pose.shape == (4, 4):
        R_cam = cam_pose[:3, :3]
        t_cam = cam_pose[:3, 3]
    else:
        R_cw = cam_pose[:3, :3]
        t_cw = cam_pose[:3, 3]

    # 计算world-to-camera变换，将点云从世界坐标系转换到相机坐标系
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw
    points_cam = (R_wc @ points_sample.T).T + t_wc

    # OpenCV相机坐标系: X右, Y下, Z前
    # 为了正确显示，需要翻转Y和Z轴以适配标准的右手坐标系
    # 这样可以将OpenCV坐标系 (X右,Y下,Z前) 转换为 (X右,Y上,Z后)
    flip_transform = np.diag([1, -1, -1])  # 翻转Y和Z轴
    points_cam = (flip_transform @ points_cam.T).T
    
    # 使方位角/仰角作为相对于该相机视角的拖动（在相机坐标系内做旋转）
    # yaw: 绕相机Y轴（左右），pitch: 绕相机X轴（上下）
    try:
        if abs(azim_angle) > 1e-6 or abs(elev_angle) > 1e-6:
            R_yaw = R.from_euler('y', azim_angle, degrees=True).as_matrix()
            R_pitch = R.from_euler('x', elev_angle, degrees=True).as_matrix()
            R_rel = R_yaw @ R_pitch
            points_cam = (R_rel @ points_cam.T).T
            # 更新视图变换：在world-to-camera的基础上叠加相对旋转
            view_R_cam = R_rel @ R_wc
            view_t_cam = R_rel @ t_wc
        else:
            view_R_cam = R_wc
            view_t_cam = t_wc
    except Exception:
        # 若旋转计算失败，回退为不旋转
        view_R_cam = R_wc
        view_t_cam = t_wc
    
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
        point_size = max(0.03, min(0.15, 40.0 / max_range))
    else:
        point_size = 1.0
    
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
            alpha=0.8, 
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
    if show_all_cameras:
        # 显示所有相机
        x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam = _draw_cameras_visualization(
            ax, camera_centers, camera_poses, cam_idx, R_cam, t_cam, max_range, show_cameras=True
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
    elif show_camera_axes:
        # 只显示当前相机（保持原有逻辑兼容性）
        single_camera_centers = np.array([camera_centers[cam_idx]])
        single_camera_poses = np.array([camera_poses[cam_idx]])
        
        x_min_cam, x_max_cam, y_min_cam, y_max_cam, z_min_cam, z_max_cam = _draw_cameras_visualization(
            ax, single_camera_centers, single_camera_poses, 0, R_cam, t_cam, max_range, show_cameras=True
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
        # 不显示相机坐标轴时的边界
        x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
        y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
        z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
    
    # 设置视角，添加一些默认偏移以避免完全正视的空白视图
    base_elev = 15.0
    elev = base_elev + elev_angle
    azim = 225.0 + azim_angle
    
    # 如果角度导致完全正视或者在特殊位置，稍作调整
    if abs(elev_angle) < 1e-3 and abs(azim_angle) < 1e-3:
        # 当两个角度都为0时，稍作调整避免空白视图
        elev += 5.0  # 稍微向上倾斜
        azim += 5.0  # 稍微旋转
    
    ax.view_init(elev=elev, azim=azim)
    
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


def generate_camera_views(results, masks, imgs_rgb_tensor, max_views_per_camera=15):
    """生成多视角图片"""
    try:
        # 准备点云和相机数据
        points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
            results, masks, imgs_rgb_tensor
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
        
        for cam_idx in range(min(2, len(camera_centers))):  # 只处理前2个相机
            # 根据max_views_per_camera限制视角数量
            limited_view_angles = view_angles[:max_views_per_camera]
            
            for azim_offset, elev_offset, view_name in limited_view_angles:
                # 判断是否显示相机坐标轴（只在几个关键视角显示）
                show_camera_axes = view_name in ["camera_front", "camera_left_30", "camera_right_30"]
                
                # 补偿90度：由于坐标系翻转，需要在仰角上加90度使(0,0)对应正面
                adjusted_elev = elev_offset + 100.0
                
                # 创建视角图片（显示所有相机）
                img_b64 = _create_view_image(
                    points_sample, colors_sample, camera_centers, camera_poses,
                    cam_idx, azim_offset, adjusted_elev, view_name, show_camera_axes, show_all_cameras=True
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


def generate_custom_angle_views(results, masks, imgs_rgb_tensor, azimuth_angle, elevation_angle):
    """
    根据自定义角度生成视角图片
    
    Args:
        results: Pi3模型的推理结果
        masks: 点云掩码
        imgs_rgb_tensor: RGB图像张量
        azimuth_angle: 方位角（左右旋转），单位：度
        elevation_angle: 仰角（上下旋转），单位：度
        
    Returns:
        生成的视角图片列表
    """
    try:
        # 准备点云和相机数据（复用共同逻辑）
        points_sample, colors_sample, camera_centers, camera_poses = _prepare_points_and_cameras(
            results, masks, imgs_rgb_tensor
        )
        
        view_images = []
        
        # 只处理第一个相机的视角
        cam_idx = 0
        view_name = f"custom_azim_{azimuth_angle}_elev_{elevation_angle}"

        # 补偿100度：由于坐标系翻转，需要在仰角上加100度使(0,0)对应正面
        adjusted_elevation = elevation_angle + 100.0
        
        # 创建自定义角度视角图片（显示所有相机）
        img_b64 = _create_view_image(
            points_sample, colors_sample, camera_centers, camera_poses,
            cam_idx, azimuth_angle, adjusted_elevation, view_name, show_camera_axes=False, show_all_cameras=True
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
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/pi3/',
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