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
import matplotlib.pyplot as plt
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
    """加载Pi3模型"""
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
            logger.info(f"正在从 {checkpoint_path} 加载模型权重...")
            try:
                weight = load_file(checkpoint_path)
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
        logger.info(f"健康检查结果：{status}")
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
        
        logger.info(f"推理成功，输出包含：{list(results.keys())}")
        logger.info("测试推理完成")
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
        
        # 获取可选参数
        conf_threshold = data.get('conf_threshold', 0.1)  # 置信度阈值
        rtol = data.get('rtol', 0.03)  # 深度边缘检测阈值
        generate_views = data.get('generate_views', True)  # 是否生成多视角图片
        max_views_per_camera = data.get('max_views_per_camera', 7)  # 减少默认视角数量以提高性能
        
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
                
                if new_h != h or new_w != w:
                    logger.info(f"调整图像尺寸从 {h}x{w} 到 {new_h}x{new_w}")
                    image_bgr = cv2.resize(image_bgr, (new_w, new_h))
                
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
        
        response_data = {
            "success": True,
            "ply_file": ply_b64,
            "ply_filename": ply_filename,
            "points_count": masks.sum().item(),
            "camera_views": []
        }
        
        # 生成多视角图片（可选）
        if generate_views:
            try:
                view_images = generate_camera_views(results, masks, imgs_rgb_tensor, max_views_per_camera)  # 传递视角限制参数
                response_data["camera_views"] = view_images
            except Exception as e:
                logger.warning(f"生成多视角图片失败：{e}")
        
        logger.info("Pi3重建完成")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

def generate_camera_views(results, masks, imgs_rgb_tensor, max_views_per_camera=15):
    """生成多视角图片"""
    try:
        # 获取点云和相机位置
        points_3d = results['points'][0][masks].cpu().numpy()
        camera_poses = results['camera_poses'][0].cpu().numpy()
        colors_3d = imgs_rgb_tensor.permute(0, 2, 3, 1)[masks].cpu().numpy()  # 使用RGB颜色数据
        
        # 应用官方的场景旋转 (Y轴100°, X轴155°)
        r_y = R.from_euler('y', 100, degrees=True)
        r_x = R.from_euler('x', 155, degrees=True)
        official_rotation = r_x * r_y
        points_3d = official_rotation.apply(points_3d)
        
        # 提取相机位置并应用官方旋转
        camera_centers = []
        for pose in camera_poses:
            if pose.shape == (4, 4):
                R_cam = pose[:3, :3]
                t_cam = pose[:3, 3]
            else:
                R_cam = pose[:, :3]
                t_cam = pose[:, 3]
            
            camera_center = -R_cam.T @ t_cam
            camera_center = official_rotation.apply(camera_center.reshape(1, -1))[0]
            camera_centers.append(camera_center)
        
        camera_centers = np.array(camera_centers)
        
        # 子采样点云以提高渲染性能，最多可视化100000个点
        max_points_to_visualize = 100000
        if len(points_3d) > max_points_to_visualize:
            indices = np.random.choice(len(points_3d), max_points_to_visualize, replace=False)
            points_sample = points_3d[indices]
            colors_sample = colors_3d[indices]
            logger.info(f"点云子采样：从 {len(points_3d)} 个点中采样了 {max_points_to_visualize} 个点用于可视化")
        else:
            points_sample = points_3d
            colors_sample = colors_3d
            logger.info(f"使用全部 {len(points_3d)} 个点进行可视化")
        
        # 生成关键视角，减少数量以提高性能
        view_angles = [
            (0, 0, "camera_front"),           # 正面
            (-30, 0, "camera_left_30"),       # 左30度
            (30, 0, "camera_right_30"),       # 右30度
            (-45, 0, "camera_left_45"),       # 左45度
            (45, 0, "camera_right_45"),       # 右45度
            (0, 10, "camera_front_up"),       # 正面向上10度
            (0, -10, "camera_front_down"),    # 正面向下10度
        ]
        
        view_images = []
        
        for cam_idx, (cam_center, cam_pose) in enumerate(zip(camera_centers[:2], camera_poses[:2])):  # 只处理前2个相机
            # 提取相机的旋转矩阵和平移
            if cam_pose.shape == (4, 4):
                R_cam = cam_pose[:3, :3]
                t_cam = cam_pose[:3, 3]
            else:
                R_cam = cam_pose[:3, :3]
                t_cam = cam_pose[:3, 3]
            
            # 将点云从世界坐标系转换到相机坐标系
            points_cam = (R_cam @ points_sample.T).T + t_cam
            
            # 计算点云的实际范围，用于自适应缩放
            x_range = points_cam[:, 0].max() - points_cam[:, 0].min()
            y_range = points_cam[:, 1].max() - points_cam[:, 1].min()
            z_range = points_cam[:, 2].max() - points_cam[:, 2].min()
            max_range = max(x_range, y_range, z_range)
            
            # 计算点的大小，确保无论点云大小如何都能清晰可见
            if max_range > 0:
                # 根据点云范围自适应调整点的大小
                point_size = max(0.5, min(3.0, 50.0 / max_range))
            else:
                point_size = 1.0
                
            logger.info(f"相机 {cam_idx + 1}: 点云范围 {max_range:.3f}, 点大小 {point_size:.2f}")
            
            # 根据max_views_per_camera限制视角数量
            limited_view_angles = view_angles[:max_views_per_camera]
            
            for angle_idx, (azim_offset, elev_offset, view_name) in enumerate(limited_view_angles):
                # 创建图形，增加分辨率
                fig = plt.figure(figsize=(10, 8), dpi=120)
                ax = fig.add_subplot(111, projection='3d')
                
                # 绘制点云，使用自适应点大小
                ax.scatter(
                    points_cam[:, 0], points_cam[:, 1], points_cam[:, 2],
                    c=colors_sample,
                    s=point_size,  # 使用自适应的点大小
                    alpha=0.8,     # 降低透明度以提高渲染速度
                    edgecolors='none'  # 去除边缘以提高性能
                )
                
                # 只在几个关键视角显示相机坐标系以减少计算量
                if view_name in ["camera_front", "camera_left_30", "camera_right_30"]:
                    # 绘制相机位置和朝向
                    for i, (center, pose) in enumerate(zip(camera_centers[:len(camera_poses)], camera_poses)):
                        # 将相机中心转换到当前相机坐标系
                        if cam_pose.shape == (4, 4):
                            current_R_cam = cam_pose[:3, :3]
                            current_t_cam = cam_pose[:3, 3]
                        else:
                            current_R_cam = cam_pose[:3, :3]
                            current_t_cam = cam_pose[:3, 3]
                        
                        cam_center_in_view = (current_R_cam @ center.T).T + current_t_cam
                        
                        # 绘制相机位置（红色球体）
                        ax.scatter(cam_center_in_view[0], cam_center_in_view[1], cam_center_in_view[2], 
                                  c='red', s=80, marker='o', alpha=1.0)
                        
                        # 计算相机坐标系的轴向量
                        if pose.shape == (4, 4):
                            R_pose = pose[:3, :3]
                        else:
                            R_pose = pose[:, :3]
                        
                        # 应用官方旋转到相机姿态
                        R_pose_rotated = official_rotation.as_matrix() @ R_pose
                        
                        # 转换到当前视图坐标系
                        R_pose_in_view = current_R_cam @ R_pose_rotated
                        
                        # 计算坐标轴的长度（基于点云范围的比例）
                        axis_length = max_range * 0.12  # 稍微减小坐标轴长度
                        
                        # 绘制相机坐标系的三个轴（减少线宽以提高性能）
                        # X轴 (红色)
                        x_axis = R_pose_in_view[:, 0] * axis_length
                        ax.plot([cam_center_in_view[0], cam_center_in_view[0] + x_axis[0]],
                               [cam_center_in_view[1], cam_center_in_view[1] + x_axis[1]],
                               [cam_center_in_view[2], cam_center_in_view[2] + x_axis[2]], 'r-', linewidth=2)
                        
                        # Y轴 (绿色)
                        y_axis = R_pose_in_view[:, 1] * axis_length
                        ax.plot([cam_center_in_view[0], cam_center_in_view[0] + y_axis[0]],
                               [cam_center_in_view[1], cam_center_in_view[1] + y_axis[1]],
                               [cam_center_in_view[2], cam_center_in_view[2] + y_axis[2]], 'g-', linewidth=2)
                        
                        # Z轴 (蓝色)
                        z_axis = R_pose_in_view[:, 2] * axis_length
                        ax.plot([cam_center_in_view[0], cam_center_in_view[0] + z_axis[0]],
                               [cam_center_in_view[1], cam_center_in_view[1] + z_axis[1]],
                               [cam_center_in_view[2], cam_center_in_view[2] + z_axis[2]], 'b-', linewidth=2)
                        
                        # 添加相机编号标签
                        ax.text(cam_center_in_view[0], cam_center_in_view[1], cam_center_in_view[2] + axis_length * 0.3, 
                               f'C{i+1}', fontsize=10, color='black', weight='bold')
                
                # 设置视角
                base_elev = 15.0
                elev = base_elev + elev_offset
                azim = 225.0 + azim_offset
                ax.view_init(elev=elev, azim=azim)
                
                # 计算点云的紧密边界框，添加适当边距
                margin_factor = 0.12  # 适中的边距
                x_min, x_max = points_cam[:, 0].min(), points_cam[:, 0].max()
                y_min, y_max = points_cam[:, 1].min(), points_cam[:, 1].max()
                z_min, z_max = points_cam[:, 2].min(), points_cam[:, 2].max()
                
                # 只在显示相机坐标系的视角中考虑相机位置
                if view_name in ["camera_front", "camera_left_30", "camera_right_30"]:
                    # 将相机位置也考虑进边界计算
                    for center in camera_centers[:len(camera_poses)]:
                        # 转换相机中心到当前相机坐标系
                        if cam_pose.shape == (4, 4):
                            current_R_cam = cam_pose[:3, :3]
                            current_t_cam = cam_pose[:3, 3]
                        else:
                            current_R_cam = cam_pose[:3, :3]
                            current_t_cam = cam_pose[:3, 3]
                        
                        cam_center_in_view = (current_R_cam @ center.T).T + current_t_cam
                        axis_length = max_range * 0.12
                        
                        # 扩展边界以包含相机和其坐标轴
                        x_min = min(x_min, cam_center_in_view[0] - axis_length)
                        x_max = max(x_max, cam_center_in_view[0] + axis_length)
                        y_min = min(y_min, cam_center_in_view[1] - axis_length)
                        y_max = max(y_max, cam_center_in_view[1] + axis_length)
                        z_min = min(z_min, cam_center_in_view[2] - axis_length)
                        z_max = max(z_max, cam_center_in_view[2] + axis_length)
                
                x_margin = (x_max - x_min) * margin_factor
                y_margin = (y_max - y_min) * margin_factor
                z_margin = (z_max - z_min) * margin_factor
                
                # 设置紧密的坐标轴范围以放大点云显示
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                ax.set_zlim(z_min - z_margin, z_max + z_margin)
                
                # 设置坐标轴颜色对应朝向箭头
                ax.xaxis.label.set_color('red')
                ax.yaxis.label.set_color('green') 
                ax.zaxis.label.set_color('blue')
                
                # 去除坐标轴和刻度以获得更清晰的点云视图
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                
                # 设置背景为白色
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('white')
                ax.yaxis.pane.set_edgecolor('white')
                ax.zaxis.pane.set_edgecolor('white')

                # 保存为字节流，优化设置以提高性能
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                           pad_inches=0.03, facecolor='white', edgecolor='none')
                buf.seek(0)
                
                # 编码为base64
                img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                view_images.append({
                    "camera": cam_idx + 1,
                    "view": view_name,
                    "image": img_b64
                })
                
                plt.close(fig)
                buf.close()
        
        return view_images
    
    except Exception as e:
        logger.error(f"生成视角图片失败：{e}")
        return []

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Pi3 3D Reconstruction Server')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/pi3/model.safetensors',
                        help='Path to Pi3 model checkpoint (default: checkpoints/pi3/model.safetensors)')
    parser.add_argument('--port', type=int, default=20021,
                        help='Port to run the server on (default: 20021)')
    
    args = parser.parse_args()
    
    logger.info("正在启动Pi3服务器...")
    logger.info(f"模型路径: {args.checkpoint_path}")
    logger.info(f"服务端口: {args.port}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"找不到模型文件：{args.checkpoint_path}")
        exit(1)
    
    # 加载指定模型
    if not load_model(checkpoint_path=args.checkpoint_path):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=args.port, debug=False) 