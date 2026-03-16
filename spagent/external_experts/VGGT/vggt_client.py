import base64
import cv2
import requests
import os
import logging
from typing import List, Optional, Dict, Any
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VGGTClient:
    """VGGT 3D重建客户端"""
    
    def __init__(self, server_url: str = "http://localhost:20032"):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:20032'
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5分钟超时，因为3D重建需要较长时间
        
    def health_check(self) -> Optional[Dict[str, Any]]:
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"健康检查失败，状态码：{response.status_code}")
                return None
        except Exception as e:
            logger.error(f"健康检查失败：{e}")
            return None
    
    def test_infer(self) -> Optional[Dict[str, Any]]:
        """使用服务器内置的测试数据进行推理"""
        try:
            response = self.session.get(f"{self.server_url}/test")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"测试推理失败，状态码：{response.status_code}")
                return None
        except Exception as e:
            logger.error(f"测试推理失败：{e}")
            return None
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """将图片编码为base64字符串"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在：{image_path}")
                return None
                
            # 使用cv2读取图片
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                logger.error(f"无法读取图片：{image_path}")
                return None
            
            # 编码为JPEG格式
            _, buffer = cv2.imencode('.jpg', image_bgr)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图片失败：{e}")
            return None
    
    def _encode_video_frames(self, video_path: str, interval: int = 10, 
                            save_frames: bool = False, 
                            output_dir: str = "outputs/video_frames") -> Optional[tuple]:
        """从视频中提取帧并编码为base64"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在：{video_path}")
                return None
            
            if not video_path.lower().endswith('.mp4'):
                logger.error(f"只支持mp4格式的视频文件：{video_path}")
                return None
            
            logger.info(f"正在从视频提取帧：{video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件：{video_path}")
                return None
            
            if save_frames:
                os.makedirs(output_dir, exist_ok=True)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(frame_output_dir, exist_ok=True)
                logger.info(f"将保存提取的帧到：{frame_output_dir}")
            
            encoded_frames = []
            frame_idx = 0
            extracted_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 调整图像尺寸确保是patch大小(14)的倍数
                    h, w = rgb_frame.shape[:2]
                    patch_size = 14
                    new_h = ((h + patch_size - 1) // patch_size) * patch_size
                    new_w = ((w + patch_size - 1) // patch_size) * patch_size
                    
                    if new_h != h or new_w != w:
                        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
                    
                    if save_frames:
                        frame_filename = f"frame_{extracted_frame_count:04d}_idx_{frame_idx:06d}.jpg"
                        frame_path = os.path.join(frame_output_dir, frame_filename)
                        cv2.imwrite(frame_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                        logger.info(f"保存帧 {extracted_frame_count + 1}: {frame_filename}")
                    
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    encoded_frames.append(frame_b64)
                    extracted_frame_count += 1
                
                frame_idx += 1
            
            cap.release()
            
            if not encoded_frames:
                logger.error("视频中没有提取到任何帧")
                return None
            
            logger.info(f"从视频中提取了 {len(encoded_frames)} 帧")
            if save_frames:
                logger.info(f"所有帧已保存到：{frame_output_dir}")
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            return (encoded_frames, video_name)
            
        except Exception as e:
            logger.error(f"处理视频失败：{e}")
            return None
    
    def infer_from_images(self, 
                         image_paths: List[str], 
                         conf_threshold: float = 50.0,
                         generate_views: bool = True,
                         use_filename: bool = True,
                         azimuth_angle: Optional[float] = None,
                         elevation_angle: Optional[float] = None,
                         rotation_reference_camera: int = 1,
                         camera_view: bool = False,
                         remove_outliers: bool = True) -> Optional[Dict[str, Any]]:
        """
        从图片列表进行3D重建
        
        Args:
            image_paths: 图片路径列表
            conf_threshold: 置信度百分位阈值（百分比，0-100）
            generate_views: 是否生成多视角图片
            use_filename: 是否使用文件名来命名输出文件
            azimuth_angle: 自定义方位角（左右旋转），单位：度
            elevation_angle: 自定义仰角（上下旋转），单位：度
            rotation_reference_camera: 参考相机索引（1-based）
            camera_view: 是否使用相机视角模式
            remove_outliers: 是否移除离群点
            
        Returns:
            重建结果字典，包含PLY文件和多视角图片
        """
        try:
                       
            # 编码所有图片
            encoded_images = []
            image_names = []
            for img_path in image_paths:
                encoded = self._encode_image(img_path)
                if encoded:
                    encoded_images.append(encoded)
                    if use_filename:
                        image_names.append(img_path)
                else:
                    logger.error(f"无法编码图片：{img_path}")
                    return None
            
            if not encoded_images:
                logger.error("没有成功编码的图片")
                return None
            
            # 构建请求数据
            request_data = {
                "images": encoded_images,
                "conf_threshold": conf_threshold,
                "generate_views": generate_views,
                "rotation_reference_camera": rotation_reference_camera,
                "camera_view": camera_view,
                "remove_outliers": remove_outliers
            }
            
            if use_filename and image_names:
                request_data["image_names"] = image_names
            
            if azimuth_angle is not None and elevation_angle is not None:
                request_data["azimuth_angle"] = azimuth_angle
                request_data["elevation_angle"] = elevation_angle
            
            # 发送请求
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):                    
                    return result
                else:
                    logger.error(f"3D重建失败：{result.get('error', '未知错误')}")
                    return None
            else:
                logger.error(f"请求失败，状态码：{response.status_code}")
                logger.error(f"响应内容：{response.text}")
                return None
                
        except Exception as e:
            logger.error(f"3D重建请求失败：{e}")
            logger.error(traceback.format_exc())
            return None
    
    def infer_from_video(self, 
                        video_path: str, 
                        interval: int = 10,
                        conf_threshold: float = 50.0,
                        generate_views: bool = True,
                        max_views_per_camera: int = 4,
                        azimuth_angle: Optional[float] = None,
                        elevation_angle: Optional[float] = None,
                        rotation_reference_camera: int = 1,
                        camera_view: bool = False,
                        remove_outliers: bool = True,
                        save_frames: bool = False,
                        frames_output_dir: str = "outputs/video_frames") -> Optional[Dict[str, Any]]:
        """
        从视频文件进行3D重建
        
        Args:
            video_path: 视频文件路径（支持mp4格式）
            interval: 帧采样间隔（每interval帧提取一帧）
            conf_threshold: 置信度百分位阈值（百分比，0-100）
            generate_views: 是否生成多视角图片
            max_views_per_camera: 每个相机最多生成的视角图片数量
            azimuth_angle: 自定义方位角（左右旋转），单位：度
            elevation_angle: 自定义仰角（上下旋转），单位：度
            rotation_reference_camera: 参考相机索引（1-based）
            camera_view: 是否使用相机视角模式
            remove_outliers: 是否移除离群点
            save_frames: 是否保存提取的视频帧到本地
            frames_output_dir: 保存视频帧的输出目录
            
        Returns:
            重建结果字典，包含PLY文件和多视角图片
        """
        try:
            # 从视频提取帧
            result = self._encode_video_frames(video_path, interval, 
                                               save_frames=save_frames, 
                                               output_dir=frames_output_dir)
            if not result:
                return None
            
            encoded_frames, video_name = result
            
            # 构建请求数据
            request_data = {
                "images": encoded_frames,
                "conf_threshold": conf_threshold,
                "generate_views": generate_views,
                "image_names": [video_name],
                "rotation_reference_camera": rotation_reference_camera,
                "camera_view": camera_view,
                "remove_outliers": remove_outliers
            }
            
            if azimuth_angle is not None and elevation_angle is not None:
                request_data["azimuth_angle"] = azimuth_angle
                request_data["elevation_angle"] = elevation_angle
                view_mode = "相机视角" if camera_view else "全局视角"
                logger.info(f"使用自定义角度: 方位角={azimuth_angle}°, 仰角={elevation_angle}° ref_cam={rotation_reference_camera} 模式={view_mode}")
            else:
                request_data["max_views_per_camera"] = max_views_per_camera
            
            logger.info(f"正在发送 {len(encoded_frames)} 帧进行VGGT 3D重建...")
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("视频VGGT 3D重建成功！")
                    
                    # 限制每个相机的视角图片数量
                    if "camera_views" in result and result["camera_views"]:
                        original_views = result["camera_views"]
                        limited_views = []
                        
                        camera_groups = {}
                        for view in original_views:
                            camera_id = view.get("camera", 1)
                            if camera_id not in camera_groups:
                                camera_groups[camera_id] = []
                            camera_groups[camera_id].append(view)
                        
                        for camera_id, views in camera_groups.items():
                            limited_camera_views = views[:max_views_per_camera]
                            limited_views.extend(limited_camera_views)
                            logger.info(f"相机 {camera_id}: 从 {len(views)} 个视角中保留 {len(limited_camera_views)} 个")
                        
                        result["camera_views"] = limited_views
                    
                    return result
                else:
                    logger.error(f"视频VGGT 3D重建失败：{result.get('error', '未知错误')}")
                    return None
            else:
                logger.error(f"请求失败，状态码：{response.status_code}")
                logger.error(f"响应内容：{response.text}")
                return None
                
        except Exception as e:
            logger.error(f"视频VGGT 3D重建请求失败：{e}")
            logger.error(traceback.format_exc())
            return None
    
    def save_results(self, result: Dict[str, Any], output_dir: str = "outputs", 
                    rotation_reference_camera: int = 1, camera_view: bool = False) -> bool:
        """
        保存重建结果到本地
        
        Args:
            result: 重建结果字典
            output_dir: 输出目录
            rotation_reference_camera: 参考相机索引
            camera_view: 是否使用相机视角模式
            
        Returns:
            保存是否成功
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建文件名后缀
            suffix = ""
            if rotation_reference_camera != 1:
                suffix += f"_refcam{rotation_reference_camera}"
            if camera_view:
                suffix += "_camview"
            
            # 保存PLY文件
            if "ply_file" in result:
                ply_filename = result.get("ply_filename", "vggt_result.ply")
                ply_path = os.path.join(output_dir, ply_filename)
                
                ply_data = base64.b64decode(result["ply_file"])
                with open(ply_path, 'wb') as f:
                    f.write(ply_data)
                logger.info(f"PLY文件已保存：{ply_path}")
            
            # 保存多视角图片
            if "camera_views" in result and result["camera_views"]:
                views_dir = os.path.join(output_dir, "camera_views")
                os.makedirs(views_dir, exist_ok=True)
                
                for view_data in result["camera_views"]:
                    camera = view_data.get("camera", 1)
                    azimuth = view_data.get("azimuth_angle", 0)
                    elevation = view_data.get("elevation_angle", 0)
                    
                    img_filename = f"vggt_camera_{camera:02d}_azim{azimuth:.1f}_elev{elevation:.1f}{suffix}.png"
                    img_path = os.path.join(views_dir, img_filename)
                    
                    img_data = base64.b64decode(view_data["image"])
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                logger.info(f"多视角图片已保存到：{views_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"保存结果失败：{e}")
            return False


if __name__ == "__main__":
    # 测试VGGT客户端
    client = VGGTClient()
    
    # 1. 健康检查
    logger.info("\n=== 执行健康检查 ===")
    health = client.health_check()
    if health:
        logger.info(f"服务器状态: {health}")
    else:
        logger.error("服务器健康检查失败，退出程序")
        exit(1)
    
    # 2. 测试推理
    logger.info("\n=== 执行测试推理 ===")
    test_result = client.test_infer()
    if test_result:
        logger.info(f"测试推理成功: {test_result}")
    else:
        logger.error("测试推理失败，退出程序")
        exit(1)
    
    # 3. 处理图片
    logger.info("\n=== 处理图片 ===")
    
    test_images = [
        "dataset/mindcube/data/other_all_image/among/ball_060/front_008.jpg", "dataset/mindcube/data/other_all_image/among/ball_060/right_234.jpg"
    ]
    
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if existing_images:
        logger.info(f"使用测试图片：{existing_images}")
        result = client.infer_from_images(
            image_paths=existing_images,
            conf_threshold=50.0,
            generate_views=True,
            azimuth_angle=45,
            elevation_angle=0,
            rotation_reference_camera=1,
            camera_view=False
        )
        
        if result:
            logger.info("VGGT 3D重建成功！")
            logger.info(f"- 点云数量: {result.get('points_count', '未知')}")
            logger.info(f"- PLY文件名: {result.get('ply_filename', '未知')}")
            logger.info(f"- 生成视角数: {len(result.get('camera_views', []))}")
            
            if client.save_results(result, "outputs/vggt_test"):
                logger.info("结果保存成功！")
            else:
                logger.error("结果保存失败")
        else:
            logger.error("VGGT 3D重建失败")
    else:
        logger.warning("没有找到可用的测试图片")
        logger.info("可用的测试图片路径：")
        for test_image in test_images:
            logger.info(f"  - {test_image}")
    
    # 4. 处理视频文件
    logger.info("\n=== 处理视频文件 ===")
    
    test_videos = ["dataset/VSI_videos/arkitscenes_41069048.mp4"]
    
    video_path = None
    for test_video in test_videos:
        if os.path.exists(test_video):
            video_path = test_video
            break
    
    if video_path:
        logger.info(f"使用测试视频：{video_path}")
        video_result = client.infer_from_video(
            video_path=video_path,
            interval=200,
            conf_threshold=50.0,
            generate_views=True,
            max_views_per_camera=5,
            azimuth_angle=0,
            elevation_angle=-45,
            rotation_reference_camera=1,
            camera_view=False,
            save_frames=False
        )
        
        if video_result:
            logger.info("视频VGGT 3D重建成功！")
            logger.info(f"- 点云数量: {video_result.get('points_count', '未知')}")
            logger.info(f"- PLY文件名: {video_result.get('ply_filename', '未知')}")
            logger.info(f"- 生成视角数: {len(video_result.get('camera_views', []))}")
            
            if client.save_results(video_result, "outputs/vggt_video"):
                logger.info("视频重建结果保存成功！")
            else:
                logger.error("视频重建结果保存失败")
        else:
            logger.error("视频VGGT 3D重建失败")
    else:
        logger.warning("没有找到可用的测试视频")
