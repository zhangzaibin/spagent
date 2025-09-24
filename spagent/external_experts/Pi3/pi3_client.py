import base64
import cv2
import requests
import numpy as np
import os
import logging
import time
from typing import List, Optional, Dict, Any
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Pi3Client:
    """Pi3 3D重建客户端"""
    
    def __init__(self, server_url: str = "http://localhost:20030"):
        """
        初始化客户端
        
        Args:
            server_url: 服务器地址，如 'http://localhost:20030'
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
                
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图片失败：{e}")
            return None
    
    def _encode_video_frames(self, video_path: str, interval: int = 10) -> Optional[tuple[List[str], str]]:
        """从视频中提取帧并编码为base64"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在：{video_path}")
                return None
            
            # 检查文件是否为mp4格式
            if not video_path.lower().endswith('.mp4'):
                logger.error(f"只支持mp4格式的视频文件：{video_path}")
                return None
            
            logger.info(f"正在从视频提取帧：{video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件：{video_path}")
                return None
            
            encoded_frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % interval == 0:
                    # 转换BGR到RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 调整图像尺寸确保是patch大小(14)的倍数
                    h, w = rgb_frame.shape[:2]
                    patch_size = 14
                    
                    # 计算最接近的有效尺寸
                    new_h = ((h + patch_size - 1) // patch_size) * patch_size
                    new_w = ((w + patch_size - 1) // patch_size) * patch_size
                    
                    if new_h != h or new_w != w:
                        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
                    
                    # 编码为JPEG然后转base64
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    encoded_frames.append(frame_b64)
                
                frame_idx += 1
            
            cap.release()
            
            if not encoded_frames:
                logger.error("视频中没有提取到任何帧")
                return None
            
            logger.info(f"从视频中提取了 {len(encoded_frames)} 帧")
            
            # 生成基于视频文件名的名称
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            return (encoded_frames, video_name)
            
        except Exception as e:
            logger.error(f"处理视频失败：{e}")
            return None
    
    def infer_from_images(self, 
                         image_paths: List[str], 
                         conf_threshold: float = 0.1,
                         rtol: float = 0.03,
                         generate_views: bool = True,
                         use_filename: bool = True,
                         azimuth_angle: Optional[float] = None,
                         elevation_angle: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        从图片列表进行3D重建
        
        Args:
            image_paths: 图片路径列表
            conf_threshold: 置信度阈值
            rtol: 深度边缘检测阈值
            generate_views: 是否生成多视角图片
            use_filename: 是否使用文件名来命名输出文件
            azimuth_angle: 自定义方位角（左右旋转），单位：度。如果提供，将生成该角度的视角图片
            elevation_angle: 自定义仰角（上下旋转），单位：度。如果提供，将生成该角度的视角图片
            
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
                        image_names.append(os.path.basename(img_path))
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
                "rtol": rtol,
                "generate_views": generate_views
            }
            
            # 如果使用文件名，添加到请求数据中
            if use_filename and image_names:
                request_data["image_names"] = image_names
            
            # 如果提供了自定义角度，添加到请求数据中
            if azimuth_angle is not None and elevation_angle is not None:
                request_data["azimuth_angle"] = azimuth_angle
                request_data["elevation_angle"] = elevation_angle
                logger.info(f"使用自定义角度: 方位角={azimuth_angle}°, 仰角={elevation_angle}°")
            
            # 发送请求
            logger.info(f"正在发送 {len(encoded_images)} 张图片进行3D重建...")
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("3D重建成功！")
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
            logger.error(f"错误追踪：{traceback.format_exc()}")
            return None
    
    def infer_from_video(self, 
                        video_path: str, 
                        interval: int = 10,
                        conf_threshold: float = 0.1,
                        rtol: float = 0.03,
                        generate_views: bool = True,
                        max_views_per_camera: int = 4,
                        azimuth_angle: Optional[float] = None,
                        elevation_angle: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        从视频文件进行3D重建
        
        Args:
            video_path: 视频文件路径（支持mp4格式）
            interval: 帧采样间隔（每interval帧提取一帧）
            conf_threshold: 置信度阈值
            rtol: 深度边缘检测阈值
            generate_views: 是否生成多视角图片
            max_views_per_camera: 每个相机最多生成的视角图片数量（仅在未提供自定义角度时使用）
            azimuth_angle: 自定义方位角（左右旋转），单位：度。如果提供，将生成该角度的视角图片
            elevation_angle: 自定义仰角（上下旋转），单位：度。如果提供，将生成该角度的视角图片
            
        Returns:
            重建结果字典，包含PLY文件和多视角图片
        """
        try:
            # 从视频提取帧
            result = self._encode_video_frames(video_path, interval)
            if not result:
                return None
            
            encoded_frames, video_name = result
            
            # 构建请求数据
            request_data = {
                "images": encoded_frames,
                "conf_threshold": conf_threshold,
                "rtol": rtol,
                "generate_views": generate_views,
                "image_names": [video_name]  # 使用视频名称作为文件名
            }
            
            # 如果提供了自定义角度，添加到请求数据中，否则使用max_views_per_camera
            if azimuth_angle is not None and elevation_angle is not None:
                request_data["azimuth_angle"] = azimuth_angle
                request_data["elevation_angle"] = elevation_angle
                logger.info(f"使用自定义角度: 方位角={azimuth_angle}°, 仰角={elevation_angle}°")
            else:
                request_data["max_views_per_camera"] = max_views_per_camera  # 添加视角限制参数
            
            # 发送请求
            logger.info(f"正在发送 {len(encoded_frames)} 帧进行3D重建...")
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("视频3D重建成功！")
                    
                    # 限制每个相机的视角图片数量
                    if "camera_views" in result and result["camera_views"]:
                        original_views = result["camera_views"]
                        limited_views = []
                        
                        # 按相机分组
                        camera_groups = {}
                        for view in original_views:
                            camera_id = view.get("camera", 1)
                            if camera_id not in camera_groups:
                                camera_groups[camera_id] = []
                            camera_groups[camera_id].append(view)
                        
                        # 每个相机只保留指定数量的视角
                        for camera_id, views in camera_groups.items():
                            limited_camera_views = views[:max_views_per_camera]
                            limited_views.extend(limited_camera_views)
                            logger.info(f"相机 {camera_id}: 从 {len(views)} 个视角中保留 {len(limited_camera_views)} 个")
                        
                        result["camera_views"] = limited_views
                    
                    return result
                else:
                    logger.error(f"视频3D重建失败：{result.get('error', '未知错误')}")
                    return None
            else:
                logger.error(f"请求失败，状态码：{response.status_code}")
                logger.error(f"响应内容：{response.text}")
                return None
                
        except Exception as e:
            logger.error(f"视频3D重建请求失败：{e}")
            logger.error(f"错误追踪：{traceback.format_exc()}")
            return None
    
    def save_results(self, result: Dict[str, Any], output_dir: str = "outputs") -> bool:
        """
        保存重建结果到本地
        
        Args:
            result: 重建结果字典
            output_dir: 输出目录
            
        Returns:
            保存是否成功
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存PLY文件
            if "ply_file" in result:
                ply_filename = result.get("ply_filename", "result.ply")
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
                    view_name = view_data.get("view", "unknown")
                    
                    img_filename = f"camera_{camera:02d}_{view_name}.png"
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
    # 测试Pi3客户端
    client = Pi3Client()
    
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
    
    # 3. 处理单张图片
    logger.info("\n=== 处理单张图片 ===")
    
    # 检查是否有可用的测试图片
    test_images = ["/home/ubuntu/projects/spagent/dataset/BLINK/0ec2af0b0d5fc498743ebd74f94452fae426617a1597200fddfd9b4bacdd5824.jpg"]
    
    single_image_path = None
    for test_image in test_images:
        if os.path.exists(test_image):
            single_image_path = test_image
            break
    
    if single_image_path:
        logger.info(f"使用单张测试图片：{single_image_path}")
        single_result = client.infer_from_images(
            image_paths=[single_image_path],
            conf_threshold=0.1,
            rtol=0.03,
            generate_views=True
        )
        
        if single_result:
            logger.info("单张图片3D重建成功！")
            logger.info(f"- 点云数量: {single_result.get('points_count', '未知')}")
            logger.info(f"- PLY文件名: {single_result.get('ply_filename', '未知')}")
            logger.info(f"- 生成视角数: {len(single_result.get('camera_views', []))}")
            
            # 保存结果
            if client.save_results(single_result, "outputs/single_image"):
                logger.info("单张图片结果保存成功！")
            else:
                logger.error("单张图片结果保存失败")
        else:
            logger.error("单张图片3D重建失败")
        
        # 4. 测试自定义角度
        logger.info("\n=== 测试自定义角度生成 ===")
        custom_result = client.infer_from_images(
            image_paths=[single_image_path],
            conf_threshold=0.1,
            rtol=0.03,
            generate_views=True,
            azimuth_angle=0,  # 左右转
            elevation_angle=10  # 上下转
        )
        
        if custom_result:
            logger.info("自定义角度3D重建成功！")
            logger.info(f"- 点云数量: {custom_result.get('points_count', '未知')}")
            logger.info(f"- PLY文件名: {custom_result.get('ply_filename', '未知')}")
            logger.info(f"- 生成视角数: {len(custom_result.get('camera_views', []))}")
            
            # 打印角度信息
            for view in custom_result.get('camera_views', []):
                if 'azimuth_angle' in view and 'elevation_angle' in view:
                    logger.info(f"- 视角角度: 方位角={view['azimuth_angle']}°, 仰角={view['elevation_angle']}°")
            
            # 保存结果
            if client.save_results(custom_result, "outputs/custom_angle"):
                logger.info("自定义角度结果保存成功！")
            else:
                logger.error("自定义角度结果保存失败")
        else:
            logger.error("自定义角度3D重建失败")
    else:
        logger.warning("没有找到可用的测试图片")
        logger.info("可用的测试图片路径：")
        for test_image in test_images:
            logger.info(f"  - {test_image}")
    
    # # 4. 处理视频文件
    # logger.info("\n=== 处理视频文件 ===")
    
    # # 检查是否有可用的测试视频
    # test_videos = ["assets/cartoon_horse.mp4"]
    
    # video_path = None
    # for test_video in test_videos:
    #     if os.path.exists(test_video):
    #         video_path = test_video
    #         break
    
    # if video_path:
    #     logger.info(f"使用测试视频：{video_path}")
    #     video_result = client.infer_from_video(
    #         video_path=video_path,
    #         interval=10,  # 每10帧提取一帧
    #         conf_threshold=0.1,
    #         rtol=0.03,
    #         generate_views=True,
    #         max_views_per_camera=4  # 每个相机最多4张视角图
    #     )
        
    #     if video_result:
    #         logger.info("视频3D重建成功！")
    #         logger.info(f"- 点云数量: {video_result.get('points_count', '未知')}")
    #         logger.info(f"- PLY文件名: {video_result.get('ply_filename', '未知')}")
    #         logger.info(f"- 生成视角数: {len(video_result.get('camera_views', []))}")
            
    #         # 保存结果
    #         if client.save_results(video_result, "outputs/video"):
    #             logger.info("视频重建结果保存成功！")
    #         else:
    #             logger.error("视频重建结果保存失败")
    #     else:
    #         logger.error("视频3D重建失败")
    # else:
    #     logger.warning("没有找到可用的测试视频")
    #     logger.info("可用的测试视频路径：")
    #     for test_video in test_videos:
    #         logger.info(f"  - {test_video}")
