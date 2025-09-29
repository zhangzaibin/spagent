#!/usr/bin/env python3
"""
Pi3 3D重建简化测试脚本
专门用于生成指定角度的视角图片进行调试
"""

import base64
import requests
import os
import logging
import time
import argparse
import json
from typing import List, Optional, Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplePi3Tester:
    """简化的Pi3测试器，专注于角度调试"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:20030"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300
        
    def encode_image(self, image_path: str) -> Optional[str]:
        """编码图片为base64"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"图片不存在: {image_path}")
                return None
                
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图片失败: {e}")
            return None
    
    def test_angle(self, 
                   image_paths: List[str],
                   azimuth_angle: float,
                   elevation_angle: float,
                   output_dir: str = None) -> bool:
        """
        测试指定角度的3D重建
        
        Args:
            image_paths: 图片路径列表
            azimuth_angle: 方位角（度）
            elevation_angle: 仰角（度）
            output_dir: 输出目录
        """
        try:
            logger.info(f"开始测试角度: 方位角={azimuth_angle}°, 仰角={elevation_angle}°")
            
            # 编码图片
            encoded_images = []
            image_names = []
            
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    logger.error(f"图片不存在: {img_path}")
                    continue
                    
                encoded = self.encode_image(img_path)
                if encoded:
                    encoded_images.append(encoded)
                    image_names.append(os.path.basename(img_path))
                    logger.info(f"✓ 编码成功: {os.path.basename(img_path)}")
            
            if not encoded_images:
                logger.error("没有有效的图片")
                return False
            
            # 构建请求
            request_data = {
                "images": encoded_images,
                "image_names": image_names,
                "generate_views": True,
                "azimuth_angle": azimuth_angle,
                "elevation_angle": elevation_angle,
                "conf_threshold": 0.1,
                "rtol": 0.03
            }
            
            # 发送请求
            logger.info("发送推理请求...")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            )
            
            end_time = time.time()
            logger.info(f"推理完成，耗时: {end_time - start_time:.2f}秒")
            
            if response.status_code != 200:
                logger.error(f"请求失败，状态码: {response.status_code}")
                logger.error(f"响应: {response.text}")
                return False
                
            result = response.json()
            if not result.get("success"):
                logger.error(f"推理失败: {result.get('error', '未知错误')}")
                return False
            
            logger.info("✓ 推理成功!")
            logger.info(f"- 点云数量: {result.get('points_count', 0)}")
            logger.info(f"- 生成视角数: {len(result.get('camera_views', []))}")
            
            # 保存结果
            if output_dir is None:
                output_dir = f"debug_angle_{azimuth_angle}_{elevation_angle}"
            
            self.save_results(result, output_dir, azimuth_angle, elevation_angle)
            return True
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            return False
    
    def save_results(self, result: Dict[str, Any], output_dir: str, azim: float, elev: float):
        """保存结果"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存PLY文件
            if "ply_file" in result:
                ply_filename = result.get("ply_filename", "result.ply")
                ply_path = os.path.join(output_dir, ply_filename)
                
                ply_data = base64.b64decode(result["ply_file"])
                with open(ply_path, 'wb') as f:
                    f.write(ply_data)
                logger.info(f"PLY文件保存: {ply_path}")
            
            # 保存视角图片
            if "camera_views" in result and result["camera_views"]:
                for i, view_data in enumerate(result["camera_views"]):
                    img_filename = f"view_azim_{azim}_elev_{elev}_{i+1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    img_data = base64.b64decode(view_data["image"])
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    logger.info(f"视角图片保存: {img_filename}")
            
            # 保存调试信息
            debug_info = {
                "azimuth_angle": azim,
                "elevation_angle": elev,
                "points_count": result.get("points_count", 0),
                "views_generated": len(result.get("camera_views", [])),
                "ply_filename": result.get("ply_filename", ""),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            debug_path = os.path.join(output_dir, "debug_info.json")
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"调试信息保存: {debug_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='Pi3角度调试测试')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:20030',
                        help='Pi3服务器地址')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                        help='图片路径列表')
    parser.add_argument('--azimuth', type=float, required=True,
                        help='方位角（度）')
    parser.add_argument('--elevation', type=float, required=True,
                        help='仰角（度）')
    parser.add_argument('--output', type=str,
                        help='输出目录（可选）')
    
    args = parser.parse_args()
    
    # 验证图片文件
    valid_images = []
    for img_path in args.images:
        if os.path.exists(img_path):
            valid_images.append(img_path)
            logger.info(f"✓ 找到图片: {img_path}")
        else:
            logger.warning(f"✗ 图片不存在: {img_path}")
    
    if not valid_images:
        logger.error("没有有效的图片文件")
        return
    
    # 创建测试器并运行
    tester = SimplePi3Tester(server_url=args.server_url)
    
    success = tester.test_angle(
        image_paths=valid_images,
        azimuth_angle=args.azimuth,
        elevation_angle=args.elevation,
        output_dir=args.output
    )
    
    if success:
        output_dir = args.output or f"debug_angle_{args.azimuth}_{args.elevation}"
        logger.info(f"\n🎉 测试完成! 结果保存在: {output_dir}/")
        logger.info("生成的文件:")
        logger.info("- *.ply (3D点云文件)")
        logger.info("- view_*.png (视角图片)")
        logger.info("- debug_info.json (调试信息)")
    else:
        logger.error("\n❌ 测试失败")

if __name__ == "__main__":
    main()
