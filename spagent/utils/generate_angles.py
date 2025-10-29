"""
批量生成不同角度的视角图片
从JSONL文件中读取图片列表，为每组图片生成多个角度的视角
"""

import os
import sys
import logging
import base64
import argparse
import json
from typing import List, Tuple, Dict
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spagent.external_experts.Pi3.pi3_client import Pi3Client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_scene_id(image_path: str) -> str:
    """
    从图片路径中提取scene ID，适配多种数据集格式
    
    Args:
        image_path: 图片路径，支持多种格式:
            - VLM-3R/scannet: "VLM-3R/scannet_frames_25k/scene0296_01/color/000000.jpg"
            - 带目录结构: "dataset/scene123/img0001.jpg"
            - 普通路径: "image.jpg"
        
    Returns:
        scene ID字符串，如 "scene0296_01" 或 "scene123_img0001"
    """
    # 1. 尝试提取scene ID（在scannet数据集中）
    if 'scannet' in image_path.lower() or 'scene' in image_path.lower():
        parts = image_path.split('/')
        for part in parts:
            if part.startswith('scene') and '_' in part:
                return part
    
    # 2. 尝试从路径中找到包含数字的目录名（通常是场景ID）
    path_parts = image_path.split('/')
    for part in reversed(path_parts[:-1]):  # 从后往前找，跳过文件名
        # 如果目录名包含数字或常见的场景标识符
        if any(c.isdigit() for c in part) or part.lower() in ['scene', 'view', 'camera']:
            # 提取文件名
            filename = os.path.splitext(os.path.basename(image_path))[0]
            # 组合目录名和文件名
            if filename and filename != part:
                return f"{part}_{filename}"
            return part
    
    # 3. Fallback: 使用父目录名和文件名的组合
    path_parts = os.path.dirname(image_path).split('/')
    if len(path_parts) >= 2:
        prominent_part = path_parts[-2]  # 倒数第二个部分
        filename = os.path.splitext(os.path.basename(image_path))[0]
        if prominent_part and prominent_part != '.' and prominent_part != '':
            return f"{prominent_part}_{filename}"
        elif filename:
            return filename
    
    # 4. 最后的fallback: 只使用文件名（不含扩展名）
    return os.path.splitext(os.path.basename(image_path))[0]


def load_images_from_jsonl(jsonl_path: str) -> List[Tuple[List[str], int, str]]:
    """
    从JSONL文件中读取图片列表
    
    Args:
        jsonl_path: JSONL文件路径
        
    Returns:
        图片列表 [(images, item_id, scene_id), ...]，其中images是图片路径列表
    """
    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL文件不存在: {jsonl_path}")
        return []
    
    image_lists = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                # 支持"images"和"image"两种键名
                images = data.get('images') or data.get('image', [])
                if isinstance(images, list) and len(images) >= 2:
                    # 提取scene ID用于命名
                    scene_id = extract_scene_id(images[0])
                    # 使用所有图片进行3D重建
                    image_lists.append((images, idx, scene_id))
                    logger.info(f"读取第 {idx} 条: {len(images)} 张图片, scene_id={scene_id}")
                else:
                    logger.warning(f"第 {idx} 条缺少images或图片数量不足，跳过")
            except json.JSONDecodeError as e:
                logger.error(f"第 {idx} 行JSON解析失败: {e}")
                continue
    
    logger.info(f"从JSONL文件读取了 {len(image_lists)} 条有效图片记录")
    return image_lists


def generate_angle_specifications() -> List[Tuple[float, float, str]]:
    """
    生成所有角度规格
    使用指定的角度组合
    
    Returns:
        角度列表 [(azimuth, elevation, name), ...]
    """
    angles = []
    
    # 正面 (0, 0)
    angles.append((0, 0, "front_0_0"))
    
    # 左侧: -45, -90, -135
    angles.append((-45, 0, "left_45_0"))
    angles.append((-90, 0, "left_90_0"))
    angles.append((-135, 0, "left_135_0"))
    
    # 右侧: 45, 90, 180, 135
    angles.append((45, 0, "right_45_0"))
    angles.append((90, 0, "right_90_0"))
    angles.append((180, 0, "right_180_0"))
    angles.append((135, 0, "right_135_0"))
    
    # 上侧: 45, 60
    angles.append((0, 45, "up_0_45"))
    angles.append((0, 60, "up_0_60"))
    
    # 下侧: -45
    angles.append((0, -45, "down_0_45"))
    
    # 对角: (45, 30), (-45, 30)
    angles.append((45, 30, "diag_45_30"))
    angles.append((-45, 30, "diag_neg45_30"))
    
    logger.info(f"生成了 {len(angles)} 个角度规格")
    return angles


def generate_views_for_images(client: Pi3Client, 
                             images: List[str],
                             angles: List[Tuple[float, float, str]],
                             output_dir: str,
                             item_id: int,
                             scene_id: str,
                             port: int = None) -> bool:
    """
    为一组图片生成所有角度的视角
    
    Args:
        client: Pi3客户端
        images: 图片路径列表（至少需要2张）
        angles: 角度列表
        output_dir: 输出目录(所有图片都存在这里)
        item_id: 条目ID
        scene_id: Scene ID（用于命名输出文件）
        port: 使用的端口号（用于日志显示）
        
    Returns:
        是否成功
    """
    if len(images) < 2:
        logger.error(f"条目 {item_id} (scene: {scene_id}) 图片数量不足（需要至少2张）")
        return False
    
    port_info = f" [端口:{port}]" if port else ""
    logger.info(f"\n{'='*60}")
    logger.info(f"处理条目 {item_id} (scene: {scene_id}): {len(images)} 张图片{port_info}")
    for i, img_path in enumerate(images, 1):
        logger.info(f"  图片{i}: {os.path.basename(img_path)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"{'='*60}")
    
    # 首先进行一次完整的3D重建(会缓存PLY和相机位姿)
    logger.info("步骤1: 执行初始3D重建并缓存...")
    initial_start = time.time()
    
    initial_result = client.infer_from_images(
        image_paths=images,  # 使用所有图片进行3D重建
        conf_threshold=0.08,
        rtol=0.02,
        generate_views=False,  # 第一次不生成视角，只重建和缓存
        use_filename=True
    )
    
    if not initial_result:
        logger.error(f"条目 {item_id} 初始重建失败，跳过")
        return False
    
    initial_time = time.time() - initial_start
    logger.info(f"初始重建完成，耗时: {initial_time:.2f}秒")
    logger.info(f"点云数量: {initial_result.get('points_count', '未知')}")
    logger.info(f"是否使用缓存: {initial_result.get('cached', False)}")
    
    # 为每个角度生成视角图片
    logger.info(f"\n步骤2: 生成 {len(angles)} 个角度的视角图片...")
    
    success_count = 0
    total_view_time = 0
    
    for idx, (azimuth, elevation, angle_name) in enumerate(angles, 1):
        view_start = time.time()
        
        logger.info(f"\n[{idx}/{len(angles)}] 生成角度: 方位角={azimuth}°, 仰角={elevation}° ({angle_name})")
        
        # 请求生成特定角度的视角
        result = client.infer_from_images(
            image_paths=images,  # 使用所有图片生成视角
            conf_threshold=0.08,
            rtol=0.02,
            generate_views=True,
            use_filename=True,
            azimuth_angle=azimuth,
            elevation_angle=elevation
        )
        
        if result and result.get("success"):
            # 保存视角图片，使用scene_id命名
            if "camera_views" in result and result["camera_views"]:
                for view_data in result["camera_views"]:
                    camera = view_data.get("camera", 1)
                    # 使用scene_id作为文件名的一部分，确保唯一性
                    view_img_name = f"pi3_{scene_id}_cam{camera}_azim{float(azimuth):.1f}_elev{float(elevation):.1f}.png"
                    view_img_path = os.path.join(output_dir, view_img_name)
                    
                    img_data = base64.b64decode(view_data["image"])
                    with open(view_img_path, 'wb') as f:
                        f.write(img_data)
                
                view_time = time.time() - view_start
                total_view_time += view_time
                success_count += 1
                
                cached = result.get("cached", False)
                logger.info(f"  ✓ 成功生成 (耗时: {view_time:.2f}秒, 缓存命中: {cached})")
            else:
                logger.warning(f"  ✗ 未生成视角图片")
        else:
            logger.error(f"  ✗ 生成失败")
        
        # 每10个角度输出一次进度
        if idx % 10 == 0:
            avg_time = total_view_time / success_count if success_count > 0 else 0
            remaining = len(angles) - idx
            estimated_time = avg_time * remaining
            logger.info(f"\n进度: {idx}/{len(angles)} ({idx/len(angles)*100:.1f}%)")
            logger.info(f"平均耗时: {avg_time:.2f}秒/角度")
            logger.info(f"预计剩余时间: {estimated_time:.1f}秒 ({estimated_time/60:.1f}分钟)")
    
    # 统计信息
    total_time = initial_time + total_view_time
    logger.info(f"\n{'='*60}")
    logger.info(f"条目 {item_id} (scene: {scene_id}) 处理完成!")
    logger.info(f"总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
    logger.info(f"初始重建: {initial_time:.2f}秒")
    logger.info(f"视角生成: {total_view_time:.2f}秒")
    logger.info(f"成功生成: {success_count}/{len(angles)} 个角度")
    logger.info(f"平均耗时: {total_view_time/success_count if success_count > 0 else 0:.2f}秒/角度")
    logger.info(f"文件保存在: {output_dir}/pi3_{scene_id}_*.png")
    logger.info(f"{'='*60}\n")
    
    return success_count > 0


def process_images_with_client(client: Pi3Client, 
                               images: List[str], 
                               angles: List[Tuple[float, float, str]], 
                               output_dir: str,
                               item_id: int,
                               scene_id: str,
                               port: int) -> Tuple[bool, int, str]:
    """
    使用指定客户端处理图片组（用于线程池）
    
    Returns:
        (success, item_id, scene_id)
    """
    success = generate_views_for_images(client, images, angles, output_dir, item_id, scene_id, port=port)
    return success, item_id, scene_id


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量生成不同角度的视角图片')
    parser.add_argument('--jsonl', type=str, default='dataset/VLM-3R/filtered/vlm3r_dataset.jsonl',
                       help='JSONL文件路径，包含图片列表')
    parser.add_argument('--ports', type=int, nargs='+', required=True,
                       help='Pi3服务器端口列表，例如: --ports 20030 20032 20034 20036')
    parser.add_argument('--host', type=str, default='10.7.33.15',
                       help='服务器主机地址 (默认: 10.7.33.15)')
    parser.add_argument('--output-dir', type=str, default='outputs/angle_views',
                       help='输出目录路径 (默认: outputs/angle_views)')
    
    args = parser.parse_args()
    
    # 配置参数
    JSONL_PATH = args.jsonl
    OUTPUT_DIR = args.output_dir
    PORTS = args.ports
    HOST = args.host
    
    logger.info("="*80)
    logger.info("批量生成角度视角图片工具（并行模式）")
    logger.info("="*80)
    logger.info(f"JSONL文件: {JSONL_PATH}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"服务器主机: {HOST}")
    logger.info(f"服务器端口: {PORTS} (共 {len(PORTS)} 个，并行模式)")
    logger.info("="*80 + "\n")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化所有客户端
    logger.info("初始化Pi3客户端...")
    clients = []
    for port in PORTS:
        server_url = f"http://{HOST}:{port}"
        client = Pi3Client(server_url=server_url)
        
        # 健康检查
        logger.info(f"检查服务器 {server_url}...")
        health = client.health_check()
        if not health:
            logger.error(f"服务器 {server_url} 健康检查失败，跳过")
            continue
        
        logger.info(f"服务器 {server_url} 状态正常")
        clients.append((client, port))
    
    if not clients:
        logger.error("没有可用的服务器，程序退出")
        return 1
    
    num_threads = len(clients)
    logger.info(f"成功初始化 {num_threads} 个客户端")
    logger.info(f"将使用 {num_threads} 个并行线程（每个端口1个线程）\n")
    
    # 从JSONL文件加载图片列表
    logger.info(f"从JSONL文件加载图片列表: {JSONL_PATH}")
    image_data_list = load_images_from_jsonl(JSONL_PATH)
    
    if not image_data_list:
        logger.error("未找到任何有效的图片记录")
        return 1
    
    logger.info(f"共加载 {len(image_data_list)} 组图片\n")
    
    # 生成角度规格
    angles = generate_angle_specifications()
    
    # 显示角度信息示例
    logger.info("角度规格示例:")
    logger.info(f"  正面: (0, 0)")
    logger.info(f"  左侧: (-45, 0), (-90, 0), (-135, 0)")
    logger.info(f"  右侧: (45, 0), (90, 0), (180, 0), (135, 0)")
    logger.info(f"  上侧: (0, 45), (0, 60)")
    logger.info(f"  下侧: (0, -45)")
    logger.info(f"  对角: (45, 30), (-45, 30)")
    logger.info(f"总计: {len(angles)} 个角度\n")
    
    # 并行处理图片组
    total_start = time.time()
    success_count = 0
    
    logger.info(f"开始并行处理 {len(image_data_list)} 组图片，使用 {num_threads} 个并发线程...\n")
    
    # 将图片组分配给不同的客户端（轮询分配）
    tasks = []
    for idx, (images, item_id, scene_id) in enumerate(image_data_list):
        client, port = clients[idx % len(clients)]
        tasks.append((client, images, item_id, scene_id, port))
    
    # 使用线程池并行执行（线程数 = 可用端口数）
    completed = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有任务
        future_to_data = {
            executor.submit(process_images_with_client, client, images, angles, OUTPUT_DIR, item_id, scene_id, port): 
            (images, item_id, scene_id, port) for client, images, item_id, scene_id, port in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_data):
            images, item_id, scene_id, port = future_to_data[future]
            completed += 1
            
            try:
                success, result_item_id, result_scene_id = future.result()
                if success:
                    success_count += 1
                    logger.info(f"✓ [{completed}/{len(image_data_list)}] 条目 {result_item_id} (scene: {result_scene_id}) 处理完成 (端口:{port})")
                else:
                    logger.error(f"✗ [{completed}/{len(image_data_list)}] 条目 {result_item_id} (scene: {result_scene_id}) 处理失败 (端口:{port})")
            except Exception as e:
                logger.error(f"✗ [{completed}/{len(image_data_list)}] 条目 {item_id} (scene: {scene_id}) 处理出错: {e} (端口:{port})")
            
            # 输出进度
            if completed % 5 == 0 or completed == len(image_data_list):
                elapsed = time.time() - total_start
                avg_time_per_item = elapsed / completed
                remaining_items = len(image_data_list) - completed
                estimated_remaining = avg_time_per_item * remaining_items
                
                logger.info(f"\n总体进度: {completed}/{len(image_data_list)} ({completed/len(image_data_list)*100:.1f}%)")
                logger.info(f"已用时间: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
                if remaining_items > 0:
                    logger.info(f"预计剩余: {estimated_remaining:.1f}秒 ({estimated_remaining/60:.1f}分钟)")
                logger.info("")
    
    # 最终统计
    total_time = time.time() - total_start
    logger.info("\n" + "="*80)
    logger.info("批量处理完成!")
    logger.info("="*80)
    logger.info(f"总耗时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    logger.info(f"处理条目: {success_count}/{len(image_data_list)}")
    logger.info(f"每条平均耗时: {total_time/len(image_data_list) if image_data_list else 0:.1f}秒")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"每组图片生成: {len(angles)} 个角度视角")
    logger.info(f"文件命名格式: pi3_{{scene_id}}_cam{{camera}}_azim{{azimuth}}_elev{{elevation}}.png")
    logger.info("="*80)
    
    return 0 if success_count == len(image_data_list) else 1


if __name__ == "__main__":
    sys.exit(main())
