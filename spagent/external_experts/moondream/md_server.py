import base64
import cv2
import io
import logging
import numpy as np
import os
import argparse
import json
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import traceback
import moondream as md
import hashlib

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型
model = None

def load_model():
    """加载Moondream模型"""
    global model
    try:
        logger.info("正在加载Moondream模型...")
        api_key = os.getenv("MOONDREAM_API_KEY")
        if not api_key:
            logger.error("环境变量MOONDREAM_API_KEY未设置")
            return False
            
        model = md.vl(api_key=api_key)
        logger.info("Moondream模型加载完成！")
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
            "api_key_set": os.getenv("MOONDREAM_API_KEY") is not None
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
        if model is None:
            return jsonify({"error": "模型未加载"}), 500
            
        # 创建测试图像
        logger.info("正在创建测试图像...")
        test_image = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(test_image)
        draw.rectangle([64, 64, 192, 192], fill='blue')
        draw.text((128, 200), "Test", fill='black')
        
        # 测试caption功能
        logger.info("正在进行测试推理...")
        result = model.caption(test_image)
        caption_text = result['caption']
        
        logger.info("测试推理完成")
        return jsonify({
            "success": True,
            "message": "测试推理成功",
            "caption": caption_text
        })
        
    except Exception as e:
        logger.error(f"测试推理失败：{e}")
        logger.error(f"错误追踪：{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def save_annotated_image(image, detection_boxes=None, point_coords=None, filename_prefix="annotated"):
    """创建标注后的图像并返回base64编码"""
    try:
        # 创建图像副本用于标注
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        w, h = overlay.size
        
        # 绘制检测框
        if detection_boxes:
            for box in detection_boxes:
                draw.rectangle([
                    int(box['x_min'] * w),
                    int(box['y_min'] * h),
                    int(box['x_max'] * w),
                    int(box['y_max'] * h)
                ], outline='red', width=3)
        
        # 绘制点
        if point_coords:
            for pt in point_coords:
                r = 4
                draw.ellipse([
                    int(pt['x'] * w) - r, int(pt['y'] * h) - r,
                    int(pt['x'] * w) + r, int(pt['y'] * h) + r
                ], fill='blue')
        
        # 将PIL图像转换为numpy数组，然后编码为base64
        import cv2
        overlay_np = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', overlay_np)
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        logger.info("标注图像已创建并编码")
        return annotated_b64
        
    except Exception as e:
        logger.error(f"创建标注图像失败：{e}")
        return None

def save_multi_object_annotated_image(image, multi_points, color_mapping, filename_prefix="multi_annotated"):
    """创建多对象标注图像，使用不同颜色"""
    try:
        # 创建图像副本用于标注
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        w, h = overlay.size
        
        # 定义颜色列表（RGB格式）
        colors = [
            (255, 0, 0),     # 红色
            (0, 255, 0),     # 绿色
            (0, 0, 255),     # 蓝色
            (255, 255, 0),   # 黄色
            (255, 0, 255),   # 紫色
            (0, 255, 255),   # 青色
            (255, 165, 0),   # 橙色
            (128, 0, 128),   # 紫罗兰
            (255, 192, 203), # 粉色
            (165, 42, 42),   # 棕色
        ]
        
        # 为每个对象分配颜色
        color_assignment = {}
        for i, obj_name in enumerate(multi_points.keys()):
            color_assignment[obj_name] = colors[i % len(colors)]
        
        # 绘制每个对象的点
        for obj_name, points in multi_points.items():
            color = color_assignment[obj_name]
            for pt in points:
                r = 6  # 稍微大一点的点
                x, y = int(pt['x'] * w), int(pt['y'] * h)
                
                # 绘制实心圆
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
                # 绘制边框让点更明显
                draw.ellipse([x - r, y - r, x + r, y + r], outline=(0, 0, 0), width=1)
        
        # 更新颜色映射信息
        for obj_name in multi_points.keys():
            color = color_assignment[obj_name]
            color_mapping[obj_name] = f"RGB{color}"
        
        # 将PIL图像转换为numpy数组，然后编码为base64
        import cv2
        overlay_np = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', overlay_np)
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        logger.info("多对象标注图像已创建并编码")
        return annotated_b64
        
    except Exception as e:
        logger.error(f"创建多对象标注图像失败：{e}")
        return None

@app.route('/infer', methods=['POST'])
def infer():
    """Moondream推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "缺少图像数据"}), 400
        
        if 'task' not in data:
            return jsonify({"error": "缺少任务类型"}), 400
        
        task = data['task']
        if task not in ['caption', 'query', 'detect', 'point']:
            return jsonify({"error": "无效的任务类型，支持: caption, query, detect, point"}), 400
        
        # 解码base64图像
        try:
            image_bytes = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"图像数据无效: {str(e)}"}), 400
        
        result = {}
        annotated_image_b64 = None
        
        # 根据任务类型执行相应操作
        if task == 'caption':
            logger.info("正在生成图像描述...")
            caption_result = model.caption(image)
            result = {
                "task": "caption",
                "caption": caption_result['caption']
            }
            
        elif task == 'query':
            if 'question' not in data:
                return jsonify({"error": "query任务需要question参数"}), 400
            
            question = data['question']
            logger.info(f"正在回答问题：{question}")
            query_result = model.query(image, question)
            result = {
                "task": "query",
                "question": question,
                "answer": query_result['answer']
            }
            
        elif task == 'detect':
            if 'object' not in data:
                return jsonify({"error": "detect任务需要object参数"}), 400
            
            object_name = data['object']
            logger.info(f"正在检测对象：{object_name}")
            detection_result = model.detect(image, object_name)
            detection_boxes = detection_result['objects']
            
            # 创建标注图像
            annotated_image_b64 = save_annotated_image(
                image, 
                detection_boxes=detection_boxes,
                filename_prefix="detection"
            )
            
            result = {
                "task": "detect",
                "object": object_name,
                "detections": detection_boxes,
                "annotated_image": annotated_image_b64
            }
            
        elif task == 'point':
            if 'object' not in data:
                return jsonify({"error": "point任务需要object参数"}), 400
            
            object_input = data['object']
            logger.info(f"正在定位对象：{object_input}")
            
            # 解析对象输入，支持单个对象或逗号分隔的多个对象
            object_names = [name.strip() for name in object_input.split(',')]
            is_multi_object = len(object_names) > 1
            
            if is_multi_object:
                # 多对象定位
                logger.info(f"检测到多对象输入：{object_names}")
                multi_points = {}
                color_mapping = {}
                total_points = 0
                
                # 对每个对象进行定位
                for obj_name in object_names:
                    try:
                        point_result = model.point(image, obj_name)
                        obj_points = point_result['points']
                        multi_points[obj_name] = obj_points
                        total_points += len(obj_points)
                        
                        logger.info(f"定位到 {len(obj_points)} 个 '{obj_name}' 的关键点")
                        
                    except Exception as e:
                        logger.warning(f"定位对象 '{obj_name}' 时出错: {e}")
                        multi_points[obj_name] = []
                
                # 创建多对象标注图像
                annotated_image_b64 = save_multi_object_annotated_image(
                    image,
                    multi_points,
                    color_mapping,
                    filename_prefix="multi_pointing"
                )
                
                result = {
                    "task": "point",
                    "object": object_input,
                    "objects": object_names,
                    "is_multi_object": True,
                    "all_points": multi_points,
                    "total_points": total_points,
                    "color_mapping": color_mapping,
                    "annotated_image": annotated_image_b64
                }
            else:
                # 单对象定位
                object_name = object_names[0]
                point_result = model.point(image, object_name)
                point_coords = point_result['points']
                
                # 创建标注图像
                annotated_image_b64 = save_annotated_image(
                    image,
                    point_coords=point_coords,
                    filename_prefix="pointing"
                )
                
                result = {
                    "task": "point",
                    "object": object_name,
                    "is_multi_object": False,
                    "points": point_coords,
                    "annotated_image": annotated_image_b64
                }
        
        logger.info(f"{task}任务完成")
        return jsonify({
            "success": True,
            **result
        })
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        logger.error(f"错误追踪：{traceback.format_exc()}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Moondream Server')
    parser.add_argument('--port', type=int, default=20024,
                        help='Port to run the server on (default: 20024)')
    
    args = parser.parse_args()
    
    logger.info("正在启动Moondream服务器...")
    logger.info(f"服务端口: {args.port}")
    
    # 加载模型
    if not load_model():
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=args.port, debug=False)
