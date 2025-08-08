import base64
import io
import logging
import os
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLOE

# 模拟或真实模型推理接口
# from mock_supervision_service import MockSupervisionService  # 你可以替换为真实推理函数
# service = MockSupervisionService()

from yoloe_annotator import Annotator
service = Annotator()

# Flask 实例
app = Flask(__name__)

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    try:
        status = {
            "status": "ok",
            "available_tasks": ["image", "video"],
            "available_endpoints": ["/infer", "/infer_video", "/test"],
            "mock_mode": False,
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"健康检查失败：{e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test_infer():
    """测试推理接口（使用服务器内置图片）"""
    try:
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        test_img[:, :] = (0, 255, 0)

        annotated = cv2.putText(test_img.copy(), "TEST", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)

        _, buffer = cv2.imencode(".jpg", annotated)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({
            "success": True,
            "annotated_image": img_b64,
            "shape": annotated.shape
        })

    except Exception as e:
        logger.error(f"测试推理失败: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer_video", methods=["POST"])
def infer_video():
    """视频文件推理接口"""
    try:
        data = request.get_json()
        if not data or 'video' not in data:
            return jsonify({"success": False, "error": "缺少视频数据"}), 400

        task = data.get('task', 'video')
        model_name = data.get('model_name', 'yoloe-v8l-seg.pt')
        class_names = data.get('class_names', ["person", "car"])
        output_filename = data.get('output_filename', 'output.mp4')
        
        # 创建模型实例
        model = YOLOE(model_name)
        model.set_classes(class_names, model.get_text_pe(class_names))

        # 解码视频文件
        try:
            video_bytes = base64.b64decode(data['video'])
            
            # 保存临时视频文件
            temp_input_path = f"/tmp/temp_input_{os.getpid()}.mp4"
            temp_output_path = f"/tmp/temp_output_{os.getpid()}.mp4"
            
            with open(temp_input_path, 'wb') as f:
                f.write(video_bytes)
                
        except Exception:
            return jsonify({"success": False, "error": "视频数据格式错误"}), 400

        logger.info(f"接收到视频处理请求 - task: {task}, model: {model_name}, 文件大小: {len(video_bytes)/1024/1024:.1f}MB")

        # 调用yoloe_annotator处理整个视频文件
        try:
            result_path = service(temp_input_path, model, task=task, target_path=temp_output_path)
            
            # 读取处理后的视频
            with open(result_path, 'rb') as f:
                processed_video_bytes = f.read()
            
            # 编码返回视频
            video_b64 = base64.b64encode(processed_video_bytes).decode('utf-8')
            
            # 清理临时文件
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            return jsonify({
                "success": True,
                "annotated_video": video_b64,
                "task": task,
                "model_used": model_name,
                "classes": class_names
            })
            
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            raise e

    except Exception as e:
        logger.error(f"视频处理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    """图像/视频推理接口（多任务标注）"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "缺少图像数据"}), 400

        task = data.get('task', 'image')  # 'image' 或 'video'
        model_name = data.get('model_name', 'yoloe-v8l-seg.pt')  # 默认模型
        class_names = data.get('class_names', ["person", "car"])  # 默认类别
        
        # 创建模型实例
        model = YOLOE(model_name)
        model.set_classes(class_names, model.get_text_pe(class_names))

        # 解码图像
        try:
            image_bytes = base64.b64decode(data['image'])
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image_np is None:
                raise ValueError("无法解码图像")
        except Exception:
            return jsonify({"success": False, "error": "图像数据格式错误"}), 400

        logger.info(f"接收到推理请求 - task: {task}, model: {model_name}, image shape: {image_np.shape}")

        # 调用yoloe_annotator进行推理
        # 注意：对于视频任务，我们仍然处理单帧，但使用视频模式的标注
        annotated_image = service(image_np, model, task=task)

        # 编码返回图像
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "annotated_image": img_b64,
            "shape": annotated_image.shape,
            "task": task,
            "model_used": model_name,
            "classes": class_names
        })

    except Exception as e:
        logger.error(f"推理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    logger.info("启动标注服务中...")
    app.run(host='0.0.0.0', port=8000, debug=False)
