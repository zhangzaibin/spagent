import base64
import io
import logging
import os
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO

# 模拟或真实模型推理接口
# from mock_supervision_service import MockSupervisionService  # 你可以替换为真实推理函数
# service = MockSupervisionService()

from annotator import Annotator
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
            "available_tasks": ["image_det", "image_seg", "keypoints", "video_track"],
            "mock_mode": True,
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


@app.route("/infer", methods=["POST"])
def infer():
    """图像推理接口（多任务标注）"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "缺少图像数据"}), 400

        task = data.get('task', 'image_det')  # 默认任务
        model_name = data.get('model_name', 'mock_model.pt')  # 默认模型
        model = YOLO(model_name)

        # 解码图像
        try:
            image_bytes = base64.b64decode(data['image'])
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image_np is None:
                raise ValueError("无法解码图像")
        except Exception:
            return jsonify({"success": False, "error": "图像数据格式错误"}), 400

        logger.info(f"接收到推理请求 - task: {task}, model: {model_name}, image shape: {image_np.shape}")

        # 调用推理函数
        annotated_image = service(image_np, task=task, model=model)

        # 编码返回图像
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "annotated_image": img_b64,
            "shape": annotated_image.shape
        })

    except Exception as e:
        logger.error(f"推理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    logger.info("启动标注服务中...")
    app.run(host='0.0.0.0', port=8000, debug=False)
