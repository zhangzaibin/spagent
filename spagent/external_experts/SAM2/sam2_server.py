import base64
import cv2
import io
import logging
import numpy as np
import torch
import os
from flask import Flask, request, jsonify
from PIL import Image
import traceback
from ultralytics import SAM

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 添加文件处理器
file_handler = logging.FileHandler('sam_server.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

app = Flask(__name__)

# 全局变量存储模型和配置
model = None
model_name = None
model_configs = {
    'sam2.1_t': {'model_type': 'sam2.1_t.pt'},
    'sam2.1_s': {'model_type': 'sam2.1_s.pt'},
    'sam2.1_b': {'model_type': 'sam2.1_b.pt'},
    'sam2.1_l': {'model_type': 'sam2.1_l.pt'}
}


def load_model(model_type='sam2.1_t', model_path=None):
    """加载SAM2模型"""
    global model
    global model_name
    try:
        logger.info(f"正在加载模型（类型：{model_type}）...")
        model_name = model_type
        
        model = SAM(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"使用设备：{device}")
        
        model = model.to(device).eval()
        
        logger.info("模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"模型加载失败：{e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        model_type = None
        if model is not None:
            model_type = model.type if hasattr(model, 'type') else '未知'
            
        status = {
            "status": "健康",
            "model_name": model_name if model_name else None,
            "device": str(next(model.model.parameters()).device) if model is not None else None
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
        # 创建测试图像
        logger.info("正在创建测试图像...")
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (64, 64), (192, 192), (255, 255, 255), -1)
        
        # 运行推理
        logger.info("正在进行测试推理...")
        if model is None:
            raise ValueError("模型未加载")
            
        results = model.predict(
            test_image,
            points=[[128, 128]],  # 中心点
            labels=[1],  # 前景点
            conf=0.5,
            save=False
        )
        
        logger.info("测试推理完成")
        return jsonify({
            "success": True,
            "message": "测试推理成功",
            "shape": test_image.shape
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
    """图像分割推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "缺少图像数据"}), 400
        
        # 解码base64图像
        try:
            image_bytes = base64.b64decode(data['image'])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except:
            return jsonify({"error": "图像数据无效"}), 400
        
        # 获取提示信息
        prompts = {}
        if 'point_coords' in data:
            prompts['points'] = data['point_coords']
            prompts['labels'] = data.get('point_labels', [1] * len(data['point_coords']))
        
        if 'box' in data:
            prompts['bboxes'] = data['box']  # [x1, y1, x2, y2]
            

        # 运行推理
        logger.info("正在进行分割...")
        results = model.predict(
            image,
            **prompts,
            conf=data.get('conf', 0.5),
            save=False
        )
        
        # 获取掩码
        masks = results[0].masks
        if masks is None:
            return jsonify({"error": "未检测到目标"}), 400
            
        # 将掩码转换为图像格式
        mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in masks:
            mask_array = mask.data.detach().cpu().numpy().squeeze()
            mask_image = np.logical_or(mask_image, mask_array)
        mask_image = (mask_image * 255).astype(np.uint8)
        
        # 编码结果为base64
        _, buffer = cv2.imencode('.png', mask_image)
        mask_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        logger.info("分割完成")
        return jsonify({
            "success": True,
            "mask": mask_b64,
            "shape": mask_image.shape
        })
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

@app.route('/infer_video', methods=['POST'])
def infer_video():
    """视频分割推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'video' not in data:
            return jsonify({"error": "缺少视频数据"}), 400
        
        # 解码base64视频
        try:
            video_bytes = base64.b64decode(data['video'])
            temp_video = 'temp_video.mp4'
            with open(temp_video, 'wb') as f:
                f.write(video_bytes)
            
            cap = cv2.VideoCapture(temp_video)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
        except:
            return jsonify({"error": "视频数据无效"}), 400
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 准备输出视频
        output_video = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
        
        # 获取提示信息（仅用于第一帧）
        prompts = {}
        if 'point_coords' in data:
            prompts['points'] = data['point_coords']
            prompts['labels'] = data.get('point_labels', [1] * len(data['point_coords']))
        
        if 'box' in data:
            prompts['box'] = data['box']
            
        
        # 处理视频
        logger.info("开始处理视频...")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 对第一帧使用提供的提示信息
            if frame_count == 0:
                results = model.track(frame, **prompts, conf=data.get('conf', 0.5), persist=True)
            else:
                results = model.track(frame, conf=data.get('conf', 0.5), persist=True)
            
            # 绘制结果
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frame_count += 1
            
        cap.release()
        out.release()
        os.remove(temp_video)
        
        # 将输出视频编码为base64
        with open(output_video, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove(output_video)
        
        logger.info("视频处理完成")
        return jsonify({
            "success": True,
            "video": video_b64,
            "frames": frame_count,
            "fps": fps,
            "size": [frame_width, frame_height]
        })
        
    except Exception as e:
        logger.error(f"视频处理失败：{e}")
        return jsonify({"error": f"视频处理失败：{str(e)}"}), 500

if __name__ == '__main__':
    logger.info("正在启动服务器...")
    model_path = 'spagent/external_experts/SAM2/checkpoints/sam2.1_b.pt'
    # 加载默认模型
    if not load_model(model_type='sam2.1_b', model_path=model_path):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    # 启动Flask服务器
    app.run(host='127.0.0.1', port=5000, debug=False) 