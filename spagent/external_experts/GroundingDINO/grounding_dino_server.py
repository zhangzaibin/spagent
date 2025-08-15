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
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T


# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 添加文件处理器
file_handler = logging.FileHandler('grounding_dino_server.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

app = Flask(__name__)

# 全局变量存储模型和配置
model = None
model_name = None
transform = None

def load_grounding_dino_model(model_path=None):
    """加载Grounding DINO模型"""
    global model
    global model_name
    global transform
    
    try:
        logger.info("正在加载Grounding DINO模型...")
        
        # 默认模型路径
        if model_path is None:
            model_path = "groundingdino_swinb_cogcoor.pth"
        
        model_name = "grounding_dino"
        
        # 加载模型
        model = load_model("spagent/external_experts/GroundingDINO/configs/GroundingDINO_SwinB_cfg.py", model_path)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # 创建图像变换
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info(f"使用设备：{device}")
        logger.info("Grounding DINO模型加载完成！")
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
            "model_name": model_name if model_name else None,
            # "device": str(next(model.parameters()).device) if model is not None else None
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
        
        # 转换为PIL图像
        test_image_pil = Image.fromarray(test_image)
        
        # 运行推理
        logger.info("正在进行测试推理...")
        if model is None:
            raise ValueError("模型未加载")
        
        # 测试文本提示
        text_prompt = "white rectangle"
        box_threshold = 0.35
        text_threshold = 0.25
        
        # 转换图像
        image_transformed, _ = transform(test_image_pil, None)
        
        # 预测
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        logger.info("测试推理完成")
        return jsonify({
            "success": True,
            "message": "测试推理成功",
            "shape": test_image.shape,
            "detected_objects": len(boxes) if boxes is not None else 0
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
    """目标检测推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "缺少图像数据"}), 400
        
        if 'text_prompt' not in data:
            return jsonify({"error": "缺少文本提示"}), 400
        
        # 解码base64图像
        try:
            image_bytes = base64.b64decode(data['image'])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
        except:
            return jsonify({"error": "图像数据无效"}), 400
        
        # 获取参数
        text_prompt = data['text_prompt']
        box_threshold = data.get('box_threshold', 0.35)
        text_threshold = data.get('text_threshold', 0.25)
        
        # 运行推理
        logger.info(f"正在进行目标检测，文本提示：{text_prompt}")
        
        # 转换图像
        image_transformed, _ = transform(image_pil, None)
        
        # 预测
        boxes, logits, phrases = predict(
            model=model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        logger.info(f"boxes: {boxes}")
        
        if boxes is None or len(boxes) == 0:
            return jsonify({
                "success": False,
                "detections": [],
                "message": "未检测到目标"
            })
        
        # 处理检测结果
        detections = []
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # 转换坐标格式 [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.cpu().numpy()
            
            detection = {
                "id": i,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(logit),
                "label": phrase
            }
            detections.append(detection)
        
        # 创建标注图像
        annotated_image = annotate(image_rgb, boxes, logits, phrases)
        # annotated_image_cv = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
        
        # 编码结果为base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        logger.info(f"检测完成，发现 {len(detections)} 个目标")
        return jsonify({
            "success": True,
            "detections": detections,
            "annotated_image": annotated_b64,
            "shape": image_rgb.shape
        })
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

@app.route('/infer_video', methods=['POST'])
def infer_video():
    """视频目标检测推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'video' not in data:
            return jsonify({"error": "缺少视频数据"}), 400
        
        if 'text_prompt' not in data:
            return jsonify({"error": "缺少文本提示"}), 400
        
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
        
        # 获取参数
        text_prompt = data['text_prompt']
        box_threshold = data.get('box_threshold', 0.35)
        text_threshold = data.get('text_threshold', 0.25)
        
        # 处理视频
        logger.info(f"开始处理视频，文本提示：{text_prompt}")
        frame_count = 0
        total_detections = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为PIL图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # 转换图像
            image_transformed, _ = transform(frame_pil, None)
            
            # 预测
            boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # 创建标注图像
            if boxes is not None and len(boxes) > 0:
                annotated_frame = annotate(frame_pil, boxes, logits, phrases)
                annotated_frame_cv = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
                out.write(annotated_frame_cv)
                total_detections += len(boxes)
            else:
                out.write(frame)
            
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
            "size": [frame_width, frame_height],
            "total_detections": total_detections
        })
        
    except Exception as e:
        logger.error(f"视频处理失败：{e}")
        return jsonify({"error": f"视频处理失败：{str(e)}"}), 500

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Grounding DINO Server')
    parser.add_argument('--model_path', type=str, default='checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth',
                        help='Path to Grounding DINO model checkpoint (default: checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth)')
    parser.add_argument('--port', type=int, default=20022,
                        help='Port to run the server on (default: 20022)')
    
    args = parser.parse_args()
    
    logger.info("正在启动Grounding DINO服务器...")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"服务端口: {args.port}")
    
    # 加载指定模型
    if not load_grounding_dino_model(args.model_path):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=args.port, debug=False) 