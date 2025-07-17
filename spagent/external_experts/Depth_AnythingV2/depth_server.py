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

from depth_anything_v2.dpt import DepthAnythingV2

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型和配置
model = None
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def load_model(encoder='vitl', checkpoint_path=None):
    """加载深度估计模型"""
    global model
    try:
        logger.info(f"正在加载模型（编码器类型：{encoder}）...")
        
        # 创建模型实例
        logger.info(f"正在创建模型实例，配置：{model_configs[encoder]}")
        model = DepthAnythingV2(**model_configs[encoder])
        
        # 加载检查点
        if checkpoint_path:
            logger.info(f"正在从 {checkpoint_path} 加载模型权重...")
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("模型权重加载成功")
            except Exception as e:
                logger.error(f"加载模型权重失败：{e}")
                return False
            
        # 设置设备和评估模式
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        logger.info(f"使用设备：{device}")
        
        model = model.to(device).eval()
        
        # 使用测试输入验证模型（输入尺寸必须是patch大小14的倍数）
        try:
            logger.info("正在使用测试数据验证模型...")
            # 使用224x224的输入尺寸（是14的倍数）
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            logger.info("模型验证成功！")
        except Exception as e:
            logger.error(f"模型验证失败：{e}")
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
        model_type = None
        if model is not None:
            model_type = model.encoder if hasattr(model, 'encoder') else '未知'
            
        status = {
            "status": "健康",
            "model_loaded": model is not None,
            "model_type": model_type,
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
        # 创建测试图像（黑白渐变）
        logger.info("正在创建测试图像...")
        test_image = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            test_image[:, i] = i
        
        # 运行推理
        logger.info("正在进行测试推理...")
        with torch.no_grad():
            depth = model.infer_image(test_image)
            logger.info(f"推理成功，输出尺寸：{depth.shape}")
        
        logger.info("测试推理完成")
        return jsonify({
            "success": True,
            "message": "测试推理成功",
            "shape": list(depth.shape)
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
    """深度估计推理接口"""
    global model
    
    if model is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "缺少图像数据"}), 400
        
        # 获取可选参数
        input_size = data.get('input_size', 518)  # 默认输入尺寸
        return_colored = data.get('return_colored', True)  # 是否返回彩色深度图
        
        # 解码base64图像
        try:
            image_bytes = base64.b64decode(data['image'])
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        except:
            return jsonify({"error": "图像数据无效"}), 400
        
        # 运行推理
        logger.info("正在进行深度估计...")
        with torch.no_grad():
            depth = model.infer_image(image, input_size)
        
        # 归一化深度图
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # 如果需要，应用颜色映射
        if return_colored:
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # 编码结果为base64
        _, buffer = cv2.imencode('.png', depth)
        depth_b64 = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("深度估计完成")
        return jsonify({
            "success": True,
            "depth_map": depth_b64,
            "shape": depth.shape
        })
        
    except Exception as e:
        logger.error(f"推理失败：{e}")
        return jsonify({"error": f"推理失败：{str(e)}"}), 500

if __name__ == '__main__':
    logger.info("正在启动服务器...")
    
    # 加载默认模型
    checkpoint_path = 'checkpoints/depth_anything_v2_vitb.pth'
    logger.info(f"正在从 {checkpoint_path} 加载模型")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"找不到模型文件：{checkpoint_path}")
        exit(1)
    
    if not load_model(encoder='vitb', checkpoint_path=checkpoint_path):
        logger.error("无法启动服务器：模型加载失败")
        exit(1)
    
    logger.info("模型加载成功，正在启动服务器...")
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=5000, debug=False) 