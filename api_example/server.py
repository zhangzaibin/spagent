'''
这是我们在做具身智能部署api的代码, 将模型在本地部署成一个api，
这样我可以从别的终端和机器访问，比较方便。大家部署api的代码可以和这套部署的范式对齐。
'''

import dataclasses
import pickle
import base64
import logging
from flask import Flask, request, jsonify
import jax
import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量保存加载的模型
policy = None

def load_model():
    """加载模型"""
    global policy
    try:
        logger.info("开始加载模型...")
        config = _config.get_config("pi0_fast_droid_train_lora")
        checkpoint_dir = "/18141169908/xjl/openpi/checkpoints/pi0_fast_droid_train_lora/train_8000_stack_agent1_50/7999"
        # for stack cube horizon为8
        # checkpoint_dir = "/18141169908/xjl/openpi/checkpoints/pi0_fast_droid_train_lora/train_8000_agent1_stack/7999"
        
        # 创建训练好的策略
        policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        logger.info("模型加载成功!")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": policy is not None
    })

@app.route('/infer', methods=['POST'])
def infer():
    """推理接口"""
    global policy
    
    if policy is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if 'example' not in data:
            return jsonify({"error": "缺少example数据"}), 400
        
        # 解码example数据 (假设客户端会发送base64编码的pickle数据)
        example_b64 = data['example']
        example_bytes = base64.b64decode(example_b64)
        example = pickle.loads(example_bytes)
        
        logger.info("收到推理请求")
        print(example)
        
        # 运行推理
        result = policy.infer(example)
        
        # 将结果序列化
        result_bytes = pickle.dumps(result)
        result_b64 = base64.b64encode(result_bytes).decode('utf-8')
        
        logger.info("推理完成")
        
        return jsonify({
            "success": True,
            "result": result_b64,
            "actions_shape": list(result["actions"].shape) if "actions" in result else None
        })
        
    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        return jsonify({"error": f"推理失败: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    """测试接口，使用示例数据进行推理"""
    global policy
    
    if policy is None:
        return jsonify({"error": "模型未加载"}), 500
    
    try:
        # 创建示例数据
        example = droid_policy.make_droid_example()
        
        # 运行推理
        result = policy.infer(example)
        
        return jsonify({
            "success": True,
            "message": "测试推理成功",
            "actions_shape": list(result["actions"].shape) if "actions" in result else None
        })
        
    except Exception as e:
        logger.error(f"测试推理出错: {e}")
        return jsonify({"error": f"测试失败: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("启动服务器...")
    
    # 启动时加载模型
    if not load_model():
        logger.error("无法启动服务器: 模型加载失败")
        exit(1)
    
    # 启动Flask服务器
    app.run(host='0.0.0.0', port=20010, debug=False) 