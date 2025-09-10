# Moondream

## 📁 文件夹结构

```
moondream/
├── md_server.py           # Moondream服务器端
├── md_client.py           # Moondream客户端
├── md_local.py           # 本地部署moondream
├── README.md             # 说明文档
├── __init__.py           # 模块初始化文件
└── __pycache__/          # Python缓存文件
```

## 📚 官方资源
- **官方网站**: [Moondream](https://moondream.ai/)
- **官方文档**: [Moondream API Documentation](https://docs.moondream.ai/)

## 🛠️ 安装要求

```bash
pip install moondream 
```

## ⚙️ 环境配置

设置Moondream API密钥：

```bash
export MOONDREAM_API_KEY="your_api_key"
```

## 🎯 使用方法

### 启动服务器

```bash
cd spagent/external_experts/moondream
python md_server.py --port 20022
```

### 使用客户端

```python
from spagent.external_experts.moondream import MoondreamClient
```

## ⚙️ 端口配置

默认端口: 20021

可以通过命令行参数修改：
```bash
python md_server.py --port 8080
```
