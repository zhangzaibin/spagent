# Supervision + YOLOE 目标检测

## 📁 文件夹结构

```
supervision/
├── __init__.py                    # 初始化文件
├── _service.py                    # 服务基类
├── annotator.py                   # 标注工具
├── supervision_server.py          # Supervision服务器端
├── supervision_client.py          # Supervision客户端
├── sv_yoloe_server.py             # YOLOE服务器端
├── sv_yoloe_client.py             # YOLOE客户端
├── yoloe_annotator.py             # YOLOE标注工具
├── yoloe_test.py                  # YOLOE测试文件
├── download_weights.py            # YOLOE权重下载脚本
├── mock_supervision_service.py    # 模拟服务（测试用）
├── README.md                      # 说明文档
├── __pycache__/                   # Python缓存文件
└── checkpoints/                   # 模型权重文件夹
    ├── .cache/                    # 缓存文件
    ├── yoloe-v8l-seg.pt          # YOLOE v8 large分割模型
    └── yoloe-v8l-seg-pf.pt       # YOLOE v8 large分割模型(优化版)
```

## 📚 官方资源
- **官方仓库**: [Supervision GitHub](https://github.com/roboflow/supervision)
- **文档**: [Supervision Documentation](https://supervision.roboflow.com/)

## 📦 安装
官方链接 [Supervision GitHub](https://github.com/roboflow/supervision)

安装supervision：
```bash
pip install supervision
```

## 📥 权重下载

### Supervision权重
运行server和client的时候会自动下载相关模型

### YOLOE权重
运行权重下载脚本：
```bash
python download_weights.py
```

## 📊 可用模型

| 模型文件 | 功能 | 用途 |
|----------|------|------|
| yoloe-v8l-seg.pt | YOLOE v8 Large 分割 | 高精度目标检测和分割 |
| yoloe-v8l-seg-pf.pt | YOLOE v8 Large 分割(优化版) | 性能优化的分割模型 |