# Depth-Anything-V2 深度估计

## 📁 文件夹结构

```
Depth_AnythingV2/
├── __init__.py                 # 初始化文件
├── _service.py                 # 服务基类
├── depth_server.py             # 深度估计服务器端
├── depth_client.py             # 深度估计客户端
├── mock_depth_service.py       # 模拟服务（测试用）
├── README.md                   # 说明文档
├── __pycache__/                # Python缓存文件
├── checkpoints/                # 模型权重文件夹
│   ├── depth_anything_v2_vits.pth  # Small模型
│   ├── depth_anything_v2_vitb.pth  # Base模型
│   └── depth_anything_v2_vitl.pth  # Large模型
└── depth_anything_v2/          # 模型代码
    ├── dinov2.py
    ├── dpt.py
    ├── dinov2_layers/
    ├── util/
    └── __pycache__/
```

## 📚 官方资源
- **官方仓库**: [Depth-Anything-V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
- **论文**: [Depth Anything V2](https://arxiv.org/abs/2406.09414)

## 📝 组件说明

- **depth_server.py**: 服务器端，运行client或workflow之前需要先运行
- **depth_client.py**: 真实的客户端，部署完成后从depth_client中导入infer函数在workflow中调用
- **mock_depth_service.py**: 模拟客户端，用于调试，后续会被真实client替代
- **_service.py**: 集成了server和client的类，可以直接运行测试

## 📥 模型权重下载
### 手动下载权重文件
```bash
cd checkpoints/

# Depth-Anything-V2-Small
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

# Depth-Anything-V2-Base  
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# Depth-Anything-V2-Large
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

## 📊 模型规格

| 模型 | 骨干网络 | 参数量 | 文件大小 | 推理速度 | 精度 |
|------|----------|--------|----------|----------|------|
| Small | ViT-S | ~25M | ~100MB | 快 | 良好 |
| Base | ViT-B | ~97M | ~390MB | 中等 | 高 |
| Large | ViT-L | ~335M | ~1.3GB | 慢 | 很高 |

## 🚀 快速开始

### 1. 启动服务器
```bash
python depth_server.py  --port 8080
```

### 2. 测试客户端
```bash
python depth_client.py
```

注意：图片路径在客户端代码中配置
