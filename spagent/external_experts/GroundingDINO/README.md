# GroundingDINO 目标检测

## 📁 文件夹结构

```
GroundingDINO/
├── grounding_dino_server.py       # GroundingDINO服务器端
├── grounding_dino_client.py       # GroundingDINO客户端
├── README.md                      # 说明文档
├── __pycache__/                   # Python缓存文件
├── checkpoints/                   # 模型权重文件夹
│   └── groundingdino_swinb_cogcoor.pth
└── configs/                       # 模型配置文件夹
    └── GroundingDINO_SwinB_cfg.py
```

## 📚 官方资源
- **官方仓库**: [GroundingDINO GitHub](https://github.com/IDEA-Research/GroundingDINO)
- **论文**: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

## 📦 安装
```bash
pip install groundingdino_py
```

## 📥 权重下载
模型权重下载链接：
```bash
cd checkpoints/
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

## 📊 模型信息

| 模型 | 骨干网络 | 权重文件 | 用途 |
|------|----------|----------|------|
| GroundingDINO | Swin-B | groundingdino_swinb_cogcoor.pth | 开放词汇目标检测 |