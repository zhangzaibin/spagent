# SAM2 (Segment Anything Model 2)

## 📁 文件夹结构

```
SAM2/
├── sam2_server.py          # SAM2服务器端
├── sam2_client.py          # SAM2客户端  
├── README.md              # 说明文档
├── __pycache__/           # Python缓存文件
└── checkpoints/           # 模型权重文件夹
    ├── sam2.1_b.pt
    ├── sam2.1_l.pt
    └── sam2.1_s.pt
    └── sam2.1_t.pt
```

## 📚 官方资源
- **官方仓库**: [SAM2 GitHub](https://github.com/facebookresearch/sam2)
- **论文**: [SAM 2: Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)

## 📥 权重下载

#### 使用官方脚本（推荐）
```bash
cd checkpoints/
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```

#### 手动下载
```bash
cd checkpoints/

# SAM2.1 Hiera Large (推荐)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# SAM2.1 Hiera Base+ 
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# SAM2.1 Hiera Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

## 📊 模型规格

| 模型 | 参数量 | 文件大小 | 用途 |
|------|--------|----------|------|
| Hiera Large | ~224M | ~900MB | 高精度 |
| Hiera Base+ | ~80M | ~320MB | 平衡性能 |
| Hiera Small | ~46M | ~185MB | 快速推理 |