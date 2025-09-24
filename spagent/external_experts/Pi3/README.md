# Pi3 3D重建服务

基于Pi3模型的3D重建服务，提供server/client架构和集成服务接口。

## 文件结构

```
Pi3/
├── pi3                   # 运行代码
├── example.py            # 原始Pi3运行代码(参考)
├── pi3_server.py         # Flask服务器
├── pi3_client.py         # 客户端
└── README.md            # 本文件
```

## 权重

下载链接 [link](https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors)

## 环境

torch==2.5.1

torchvision==0.20.1

numpy==1.26.4


## 快速开始

### 1. 准备模型权重

将Pi3模型权重文件放置在 `checkpoints/model.safetensors`

### 2. 运行示例

```bash
# 启动服务器(终端1)
python pi3_server.py --checkpoint_path spagent/external_experts/checkpoints/pi3/model.safetensors --port 20030

# 使用客户端(终端2)
python pi3_client.py
```

### 3. 将生成的ply文件转为可视化的文件
```bash
python spagent/utils/ply_to_html_viewer.py  xxx.ply  --output {html文件路径}/xxx.html  --max_points 100000
```