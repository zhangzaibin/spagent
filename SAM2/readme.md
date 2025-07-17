## SAM2 服务部署说明

### 模型下载（用的是最新的SAM2.1模型）
测试时用的最小的tiny模型，服务器端可自行替换本地下载好的模型

请从以下链接下载SAM2模型权重文件：
- SAM2.1-Tiny: [https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_t.pt]
- SAM2.1-Small: [https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_s.pt]
- SAM2.1-Base: [https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt]
- SAM2.1-Large: [https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_l.pt]

### 使用说明
0.assets文件夹中存放了测试图片和视频

1. 首先启动服务器：
```bash
python sam_server.py
```
服务器默认将在本地5000端口启动。[http://127.0.0.1:5000]

2. 然后在新的终端中运行客户端：
```bash
python sam_client.py
```

### 功能说明
本服务提供以下功能：

1. 图像分割
   - 支持点提示：通过点击指定前景和背景
   - 支持框选提示：通过框选目标区域
   - 支持文本提示：通过文本描述目标
   - 支持多个目标的同时分割
   - 实时预览分割结果

2. 视频分割（新功能）
   - 支持视频目标分割和跟踪
   - 仅需在第一帧提供提示信息
   - 自动跟踪后续帧中的目标
   - 支持多种提示方式（点击、框选、文本）
   - 实时可视化分割结果
   - 自动保存处理后的视频

3. 批量处理
   - 支持对多张图片进行批量分割
   - 支持对多个视频进行批量处理
   - 自动保存所有结果

### API接口说明
服务器提供以下HTTP接口：
- `/health`：健康检查接口
- `/test`：测试接口，使用内置测试图像
- `/infer`：图像分割接口，接受图像和提示信息
- `/infer_video`：视频分割接口，接受视频和提示信息（新增）

### 使用示例

1. 图像分割示例：
```python
# 使用点提示进行分割
prompts = {
    'point_coords': [[100, 100]],  # 点击坐标
    'point_labels': [1]  # 1表示前景，0表示背景
}
result = client.infer('image.jpg', prompts)
```

2. 视频分割示例：
```python
# 使用点提示对视频进行分割
prompts = {
    'point_coords': [[100, 100]],  # 第一帧的点击坐标
    'point_labels': [1]  # 1表示前景点
}
output_path = client.infer_video('video.mp4', prompts)
```

### 注意事项
1. 首次运行时会自动下载模型，请确保网络连接正常
2. 建议使用GPU进行推理以获得更好的性能
3. 如遇到内存不足，可以调整batch_size或输入图像尺寸
4. 视频处理可能需要较长时间，请耐心等待
5. 视频文件建议使用MP4格式，其他格式可能需要额外编解码器

### 常见问题
1. 如果遇到模型加载失败，请检查模型文件是否正确放置
2. 如果服务无法启动，请检查端口是否被占用
3. 如果推理速度较慢，请检查是否正确使用了GPU（和显卡性能有关）
4. 如果视频处理失败，请检查视频格式是否支持
5. 如果内存占用过高，请尝试降低视频分辨率或帧率 