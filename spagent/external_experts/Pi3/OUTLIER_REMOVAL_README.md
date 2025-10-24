# Pi3 离群点移除功能说明

## 概述

Pi3服务器现在支持自动移除点云中的离群点，这可以显著提升点云可视化的质量和准确性。该功能使用**统计离群点移除（Statistical Outlier Removal, SOR）**方法。

## 工作原理

统计离群点移除方法的工作流程：

1. 对每个点，计算它到最近的k个邻居的平均距离
2. 计算所有点的平均距离的全局均值和标准差
3. 移除那些平均距离超过 `mean + std_threshold × std` 的点

离群点通常是距离其他点较远的孤立点，它们可能由以下原因产生：
- 重建算法的噪声
- 传感器误差
- 场景边界的不确定性

## API 使用方法

### 参数说明

在调用 `/infer` API 时，可以使用以下参数控制离群点移除：

```python
{
    "images": [...],  # 必需：base64编码的图像列表
    
    # 离群点移除参数（可选）
    "remove_outliers": True,      # 是否启用离群点移除，默认为True
    "k_neighbors": 20,            # KNN邻居数量，默认为20
    "std_threshold": 2.0          # 标准差阈值，默认为2.0
}
```

### 参数详解

#### `remove_outliers` (布尔值，默认: True)
- `True`: 启用离群点移除
- `False`: 禁用离群点移除，保留所有点

#### `k_neighbors` (整数，默认: 20)
- 用于计算每个点邻域的最近邻数量
- 较大的值：更稳健但可能过于保守（可能保留一些离群点）
- 较小的值：更激进但可能误删正常点
- **推荐范围**: 10-50

#### `std_threshold` (浮点数，默认: 2.0)
- 标准差阈值倍数，控制过滤的严格程度
- 较大的值（如3.0）：保留更多点，只移除明显的离群点
- 较小的值（如1.0）：更激进地移除离群点，但可能误删正常点
- **推荐范围**: 1.5-3.0

## 使用示例

### Python 客户端示例

```python
import requests
import base64
import json

# 读取图像并编码
def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# 准备请求数据
data = {
    "images": [
        encode_image("image1.jpg"),
        encode_image("image2.jpg"),
    ],
    
    # 启用离群点移除（使用默认参数）
    "remove_outliers": True,
    
    # 或者自定义参数以获得更激进的过滤
    "k_neighbors": 30,        # 使用30个邻居
    "std_threshold": 1.5      # 更严格的阈值
}

# 发送请求
response = requests.post('http://localhost:20021/infer', json=data)
result = response.json()

print(f"原始点数可能被过滤到: {result['points_count']} 个点")
```

### cURL 示例

```bash
curl -X POST http://localhost:20021/infer \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64_image_1>", "<base64_image_2>"],
    "remove_outliers": true,
    "k_neighbors": 20,
    "std_threshold": 2.0
  }'
```

## 效果评估

离群点移除后，您会看到：

1. **日志输出**显示移除的点数和百分比：
   ```
   离群点移除完成: 原始点数=500000, 移除点数=25000 (5.00%), 保留点数=475000
   ```

2. **返回的PLY文件**和**可视化图片**都使用过滤后的点云

3. **更清晰的可视化效果**：
   - 移除了飘散在空中的噪声点
   - 场景边界更加清晰
   - 整体点云更加紧凑和准确

## 性能考虑

- 离群点移除会增加一定的处理时间，特别是对大型点云（> 100万点）
- 典型性能影响：增加1-3秒的处理时间
- 如果性能是关键考虑因素，可以：
  - 设置 `"remove_outliers": false` 禁用此功能
  - 减小 `k_neighbors` 值（如使用10或15）

## 调优建议

### 场景1：高质量重建，允许更长处理时间
```python
{
    "remove_outliers": True,
    "k_neighbors": 30,
    "std_threshold": 1.8
}
```

### 场景2：快速预览，平衡质量和速度
```python
{
    "remove_outliers": True,
    "k_neighbors": 15,
    "std_threshold": 2.0
}
```

### 场景3：保留最多细节，只移除明显离群点
```python
{
    "remove_outliers": True,
    "k_neighbors": 20,
    "std_threshold": 3.0
}
```

### 场景4：禁用离群点移除
```python
{
    "remove_outliers": False
}
```

## 注意事项

1. 如果点云数量小于 `k_neighbors`，离群点移除会自动跳过
2. 如果离群点移除失败（异常情况），系统会自动回退到使用原始点云
3. 离群点移除会同时应用到PLY文件和可视化图片

## 依赖要求

确保已安装 `scikit-learn`：

```bash
pip install scikit-learn
```

或使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```






