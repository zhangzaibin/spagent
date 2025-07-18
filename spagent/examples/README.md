# BLINK Dataset Evaluation

这个目录包含了用于评估GPT-4o-mini在BLINK数据集上表现的脚本。

## 文件说明

- `straight_evaluation_gpt.py`: 主要的评估脚本
- `test_evaluation.py`: 测试脚本，用于验证功能是否正常
- `README.md`: 本说明文件

## 功能特性

1. **数据加载**: 从JSONL格式加载BLINK数据集
2. **图像推理**: 使用GPT-4o-mini对图像进行推理
3. **答案标准化**: 自动标准化答案格式（A, B, C, D）
4. **准确率计算**: 计算整体准确率和各任务类型的准确率
5. **性能统计**: 记录推理时间和成功率
6. **结果保存**: 将评估结果保存为JSON文件

## 使用方法

### 1. 环境准备

确保你已经安装了必要的依赖：

```bash
pip install openai tqdm
```

### 2. 数据准备

确保你的数据文件结构如下：

```
dataset/
├── blink_data.jsonl          # BLINK数据集文件
└── BLINK/                    # 图像文件夹
    ├── 02bf928316cf55ddda3d9e938b89f7624db742364c4dd89eb4e3fddb55f51f9a.jpg
    ├── ebb9c1c41b0fe3ff0d65cfc4ef3e2d26e4aefba3be654213a2aeab56d6546443.jpg
    └── ...
```

### 3. 运行测试

首先运行测试脚本确保一切正常：

```bash
cd spagent/examples
python test_evaluation.py
```

### 4. 运行评估

运行完整的评估：

```bash
python straight_evaluation_gpt.py
```

### 5. 自定义评估

你也可以在代码中修改参数：

```python
# 修改模型
model = "gpt-4o-mini"  # 或其他支持的模型

# 限制评估样本数（用于测试）
max_samples = 10  # 只评估前10个样本

# 修改数据路径
data_path = "your_data_path.jsonl"
image_base_path = "your_image_path"
```

## 输出结果

评估完成后，你会看到：

1. **控制台输出**: 包含详细的评估统计信息
2. **JSON文件**: 保存完整的评估结果

### 控制台输出示例

```
============================================================
BLINK DATASET EVALUATION RESULTS
============================================================
Model: gpt-4o-mini
Total samples: 386
Successful samples: 380
Failed samples: 6
Overall accuracy: 0.8234 (82.34%)
Average inference time: 2.45 seconds
Total inference time: 931.00 seconds

Task-wise Statistics:
----------------------------------------
Spatial_Relation    : 0.8156 (198/243)
Counting            : 0.8478 (121/143)
```

### JSON输出文件

结果会保存为 `blink_evaluation_results_gpt_4o_mini.json`，包含：

- 整体统计信息
- 各任务类型的准确率
- 失败的样本详情
- 推理时间统计

## 配置说明

### 模型参数

在 `straight_evaluation_gpt.py` 中可以调整以下参数：

- `temperature`: 设置为0.0以获得确定性输出
- `max_tokens`: 限制输出长度，提高效率
- `model`: 选择不同的GPT模型

### 错误处理

脚本包含完善的错误处理：

- 图像文件不存在
- 数据格式错误
- API调用失败
- 网络连接问题

## 注意事项

1. **API密钥**: 确保设置了正确的OpenAI API密钥
2. **网络连接**: 需要稳定的网络连接访问OpenAI API
3. **费用**: 使用GPT-4o-mini会产生API调用费用
4. **时间**: 完整评估可能需要较长时间，建议先用少量样本测试

## 故障排除

### 常见问题

1. **ImportError**: 确保项目路径正确设置
2. **FileNotFoundError**: 检查数据文件和图像路径
3. **APIError**: 检查API密钥和网络连接
4. **MemoryError**: 减少 `max_samples` 参数

### 调试模式

可以设置 `max_samples = 5` 来快速测试脚本是否正常工作。 