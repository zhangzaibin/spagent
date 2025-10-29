# 数据采集功能使用说明（已修复）

## 问题原因

之前的版本中 `auto_save=False` 导致数据虽然被采集但没有保存到磁盘。

## ✅ 已修复

现在 `auto_save=True`，数据会自动保存。

## 使用方法

### 方法 1：使用修改后的 evaluate_img.py

```bash
# 启用数据采集
python examples/evaluation/evaluate_img.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --max_iterations 3 \
    --enable_data_collection
```

**关键参数**：
- `--enable_data_collection`：启用数据采集（必须）
- `--data_output_dir`：指定输出目录（可选，默认自动生成）

### 方法 2：测试脚本

```bash
# 运行测试脚本验证数据采集功能
python test_data_collection.py
```

## 验证数据是否保存成功

运行完成后，检查以下内容：

### 1. 查看统计信息
```bash
cat training_data/*/statistics.json
```

应该看到：
```json
{
  "total_sessions": 3,
  "successful_sessions": 3,
  "total_samples": 11,
  "success_rate": 1.0
}
```

### 2. 检查 sessions 目录
```bash
ls -la training_data/*/sessions/
```

应该看到多个 session 目录（不是空的）：
```
drwxr-xr-x session_20250124_155655_abc123/
drwxr-xr-x session_20250124_155700_def456/
drwxr-xr-x session_20250124_155705_ghi789/
```

### 3. 查看单个会话
```bash
ls -la training_data/*/sessions/session_*/
```

应该看到：
```
session_metadata.json
samples/
  sample_1.json
  sample_2.json
  sample_3.json
images/
  original_image.jpg
  pi3_result.png
```

### 4. 检查导出文件
```bash
# 检查行数
wc -l training_data/*/train.jsonl

# 查看内容
head -n 1 training_data/*/train.jsonl | jq
```

## 如果还是没有数据

### 调试步骤

1. **检查是否有 answer 标签**

数据采集只保存成功的会话（有 `<answer>` 标签）。检查模型输出：

```python
# 在评估脚本中添加打印
print(f"Response: {result['answer']}")
print(f"Has answer tag: {'<answer>' in result['answer']}")
```

2. **检查日志输出**

运行时应该看到：
```
✓ Data collection enabled: training_data/...
...
Data Collection Summary
Total sessions:      3
Successful sessions: 3
Total samples:       11
✓ Exported to training_data/.../train.jsonl
```

3. **手动测试**

使用 `test_data_collection.py` 进行单独测试：

```bash
# 修改脚本中的 test_image 路径为你的测试图片
python test_data_collection.py
```

4. **查看完整错误信息**

如果有错误，会在终端显示完整的 traceback。

## 常见问题

### Q1: sessions 目录是空的
**原因**：所有会话都失败了（没有 `<answer>` 标签）

**解决**：
- 检查模型是否正确返回答案
- 查看 statistics.json 中的 `failed_sessions`
- 临时设置 `auto_save=True` 以保存所有会话

### Q2: 导出文件是空的
**原因**：sessions 目录中没有成功的会话

**解决**：
- 先确认 sessions 目录有数据
- 检查 session_metadata.json 中的 `success` 字段

### Q3: 提示权限错误
**解决**：
```bash
chmod -R 755 training_data/
```

## 数据格式说明

### session_metadata.json
```json
{
  "session_id": "session_xxx",
  "question": "原始问题",
  "success": true,
  "final_answer": "最终答案",
  "num_iterations": 3,
  "samples": [
    {
      "sample_id": "session_xxx_iter_1",
      "iteration": 1,
      "images": ["image.jpg"],
      "prompt": "完整的prompt...",
      "response": "模型回复...",
      "context": {...}
    }
  ]
}
```

### train.jsonl（每行一个样本）
```jsonl
{"sample_id":"session_xxx_iter_1","iteration":1,"images":["img.jpg"],"prompt":"...","response":"..."}
{"sample_id":"session_xxx_iter_2","iteration":2,"images":["img.jpg","depth.jpg"],"prompt":"...","response":"..."}
```

### train_sharegpt.json
```json
[
  {
    "id": "session_xxx_iter_1",
    "images": ["img1.jpg"],
    "conversations": [
      {"from": "human", "value": "问题"},
      {"from": "gpt", "value": "回答"}
    ]
  }
]
```

## 完整示例

```bash
# 1. 运行评估并采集数据
python examples/evaluation/evaluate_img.py \
    --data_path dataset/cvbench_data.jsonl \
    --max_samples 10 \
    --model gpt-4o \
    --max_iterations 3 \
    --enable_data_collection

# 2. 检查结果
ls -la training_data/depth_detection_segmentation_*/

# 3. 查看统计
cat training_data/*/statistics.json

# 4. 查看会话数量
ls training_data/*/sessions/ | wc -l

# 5. 检查导出数据
wc -l training_data/*/train.jsonl
head -n 1 training_data/*/train.jsonl | jq

# 6. 查看第一个会话详情
cat training_data/*/sessions/session_*/session_metadata.json | jq
```

## 修改内容总结

### 已修改的文件：

1. ✅ `examples/evaluation/spagent_evaluation.py`
   - 添加 `data_collector` 参数
   - 在函数结束时导出数据

2. ✅ `examples/evaluation/evaluate_img.py`
   - 导入 `DataCollector`
   - 添加命令行参数
   - 创建并传递 `DataCollector`

3. ✅ `examples/evaluation/evaluate_img_with_data_collection.py`
   - 修复 `auto_save=False` → `auto_save=True`

### 新增文件：

4. ✅ `test_data_collection.py`
   - 快速测试脚本

## 联系方式

如果还有问题，请提供：
1. 运行命令
2. 终端完整输出
3. statistics.json 内容
4. sessions 目录的 ls 输出


