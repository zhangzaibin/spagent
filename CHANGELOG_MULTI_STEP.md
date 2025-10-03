# 多步Workflow功能更新说明

## 概述

成功实现了SPAgent的多步workflow功能，允许模型进行多轮工具调用和推理，特别适合需要从多个角度分析3D结构的任务。

## 核心改动

### 1. `spagent/core/spagent.py`

**`solve_problem()` 方法增强**
- ✅ 新增 `max_iterations` 参数（默认值改为3）
- ✅ 实现多步迭代循环，允许模型多次调用工具
- ✅ 每次迭代后检查是否有 `<answer>` 标签，支持提前终止
- ✅ 追踪所有迭代的工具调用、结果和生成的图像
- ✅ 自动更新当前图像列表，将新生成的图像用于下次推理

**新增 `_create_continuation_prompt()` 方法**
- ✅ 为后续迭代创建专门的prompt
- ✅ 显示之前的工具执行结果和使用的角度
- ✅ 明确告知剩余迭代次数
- ✅ 指导模型选择不同的视角或提供最终答案

**返回值增强**
- ✅ 新增 `iterations` 字段：记录实际执行的迭代次数
- ✅ 工具结果添加 `_iterN` 后缀区分不同迭代
- ✅ 累积所有迭代生成的图像

### 2. `spagent/core/prompts.py`

**`create_system_prompt()` 更新**
- ✅ 新增"Multi-Step Workflow"章节
- ✅ 明确说明可以多轮调用工具
- ✅ 详细说明Pi3工具的角度参数使用方法
- ✅ 提供常用视角示例（正面、左侧、右侧、顶部等）

**`create_user_prompt()` 更新**
- ✅ 新增"Important Notes"部分
- ✅ 强调可以多次调用工具获取不同视角
- ✅ 指导何时提供最终答案

### 3. `spagent/tools/pi3_tool.py`

**工具描述更新**
- ✅ 强调支持"CUSTOM viewing angles"
- ✅ 详细说明 `azimuth_angle` 和 `elevation_angle` 参数
- ✅ 提供常用角度组合示例
- ✅ 明确说明可以多次调用以获得全面分析

**参数schema更新**
- ✅ 启用 `azimuth_angle` 参数（-180° 到 180°）
- ✅ 启用 `elevation_angle` 参数（-90° 到 90°）
- ✅ 两个参数都设为可选，默认值为0
- ✅ 添加详细的参数说明

**`call()` 方法更新**
- ✅ 将角度参数从注释状态恢复
- ✅ 默认值设为 0（正面视图）
- ✅ 参数验证和类型转换
- ✅ 将角度信息包含在返回结果中

## 工作流程示例

```
用户提问: "从多个角度分析这个3D物体"

迭代1:
  模型思考 → "我需要先看正面视图"
  工具调用 → pi3_tool(azimuth=0, elevation=0)
  生成图像 → front_view.png

迭代2:
  模型看到正面图 → "我需要看左侧"
  工具调用 → pi3_tool(azimuth=-45, elevation=0)
  生成图像 → left_view.png

迭代3:
  模型看到正面+左侧图 → "我需要看顶部"
  工具调用 → pi3_tool(azimuth=0, elevation=45)
  生成图像 → top_view.png

最终推理:
  模型综合所有视角 → 提供完整分析
```

## 关键特性

### 1. 向后兼容
- 虽然默认 `max_iterations=3`，但如果模型第一次就提供答案，仍会正常结束
- 所有现有代码无需修改

### 2. 智能终止
- 模型可以在任何时候通过输出 `<answer>` 标签提前终止
- 达到 `max_iterations` 后自动终止

### 3. 状态追踪
- 每次迭代的工具调用都被记录
- 所有生成的图像都被保存和追踪
- 工具结果包含详细的角度信息

### 4. 清晰的Prompt指导
- 系统prompt明确说明多步能力
- Continuation prompt显示已使用的角度
- 提供常用视角建议

## 使用示例

### 基础用法（仍然兼容）
```python
result = agent.solve_problem(
    image_path="test.jpg",
    question="分析这个物体"
)
# 如果模型第一次就能回答，只会执行1次迭代
```

### 多步分析
```python
result = agent.solve_problem(
    image_path="test.jpg",
    question="从多个角度分析这个3D结构",
    max_iterations=3
)

print(f"实际迭代: {result['iterations']}")
print(f"工具调用: {len(result['tool_calls'])}")
print(f"生成图像: {len(result['additional_images'])}")
```

### 指定具体角度
```python
question = """
请生成以下角度的3D渲染：
1. 正面 (azimuth=0, elevation=0)
2. 左侧45度 (azimuth=-45, elevation=0)
3. 俯视 (azimuth=0, elevation=45)

然后综合分析。
"""

result = agent.solve_problem(
    image_path="test.jpg",
    question=question,
    max_iterations=3
)
```

## 角度参数说明

### azimuth_angle (方位角)
- **范围**: -180° 到 180°
- **默认**: 0° (正面视图)
- **含义**: 水平旋转角度
  - 负值：向左旋转
  - 正值：向右旋转
  - 示例：-45°(左侧), 0°(正面), 45°(右侧), 180°(背面)

### elevation_angle (仰角)
- **范围**: -90° 到 90°
- **默认**: 0° (水平视图)
- **含义**: 垂直旋转角度
  - 负值：向下看
  - 正值：向上看
  - 示例：-45°(俯视), 0°(水平), 45°(仰视)

### 常用视角组合

| 视角 | azimuth | elevation | 说明 |
|------|---------|-----------|------|
| 正面 | 0 | 0 | 标准正面视图 |
| 左侧 | -45 | 0 | 从左侧观察 |
| 右侧 | 45 | 0 | 从右侧观察 |
| 背面 | 180 | 0 | 从背后观察 |
| 顶部 | 0 | 45 | 从上方观察 |
| 底部 | 0 | -45 | 从下方观察 |
| 左上 | -45 | 30 | 左上角视角 |
| 右上 | 45 | 30 | 右上角视角 |

## 配置建议

### 不同任务的迭代次数建议

- **简单问答**: `max_iterations=1` - 单次推理即可
- **3D分析**: `max_iterations=3` - 2-3个角度足够
- **复杂结构分析**: `max_iterations=5` - 需要多个角度全面理解
- **细节检查**: `max_iterations=4` - 可能需要回到某些角度细看

### 性能考虑

每次迭代都会：
1. 调用模型推理（耗时）
2. 执行工具（Pi3重建耗时）
3. 生成并保存图像

建议：
- 根据任务复杂度合理设置 `max_iterations`
- 简单任务不要设置过高
- 利用模型的提前终止能力

## 测试

运行测试脚本验证功能：

```bash
python test_multi_step.py
```

测试包括：
1. 多步workflow测试（3次迭代）
2. 向后兼容性测试（单次迭代）
3. Mock服务测试（无需真实Pi3服务器）

## 文档

- **完整说明**: `MULTI_STEP_WORKFLOW.md`
- **使用示例**: `examples/multi_step_workflow_example.py`
- **测试脚本**: `test_multi_step.py`

## 未来改进建议

1. **自适应迭代次数**: 根据任务复杂度自动调整
2. **角度推荐系统**: 根据已有视角自动推荐下一个最有用的角度
3. **并行工具调用**: 在一次迭代中同时请求多个角度
4. **视角质量评估**: 自动评估哪些视角最有帮助

## 注意事项

1. 确保Pi3服务器正常运行（如果使用真实服务）
2. 生成的图像会保存在 `outputs/` 目录
3. 每次迭代都会记录详细日志
4. 工具结果中的角度信息可用于调试

---

**更新时间**: 2025-10-03
**版本**: v1.0
**状态**: ✅ 已完成并测试

