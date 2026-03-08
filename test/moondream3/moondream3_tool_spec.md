# Moondream3 Tool 说明 / Moondream3 Tool Spec

## 工具在做什么 / What the tool does

**Moondream3** 是视觉语言模型（VLM）工具，用于**根据图片回答问题**（Visual Question Answering, VQA）：输入一张图片路径和一个自然语言问题，返回模型基于图像内容给出的答案。

- **输入**：`image_path`（图片路径）+ `question`（问题文本）
- **输出**：成功时返回 `answer` 及 `result`；失败时返回 `success: false` 和 `error`。

---

## Parameter Schema Format 参数模式格式

在 OpenAI 函数调用格式中使用 JSON Schema：

Use JSON Schema in OpenAI function-calling format:

```json
{
    "type": "object",
    "properties": {
        "image_path": {
            "type": "string",
            "description": "Path to the input image file."
        },
        "question": {
            "type": "string",
            "description": "Question about the image to ask Moondream3."
        }
    },
    "required": ["image_path", "question"]
}
```

### 示例输入 JSON / Example input arguments

**单次调用示例：**

```json
{
    "image_path": "dataset/sample.jpg",
    "question": "What is in this image?"
}
```

**完整示例（多种用法）：** 见 `example_input.json`。

---

## Return Format 返回格式

`call()` 必须返回一个至少包含以下内容的字典：

`call()` must return a dictionary with at least:

| Field 字段 | Type 类型 | Required 必需的 | Description 描述 |
|------------|-----------|-----------------|------------------|
| success | bool 布尔值 | Yes 是 | Whether the call succeeded. 调用是否成功。 |
| result | Any 任意 | Recommended 推荐 | Main output (nested dict, contains answer). 主要输出（嵌套字典，含 answer）。 |
| error | str 字符串 | If success=False 当 success=False 时 | Error message. 错误信息。 |

### Example success 成功示例

```json
{
    "success": true,
    "result": {
        "answer": "Based on the image, the answer to \"What is in this image?\" would depend on the visual content. (Mock response.)",
        "success": true
    },
    "answer": "Based on the image, the answer to \"What is in this image?\" would depend on the visual content. (Mock response.)"
}
```

### Example failure 失败示例

```json
{
    "success": false,
    "error": "Image file not found: /path/to/missing.jpg"
}
```

```json
{
    "success": false,
    "error": "Question is required."
}
```
