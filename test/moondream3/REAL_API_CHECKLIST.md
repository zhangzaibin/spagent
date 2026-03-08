# Moondream3 真实 API 测试 — 需要准备什么

要**真实调用模型**（不是 mock），需要满足下面几项。

---

## 1. 已安装的依赖

- **Python 3**（项目已有）
- **requests**：`pip install requests`
- （可选）项目根目录的 `.venv` 已激活：`F:\lab\spagent\.venv\Scripts\activate`

---

## 2. Moondream Station 已启动

- 在**另一个终端**里运行：
  ```powershell
  moondream-station
  ```
- 等出现 **Service: Running**、**API Endpoint: http://localhost:2020/v1**
- 若未启动，测试会报：无法连接 `http://localhost:2020/v1` 或超时

---

## 3. 输入图片存在

- `example_input.json` 里的 `image_path` 必须是**本机真实路径**
- 例如：`"image_path": "F:\\photo\\2023.1.20\\1.jpg"`
- 若该文件不存在，会报：`Image file not found: ...`

---

## 4. 运行测试的方式

在 **`F:\lab\spagent\test\moondream3`** 目录下：

```powershell
# 诊断：检查缺什么（Station、图片、依赖）
python test_real_api.py

# 真实调用并输出 JSON 结果
python test_moondream3_tool.py --json --output result.json --no-mock
```

---

## 5. 若返回 success 但 answer 为空

- 说明请求已发到 Station 且 HTTP 成功，但响应里没有拿到 `answer`
- 可能原因：
  - Station 返回的 JSON 里答案字段不叫 `answer`（需对照官方文档或抓包看真实字段名）
  - 模型返回了空字符串
- 可打开 `test_real_api.py` 看 `tool.call()` 的完整返回，或抓包看 `http://localhost:2020/v1/query` 的响应 body

---

## 快速自检清单

| 项           | 检查方式 |
|--------------|----------|
| requests     | `python -c "import requests; print('OK')"` |
| Station 在跑 | 另开终端执行 `moondream-station`，看到 API Endpoint |
| 图片存在     | 打开 `example_input.json` 里的路径，确认文件存在 |
| 真实调用     | `python test_moondream3_tool.py --json --no-mock` |
