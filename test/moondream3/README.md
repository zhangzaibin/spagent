# Moondream3 测试说明

## 输入 / 输出

- **输入**：`example_input.json`（同目录），包含 `image_path` 和 `question`。
- **输出**：使用 `--output result.json` 时，结果写入同目录的 `result.json`。

---

## 1. Mock 测试（不连真实服务）

不启动任何服务，用假数据跑通流程：

```powershell
cd F:\lab\spagent\test\moondream3
python test_moondream3_tool.py --json --output result.json
```

结果里的 `tool_call_result` 是固定文案，**不会**根据图片真正数人数。

---

## 2. 真实服务测试（会真正看图回答）

### 2.1 启动 Moondream Station（推荐）

在终端运行：

```powershell
moondream-station
```

启动后选择模型（如 Moondream 3 Preview），等出现 **Service: Running**、**API Endpoint: http://localhost:2020/v1** 后再做测试。

### 2.2 用真实服务跑测试

在 `test/moondream3` 目录下执行：

```powershell
cd F:\lab\spagent\test\moondream3
python test_moondream3_tool.py --json --output result.json --no-mock
```

默认会连 **http://localhost:2020/v1**（Moondream Station）。若 Station 用了别的端口或地址，可指定：

```powershell
python test_moondream3_tool.py --json --output result.json --no-mock --server_url http://localhost:2020/v1
```

- `--no-mock`：使用真实服务。
- `--server_url`：默认 `http://localhost:2020/v1`（Moondream Station）。

此时会用 `example_input.json` 里的图片和问题请求真实模型，**会真正数人数**，结果在 `result.json` 的 `tool_call_result` 里。

### 2.3 使用项目自带的 md_server（可选）

若不用 Station，可用项目里的 Moondream 服务（端口 20024）：

```powershell
cd F:\lab\spagent
set MOONDREAM_API_KEY=你的API密钥
python spagent/external_experts/moondream/md_server.py --port 20024
```

测试时指定该地址：

```powershell
python test_moondream3_tool.py --json --output result.json --no-mock --server_url http://localhost:20024
```

### 2.3 确保图片存在

`example_input.json` 里的 `image_path` 必须是本机存在的路径，例如：

```json
{
    "image_path": "F:\\photo\\2023.1.20\\1.jpg",
    "question": "count the number of people"
}
```

若路径错误或文件不存在，`tool_call_result` 里会返回 `success: false` 和 `error: "Image file not found: ..."`。

---

## 参数汇总

| 参数 | 说明 |
|------|------|
| `--json` | 以 JSON 形式输出测试结果 |
| `--output result.json` | 将 JSON 写入当前目录的 result.json |
| `--no-mock` | 使用真实 Moondream/Moondream3 服务 |
| `--server_url URL` | 真实服务地址，默认 `http://localhost:20024` |
| `--image 路径` | 指定图片路径做单张图测试（可选） |
