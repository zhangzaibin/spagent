# Depth Anything 3 — 测试目录（仅测试）

本目录**仅用于对 Depth Anything 3 Tool 做单元/集成测试**。  
**真正被调用、需要注册到 SPAgent 的 Tool 定义在** `spagent/tools/depth_anything3_tool.py`，使用与集成请以该文件为准，并遵循 [docs/ADDING_NEW_TOOLS.md](../../docs/ADDING_NEW_TOOLS.md)。

---

## 1. Tool 定义与调用位置（真实部署）

| 用途           | 位置 |
|----------------|------|
| **Tool 定义**  | `spagent/tools/depth_anything3_tool.py` |
| **导出**       | `spagent/tools/__init__.py`（`DepthAnything3Tool`） |
| **Server**     | `spagent/external_experts/depth_anything3/depth_anything3_server.py`（与 Pi3X 一致，需单独启动） |
| **Client/Mock**| `spagent/external_experts/depth_anything3/`（HTTP client + mock） |

与 Pi3X / PR #124 一致：**模型在独立 Server 进程中运行**，Tool 仅通过 HTTP 调用 Server。需先启动 Server，再在 agent 中注册 Tool（传入 `server_url`）。

在 agent 中注册与调用示例：

```python
from spagent.tools import DepthAnything3Tool

# 先启动 Server（另起终端）:
# python -m spagent.external_experts.depth_anything3.depth_anything3_server --checkpoint_path depth-anything/DA3MONO-LARGE --port 20032

tool = DepthAnything3Tool(
    use_mock=False,
    server_url="http://localhost:20032",
    save_dir="test/depth_anything3/outputs",
)
agent.add_tool(tool)
```

---

## 2. 本目录内容（仅测试）

- **test_depth_anything3_tool.py**：对 `DepthAnything3Tool` 的 pytest 用例（mock + 可选 real）。
- **outputs/**：测试生成的深度图输出（相对路径 `test/depth_anything3/outputs`）。
- **INSTALL.md**：安装 Depth Anything 3 依赖的步骤（仅用于跑“真实模型”测试时）。

---

## 3. 如何跑测试

在 **spagent 项目根目录** 下执行（使用相对路径）：

```bash
# Mock 测试（无需安装 Depth Anything 3）
pytest test/depth_anything3/test_depth_anything3_tool.py -v

# 或直接运行脚本（默认 mock）
python test/depth_anything3/test_depth_anything3_tool.py
```

真实模型测试（需**先启动 Depth Anything 3 Server**，与 Pi3X 一致）：

```bash
# 终端 1：启动 Server（需先按 INSTALL.md 安装 depth-anything-3）
python -m spagent.external_experts.depth_anything3.depth_anything3_server --checkpoint_path depth-anything/DA3MONO-LARGE --port 20032

# 终端 2：跑测试（可选设置 DEPTH_ANYTHING3_SERVER_URL，默认 http://localhost:20032）
python test/depth_anything3/test_depth_anything3_tool.py --real
# 或
pytest test/depth_anything3/test_depth_anything3_tool.py::test_depth_anything3_tool_real -v -s
```

测试输入图使用相对路径：`assets/example.png`；输出目录：`test/depth_anything3/outputs`。

---

## 4. 规范与参考

- Tool 接口、返回格式、注册方式等以 ** [docs/ADDING_NEW_TOOLS.md](../../docs/ADDING_NEW_TOOLS.md) ** 为准。
- 本目录不定义或暴露任何供 agent 调用的入口；所有真实调用均在 `spagent/tools/depth_anything3_tool.py` 中完成。
