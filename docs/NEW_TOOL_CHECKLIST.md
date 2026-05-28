# New Tool Contributor Checklist

Practical file-by-file checklist for adding a new external expert tool to SPAgent.
For architecture concepts see [ADDING_NEW_TOOLS.md](ADDING_NEW_TOOLS.md).

---

## 1. Decide deployment pattern

| Pattern | When to use | Examples |
|---------|-------------|---------|
| **Local only** | Source can be vendored into repo; simple setup | CountGD, WildDet3D (initial) |
| **Local + Server/Client** | Heavy model; want to share GPU across agents | CountGD, WildDet3D |
| **API only** | Cloud API, no local weights | Veo, Sora |

---

## 2. Files to create

### `spagent/external_experts/<ToolName>/`

| File | Purpose |
|------|---------|
| `__init__.py` | Export `LocalClient` (and `Client` if server mode) |
| `<toolname>_local.py` | Lazy-loading local inference class with a `run()` / `detect()` / `count()` method |
| `<toolname>_server.py` | Flask server wrapping the local client (port assigned below) |
| `<toolname>_client.py` | HTTP client: encode image as base64, POST to server, save annotated image locally |

**Port assignment** (pick the next unused one):

| Port | Tool |
|------|------|
| 20019 | Depth AnythingV2 |
| 20020 | SAM2 |
| 20021 | Pi3 |
| 20022 | GroundingDINO / VGGT / MapAnything |
| 20024 | Moondream |
| 20025 | Molmo2 |
| 20026 | CountGD |
| 20027 | WildDet3D |
| 20031 | Pi3X |
| 20034 | OrientAnythingV2 / VACE |
| 20035 | OneFormer |
| **next** | **Your tool** |

**`_local.py` pattern:**
```python
class MyToolLocalClient:
    def __init__(self, checkpoint=None, device="cuda"):
        self.checkpoint = checkpoint or os.environ.get("MYTOOL_CHECKPOINT")
        if not self.checkpoint:
            raise EnvironmentError("Set MYTOOL_CHECKPOINT or pass checkpoint=")
        self._model = None

    def _ensure_model_loaded(self): ...  # lazy load

    def run(self, image_path, **kwargs) -> Dict:
        ...
        return {"success": True, "output_path": ..., "description": ...}
```

**`_server.py` pattern:**
```python
_HERE = Path(__file__).resolve().parent
_SPAGENT = _HERE.parents[1]
if str(_SPAGENT) not in sys.path:
    sys.path.insert(0, str(_SPAGENT))

from external_experts.MyTool.mytool_local import MyToolLocalClient
from flask import Flask, jsonify, request

app = Flask(__name__)
_client = None  # set at startup

@app.route("/health", methods=["GET"])
def health_check(): ...

@app.route("/infer", methods=["POST"])
def infer():
    # decode base64 image → tempfile → _client.run() → encode result image → jsonify
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", ...)
    parser.add_argument("--port", type=int, default=<PORT>)
    parser.add_argument("--device", default="cuda")
    ...
    _client = MyToolLocalClient(...)
    _client._ensure_model_loaded()
    app.run(host="0.0.0.0", port=args.port, debug=False)
```

**`_client.py` pattern:**
```python
class MyToolClient:
    def __init__(self, server_url, timeout=60):
        self.server_url = server_url.rstrip("/")

    def run(self, image_path, **kwargs) -> Dict:
        # cv2.imread → base64 → POST /infer → save annotated image → return result
        ...
```

---

### `spagent/tools/<toolname>_tool.py`

```python
class MyTool(Tool):
    def __init__(self, checkpoint=None, device="cuda", server_url=None, use_mock=False):
        super().__init__(name="mytool", description="...")
        self._server_url = server_url
        self._client = None
        self._client_kwargs = dict(checkpoint=checkpoint, device=device)

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockMyToolClient()
        elif self._server_url:
            from external_experts.MyTool.mytool_client import MyToolClient
            self._client = MyToolClient(self._server_url)
        else:
            from external_experts.MyTool.mytool_local import MyToolLocalClient
            self._client = MyToolLocalClient(**self._client_kwargs)

    def call(self, image_path, ...) -> Dict:
        ...
        raw = self._client.run(image_path=image_path, ...)
        if raw.get("success"):
            raw["result"] = {...}  # required by ADDING_NEW_TOOLS.md
        return raw

class _MockMyToolClient:
    def run(self, image_path, **kwargs) -> Dict:
        return {"success": True, "result": {...}, "output_path": image_path, "description": "[mock] ..."}
```

---

## 3. Files to modify

### `spagent/tools/__init__.py`
```python
from .mytool_tool import MyTool          # add import
__all__ = [..., 'MyTool']               # add to __all__
```

### `examples/evaluation/evaluate_img.py`
```python
from spagent.tools import MyTool         # add import

TOOL_CONFIGS = {
    ...
    "mytool": [MyTool(device="cuda")],   # add entry
}
```

### `test/test_tool.py`

1. Add `test_mytool()` function (follow the pattern of `test_countgd` or `test_wilddet3d`)
2. Add `"mytool"` to `choices=` in `--tool` argument
3. Add `"mytool"` to `image_required_tools` set (if image is required)
4. Add `elif args.tool == "mytool":` dispatch block
5. Add usage examples to the module docstring at the top

### `docs/Tool/TOOL_USING.md`

- [ ] **Directory tree** — add `├── MyTool/ # description (local or server port XXXXX)`
- [ ] **Tool overview table** — add row with Tool Name, Class, Function, Deployment, Parameters
- [ ] **Detailed section** (`### N. MyTool - ...`) — setup, features, local usage, server usage, test commands, returns, resources

### `readme.md`

- [ ] **Project Structure table** — add to Tools cell and External Experts cell
- [ ] **Expert Tools table** — add row
- [ ] **Setup section** — add setup instructions (env vars, checkpoint download, pip deps)

### `.gitignore`

Add any large files or directories specific to this tool (checkpoints, generated outputs, etc.)

---

## 4. Branch and PR workflow

```bash
# 1. Branch from main
git checkout main && git pull origin main
git checkout -b feat/<toolname>

# 2. Implement, commit incrementally
git add <files> && git commit -m "feat: add <ToolName> ..."

# 3. Push and open PR
git push origin feat/<toolname>
```

**PR comment should include:**
- Summary (what it does, deployment modes)
- What's included (new files, modified files)
- Setup (deps, checkpoint download command)
- Usage examples (local, server, mock)
- Test commands

---

## 5. Quick compatibility checklist

- [ ] `torch.load(..., weights_only=False)` if checkpoint uses non-tensor globals (PyTorch ≥ 2.6)
- [ ] Custom CUDA ops: add `_OP_AVAILABLE` flag and fall back to pure-PyTorch path
- [ ] `result` field present in `call()` return dict (required by `ADDING_NEW_TOOLS.md`)
- [ ] Mock client returns same keys as real client
- [ ] Server path setup uses `Path(__file__).resolve().parents[N]` so it works from any directory
- [ ] No binary files committed (`.zip`, `.pt`, `.pth`, model weights)
