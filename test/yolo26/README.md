# YOLO26 Tool Real Inference Guide

This document explains how to run `YOLO26Tool` with real model weights locally, and how to execute the real integration test (not mocked).

## 1. Prerequisites

- Run commands from the repository root: `spagent/`
- Your Python environment has required packages installed (at least `pytest`, `ultralytics`, and `opencv-python`)
- YOLO26 weights are available

By default, the test reads weights from this relative path:

- `weights/yolo26n.pt`

If your weights are stored elsewhere, override the path with an environment variable (see below).

## 2. Run the Real Integration Test

Test file: `test/yolo26/test_yolo26_tool_real.py`

This test is disabled by default and must be explicitly enabled:

```bash
RUN_REAL_YOLO26_TEST=1 python -m pytest -q test/yolo26/test_yolo26_tool_real.py
```

Optional environment variables:

- `YOLO26_MODEL_PATH`: custom weights path (relative path example: `weights/yolo26n.pt`)
- `YOLO26_DEVICE`: inference device, default is `cpu`, can be `cuda:0`

Example:

```bash
RUN_REAL_YOLO26_TEST=1 \
YOLO26_MODEL_PATH=weights/yolo26n.pt \
YOLO26_DEVICE=cpu \
python -m pytest -q test/yolo26/test_yolo26_tool_real.py
```

Test input and output:

- Input image: `test/yolo26/assets/bus.png`
- Annotated output directory: `test/yolo26/outputs/yolo26/`

## 3. Call YOLO26Tool Directly (Without pytest)

You can run real inference directly with this script:

```bash
python - <<'PY'
from spagent.tools.yolo26_tool import YOLO26Tool

tool = YOLO26Tool(
    model_path="weights/yolo26n.pt",
    device="cpu",
    conf=0.25,
    iou=0.45,
    max_det=100,
    save_annotated=True,
    output_dir="test/yolo26/outputs/yolo26",
)

result = tool.call(
    image_path="test/yolo26/assets/bus.png",
    conf=0.25,
    save_annotated=True,
)

print(result)
PY
```

On success, the returned object includes:

- `success: True`
- `result.num_detections`
- `result.detections`
- `output_path` (when `save_annotated=True`)

## 4. Troubleshooting

- Weights not found: verify `weights/yolo26n.pt` exists, or set `YOLO26_MODEL_PATH`
- No detections: check image path, `conf` threshold, and model weights
- GPU unavailable: set `YOLO26_DEVICE=cpu`
