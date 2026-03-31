# OrientAnythingTool Test Guide

This document explains how to run and test `OrientAnythingTool` inside SPAgent, including required dependencies, model assets, example commands, and expected output.

## Overview

`OrientAnythingTool` estimates the 3D orientation of an object from a single RGB image.

Given an input image, the tool predicts:

- azimuth
- polar
- rotation
- confidence

The tool is integrated into **SPAgent**, but it depends on an external local clone of the official **Orient-Anything** repository for model code and assets.

## Directory Layout

A recommended directory structure:

```

├── spagent/
│   ├── spagent/
│   │   ├── tools/
│   │   └── external_experts/orient_anything/
│   └── test/orient_anything/
│       ├── README.md
│       ├── assets/
│       │   └── bus.png
│       └── test_orient_anything_tool.py
│
└── Orient-Anything/
    ├── app.py
    ├── inference.py
    ├── utils.py
    ├── vision_tower.py
    ├── paths.py
    ├── assets/
    │   ├── axis.png
    │   └── axis.obj
    └── croplargeEX2/
        └── dino_weight.pt
```

## Dependencies

### 1. SPAgent

This tool runs inside the SPAgent project.

### 2. Clone the Orient-Anything repository

The tool imports modules directly from the official repository, so the repository must exist locally.

Example (clone outside repo; use a path that fits your setup, e.g. sibling of spagent):

```
# From spagent project root:
cd ..
git clone https://github.com/Viglong/Orient-Anything.git Orient-Anything
```

Or with an absolute path: `git clone https://github.com/Viglong/Orient-Anything.git /path/to/Orient-Anything`

### 3. Install Python dependencies

Install the Orient-Anything dependencies inside the same Python environment used by SPAgent.

```
# If you cloned to ../Orient-Anything from spagent root:
pip install -r ../Orient-Anything/requirements.txt
```

The runtime also relies on common packages such as:

- torch
- transformers
- huggingface_hub
- Pillow

### 4. Orient-Anything checkpoint

For the `large` model, the following file must exist under your Orient-Anything clone:

```
<Orient-Anything>/croplargeEX2/dino_weight.pt
```

e.g. `../Orient-Anything/croplargeEX2/dino_weight.pt` if clone is sibling of spagent.

Other optional model sizes:

```
cropsmallEX2/dino_weight.pt
cropbaseEX2/dino_weight.pt
```

### 5. Hugging Face cache for DINOv2

The backbone model used by the repo is:

```
facebook/dinov2-large
```

Ensure the Hugging Face cache contains the needed files such as:

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`

### 6. Background removal model

If background removal is enabled (`remove_background=True`), the tool will download **U2Net** automatically.

Typical location:

```
/data/sjq/.u2net/u2net.onnx
```

This download only happens once.

### 7. Required assets

These files must exist in the Orient-Anything repository:

```
/data/sjq/Orient-Anything/assets/axis.png
/data/sjq/Orient-Anything/assets/axis.obj
```

## Why an external Orient-Anything repo is needed

`OrientAnythingTool` does not reimplement the full model. It imports modules directly from the official `Orient-Anything` repository, including:

- `paths.py`
- `vision_tower.py`
- `inference.py`
- `utils.py`

Because of this, the external repository must be cloned locally and passed in through `repo_root`.

## Test Image

Place a test image in:

```
/data/sjq/spagent/test/orient_anything/assets/bus.png
```

The image should ideally contain a single prominent object.

## Environment Variables Used

The test reads the following environment variables:

| Variable                        | Description                                  |
| ------------------------------- | -------------------------------------------- |
| `RUN_REAL_ORIENT_ANYTHING_TEST` | Enable the real test                         |
| `ORIENT_ANYTHING_DEVICE`        | Device for inference (`cuda:0` or `cpu`)     |
| `ORIENT_ANYTHING_REPO_ROOT`     | Path to the local Orient-Anything repository |
| `ORIENT_ANYTHING_TEST_IMAGE`    | Path to the test image                       |
| `ORIENT_ANYTHING_MODEL_SIZE`    | Model size, default is `large`               |

## Running the Test

From the **SPAgent root directory**, run:

```
RUN_REAL_ORIENT_ANYTHING_TEST=1 \
ORIENT_ANYTHING_DEVICE=cuda:0 \
ORIENT_ANYTHING_REPO_ROOT=/data/sjq/Orient-Anything \
ORIENT_ANYTHING_TEST_IMAGE=/data/sjq/spagent/test/orient_anything/assets/bus.png \
python -m pytest test/orient_anything/test_orient_anything_tool.py -s
```

This command:

1. Enables the real test.
2. Sets the inference device.
3. Points to the local Orient-Anything repository.
4. Specifies the test image.
5. Runs the pytest file.

## First Run Behavior

On the first run, some assets may be downloaded automatically:

- Hugging Face model configs
- DINOv2 weights
- U2Net background removal model

After the first run, subsequent runs should be much faster.

## Example Successful Output

Example terminal output:

```
sjq@4029GP-TRT:~/spagent$ RUN_REAL_ORIENT_ANYTHING_TEST=1 \
> ORIENT_ANYTHING_DEVICE=cuda:0 \
> ORIENT_ANYTHING_REPO_ROOT=/data/sjq/Orient-Anything \
> ORIENT_ANYTHING_TEST_IMAGE=/data/sjq/spagent/test/orient_anything/assets/bus.png \
> python -m pytest test/orient_anything/test_orient_anything_tool.py -s
================================================================= test session starts ==================================================================
platform linux -- Python 3.10.19, pytest-9.0.2, pluggy-1.6.0
rootdir: /data/sjq/spagent
plugins: anyio-4.12.1
collected 1 item

config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 549/549 [00:00<00:00, 2.16MB/s]
model.safetensors: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 1.22G/1.22G [22:44<00:00, 892kB/s]
Downloading data from 'https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx' to file '/data/sjq/.u2net/u2net.onnx'.
100%|████████████████████████████████████████| 176M/176M [00:00<00:00, 627GB/s]

=== OrientAnythingTool result ===
{'success': True, 'result': {'azimuth': 36.0, 'polar': -1.0, 'rotation': -90.0, 'confidence': 0.9999799728393555, 'model_size': 'large', 'use_tta': False, 'remove_background': True}, 'output_path': None, 'summary': 'Estimated orientation: azimuth=36.00, polar=-1.00, rotation=-90.00, confidence=1.000.'}
.

=================================================================== warnings summary ===================================================================
test/orient_anything/test_orient_anything_tool.py::test_orient_anything_tool
  /data/sjq/anaconda3/envs/spagent/lib/python3.10/site-packages/huggingface_hub/file_download.py:797: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================================================== 1 passed, 1 warning in 1612.26s (0:26:52) =======================================================
```

## Output Format

A successful tool call returns a dictionary like this:

```
{
    "success": True,
    "result": {
        "azimuth": 36.0,
        "polar": -1.0,
        "rotation": -90.0,
        "confidence": 0.9999799728393555,
        "model_size": "large",
        "use_tta": False,
        "remove_background": True
    },
    "output_path": None,
    "summary": "Estimated orientation: azimuth=36.00, polar=-1.00, rotation=-90.00, confidence=1.000."
}
```

### Field descriptions

| Field         | Meaning                                  |
| ------------- | ---------------------------------------- |
| `success`     | Whether the tool call succeeded          |
| `result`      | Main prediction output                   |
| `azimuth`     | Predicted azimuth angle                  |
| `polar`       | Predicted polar angle                    |
| `rotation`    | Predicted in-plane rotation angle        |
| `confidence`  | Prediction confidence                    |
| `output_path` | Path to an output file, currently `None` |
| `summary`     | Human-readable result summary            |

## Notes

### Single-object images work best

The tool performs best when the image contains one clear main object.

### First run may take time

Model downloads and initialization may take several minutes on the first run.

### Warning message

You may see a warning such as:

```
FutureWarning: resume_download is deprecated
```

This warning does **not** affect correctness. The test can still pass successfully.

### Current output is numeric only

The current implementation returns numeric orientation estimates and a summary string. It does not currently save a visualization image, so `output_path` is `None`.

## Minimal Checklist

Before running the test, verify the following:

- SPAgent environment is activated
- Orient-Anything repo exists
- `axis.png` and `axis.obj` exist
- `dino_weight.pt` exists
- local Hugging Face cache for `facebook/dinov2-large` exists
- test image exists

If all conditions are met, the test should run successfully.