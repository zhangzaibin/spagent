# External Experts Module

> **中文版本**: [中文文档](TOOL_USING_ZH.md) | **English Version**: This document

The External Experts module contains specialized models for spatial intelligence tasks, including depth estimation, object detection, segmentation, 3D reconstruction, video generation, and more. All tools adopt a server/client architecture or direct API call pattern, supporting independent deployment and invocation.

## 📁 Module Structure

```
external_experts/
├── __init__.py                     # Module initialization
├── checkpoints/                    # All model weight files
│   └──depth_anything
│   └──grounding_dino
│   └──pi3
│   └──pi3x
│   └──sam2
│   └──vggt
├── GroundingDINO/                  # Open-vocabulary object detection
├── SAM2/                          # Image and video segmentation
├── Depth_AnythingV2/              # Depth estimation
├── Pi3/                           # 3D reconstruction (Pi3 & Pi3X)
├── VGGT/                          # Multi-view 3D reconstruction & camera pose estimation
├── mapanything/                   # Dense 3D reconstruction via depth estimation
├── moondream/                     # Vision language model
├── Veo/                           # Google Veo video generation (API-based)
├── Sora/                          # OpenAI Sora video generation (API-based)
└── supervision/                   # YOLO object detection and annotation tools
```

## 🛠️ Tool Overview

| Tool Name | Tool Class | Function | Main Purpose | Deployment | Main Parameters |
|---------|------------|----------|--------------|------------|----------------|
| **Depth AnythingV2** | `DepthEstimationTool` | Depth Estimation | Monocular depth estimation, analyze 3D depth relationships in images | Server (port 20019) | `image_path` |
| **SAM2** | `SegmentationTool` | Image/Video Segmentation | High-precision segmentation tasks, precisely segment objects in images | Server (port 20020) | `image_path`, `point_coords`(optional), `point_labels`(optional), `box`(optional) |
| **GroundingDINO** | `ObjectDetectionTool` | Open-vocabulary Object Detection | Detect arbitrary objects based on text descriptions | Server (port 20022) | `image_path`, `text_prompt`, `box_threshold`, `text_threshold` |
| **Moondream** | `MoondreamTool` | Vision Language Model | Image understanding and Q&A, answer natural language questions based on image content | Server (port 20024) | `image_path`, `task`, `object_name` |
| **Pi3** | `Pi3Tool` | 3D Reconstruction | Generate 3D point clouds and multi-view rendered images from images | Server (port 20030) | `image_path`, `azimuth_angle`, `elevation_angle` |
| **Pi3X** | `Pi3XTool` | 3D Reconstruction (Enhanced) | Upgraded Pi3 with smoother point clouds, metric scale, and optional multimodal conditioning | Server (port 20031) | `image_path`, `azimuth_angle`, `elevation_angle` |
| **VGGT** | `VGGTTool` | Multi-view 3D Reconstruction & Camera Pose Estimation | Reconstruct 3D point clouds and estimate camera poses from multiple images or video frames | 20032 | `image_paths`, `azimuth_angle`, `elevation_angle`, `rotation_reference_camera`, `camera_view` |
| **MapAnything** | `MapAnythingTool` | Dense 3D Reconstruction via Depth Estimation | Reconstruct dense 3D point clouds from multiple images using depth maps and camera poses | 20033 | `image_paths`, `azimuth_angle`, `elevation_angle`, `conf_percentile`, `apply_mask` |
| **Supervision** | `SupervisionTool` | Object Detection Annotation | YOLO models and visualization tools, general object detection and segmentation | Local | `image_path`, `task` ("image_det" or "image_seg") |
| **YOLO-E** | `YOLOETool` | YOLO-E Detection | High-precision detection with custom classes | Local | `image_path`, `task`, `class_names` |
| **Veo** | `VeoTool` | Video Generation | Text-to-video and image-to-video via Google Veo (Gemini API) | API (no server) | `prompt`, `image_path`(optional), `duration`, `aspect_ratio` |
| **Sora** | `SoraTool` | Video Generation | Text-to-video and image-to-video via OpenAI Sora | API (no server) | `prompt`, `image_path`(optional), `duration`, `resolution`, `aspect_ratio` |

**Usage Examples**:
- For detailed usage examples, please refer to: [Advanced Examples](../Examples/ADVANCED_EXAMPLES.md)
- For quick start guide, please refer to: [Quick Start Guide](../../readme.md#-quick-start)
- For adding custom tools: [How to Add New Tools](../ADDING_NEW_TOOLS.md)

---

## 📋 Detailed Tool Introduction

### 1. Depth AnythingV2 - Depth Estimation

**Function**: Monocular image depth estimation

**Features**:
- Three model sizes available
- High-quality depth map generation
- Support for multiple input formats

**File Structure**:
```
Depth_AnythingV2/
├── depth_server.py
├── depth_client.py
├── mock_depth_service.py
└── depth_anything_v2/
```

**Model Specifications**:
| Model | Backbone | Parameters | File Size | Inference Speed | Accuracy |
|------|----------|------------|-----------|-----------------|----------|
| Small | ViT-S | ~25M | ~100MB | Fast | Good |
| Base | ViT-B | ~97M | ~390MB | Medium | High |
| Large | ViT-L | ~335M | ~1.3GB | Slow | Very High |

**Weight Download**:
```bash
cd checkpoints/
# Small model (~25MB, fastest)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
# Base model (~100MB, balanced) - Recommended
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth
# Large model (~350MB, highest quality)
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth
```

**Resources**:
- [Official Repository](https://github.com/DepthAnything/Depth-Anything-V2)
- [Paper](https://arxiv.org/abs/2406.09414)

---

### 2. SAM2 - Image and Video Segmentation

**Function**: High-precision image and video segmentation model

**Features**:
- Support for image and video segmentation
- Multiple model sizes available
- High-precision segmentation results

**File Structure**:
```
SAM2/
├── sam2_server.py
└── sam2_client.py
```

**Model Specifications**:
| Model | Parameters | File Size | Purpose |
|------|------------|-----------|---------|
| Hiera Large | ~224M | ~900MB | High precision |
| Hiera Base+ | ~80M | ~320MB | Balanced performance |
| Hiera Small | ~46M | ~185MB | Fast inference |

**Weight Download**:
#### Using Official Script (Recommended)
```bash
cd checkpoints/
# Recommended to use official script
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/checkpoints/download_ckpts.sh
chmod +x download_ckpts.sh
./download_ckpts.sh
```

#### Manual Download
```bash
cd checkpoints/

# SAM2.1 Hiera Large (Recommended)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# SAM2.1 Hiera Base+ 
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# SAM2.1 Hiera Small
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

**Resources**:
- [Official Repository](https://github.com/facebookresearch/sam2)
- [Paper](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)

---

### 3. GroundingDINO - Open-vocabulary Object Detection

**Function**: Detect target objects in images based on natural language descriptions

**Features**:
- Support for open-vocabulary detection, no predefined categories needed
- Based on Swin-B backbone network
- Can detect arbitrary objects through text descriptions

**File Structure**:
```
GroundingDINO/
├── grounding_dino_server.py
├── grounding_dino_client.py
└── configs/
    └── GroundingDINO_SwinB_cfg.py
```

**Installation**:
```bash
pip install groundingdino_py
```

**Weight Download**:
```bash
cd checkpoints/
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

**Resources**:
- [Official Repository](https://github.com/IDEA-Research/GroundingDINO)
- [Paper](https://arxiv.org/abs/2303.05499)

**Known Issues**:

> **`transformers` Version Incompatibility**
>
> **Error:** `'BertModel' object has no attribute 'get_head_mask'`
>
> GroundingDINO requires `transformers 4.x`. If `transformers>=5.0` is installed, the BERT text encoder fails to load. Fix by downgrading:
> ```bash
> pip install "transformers==4.26.0"
> ```

> **SOCKS Proxy Not Supported**
>
> **Error:** `Unknown scheme for proxy URL URL('socks://127.0.0.1:xxxx/')`
>
> If a SOCKS proxy is set in your environment (e.g. `ALL_PROXY=socks://...`), the `bert-base-uncased` model download will fail. Unset proxy variables before starting the server:
> ```bash
> unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
> ```
> Or install SOCKS support: `pip install httpx[socks]`

---

### 4. Moondream - Vision Language Model

**Function**: Vision language understanding and image Q&A

**Features**:
- Image understanding capabilities
- Natural language interaction
- API interface support

**File Structure**:
```
moondream/
├── md_server.py          # Server side
├── md_client.py          # Client side
├── md_local.py          # Local deployment
├── __init__.py
└── __pycache__/
```

**Installation**:
```bash
pip install moondream
```

**Environment Configuration**:
```bash
export MOONDREAM_API_KEY="your_api_key"
```

**Resources**:
- [Official Website](https://moondream.ai/)
- [API Documentation](https://docs.moondream.ai/)

---

### 5. Pi3 - 3D Reconstruction Service

**Function**: 3D reconstruction based on Pi3 model, generate 3D point clouds from images

**Features**:
- High-quality 3D reconstruction
- Support for PLY format output
- Visualization support

**File Structure**:
```
Pi3/
├── pi3/                  # Runtime code
├── example.py            # Original Pi3 runtime code
├── pi3_server.py         # Flask server
└── pi3_client.py         # Client
```

**Environment Requirements**:
- torch==2.5.1
- torchvision==0.20.1
- numpy==1.26.4

**Usage**:
```bash
# Visualize generated PLY files
python spagent/utils/ply_to_html_viewer.py xxx.ply --output xxx.html --max_points 100000
```

**Weight Download**:
```bash
cd checkpoints/pi3
wget https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
```

---

### 5.1 Pi3X - Enhanced 3D Reconstruction Service

**Function**: Enhanced 3D reconstruction based on Pi3X model (upgraded version of Pi3)

**Features**:
- Smoother point cloud reconstruction (ConvHead replaces LinearPts3d, eliminates grid artifacts)
- Approximate metric scale reconstruction
- Optional multimodal conditioning (camera poses, intrinsics, depth)
- More reliable continuous confidence scoring
- Fully compatible API with Pi3 (same input/output format)

**File Structure**:
```
Pi3/
├── pi3/
│   └── models/
│       ├── pi3.py            # Original Pi3 model
│       ├── pi3x.py           # Pi3X model (enhanced)
│       └── layers/
│           ├── conv_head.py   # Convolutional upsampling head (Pi3X)
│           └── prope.py       # PRoPE positional encoding (Pi3X)
├── pi3_server.py              # Pi3 Flask server
├── pi3_client.py              # Pi3 client
├── pi3x_server.py             # Pi3X Flask server
└── pi3x_client.py             # Pi3X client
```

**Weight Download**:
```bash
mkdir -p checkpoints/pi3x
cd checkpoints/pi3x
wget https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors
```

**Resources**:
- [Official Repository](https://github.com/yyfz/Pi3)
- [Pi3X HuggingFace Weights](https://huggingface.co/yyfz233/Pi3X)

---

### 6. VGGT - Multi-view 3D Reconstruction & Camera Pose Estimation

**Function**: Reconstruct 3D point clouds and estimate camera extrinsics/intrinsics from multiple images using the VGGT-1B model

**Features**:
- Multi-view input (image list or video frames)
- Outputs dense 3D point cloud (PLY format) and multi-view rendered images
- Camera pose estimation (extrinsic & intrinsic matrices)
- Confidence-based point filtering and Mahalanobis outlier removal
- Supports custom viewing angles (azimuth & elevation)

**File Structure**:
```
VGGT/
├── vggt_server.py        # Flask server
├── vggt_client.py        # Client
└── vggt/                 # VGGT model code
```

**Weight Download**:

The model is automatically downloaded from HuggingFace on first launch:
```bash
# Automatic download (default)
# python vggt_server.py  →  downloads facebook/VGGT-1B automatically

# Or download manually and pass the path
huggingface-cli download facebook/VGGT-1B --local-dir checkpoints/vggt
```

**Resources**:
- [Official Repository](https://github.com/facebookresearch/vggt)
- [HuggingFace Model](https://huggingface.co/facebook/VGGT-1B)

---

### 7. MapAnything - Dense 3D Reconstruction via Depth Estimation

**Function**: Reconstruct dense 3D point clouds from multiple images using predicted depth maps and camera poses

**Features**:
- Dense 3D reconstruction from multi-view images or video frames
- Built-in edge filtering and confidence-based point masking
- Outputs dense 3D point cloud (PLY format) and multi-view rendered images
- Interface compatible with Pi3 for easy comparison
- Supports custom viewing angles (azimuth & elevation)

**File Structure**:
```
mapanything/
├── mapanything_server.py   # Flask server
├── mapanything_client.py   # Client
└── mapanything/            # MapAnything model code
```

**Weight Download**:

The model is automatically downloaded from HuggingFace on first launch:
```bash
# Automatic download (default)
# python mapanything_server.py  →  downloads facebook/map-anything automatically

# Or pre-download manually
huggingface-cli download facebook/map-anything --local-dir ~/.cache/huggingface/hub/models--facebook--map-anything
```

**Resources**:
- [HuggingFace Model](https://huggingface.co/facebook/map-anything)

---

### 8. Supervision - Object Detection and Annotation Tools

**Function**: YOLO object detection and visualization annotation tools

**Features**:
- Integration of multiple YOLO models
- Rich visualization tools
- Annotation and post-processing capabilities

**File Structure**:
```
supervision/
├── __init__.py
├── supervision_server.py
├── supervision_client.py
├── sv_yoloe_server.py
├── sv_yoloe_client.py
├── annotator.py
├── yoloe_annotator.py
├── yoloe_test.py 
├── download_weights.py
└── mock_supervision_service.py
```

**Installation**:
```bash
pip install supervision
```

**Available Models**:
| Model File | Function | Purpose |
|----------|----------|---------|
| yoloe-v8l-seg.pt | YOLOE v8 Large Segmentation | High-precision object detection and segmentation |
| yoloe-v8l-seg-pf.pt | YOLOE v8 Large Segmentation (Optimized) | Performance-optimized segmentation model |

**Weight Download**:
```bash
python download_weights.py
```

**Resources**:
- [Official Repository](https://github.com/roboflow/supervision)
- [Documentation](https://supervision.roboflow.com/)

---

### 9. Veo - Video Generation (Google)

**Function**: Text-to-video and image-to-video generation via Google's Veo model through the Gemini API. No local server required — calls the cloud API directly.

**Features**:
- Text-to-video (t2v) and image-to-video (i2v) generation
- Returns a `.mp4` file saved locally under `outputs/`
- Mock mode available for offline testing (no API key needed)

**File Structure**:
```
Veo/
├── __init__.py
├── veo_client.py          # Real Gemini API client
└── mock_veo_service.py    # Mock service for testing
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | ✅ | — | Text description of the video to generate |
| `image_path` | string | ❌ | None | Reference image path for image-to-video |
| `duration` | integer | ❌ | 8 | Duration in seconds (5 or 8) |
| `aspect_ratio` | string | ❌ | `"16:9"` | `"16:9"` or `"9:16"` |

**API Key Setup**:
```bash
export GOOGLE_API_KEY="your_google_api_key"
# or
export GCP_API_KEY="your_gcp_api_key"
```

**Tool Test**:
```bash
# Text-to-video (no reference image)
python test/test_tool.py --tool veo \
    --image dummy \
    --prompt "A golden retriever running on a beach at sunset" \
    --duration 8

# Image-to-video (with reference image)
python test/test_tool.py --tool veo \
    --image assets/dog.jpeg \
    --prompt "The dog starts running across the field" \
    --duration 8 \
    --aspect_ratio 16:9

# Mock mode — no API key required
python test/test_tool.py --tool veo \
    --image dummy \
    --prompt "test video" \
    --use_mock
```

**Evaluation**:
```bash
# Prepare a JSONL dataset (one sample per line):
# {"id":"1","prompt":"A sunset over the ocean","image":[],"task":"t2v","duration":8}
# {"id":"2","prompt":"The dog starts running","image":["assets/dog.jpeg"],"task":"i2v"}

# Run evaluation with real Veo API
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/veo_eval_data.jsonl \
    --model gpt-4o \
    --video_num_frames 4

# Run evaluation with mock service
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/veo_eval_data.jsonl \
    --use_mock \
    --max_samples 5

# Run on existing dataset (conversations format)
python examples/evaluation/evaluate_veo.py \
    --data_path dataset/tmp.jsonl \
    --image_base_path dataset \
    --model gpt-4o
```

**Key CLI Options**:

| Option | Default | Description |
|--------|---------|-------------|
| `--data_path` | `dataset/veo_eval_data.jsonl` | Path to JSONL evaluation dataset |
| `--image_base_path` | `.` | Base directory for resolving image paths |
| `--model` | `gpt-4o` | LLM orchestrator model |
| `--max_samples` | all | Limit number of samples |
| `--video_num_frames` | `4` | Frames extracted from generated video to feed back to model |
| `--use_mock` | false | Use mock Veo service (no API key needed) |
| `--max_iterations` | `3` | Max tool-call iterations per sample |

---

### 10. Sora - Video Generation (OpenAI)

**Function**: Text-to-video and image-to-video generation via OpenAI's Sora model. No local server required — calls the cloud API directly.

**Features**:
- Text-to-video (t2v) and image-to-video (i2v) generation
- Resolution control (480p / 720p / 1080p)
- Supports 1:1 square aspect ratio in addition to 16:9 and 9:16
- Returns a `.mp4` file saved locally under `outputs/`
- Mock mode available for offline testing (no API key needed)

**File Structure**:
```
Sora/
├── __init__.py
├── sora_client.py          # Real OpenAI API client
└── mock_sora_service.py    # Mock service for testing
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | ✅ | — | Text description of the video to generate |
| `image_path` | string | ❌ | None | Reference image path for image-to-video |
| `duration` | integer | ❌ | 10 | Duration in seconds (5–20) |
| `resolution` | string | ❌ | `"1080p"` | `"480p"`, `"720p"`, or `"1080p"` |
| `aspect_ratio` | string | ❌ | `"16:9"` | `"16:9"`, `"9:16"`, or `"1:1"` |

**API Key Setup**:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

**Tool Test**:
```bash
# Text-to-video
python test/test_tool.py --tool sora \
    --image dummy \
    --prompt "A cat sitting on a windowsill watching rain fall outside" \
    --duration 10 \
    --resolution 1080p

# Image-to-video
python test/test_tool.py --tool sora \
    --image assets/dog.jpeg \
    --prompt "The dog starts running" \
    --duration 10 \
    --aspect_ratio 16:9

# Mock mode — no API key required
python test/test_tool.py --tool sora \
    --image dummy \
    --prompt "test video" \
    --use_mock
```

**Evaluation**:
```bash
# Prepare a JSONL dataset (one sample per line):
# {"id":"1","prompt":"A cat playing with yarn","image":[],"task":"t2v","duration":10,"resolution":"1080p"}
# {"id":"2","prompt":"The dog runs","image":["assets/dog.jpeg"],"task":"i2v","resolution":"720p"}

# Run evaluation with real Sora API
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/sora_eval_data.jsonl \
    --model gpt-4o \
    --video_num_frames 4

# Run evaluation with mock service
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/sora_eval_data.jsonl \
    --use_mock \
    --max_samples 5

# Run on existing dataset (conversations format)
python examples/evaluation/evaluate_sora.py \
    --data_path dataset/tmp.jsonl \
    --image_base_path dataset \
    --model gpt-4o
```

**Key CLI Options**:

| Option | Default | Description |
|--------|---------|-------------|
| `--data_path` | `dataset/sora_eval_data.jsonl` | Path to JSONL evaluation dataset |
| `--image_base_path` | `.` | Base directory for resolving image paths |
| `--model` | `gpt-4o` | LLM orchestrator model |
| `--max_samples` | all | Limit number of samples |
| `--video_num_frames` | `4` | Frames extracted from generated video to feed back to model |
| `--use_mock` | false | Use mock Sora service (no API key needed) |
| `--max_iterations` | `3` | Max tool-call iterations per sample |

---

## 🚀 Quick Start

### 1. Environment Setup

Ensure necessary dependencies are installed:
```bash
# Requires GPU memory >= 24GB
apt-get install tmux
pip install torch torchvision
pip install groundingdino_py supervision moondream
```

Create checkpoints directory:
```bash
mkdir -p checkpoints/{grounding_dino,depth_anything,pi3,pi3x,sam2}
```

### 2. Download Model Weights

Each tool requires downloading the corresponding model weight files. Please refer to the detailed instructions for each tool.

### 3. Start Services

If you want to use real expert services instead of mock mode, start the corresponding servers as needed:
```bash
# Depth estimation service
python spagent/external_experts/Depth_AnythingV2/depth_server.py \
  --checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth \
  --port 20019

# Deploy SAM2 segmentation service
# Note: You need to rename the SAM weight file to sam2.1_b.pt, otherwise it will error
python spagent/external_experts/SAM2/sam2_server.py \
  --checkpoint_path checkpoints/sam2/sam2.1_b.pt \
  --port 20020

# Deploy Grounding DINO
# Sometimes the network cannot connect to HuggingFace, we can reset the HuggingFace source
export HF_ENDPOINT=https://hf-mirror.com

python spagent/external_experts/GroundingDINO/grounding_dino_server.py \
  --checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth \
  --port 20022

# 3D reconstruction service (Pi3)
python spagent/external_experts/Pi3/pi3_server.py \
  --checkpoint_path checkpoints/pi3/model.safetensors \
  --port 20030

# 3D reconstruction service (Pi3X - enhanced, recommended)
python spagent/external_experts/Pi3/pi3x_server.py \
  --checkpoint_path checkpoints/pi3x/model.safetensors \
  --port 20031
  
# VGGT multi-view 3D reconstruction service 
python spagent/external_experts/VGGT/vggt_server.py \
  --checkpoint_path checkpoints/vggt \
  --port 20032

# MapAnything dense 3D reconstruction service (downloads facebook/map-anything automatically)
python spagent/external_experts/mapanything/mapanything_server.py \
  --port 20033

# Vision language model service
python spagent/external_experts/moondream/md_server.py \
  --port 20024
```
