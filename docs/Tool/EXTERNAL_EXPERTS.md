# External Experts Reference

> **中文版本**: [中文文档](TOOL_USING_ZH.md) | **Deployment Guide**: [TOOL_USING.md](TOOL_USING.md)

Full list of supported external expert models in SPAgent, their default ports, and primary capabilities.

| Tool Name | Type | Main Function | Deployment | Notes |
| --- | --- | --- | --- | --- |
| **Depth-AnythingV2** | 2D | Monocular Depth Estimation | Local server (20019) | Pixel-level depth maps from 2D images |
| **SAM2** | 2D | Image Segmentation | Local server (20020) | Segment Anything Model v2; interactive or automatic segmentation |
| **GroundingDINO** | 2D | Open-vocabulary Object Detection | Local server (20022) | Exposed as two tools: `zoom_object_tool` (crop close-up for attribute inspection) and `localize_object_tool` (bbox annotation for spatial/counting). Both use adaptive threshold back-off. |
| **Moondream** | 2D | Vision Language Model | Local server (20024) | Lightweight VLM for image Q&A and description |
| **Molmo2** | 2D | Multimodal Reasoning & Point Grounding | Local server (20025) | `qa`, `caption`, and `point` tasks; optional annotated point outputs |
| **Pi3** | 3D | 3D Point Cloud Reconstruction | Local server (20030) | Generates 3D point clouds and multi-view renders from a single image |
| **Pi3X** | 3D | 3D Point Cloud Reconstruction (Enhanced) | Local server (20031) | Smoother point clouds, metric scale, optional multimodal conditioning |
| **VGGT** | 3D | Multi-view 3D Reconstruction & Pose Estimation | Local server (20032) | Reconstructs point clouds + camera extrinsics from image lists or video; uses [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) |
| **MapAnything** | 3D | Dense 3D Reconstruction via Depth | Local server (20033) | Dense point clouds from depth maps + camera poses; [facebook/map-anything](https://huggingface.co/facebook/map-anything) |
| **YOLO26** | 2D | Object Detection | Local (no server) | Fast detection via `ultralytics`; outputs optional annotated image |
| **Supervision** | 2D | Detection Annotation & Visualization | Local | YOLO + visualization for post-processing |
| **Qwen2.5-VL** | 2D | Vision-Language Detection | API / local | Grounding and localization from image-text prompts |
| **OrientAnythingV2** | 3D | Orientation & Rotation Estimation | Local server (20034) | Azimuth/elevation/rotation + symmetry; two-image relative pose; NeurIPS 2025 Spotlight |
| **Sana** | Image | Text-to-Image Generation | Local SGLang server (30000) | Uses `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`; 1–4 step inference |
| **Veo** | Video | Text/Image-to-Video | API (no server) | Google Veo via Gemini API; requires `GOOGLE_API_KEY` |
| **Sora** | Video | Text/Image-to-Video | API (no server) | OpenAI Sora; requires `OPENAI_API_KEY`; t2v, i2v, 1:1 aspect ratio |
| **WAN** | Video | Text/Image-to-Video | API (no server) | Alibaba Wan via DashScope API; requires `DASHSCOPE_API_KEY` |
| **VACE** | Video | Local First-Frame Video Generation | Local server (20035) | Wan2.1-VACE; one reference image + text → `.mp4`; no cloud API |
| **WildDet3D** | 3D | Promptable 3D Object Detection | Local (no server) | Text/box/point prompts; requires `WILDDET3D_ROOT` and `WILDDET3D_CHECKPOINT` env vars |
| **FlowSeek** | 2D | Optical Flow Estimation | Local / server (20036) | Dense per-pixel motion between two images; M (ViT-B) or T (ViT-S) variants |
| **PaddleOCR-VL-1.5** | OCR | Document OCR & Structured Recognition | Local or server (20037) | Plain OCR, table, chart, formula→LaTeX, seal recognition; auto-downloads from HuggingFace |

## Port Summary

| Port | Service |
|------|---------|
| 20019 | Depth-AnythingV2 |
| 20020 | SAM2 |
| 20022 | GroundingDINO |
| 20024 | Moondream |
| 20025 | Molmo2 |
| 20030 | Pi3 |
| 20031 | Pi3X |
| 20032 | VGGT |
| 20033 | MapAnything |
| 20034 | OrientAnythingV2 |
| 20035 | VACE |
| 20036 | FlowSeek |
| 20037 | PaddleOCR-VL-1.5 |
| 30000 | Sana (SGLang) |

For deployment instructions, see **[TOOL_USING.md](TOOL_USING.md)**.
