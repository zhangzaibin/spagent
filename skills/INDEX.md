# SPAgent skills index

> Generated from `spagent/tools/catalog.py` — do not edit by hand. Regenerate with `python -m spagent.skills sync`.

One line per skill. Read `skills/<name>/SKILL.md` for arguments, output contract, and runtime requirements BEFORE first use.

- **depth_estimation_tool** — Generate a monocular depth map for one input image to analyze relative depth, near/far ordering, and 3D layout cues. — category: depth — runtime: server
- **segment_image_tool** — Segment objects or regions in an image using SAM2. — category: segmentation — runtime: server
- **detect_objects_tool** — Zoom into a specific object by detecting it and returning a cropped close-up image. — category: detection — runtime: server
- **zoom_object_tool** — Zoom into a specific object by detecting it and returning a cropped close-up image. — category: detection — runtime: server
- **localize_object_tool** — Locate objects in an image by detecting them and drawing bounding boxes on the full scene. — category: detection — runtime: server
- **supervision_tool** — Run YOLO-based object detection or instance segmentation with Supervision visualization. — category: detection/segmentation — runtime: server
- **yoloe_detection_tool** — Detect objects with YOLO-E using user-specified custom class names. — category: detection — runtime: server
- **yolo26_tool** — Run fast local YOLO26 object detection on one image. — category: detection — runtime: local, no mock
- **qwenvl_detection_tool** — Detect objects in an image using Qwen VL 2.5. — category: detection — runtime: cloud-API
- **moondream_tool** — Lightweight vision-language tasks on one image: captioning, visual Q&A, object detection, and pointing via Moondream. — category: point_grounding — runtime: cloud-API
- **molmo2_tool** — Molmo2 point-grounding tool. — category: point_grounding — runtime: server
- **pi3_tool** — This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis. — category: 3d_reconstruction — runtime: server
- **pi3x_tool** — This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis. — category: 3d_reconstruction — runtime: server
- **vggt_tool** — This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis. — category: 3d_reconstruction — runtime: server
- **mapanything_tool** — This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis. — category: 3d_reconstruction — runtime: server
- **orient_anything_v2_tool** — Estimates the 3D orientation of objects in images using Orient Anything V2 Given a single image, returns azimuth (0-360°), elevation (-90~90°), in-plane rotation (-180~180°), and symmetry_alpha (0/1/2/4 indicating rotational symmetry order). — category: orientation — runtime: server
- **image_generation_sana_tool** — Generate an image from a text prompt using Sana. — category: image_generation — runtime: server
- **video_generation_veo_tool** — Generate a video from a text prompt (and optionally a reference image) using Google Veo. — category: video_generation — runtime: cloud-API
- **video_generation_sora_tool** — Generate a video from a text prompt (and optionally a reference image) using OpenAI Sora. — category: video_generation — runtime: cloud-API
- **video_generation_wan_tool** — Generate a video from a text prompt (and optionally a reference image) using Alibaba Wan. — category: video_generation — runtime: cloud-API
- **video_generation_vace_tool** — Generate a video from one reference image and a text prompt via the local VACE first-frame pipeline; returns the path to the generated .mp4. — category: video_generation — runtime: server
- **flowseek_tool** — FlowSeek: optical flow estimation between two images. — category: optical_flow — runtime: local
- **paddleocr_vl_tool** — PaddleOCR-VL-1.5: document-level OCR and structured recognition. — category: ocr — runtime: local
- **wilddet3d_tool** — WildDet3D: promptable 3D object detection from a single RGB image. — category: detection — runtime: mock-only
