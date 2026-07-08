# SPAgent Tool Prompt & Input Configuration Analysis
## Overview
The SPAgent system uses a **three-layer prompt architecture**:
1. **Role Prompt** (user-replaceable) - Defined at agent level in `spagent/core/prompts.py`
2. **Tool Block** (auto-appended) - Tool list + wire format, always added by `build_system_prompt()`
3. **Workflow** (optional preset) - Multi-step iteration guidance
**Key Finding**: Individual tools do NOT have their own system/user prompts. Instead:
- **System Prompt**: Defined globally at the SPAgent level (e.g., `SPATIAL_3D_ROLE`, `GENERAL_VISION_ROLE`, `GENERATION_ROLE`)
- **User Prompt**: Created by `create_user_prompt(question, image_paths, tool_schemas=None)` at agent level (`prompts.py`)
- **Tool Description**: Each tool has a `description` field that is shown to the model in the tool list
- **Tool Parameters**: Each tool defines a JSON schema for its inputs via the `parameters` property
---
## Tool Configuration Table
> The **Tool Name** column is the *registered* name (`ToolCatalogEntry.tool_name` in `spagent/tools/catalog.py`) exposed to the model — not the Python class or filename. The generation tools are registered under **namespaced** names (`image_generation_sana_tool`, `video_generation_{veo,sora,wan,vace}_tool`).

| Tool Name | Category | Description (Shown to Model) | Input Image | Input Image 2 | Text/Query Inputs | Other Arguments | Output/Returns | Potential Enhancements |
|-----------|----------|------------------------------|-------------|---------------|-------------------|-----------------|----------------|------------------------|
| **image_generation_sana_tool** | Image Generation | Generate image from text prompt using Sana. Use for hypothetical scenes, target states, plan outcomes. | ❌ None | ❌ None | **prompt** (required, string): Text prompt describing image to generate | **size** (optional, default='1024x1024', enum: ['512x512', '1024x1024'])<br>**num_inference_steps** (optional, default=2, integer)<br>**guidance_scale** (optional, default=4.5, number)<br>**seed** (optional, default=42, integer)<br>**negative_prompt** (optional, string)<br>**n** (optional, default=1, integer): Number of images | ✅ **output_path** (string): Generated image path<br>✅ **image_paths** (list): All images when n>1<br>✅ **file_size_bytes**, **model**, **size**, **seed** | ⚠️ Intermediate diffusion steps, attention maps, latent representations |
| **zoom_object_tool** | Detection | Detect object and return cropped close-up images. Use for color, texture, material, pattern, text. | **image_path** (required, string) | ❌ None | **text_prompt** (required, string): Max 2 objects separated by '.' | **box_threshold** (optional, default=0.35)<br>**text_threshold** (optional, default=0.25) | ✅ **crop_paths** (list): Cropped images (35% context padding, resized to 512px)<br>✅ **detections** (list): Metadata per crop<br>✅ **message** / **description** (string): Text description (keys are `message`/`description`, not `summary`) | ⚠️ Original bbox coords (pixel/normalized), confidence scores, detection hierarchy |
| **localize_object_tool** | Detection | Detect objects and draw bounding boxes. Use for WHERE/HOW MANY/SPATIAL LAYOUT. | **image_path** (required, string) | ❌ None | **text_prompt** (required, string): Max 2 objects separated by '.' | **box_threshold** (optional, default=0.35)<br>**text_threshold** (optional, default=0.25) | ✅ **output_path** / **vis_path** (string): Annotated full image<br>✅ **message** / **description** (string): Position descriptions (keys are `message`/`description`, not `summary`)<br>✅ **detections** (list): bbox + labels | ⚠️ Structured JSON coords (pixel+normalized), spatial relationships graph, object counts per class |
| **detect_objects_tool** | Detection | Backward-compatible wrapper (`ObjectDetectionTool` → subclasses `ZoomObjectTool`); preserves the legacy name + `crop=` constructor API for call sites like quick_eval.py / run_spagent_vlmeval.py / test_tool.py. | **image_path** (required, string) | ❌ None | **text_prompt** (required, string): Max 2 objects separated by '.' | **box_threshold** (optional, default=0.35)<br>**text_threshold** (optional, default=0.25)<br>**crop** (constructor kwarg, default=true) | ✅ **detections** (list)<br>✅ **crop_paths** (list)<br>✅ **message** / **description** (string) | ⚠️ Same as zoom_object_tool (structured bbox coords, confidence) |
| **depth_estimation_tool** | Depth Estimation | Generate monocular depth map for relative depth, near/far ordering, 3D layout. | **image_path** (required, string) | ❌ None | ❌ None | ❌ None | ✅ **output_path** (string): Colored depth visualization<br>✅ **shape** (tuple): Dimensions<br>✅ **depth_data** (array): Raw depth values | ⚠️ Depth histogram, statistics (min/max/mean), depth-based segmentation, 3D point cloud, surface normals |
| **segment_image_tool** | Segmentation | Segment objects/regions using SAM2. Supports point, box, or automatic segmentation. | **image_path** (required, string) | ❌ None | ❌ None | **point_coords** (optional, [[x,y]])<br>**point_labels** (optional, [1/0])<br>**box** (optional, [x1,y1,x2,y2]) | ✅ **output_path** (string): Combined image<br>✅ **overlay_path** (string): Mask visualization<br>✅ **mask_path** (string): Raw binary mask<br>✅ **masks** (list): Multiple masks<br>✅ **shape** (tuple) | ⚠️ Polygon/contour vectors, mask statistics (area/perimeter), cropped objects, RLE encoding, nested masks |
| **pi3_tool** | 3D Reconstruction | 3D reconstruction to generate point clouds and visualizations from CUSTOM viewing angles. | **image_path** (required, list of strings) | ❌ None | ❌ None | **azimuth_angle** (optional, -180 to 180; only `image_path` is `required` in the schema)<br>**elevation_angle** (optional, -90 to 90)<br>**rotation_reference_camera** (optional, int, default=1)<br>**camera_view** (optional, bool, default=false) | ✅ **ply_filename** (string): 3D point cloud (.ply)<br>✅ **points_count** (int): Number of 3D points<br>✅ **camera_views** (list): [{camera, view, angles, image (base64)}] | ⚠️ Mesh reconstruction, textured 3D model, camera poses, depth per view, OBJ/FBX export, scene graph |
| **pi3x_tool** | 3D Reconstruction | Pi3X upgraded version with smoother point clouds. 3D reconstruction from images. | **image_path** (required, list of strings) | ❌ None | ❌ None | **azimuth_angle** (optional, -180 to 180; only `image_path` is `required` in the schema)<br>**elevation_angle** (optional, -90 to 90)<br>**rotation_reference_camera** (optional, default=1)<br>**camera_view** (optional, default=false) | ✅ Same as pi3_tool but with higher quality point clouds | ⚠️ All pi3_tool enhancements + metric scale info, object-level reconstruction, material properties |
| **qwenvl_detection_tool** | Detection | Detect objects using Qwen VL 2.5. Referring detection and reasoning detection. | **image_path** (required, string) | ❌ None | **text_prompt** (required, string): Object description or reasoning question | **task** (optional, default='ref_detection', enum: ['ref_detection', 'reasoning_detection']) | ✅ **boxes** (list): Normalized coords [0-1]<br>✅ **labels** (list): Text labels<br>✅ **raw_response** (string): Full model response | ⚠️ Pixel coordinates option, cropped regions, reasoning chain for reasoning_detection |
| **molmo2_tool** | Point Grounding | Point-grounding tool. Locates objects and returns annotated overlay showing exact position. | **image_path** (required, string) | ❌ None | **prompt** (required, string): Reasoning sentence like 'Point to the object to grasp next' | **save_annotated** (optional, default=true)<br>**max_new_tokens** (optional, default=200) | ✅ **points_by_image** (dict): {path: [{x,y,confidence}]}<br>✅ **output_path** (string): Annotated overlay<br>✅ **saved_paths** (list): All visualizations<br>✅ **raw_text**, **num_points** | ⚠️ Bounding boxes around points, segmentation masks, attention heatmaps, multi-scale crops |
| **yolo26_tool** | Detection | Fast local YOLO26 object detection. Returns bounding boxes, class labels, confidence scores. | **image_path** (required, string) | ❌ None | ❌ None | **conf** (optional, default=0.25, 0.0-1.0)<br>**save_annotated** (optional, default=true) | ✅ **detections** (list): [{bbox_xyxy (pixels), class_id, class_name, confidence}]<br>✅ **num_detections** (int)<br>✅ **output_path** (string): Annotated image<br>✅ **summary** (string) | ⚠️ Normalized coords [0-1], cropped object images, tracking IDs for video, IoU between detections |
| **orient_anything_v2_tool** | Orientation Estimation | Estimates 3D orientation of objects. Returns azimuth, elevation, in-plane rotation, symmetry. | **image_path** (required, string) | **image_path2** (optional, string): For relative_rotation | **object_category** (required, string): e.g., 'chair', 'car', 'bottle' | **task** (optional, default='orientation', enum: ['orientation', 'symmetry', 'relative_rotation']) | ✅ **Single image**: azimuth (0-360°), elevation (-90-90°), rotation (-180-180°), symmetry_alpha (0/1/2/4)<br>✅ **Two images**: Also rel_azimuth, rel_elevation, rel_rotation | ⚠️ Confidence scores per angle, 3D rotation matrix, quaternion format, coordinate axes overlay, canonical pose |
| **supervision_tool** | Detection/Segmentation | YOLO-based detection or instance segmentation with Supervision visualization. | **image_path** (required, string) | ❌ None | ❌ None | **task** (required, enum: ['image_det', 'image_seg']) | ✅ **boxes** (list): Bbox coordinates<br>✅ **labels** (list): Class labels<br>✅ **confidence** (list): Scores<br>✅ **masks** (list): For image_seg<br>✅ **vis_path** (string): Annotated visualization | ⚠️ Mask polygons (vector), mask-to-crop extraction, object statistics (area/perimeter) |
| **video_generation_veo_tool** | Video Generation | Generate video from text prompt using Google Veo. | ❌ None | ❌ None | **prompt** (required, string): Video description | **image_path** (optional, string): Reference image<br>**duration** (optional, default=8, 5 or 8)<br>**aspect_ratio** (optional, default='16:9') | ✅ **output_path** (string): Generated .mp4 video path | ⚠️ Individual frames as images, video metadata (fps, resolution, codec), thumbnail preview |
| **video_generation_sora_tool** | Video Generation | Generate video from text prompt using OpenAI Sora. | ❌ None | ❌ None | **prompt** (required, string): Video description | **image_path** (optional, string): Reference image<br>**duration** (optional, default=10, 5-20)<br>**resolution** (optional, default='1080p')<br>**aspect_ratio** (optional, default='16:9') | ✅ **output_path** (string): Generated .mp4 video path | ⚠️ Frame-by-frame images, video quality metrics, generation metadata |
| **video_generation_wan_tool** | Video Generation | Generate video from text prompt using Alibaba Wan (DashScope). | ❌ None | ❌ None | **prompt** (required, string): Video description | **image_path** (optional, string): Reference image<br>**duration** (optional, default=5, integer, range 3–10)<br>**aspect_ratio** (optional, default='16:9', enum: ['16:9', '9:16', '1:1']) | ✅ **output_path** (string): Generated .mp4 video path | ⚠️ Frame extraction, motion analysis, scene transitions |
| **video_generation_vace_tool** | Video Generation | Generate video from reference image via local VACE first-frame pipeline. | **image_path** (required, string): First-frame reference | ❌ None | **prompt** (required, string): Motion prompt | **base** (optional, default='wan')<br>**task** (optional, default='frameref')<br>**mode** (optional, default='firstframe') | ✅ **output_path** (string): Generated .mp4 video<br>✅ **description** (string) — note: `result_dir` is nested inside the `result` sub-dict, not top-level | ⚠️ Frame sequence as images, camera trajectory data, intermediate results<br>⚠️ **Very slow (minutes)** |
| **flowseek_tool** | Optical Flow | Optical flow estimation between two images. | **image1_path** (required, string) | **image2_path** (required, string) | ❌ None | **output_path** (optional, string): Save path | ✅ **flow_magnitude_mean** (float): Average pixel motion<br>✅ **output_path** (string): Colorized flow visualization | ⚠️ Raw flow vectors (u,v per pixel), flow arrows overlay, motion boundaries, dominant direction, object motion masks |
| **mapanything_tool** | 3D Reconstruction | 3D reconstruction using MapAnything. Similar to Pi3. | **image_path** (required, list of strings) | ❌ None | ❌ None | **azimuth_angle** (optional, -180 to 180; only `image_path` is `required` in the schema)<br>**elevation_angle** (optional, -90 to 90)<br>**rotation_reference_camera** (optional, default=1)<br>**camera_view** (optional, default=false) | ✅ Same as pi3_tool: **ply_filename**, **points_count**, **camera_views** | ⚠️ All pi3_tool enhancements + semantic labels per point, structure-from-motion metadata |
| **moondream_tool** | Point Grounding | Lightweight vision-language: pointing to objects. | **image_path** (required, string) | ❌ None | **object_name** (required, string): Single or comma-separated multiple objects | **task** (required, enum: ['point']) | ✅ **Single**: points [{x,y,confidence}], output_path<br>✅ **Multiple**: all_points {obj:[{x,y,confidence}]}, color_mapping, total_points, output_path<br>✅ Normalized coords [0-1] | ⚠️ Bounding boxes from points, crops around each point, distance metrics between objects |
| **paddleocr_vl_tool** | OCR/Document | Document-level OCR and structured recognition using PaddleOCR-VL-1.5. | **image_path** (required, string) | ❌ None | ❌ None | **task** (optional, default='ocr', enum: ['ocr', 'table', 'chart', 'formula', 'spotting', 'seal']) | ✅ **text** (string): Extracted text/structured output (format varies by task)<br>✅ **task** (string): Task mode used | ⚠️ Text bounding boxes, confidence scores, structured JSON for tables/charts, reading order, font/style metadata |
| **yoloe_detection_tool** | Detection | Detect objects with YOLO-E using custom class names. | **image_path** (required, string) | ❌ None | ❌ None | **task** (required, enum: ['image', 'video'])<br>**class_names** (required, array of strings) | ✅ **boxes** (list): Bbox coordinates<br>✅ **labels** (list): Detected class names<br>✅ **confidence** (list): Scores<br>✅ **class_names** (list): Input classes<br>✅ **vis_path** (string): Annotated visualization | ⚠️ Cropped object images, normalized coords, multiple format options (COCO, YOLO, Pascal VOC) |
| **wilddet3d_tool** | Detection (3D) | Promptable 3D object detection from single RGB image. | **image_path** (required, string) | ❌ None | **prompt_text** (optional, default='object', string): Object description; ignored when input_boxes/input_points given | **input_boxes** (optional, list): takes priority over prompt_text<br>**input_points** (optional, list): takes priority over prompt_text | ✅ Annotated image with 2D and 3D bounding boxes<br>✅ 3D location estimates | ⚠️ Structured 3D coordinates, depth estimates per object, 3D mesh generation, relative spatial positions |
| **vggt_tool** | 3D Reconstruction | 3D reconstruction using VGGT. | **image_path** (required, list of strings) | ❌ None | ❌ None | **azimuth_angle** (optional, -180 to 180; only `image_path` is `required` in the schema)<br>**elevation_angle** (optional, -90 to 90)<br>**rotation_reference_camera** (optional, default=1)<br>**camera_view** (optional, default=false) | ✅ Same as pi3_tool: **ply_filename**, **points_count**, **camera_views** | ⚠️ All pi3_tool enhancements |

---
## Category-Based Output Standards

This section defines the **minimal output contract** per tool category. The goal is a **clear, enforceable registry**: a small *required* core every tool in a category must return — including its raw informative data — plus unlimited per-tool optional extras.

### Design Philosophy

- **Required = envelope + the category's raw informative payload.** Each category has exactly one *informative* output — boxes, a mask, a depth field, a flow field, points, text, a point cloud, a media path. **That raw payload is required.** A tool that returns only a rendered picture of it (an annotated image, a colored map) is **not** compliant — the machine-readable data must be there too.
- **Minimal, though.** Exactly *one* required payload per category — the smallest thing that carries the category's information and lets tests + large-scale eval consume the result. Everything else is optional.
- **Format-flexible via ONE OF.** The raw payload may be returned in any *one of* several accepted representations (bbox as pixel `xyxy` OR normalized; a mask as array OR polygon OR file path; depth as array OR file). Consumers accept any; each tool picks what's natural. This is what keeps the required bar low without dropping the data.
- **Optional = per-tool flexibility.** Anything beyond the one required payload (confidence, extra metadata, a *second* coordinate format, statistics) is optional; consumers **must not** depend on it.
- **Rendering ≠ raw.** The rendered visualization is how a result reaches the VLM's eyes and is expected for any visual tool — but it is a *view* of the raw payload, not a substitute for it.

**Grounding — who reads what:**

| Consumer | Reads |
|----------|-------|
| Agent loop (`spagent/core/spagent.py:326-402`) | `success`; `description` (→ text shown to the VLM); a visualization among `output_path` / `vis_path` / `crop_paths` (`.mp4` in `output_path` is auto-frame-extracted) |
| Memory (`spagent/core/memory.py:229-243`) | `success`, `error`, `description` |
| Tests (`test/test_tool.py`) | lenient (payload **or** a vis path) — the required-raw contract below is the bar the tests tighten toward |

The loop consumes the envelope + a rendering; the **raw payload is required for downstream analysis, chaining, and reproducible large-scale experiments**, even where the current tests don't yet assert it.

### Universal Result Envelope (required for EVERY tool)

| Field | Type | When required | Purpose |
|-------|------|---------------|---------|
| `success` | bool | always | gates all downstream handling |
| `description` | str | always (may be empty) | tool's *draft* natural-language summary. **Ownership:** the tool emits a draft; once the render module exists, the **renderer owns the final VLM-facing text** (default projection = pass the tool draft through; richer projections synthesize from the raw payload). |
| `error` | str | when `success=False` | failure reason (logged + surfaced to the model) |
| **a rendered visualization** — one of `output_path` \| `vis_path` \| `crop_paths` \| `overlay_path` | str / list | any tool that produces an image/video | how the result reaches the VLM; a *view* of the raw payload (`.mp4` allowed in `output_path`). ⚠️ Today's agent loop consumes only `output_path`/`vis_path`/`crop_paths` — **the render module must also consume `overlay_path`** (a tool returning only `overlay_path` is contract-compliant). |

### Category Output Registry

Two required layers on top of the envelope: the **raw payload** (required; pick any *one of* the accepted forms) and, for visual tools, a **rendering**. Optional fields are free per tool.

| # | Category | Tools (registered) | Required raw payload — **ONE OF** | Common optional (per-tool) |
|---|----------|--------------------|-----------------------------------|----------------------------|
| 1 | Detection | detect_objects, zoom_object, localize_object, yolo26, qwenvl_detection, yoloe_detection, supervision (det)*, wilddet3d | accepted carriers (normative, **one of**): (a) `detections: [{bbox, label, confidence?}]` **or** (b) parallel `boxes` + `labels` arrays; each box **one of** {pixel `xyxy`, normalized `xyxy`, normalized `cxcywh`} | `confidence`, `image_width`/`height`, `class_id`, `crop_paths`, `masks`, `3d_coords` |
| 2 | Segmentation | segment_image, supervision (seg)* | a mask **one of** {binary array, mask file path, `polygon` coords, RLE} | `area`, `bbox`, `class_name`, `shape` |
| 3 | Image Generation | image_generation_sana | image path **one of** {`output_path`, `image_paths[]`} | `seed`, `model`, `size`, `file_size_bytes` |
| 4 | Video Generation | video_generation_{veo,sora,wan,vace} | `output_path` (.mp4 path) | `duration`, `resolution`, `fps`, `codec`, `result_dir`, `frame_paths` |
| 5 | 3D Reconstruction | pi3, pi3x, mapanything, vggt | point cloud **one of** {`ply_filename` path, raw points array} | `points_count`, `view_count`, `camera_views`, `camera_poses`, `mesh_path`, `scale_info` |
| 6 | Point Grounding | molmo2, moondream | `points`, each **one of** {normalized `(x,y)`, pixel `(x_px,y_px)`} | `confidence`, `labels`, `image_width`/`height`, `raw_text` |
| 7 | Depth | depth_estimation | depth field **one of** {raw array (`depth_data`), depth file path (.npy/.exr)} | `shape`, `value_range`, `confidence_map`, `normal_map` |
| 8 | Orientation | orient_anything_v2 | orientation **one of** {euler {`azimuth`,`elevation`,`rotation`}, `rotation_matrix`, `quaternion`} | `symmetry_order`, `confidence`, axes overlay |
| 9 | Optical Flow | flowseek | flow field **one of** {raw `(u,v)` array, flow file path} | `flow_magnitude_mean` + stats, `motion_boundaries`, `confidence_map` |
| 10 | OCR / Document | paddleocr_vl | `text` (string; structured string for table/chart/formula) | `text_boxes`, `confidence`, `structured_data`, `reading_order` |

The rendered visualization (envelope) is additionally expected for every visual category above.

\* **Dual-category tools** (supervision): the tool's static registry `category` is its primary one (`detection`); the **effective category is resolved per call** from the `task` argument, and the result carries a `category` field so validation/rendering key off the *result*, not the static registry entry.

> **This registry table is normative.** The Tool Configuration Table at the top of this doc and the per-category blocks below are explanatory reference; where they disagree with this table, this table wins.

### Per-category detail

Each block gives the required raw payload (one-of forms), the optional extras, and *why* it's the minimal informative unit.

#### 1. Detection
- **Required raw (one of):** `detections: [{bbox, label, confidence?}]` **or** parallel `boxes`+`labels`; each box as pixel `xyxy` **or** normalized `xyxy` **or** normalized `cxcywh`. **Rendering:** annotated image / crops.
- **Optional:** `confidence`, `image_width`/`image_height`, `class_id`, `crop_paths`, `masks`, `3d_coords`.
- **Why minimal:** a labelled box is the smallest unit that says *what* + *where*; format is unconstrained so no tool is forced to convert.
- **Canonical carrier / format (GroundingDINO tools):** the raw boxes live in the **`detections`** list — `[{id, bbox, confidence, label}]` — where `bbox` is **normalized `cxcywh` in [0,1]** (traced from `groundingdino.predict()` → `spagent/external_experts/GroundingDINO/grounding_dino_server.py:202-209`). The parallel top-level `boxes`/`labels`/`confidence` arrays are derived from `detections` via `_surface_boxes()`, so both carriers always agree; either satisfies the contract.

#### 2. Segmentation
- **Required raw (one of):** binary mask array **or** mask file path **or** `polygon` coords **or** RLE. **Rendering:** overlay / combined image.
- **Optional:** `area`, `bbox`, `class_name`, `shape`.
- **Why minimal:** the mask (in any form) is the information; area/bbox are computable from it.

#### 3. Image Generation
- **Required raw (one of):** `output_path` **or** `image_paths[]`.
- **Optional:** `seed`, `model`, `size`, `file_size_bytes`.
- **Why minimal:** the generated image *is* the payload; `seed`/`model` aid reproducibility but don't gate success.

#### 4. Video Generation
- **Required raw:** `output_path` (`.mp4` path).
- **Optional:** `duration`, `resolution`, `fps`, `codec`, `result_dir`, `frame_paths`.
- **Why minimal:** the file is the payload; the loop auto-extracts frames from it (`spagent.py:331-337`).

#### 5. 3D Reconstruction
- **Required raw (one of):** `ply_filename` (point cloud path) **or** a raw points array. **Rendering:** `output_path` rendered view(s).
- **Optional:** `points_count`, `view_count`, `camera_views`, `camera_poses`, `mesh_path`, `scale_info`.
- **Why minimal:** the point cloud is the reconstruction; poses/mesh are enrichments.

#### 6. Point Grounding
- **Required raw (one of):** `points`, each as normalized `(x,y)` **or** pixel `(x_px,y_px)`. **Rendering:** annotated overlay.
- **Optional:** `confidence`, `labels`, `image_width`/`image_height`, `raw_text`.
- **Why minimal:** the point coordinates are the payload; dims are only needed to convert between the two forms.
- **Formats in this repo:** molmo2 emits **pixel** coords (`point_utils.py` scales the model output by image size); moondream emits **normalized [0,1]** (native format). Both are accepted forms.

#### 7. Depth Estimation
- **Required raw (one of):** raw depth array (`depth_data`) **or** a depth file path (.npy/.exr). **Rendering:** `output_path` colored depth map.
- **Optional:** `shape`, `value_range`, `confidence_map`, `normal_map`.
- **Why minimal:** the depth *field* is the information — the colored map is only a rendering of it, so the array/file is required, not the picture.

#### 8. Orientation Estimation
- **Required raw (one of):** euler {`azimuth`, `elevation`, `rotation`} **or** `rotation_matrix` **or** `quaternion` (also echoed in `description`).
- **Optional:** `symmetry_order`, `confidence`, axes overlay.
- **Why minimal:** a single rotation, in any standard parametrization, is the payload.

#### 9. Optical Flow
- **Required raw (one of):** raw `(u,v)` flow array **or** a flow file path. **Rendering:** `output_path` colorized flow.
- **Optional:** `flow_magnitude_mean` + statistics, `motion_boundaries`, `confidence_map`.
- **Why minimal:** the per-pixel flow field is the information; a magnitude scalar is a *summary*, not the payload.
- **Design note (server mode):** the raw field is large (~16 MB per 1080p frame as float32), so HTTP transport should return a served `.npy`/compressed `.npz` path rather than inline base64 — and the server needs a persistent output dir (its temp files are unlinked after the response).

#### 10. OCR / Document
- **Required raw:** `text` (string; a structured string — Markdown/LaTeX/JSON — for table/chart/formula tasks).
- **Optional:** `text_boxes`, `confidence`, `structured_data`, `reading_order`, `font_metadata`.
- **Why minimal:** extracted text is the deliverable; boxes/order are enrichments.

---

### Architecture

The design delivers four things: (1) tools **categorized & modularized**, (2) a **unified per-category output format** (required + optional), (3) a **clear registry** tying category → tools → contract, and (4) a **separate parse/render module** that projects a standardized result into the vLLM-facing message.

**Two-stage data flow — produce vs. project:**

```
tool.call()                      parse/render module                 VLM
──────────                       ───────────────────                 ───
ToolResult(envelope              select fields per user config,      formatted
 + required payload              apply per-category DEFAULT          tool-output
 + all optional it can)   ──►    (or "give-everything" preset)  ──►  text/images
 [PRODUCE: rich, lossless]       [PROJECT: filtered, formatted]
```

- **Tools produce, they do not format for the model.** Each `tool.call()` returns a standardized `ToolResult`: the universal envelope + the category's required payload + whatever optional fields that tool can afford — as rich and lossless as possible. Tools may draft a `description`; the renderer owns the final VLM-facing text.
- **The parse/render module projects.** A single module converts a `ToolResult` into the vLLM tool-output format, **including only the fields the user configured**. Each category ships a **default projection** (a minimal, sensible field set) plus a **unified "all" preset** (dump everything). Config can override per category or per tool.
- **Why the split:** the same rich result can be rendered thinly for cheap large-scale eval or fully for debugging, without touching tool code; required stays minimal for tests + experiments while optional gives each tool head-room.

### Integration with the agent loop

The render module replaces the loop's current hardcoded consumption, in one place:

- **Call site:** `SPAgent`'s tool-result handling (the step loop in `spagent/core/spagent.py`) delegates to `render(result, config)` instead of hardcoding `result.get("output_path"/"vis_path"/"crop_paths")` reads: the renderer returns `(text, images)` — the text goes to memory/the continuation prompt, the images (now **including `overlay_path`** and extracted video frames) go into `iteration_additional_images`.
- **Config plumbing:** `SPAgent(render_config=...)` constructor arg (an object/dict, optionally loaded from file); overridable per `solve_problem()`/`step()` call via a `render_config=` kwarg. No global state.
- **Backward compatibility / migration:** `ToolResult` is **dict-compatible** (a `Mapping` — `.get()`/`[]`/`in` all work), so existing consumers and un-migrated dict-returning tools keep working unchanged. The renderer accepts both: a plain dict routes through a **legacy projection** that reproduces the legacy behavior exactly (description passthrough + output_path/vis_path/crop_paths). Tools migrate to `ToolResult` **one at a time**; a mixed iteration (some dict, some ToolResult) is fully supported. No consumer change is required until the last tool migrates.

### Render config: precedence & shape

Precedence (most specific wins): **per-tool > per-category > global preset**. Field lists **replace** (not merge) at whichever level is set — predictable over clever.

```python
render_config = {
    "preset": "default",                # "default" | "all"  (global fallback)
    "categories": {
        "detection": {"fields": ["labels", "boxes", "confidence"], "coords": "pixel_xyxy"},
        "depth":     {"fields": []},    # image-only projection for cheap eval
    },
    "tools": {
        "zoom_object_tool": {"fields": ["labels", "crop_paths"]},   # beats "detection"
    },
}
```

The projection controls: which payload fields appear in the text block, coordinate format conversion (via payload helpers), which visualizations attach, and whether `description` is the tool draft (default) or synthesized from the payload.

### Registry implementation sketch

**Extend the existing `TOOL_CATALOG` — do not build a parallel registry.** `spagent/tools/catalog.py` already registers construction metadata (`ToolCatalogEntry`: `key`/`cls`/`group`/`tool_name`/`default_kwargs`/`accepts_use_mock`). The coarse `group` (4 values: `2d_perception`/`vlm`/`3d`/`generation`) stays — it drives prompt-grouping. The fine **`category`** (10 values, this doc's registry table) is added for output contracts; e.g. molmo2 keeps `group="vlm"` and gains `category="point_grounding"`.

1. **`ToolCatalogEntry` gains a `category: str` field** (one of the 10 registry categories; supervision's static entry says `detection`, per-call resolution per the registry-table note). Register flowseek / paddleocr_vl / wilddet3d while at it (in scope for this branch).
2. **`CATEGORY_CONTRACTS: Dict[str, CategoryContract]`** in a new `spagent/core/tool_result.py` (or sibling): `CategoryContract(required_one_of, optional_fields, default_projection)` — the code form of this doc's registry table, single source of truth for validation + the renderer's defaults.
3. **One `ToolResult` envelope** (dict-compatible, see Integration) + **per-category typed payload** (`DetectionPayload`, `Mask`, `Points`, `Orientation`, …) whose constructor accepts *any one of* the allowed forms and exposes computed helpers (area from mask, dims from array, pixel↔normalized).
4. **Required raw payload gates category compliance**; optional fields never gate `success`.
5. **Parse/render module** (`spagent/core/render.py`): `render(result, config) -> (text, images)`; consumes `CATEGORY_CONTRACTS` defaults; legacy projection for plain dicts.

Guiding principle: **required = the category's raw informative payload (in any one accepted form) + the universal envelope; optional = everything else; the render module decides what of that reaches the model.**

### Validation plan
- **During dev — smoke tests only, < 10 GB VRAM.** Exercise the envelope/payload/render plumbing with (a) `use_mock=True` tools (zero VRAM) and (b) one tiny real local tool — **`yolo26_tool`** (local YOLO, no server, ~1–2 GB, `accepts_use_mock=False`) — to prove the real→standardized→render path end-to-end without disturbing GPU jobs.
- **What the smoke tests assert:** (1) per category: required payload present in one accepted form, envelope fields present, payload helpers round-trip (pixel↔normalized, mask↔bbox); (2) render path: default projection emits exactly the contract's `default_projection` fields, `"all"` preset emits every populated field, per-tool override beats per-category; (3) legacy: a plain-dict result renders byte-identical to the legacy loop output; (4) mixed iteration (dict tool + ToolResult tool) completes.
- **Gate:** full dev + **design review by the maintainer** before any broad run.
- **After sign-off — full matrix.** Run the complete `test/test_tool.py` suite across **all** tools (spinning up the heavier expert servers as needed).

---
## Output Processing Observations
### Key Patterns
**Legend**:
- ✅ = Currently returns this output
- ⚠️ = Potential enhancement or alternative output format
**Common Patterns**:
1. **Visualization Strategy**: Most tools return both raw data AND visualization paths
   - **Raw outputs**: Coordinates, masks, depth values, 3D points
   - **Visualizations**: Annotated images, overlays, colored depth maps
   - This dual approach supports both automated processing and human inspection
2. **Coordinate Format Inconsistency**: Detection tools vary in output format
   - **Pixel coordinates** (xyxy): yolo26_tool → Ready for cropping
   - **Normalized [0-1]**: qwenvl_tool → Resolution-independent
   - ⚠️ **No standard format** across all detection tools
3. **Image Processing Gaps**:
   - Most detection tools DON'T automatically crop detected regions (except zoom_object_tool)
   - Segmentation tools provide masks but not automatic cropping using those masks
   - 3D tools return point clouds but no automatic mesh generation
4. **Metadata Richness Varies**:
   - **Well-documented**: sana_tool (size, seed, model), yolo26_tool (confidence, class)
   - **Minimal metadata**: Some 3D tools lack camera intrinsics, scale information
### Enhancement Opportunities
a) **Standardized Output Schema**: All detection tools could return unified format with both pixel and normalized coordinates
b) **Post-Processing Utilities**: Helper functions for auto-cropping, coordinate conversion, batch processing, NMS across tools
c) **3D Output Diversity**: Export to multiple formats (.ply, .obj, .fbx, .gltf), automatic mesh generation, texture mapping
d) **Composite Tools**: Higher-level tools chaining Detection → Segmentation → Masked crop, or Detection → Depth → 3D localization
e) **Quality Metrics**: IoU, precision/recall for detection; point cloud density for 3D; FID scores for generation
---
## System Prompt Variants (Agent-Level)
Defined in `spagent/core/prompts.py`:
| Variant | Description | Used For |
|---------|-------------|----------|
| **SPATIAL_3D_ROLE** | *Generic* one-liner: "You are a helpful assistant that can analyze images and answer questions." The 3D-specific behaviour lives in the **workflow**, not the role. | Multi-view 3D reconstruction, camera movement analysis |
| **GENERAL_VISION_ROLE** | *Generic* one-liner: "You are a helpful visual assistant that can analyze images and answer questions." | General image understanding, Q&A |
| **GENERATION_ROLE** | Generation-specific role text | Image/video generation workflows |
| **ALL_TOOLS_ROLE** | Broad toolkit role spanning 2D perception, VLM, 3D, generation | When using many tools together |

> Note: only `GENERATION_ROLE` and `ALL_TOOLS_ROLE` carry task-specific text; `SPATIAL_3D_ROLE`/`GENERAL_VISION_ROLE` are near-identical generic strings. Full-template constants (`*_SYSTEM_PROMPT`) also exist as backward-compat variants.

## Workflow Variants (Agent-Level)
| Variant | Description | Continuation Hint |
|---------|-------------|-------------------|
| **SPATIAL_3D_WORKFLOW** | Multi-step 3D reconstruction workflow with angle exploration | `SPATIAL_3D_CONTINUATION_HINT` - Suggests NEW angles (NOT 0°,0°!), different cameras, ego/global views |
| **GENERAL_VISION_WORKFLOW** | Multi-round tool calling for thorough image understanding | `GENERAL_VISION_CONTINUATION_HINT` (or dynamic `build_general_vision_continuation_hint(tool_names)`) - Tool selection guide filtered to registered tools |
| **GENERATION_WORKFLOW** | Execution-oriented generation with minimal deliberation | `GENERATION_CONTINUATION_HINT` - Direct action, no repeated reflection |
| **ALL_TOOLS_WORKFLOW** | Broad workflow spanning all tool families (drives `workflow_mode="all_tools"`) | `ALL_TOOLS_CONTINUATION_HINT` + `TOOL_SELECTION_GUIDE` / dynamic `build_tool_selection_guide(tool_names)` |

> **Workflow routing**: `SPAgent(workflow_mode=...)` accepts `"default"`, `"auto"`, and `"all_tools"` (`spagent.py`, `_select_workflow` / `_resolve_workflow_prompts`). `"auto"` picks a workflow family from the registered tool set; the doc's Data Flow below describes the static build path only.

---
## Key Insights
1. **No Per-Tool Prompts**: Individual tools do NOT have their own system/user prompts. The agent constructs prompts globally.
2. **Tool Description as Guidance**: Each tool's `description` field serves as the model's guidance for when/how to use the tool.
3. **JSON Schema Parameters**: All tool inputs are defined via OpenAI function-calling format JSON schemas in the `parameters` property.
4. **Three-Layer Architecture**:
   - **Layer 1 (Role)**: What the agent is (SPATIAL_3D_ROLE, GENERAL_VISION_ROLE, etc.)
   - **Layer 2 (Tool Block)**: Auto-appended tool list + wire format
   - **Layer 3 (Workflow)**: Multi-step iteration guidance (SPATIAL_3D_WORKFLOW, etc.)
5. **Dynamic Prompt Building**: The `SPAgent.step()` method:
   - Takes user `content` (question) and `images` (paths)
   - Calls `create_user_prompt(question, image_paths, tool_schemas=None)`
   - Combines with system prompt and workflow guidance
   - Manages multi-turn conversation through `AgentMemory`
6. **Client Configuration**: Most tools have:
   - `use_mock`: Boolean for testing without real services
   - `server_url`: URL for HTTP-based expert services
   - Initialization of client in `_init_client()` method
7. **Common Patterns**:
   - **Detection tools**: Require `image_path` + `text_prompt`, optional thresholds
   - **3D reconstruction tools**: Require `image_path` (list) + angle parameters (azimuth, elevation)
   - **Generation tools**: Require text `prompt`, optional image reference
   - **Segmentation tools**: Require `image_path`, optional guidance (points, boxes)
8. **Critical Rules for 3D Tools**:
   - Input images are ALREADY at (azimuth=0°, elevation=0°)
   - DO NOT call with (0°, 0°) as it just returns the same view
   - Explore different angles: ±45°, ±90°, 180° for azimuth; ±30-60° for elevation
   - Use `rotation_reference_camera` to rotate around different camera positions
   - Use `camera_view=true` for first-person ego view vs. `false` for global bird's-eye
---
## Data Flow
```
User Question + Images
        ↓
SPAgent.step()
        ↓
[System Prompt (role + tools + workflow)] + [User Prompt (question + image paths)]
        ↓
Model Inference (VLLM)
        ↓
Parse <tool_call> from response
        ↓
Execute tool.call(**params)
        ↓
Tool Result (paths, data, etc.)
        ↓
Update AgentMemory
        ↓
Continuation Prompt (if more iterations needed)
        ↓
Final <answer> or <think> + <answer>
```
---
## Source Files Reference
- **Tool base class**: `spagent/core/tool.py`
- **All tool implementations**: `spagent/tools/*.py`
- **Prompt templates**: `spagent/core/prompts.py`
- **Agent orchestration**: `spagent/core/spagent.py`
- **Memory management**: `spagent/core/memory.py`
- **Model wrapper**: `spagent/core/model.py`
