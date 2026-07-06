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
> **Registered names & catalog status** (verified against `spagent/tools/catalog.py`):
> - The **Tool Name** column below is the *registered* name (`ToolCatalogEntry.tool_name`) exposed to the model, not the Python class or filename.
> - The four generation video tools and the Sana image tool are registered under **namespaced** names: `image_generation_sana_tool`, `video_generation_veo_tool`, `video_generation_sora_tool`, `video_generation_wan_tool`, `video_generation_vace_tool`.
> - **`flowseek_tool`, `paddleocr_vl_tool`, and `wilddet3d_tool` are NOT in `TOOL_CATALOG`** — their classes exist and are exported from `spagent/tools/__init__.py`, but they are not imported into `catalog.py`, so `build_all_tools()` never constructs them. They are documented here for completeness and marked 🚧 **unregistered** in the table.
> - `detect_objects_tool` (`ObjectDetectionTool`) **is** registered and is included below.

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
| **flowseek_tool** 🚧 | Optical Flow *(unregistered — not in TOOL_CATALOG)* | Optical flow estimation between two images. | **image1_path** (required, string) | **image2_path** (required, string) | ❌ None | **output_path** (optional, string): Save path | ✅ **flow_magnitude_mean** (float): Average pixel motion<br>✅ **output_path** (string): Colorized flow visualization | ⚠️ Raw flow vectors (u,v per pixel), flow arrows overlay, motion boundaries, dominant direction, object motion masks |
| **mapanything_tool** | 3D Reconstruction | 3D reconstruction using MapAnything. Similar to Pi3. | **image_path** (required, list of strings) | ❌ None | ❌ None | **azimuth_angle** (optional, -180 to 180; only `image_path` is `required` in the schema)<br>**elevation_angle** (optional, -90 to 90)<br>**rotation_reference_camera** (optional, default=1)<br>**camera_view** (optional, default=false) | ✅ Same as pi3_tool: **ply_filename**, **points_count**, **camera_views** | ⚠️ All pi3_tool enhancements + semantic labels per point, structure-from-motion metadata |
| **moondream_tool** | Point Grounding | Lightweight vision-language: pointing to objects. | **image_path** (required, string) | ❌ None | **object_name** (required, string): Single or comma-separated multiple objects | **task** (required, enum: ['point']) | ✅ **Single**: points [{x,y,confidence}], output_path<br>✅ **Multiple**: all_points {obj:[{x,y,confidence}]}, color_mapping, total_points, output_path<br>✅ Normalized coords [0-1] | ⚠️ Bounding boxes from points, crops around each point, distance metrics between objects |
| **paddleocr_vl_tool** 🚧 | OCR/Document *(unregistered — not in TOOL_CATALOG)* | Document-level OCR and structured recognition using PaddleOCR-VL-1.5. | **image_path** (required, string) | ❌ None | ❌ None | **task** (optional, default='ocr', enum: ['ocr', 'table', 'chart', 'formula', 'spotting', 'seal']) | ✅ **text** (string): Extracted text/structured output (format varies by task)<br>✅ **task** (string): Task mode used | ⚠️ Text bounding boxes, confidence scores, structured JSON for tables/charts, reading order, font/style metadata |
| **yoloe_detection_tool** | Detection | Detect objects with YOLO-E using custom class names. | **image_path** (required, string) | ❌ None | ❌ None | **task** (required, enum: ['image', 'video'])<br>**class_names** (required, array of strings) | ✅ **boxes** (list): Bbox coordinates<br>✅ **labels** (list): Detected class names<br>✅ **confidence** (list): Scores<br>✅ **class_names** (list): Input classes<br>✅ **vis_path** (string): Annotated visualization | ⚠️ Cropped object images, normalized coords, multiple format options (COCO, YOLO, Pascal VOC) |
| **wilddet3d_tool** 🚧 | Detection (3D) *(unregistered — not in TOOL_CATALOG)* | Promptable 3D object detection from single RGB image. | **image_path** (required, string) | ❌ None | **prompt_text** (optional, default='object', string): Object description; ignored when input_boxes/input_points given | **input_boxes** (optional, list): takes priority over prompt_text<br>**input_points** (optional, list): takes priority over prompt_text | ✅ Annotated image with 2D and 3D bounding boxes<br>✅ 3D location estimates | ⚠️ Structured 3D coordinates, depth estimates per object, 3D mesh generation, relative spatial positions |
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
| Tests (`test/test_tool.py`) | today lenient (payload **or** a vis path) — the required-raw contract below is the *intended* bar the tests should tighten toward |

The loop consumes the envelope + a rendering; the **raw payload is required for downstream analysis, chaining, and reproducible large-scale experiments**, even where the current tests don't yet assert it.

### Universal Result Envelope (required for EVERY tool)

| Field | Type | When required | Purpose |
|-------|------|---------------|---------|
| `success` | bool | always | gates all downstream handling |
| `description` | str | always (may be empty) | natural-language result fed back to the VLM |
| `error` | str | when `success=False` | failure reason (logged + surfaced to the model) |
| **a rendered visualization** — one of `output_path` \| `vis_path` \| `crop_paths` \| `overlay_path` | str / list | any tool that produces an image/video | how the result reaches the VLM; a *view* of the raw payload (`.mp4` allowed in `output_path`) |

### Category Output Registry

Two required layers on top of the envelope: the **raw payload** (required; pick any *one of* the accepted forms) and, for visual tools, a **rendering**. Optional fields are free per tool.

| # | Category | Tools (registered) | Required raw payload — **ONE OF** | Common optional (per-tool) |
|---|----------|--------------------|-----------------------------------|----------------------------|
| 1 | Detection | detect_objects, zoom_object, localize_object, yolo26, qwenvl_detection, yoloe_detection, supervision (det), wilddet3d 🚧 | `detections`/`boxes`+`labels`, each box **one of** {pixel `xyxy`, normalized `xyxy`, normalized `cxcywh`} | `confidence`, `image_width`/`height`, `class_id`, `crop_paths`, `masks`, `3d_coords` |
| 2 | Segmentation | segment_image, supervision (seg) | a mask **one of** {binary array, mask file path, `polygon` coords, RLE} | `area`, `bbox`, `class_name`, `shape` |
| 3 | Image Generation | image_generation_sana | image path **one of** {`output_path`, `image_paths[]`} | `seed`, `model`, `size`, `file_size_bytes` |
| 4 | Video Generation | video_generation_{veo,sora,wan,vace} | `output_path` (.mp4 path) | `duration`, `resolution`, `fps`, `codec`, `result_dir`, `frame_paths` |
| 5 | 3D Reconstruction | pi3, pi3x, mapanything, vggt | point cloud **one of** {`ply_filename` path, raw points array} | `points_count`, `view_count`, `camera_views`, `camera_poses`, `mesh_path`, `scale_info` |
| 6 | Point Grounding | molmo2, moondream | `points`, each **one of** {normalized `(x,y)`, pixel `(x_px,y_px)`} | `confidence`, `labels`, `image_width`/`height`, `raw_text` |
| 7 | Depth | depth_estimation | depth field **one of** {raw array (`depth_data`), depth file path (.npy/.exr)} | `shape`, `value_range`, `confidence_map`, `normal_map` |
| 8 | Orientation | orient_anything_v2 | orientation **one of** {euler {`azimuth`,`elevation`,`rotation`}, `rotation_matrix`, `quaternion`} | `symmetry_order`, `confidence`, axes overlay |
| 9 | Optical Flow | flowseek 🚧 | flow field **one of** {raw `(u,v)` array, flow file path} | `flow_magnitude_mean` + stats, `motion_boundaries`, `confidence_map` |
| 10 | OCR / Document | paddleocr_vl 🚧 | `text` (string; structured string for table/chart/formula) | `text_boxes`, `confidence`, `structured_data`, `reading_order` |

🚧 = class exists but not in `TOOL_CATALOG` (see the note at the top of this doc). The rendered visualization (envelope) is additionally expected for every visual category above.

### Per-category detail

Each block gives the required raw payload (one-of forms), the optional extras, *why* it's the minimal informative unit, and the honest compliance status against the **raw** bar.

#### 1. Detection
- **Required raw (one of):** `boxes`+`labels` with each box as pixel `xyxy` **or** normalized `xyxy` **or** normalized `cxcywh`. **Rendering:** annotated image / crops.
- **Optional:** `confidence`, `image_width`/`image_height`, `class_id`, `crop_paths`, `masks`, `3d_coords`.
- **Why minimal:** a labelled box is the smallest unit that says *what* + *where*; format is unconstrained so no tool is forced to convert.
- **Canonical carrier / format (GroundingDINO tools):** the raw boxes live in the **`detections`** list — `[{id, bbox, confidence, label}]` — where `bbox` is **normalized `cxcywh` in [0,1]** (traced from `groundingdino.predict()` → `grounding_dino_server.py:202-209`). ⚠️ The parallel top-level `boxes`/`labels`/`confidence` arrays that `zoom_object`/`localize_object` also return are populated from `raw.get("boxes"…)`, which the *real* client never provides — so **they are empty on the real path** and only filled (as pixel `xyxy`) by the missing-mock `_SimpleMock` fallback. Read `detections`, not the top-level arrays. One-line fix: derive `boxes`/`labels` from `detections` (`detection_tool.py:452-454` & `:638-640`).
- **Status:** 🟢 yolo26 (pixel), qwenvl (normalized), yoloe, supervision. 🟢 **zoom_object & localize_object** — both *do* expose `detections[].bbox` (normalized cxcywh); the earlier "crops-only / format-unclear" call was wrong (only the redundant top-level arrays are broken). 🟢 wilddet3d 🚧 — returns structured `boxes2d` (pixel xyxy) + `boxes3d` (camera frame); see §9-style note below. Net: **all detection tools carry a raw box payload.**

#### 2. Segmentation
- **Required raw (one of):** binary mask array **or** mask file path **or** `polygon` coords **or** RLE. **Rendering:** overlay / combined image.
- **Optional:** `area`, `bbox`, `class_name`, `shape`.
- **Why minimal:** the mask (in any form) is the information; area/bbox are computable from it.
- **Status:** 🟢 segment_image (`mask_path`/array), supervision (masks).

#### 3. Image Generation
- **Required raw (one of):** `output_path` **or** `image_paths[]`.
- **Optional:** `seed`, `model`, `size`, `file_size_bytes`.
- **Why minimal:** the generated image *is* the payload; `seed`/`model` aid reproducibility but don't gate success.
- **Status:** 🟢 image_generation_sana (path + rich metadata).

#### 4. Video Generation
- **Required raw:** `output_path` (`.mp4` path).
- **Optional:** `duration`, `resolution`, `fps`, `codec`, `result_dir`, `frame_paths`.
- **Why minimal:** the file is the payload; the loop auto-extracts frames from it (`spagent.py:331-337`).
- **Status:** 🟢 veo, sora, wan, vace. Enrichment TODO: probe the file for real `fps`/`codec`/`resolution`.

#### 5. 3D Reconstruction
- **Required raw (one of):** `ply_filename` (point cloud path) **or** a raw points array. **Rendering:** `output_path` rendered view(s).
- **Optional:** `points_count`, `view_count`, `camera_views`, `camera_poses`, `mesh_path`, `scale_info`.
- **Why minimal:** the point cloud is the reconstruction; poses/mesh are enrichments.
- **Status:** 🟢 pi3, pi3x, mapanything, vggt (all emit `ply_filename`).

#### 6. Point Grounding
- **Required raw (one of):** `points`, each as normalized `(x,y)` **or** pixel `(x_px,y_px)`. **Rendering:** annotated overlay.
- **Optional:** `confidence`, `labels`, `image_width`/`image_height`, `raw_text`.
- **Why minimal:** the point coordinates are the payload; dims are only needed to convert between the two forms.
- **Status:** 🟢 both. **molmo2 → pixel** coords (`point_utils.py:177-195` scale the model's 0–1000/0–100 output by width/height; the tool already loads `sizes` at `molmo2_tool.py:99-102` but doesn't surface them). **moondream → normalized [0,1]** (native Moondream format). Enrichment: molmo2 could attach `image_width`/`height` (already in hand) for free pixel↔normalized conversion.

#### 7. Depth Estimation
- **Required raw (one of):** raw depth array (`depth_data`) **or** a depth file path (.npy/.exr). **Rendering:** `output_path` colored depth map.
- **Optional:** `shape`, `value_range`, `confidence_map`, `normal_map`.
- **Why minimal:** the depth *field* is the information — the colored map is only a rendering of it, so the array/file is required, not the picture.
- **Status:** 🟢 depth_estimation (returns `depth_data` + `shape`).

#### 8. Orientation Estimation
- **Required raw (one of):** euler {`azimuth`, `elevation`, `rotation`} **or** `rotation_matrix` **or** `quaternion` (also echoed in `description`).
- **Optional:** `symmetry_order`, `confidence`, axes overlay.
- **Why minimal:** a single rotation, in any standard parametrization, is the payload.
- **Status:** 🟢 orient_anything_v2 (euler + symmetry).

#### 9. Optical Flow
- **Required raw (one of):** raw `(u,v)` flow array **or** a flow file path. **Rendering:** `output_path` colorized flow.
- **Optional:** `flow_magnitude_mean` + statistics, `motion_boundaries`, `confidence_map`.
- **Why minimal:** the per-pixel flow field is the information; a magnitude scalar is a *summary*, not the payload.
- **Status:** 🔴 flowseek 🚧 — the **only genuinely non-compliant tool**. Returns only `flow_magnitude_mean` + a visualization, **no raw flow field**. FIX (easy): the dense `(H,W,2)` `u,v` array already exists as `flow_np` at `flowseek_local.py:187` and is thrown away at `:201-213` — `np.save()` it and add `flow_path` to the return dict (1–3 lines, no blockers in **local** mode). Server mode adds two wrinkles: HTTP payload is ~16 MB/1080p frame (prefer a served `.npy` path or compressed `.npz` over base64 inlining), and the current `tempfile`+`os.unlink` cleanup (`flowseek_server.py:74-79`) would delete a returned path, so it needs a persistent/served dir.

#### 10. OCR / Document
- **Required raw:** `text` (string; a structured string — Markdown/LaTeX/JSON — for table/chart/formula tasks).
- **Optional:** `text_boxes`, `confidence`, `structured_data`, `reading_order`, `font_metadata`.
- **Why minimal:** extracted text is the deliverable; boxes/order are enrichments.
- **Status:** 🟢 paddleocr_vl 🚧 (returns `text`).

---

## Summary: Standardization Roadmap (against the required-raw bar)

Compliance is measured on the **raw payload**, not on whether a picture came back. After source-level verification of every tool, **the required-raw bar is met by all but one tool**:

- **🔴 Genuinely non-compliant (1):**
  - `flowseek_tool` 🚧 — returns `flow_magnitude_mean` + viz only; the dense `(u,v)` field is computed then discarded (`flowseek_local.py:187`→`:201-213`). **Easy fix**: `np.save(flow_np)` + return `flow_path` (local mode: 1–3 lines; server mode: use a served `.npy`/`.npz` path due to ~16 MB payload).
- **🟢 Compliant (all others) — with enrichment TODOs where the raw data is present but thin:**
  - **Detection** (yolo26, qwenvl, yoloe, supervision, zoom_object, localize_object): all carry boxes. Two follow-ups, both bugs not gaps: (1) fix the empty top-level `boxes`/`labels` arrays on the GroundingDINO path (derive from `detections`, `detection_tool.py:452-454`/`:638-640`); (2) ship a real `mock_gdino_service.py` — the current `_SimpleMock` fallback emits a *different* format (pixel `xyxy`, top-level `boxes`, no `detections`) than the real server (normalized `cxcywh` in `detections`).
  - **wilddet3d** 🚧: already returns `boxes2d` (pixel xyxy) + `boxes3d` (camera frame) + `scores`; enrichment = stop dropping `class_ids` (labels), `scores_2d`/`scores_3d`, and per-box depth from `depth_maps` (`wilddet3d_local.py:151` unpacks all 7 model outputs, surfaces only 3).
  - **Point grounding**: molmo2 (pixel) & moondream (normalized) both compliant; molmo2 can surface the `image_width`/`height` it already computes.
  - **Also compliant, no action:** segment_image, supervision (seg); image_generation_sana; all video tools; all 3D tools; depth_estimation; orient_anything_v2; paddleocr_vl.

> **Correction note:** an earlier draft marked `zoom_object`, `wilddet3d`, and `localize_object` as non-compliant/unclear. Source inspection disproved that — each already emits a structured box payload; the real issues are the two detection *bugs* above (empty top-level arrays, missing/inconsistent mock), not a missing category payload.

### Architecture (branch goal)

The branch delivers four things: (1) tools **categorized & modularized**, (2) a **unified per-category output format** (required + optional), (3) a **clear registry** tying category → tools → contract, and (4) a **separate parse/render module** that projects a standardized result into the vLLM-facing message.

**Two-stage data flow — produce vs. project:**

```
tool.call()                      parse/render module                 VLM
──────────                       ───────────────────                 ───
ToolResult(envelope              select fields per user config,      formatted
 + required payload              apply per-category DEFAULT          tool-output
 + all optional it can)   ──►    (or "give-everything" preset)  ──►  text/images
 [PRODUCE: rich, lossless]       [PROJECT: filtered, formatted]
```

- **Tools produce, they do not format for the model.** Each `tool.call()` returns a standardized `ToolResult`: the universal envelope + the category's required payload + whatever optional fields that tool can afford — as rich and lossless as possible. Tools never decide what the VLM sees.
- **The parse/render module projects.** A single module converts a `ToolResult` into the vLLM tool-output format, **including only the fields the user configured**. Each category ships a **default projection** (a minimal, sensible field set) plus a **unified "all" preset** (dump everything). Config can override per category or per tool.
- **Why the split:** the same rich result can be rendered thinly for cheap large-scale eval or fully for debugging, without touching tool code; required stays minimal for tests + experiments while optional gives each tool head-room.

### Registry implementation sketch
1. **One `ToolResult` envelope** dataclass (`success`, `description`, `error`, visualization fields) shared by every tool.
2. **Per-category typed payload** (`DetectionPayload`, `Mask`, `Points`, `Orientation`, …) whose constructor accepts *any one of* the allowed forms and exposes computed helpers (area from mask, dims from array, pixel↔normalized).
3. **Required raw payload gates category compliance**; optional fields live on the payload but never gate `success`.
4. **Parse/render module is separate & config-driven:** `render(result, config)` → vLLM message; per-category default projection + a unified all-fields preset; drives both the text summary and which visualizations attach.
5. **Registry** maps `category → {tools, required contract, optional fields, default projection}` — the single source of truth this doc's tables describe.

Guiding principle: **required = the category's raw informative payload (in any one accepted form) + the universal envelope; optional = everything else; the render module decides what of that reaches the model.**

### Validation plan
- **During dev — smoke tests only, < 10 GB VRAM.** Exercise the envelope/payload/render plumbing with (a) `use_mock=True` tools (zero VRAM) and (b) one tiny real local tool — **`yolo26_tool`** (local YOLO, no server, ~1–2 GB, `accepts_use_mock=False`) — to prove the real→standardized→render path end-to-end without disturbing GPU jobs.
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
