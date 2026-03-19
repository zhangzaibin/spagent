---
name: supervision
title: Supervision Tool
summary: >
  Perform object detection (bounding boxes) or instance segmentation
  (masks) on images to identify and analyze objects in the scene.
tool_name: supervision_tool
---

## Supervision Tool

Unified detection and segmentation tool for object analysis.

### When to Use
- Quick object detection with bounding boxes (task='image_det')
- Instance segmentation with pixel-level masks (task='image_seg')
- General scene understanding when you need to identify all objects

### Parameters
- **image_path** (string, required): The path to the input image for processing.
- **task** (string, required): The task type. Must be one of: 'image_det' for object detection (returns bounding boxes) or 'image_seg' for instance segmentation (returns pixel-level masks).

### Call Format
<tool_call>
{"name": "supervision_tool", "arguments": {"image_path": "<image_path>", "task": "<task>"}}
</tool_call>

### Output
Returns detection boxes or segmentation masks with a visualization image.
