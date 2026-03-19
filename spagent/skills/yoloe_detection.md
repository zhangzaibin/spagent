---
name: yoloe_detection
title: YOLO-E Detection
summary: >
  Advanced object detection using YOLO-E with custom class names.
  Supports both image and video processing with high accuracy.
tool_name: yoloe_detection_tool
---

## YOLO-E Detection Tool

Detect objects with user-specified class names using YOLO-World Enhanced.

### When to Use
- Detecting specific object categories you define
- High-speed detection on images or video frames
- When you need bounding-box-level detection with custom vocabulary

### Parameters
- **image_path** (string, required): The path to the input image for YOLO-E processing.
- **task** (string, required): The processing task type. Must be one of: 'image' for single image object detection, or 'video' for video frame processing.
- **class_names** (array of strings, required): List of object class names to detect (e.g., ['person', 'car', 'dog', 'cat']). YOLO-E can detect custom objects based on text descriptions.

### Call Format
<tool_call>
{"name": "yoloe_detection_tool", "arguments": {"image_path": "<image_path>", "task": "<task>", "class_names": "<class_names>"}}
</tool_call>

### Output
Returns detected bounding boxes with class labels and a visualization image.
Note: This tool only performs detection (bounding boxes), not segmentation.
