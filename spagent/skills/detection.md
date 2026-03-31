---
name: detection
title: Object Detection (GroundingDINO)
summary: >
  Detect and locate objects in images using open-vocabulary detection
  (GroundingDINO). Describe what to find in natural language.
tool_name: detect_objects_tool
---

## Object Detection Tool (GroundingDINO)

Detect objects in an image by describing them in natural language.

### When to Use
- Locating specific objects mentioned in a question
- Counting instances of a particular object type
- Finding bounding box coordinates for spatial reasoning
- Verifying whether certain objects exist in the scene

### Parameters
- **image_path** (string, required): The path to the input image for object detection.
- **text_prompt** (string, required): Text description of objects to detect. Use " . " to separate multiple object types (e.g. "person . car . dog").
- **box_threshold** (number, optional, default=0.35): Confidence threshold for box detection. Higher values return fewer but more confident detections.
- **text_threshold** (number, optional, default=0.25): Confidence threshold for text matching. Controls how strictly the detected objects must match the text description.

### Call Format
<tool_call>
{"name": "detect_objects_tool", "arguments": {"image_path": "<image_path>", "text_prompt": "<text_prompt>", "box_threshold": "<box_threshold>", "text_threshold": "<text_threshold>"}}
</tool_call>

### Output
Returns detected bounding boxes with labels and confidence scores, plus a visualization image.
