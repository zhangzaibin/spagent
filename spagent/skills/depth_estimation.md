---
name: depth_estimation
title: Depth Estimation
summary: >
  Generate depth maps to analyze 3D spatial relationships,
  relative distances, and depth ordering of objects in a scene.
tool_name: depth_estimation_tool
---

## Depth Estimation Tool

Use this tool to generate a depth map for an input image. The depth map reveals
the 3D spatial structure: closer objects appear brighter, farther objects darker.

### When to Use
- Comparing relative distances of objects in the scene
- Understanding spatial layout and depth ordering
- Analyzing occlusion relationships between objects
- Determining which object is closer to or farther from the camera

### Parameters
- **image_path** (string, required): The path to the input image for depth estimation.

### Call Format
<tool_call>
{"name": "depth_estimation_tool", "arguments": {"image_path": "<image_path>"}}
</tool_call>

### Output
Returns a depth map image saved alongside the original, plus numerical depth data.
Brighter regions = closer to camera, darker regions = farther away.
