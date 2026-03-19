---
name: segmentation
title: Image Segmentation (SAM2)
summary: >
  Segment objects in images using SAM2. Supports point-guided,
  box-guided, and automatic segmentation for precise region analysis.
tool_name: segment_image_tool
---

## Image Segmentation Tool (SAM2)

Segment objects in an image to isolate specific regions for analysis.

### When to Use
- Isolating specific objects or regions in a scene
- Counting distinct objects by their boundaries
- Analyzing object shapes, sizes, and spatial extent
- Comparing areas occupied by different objects

### Parameters
- **image_path** (string, required): The path to the input image for segmentation.
- **point_coords** (array, optional): List of point coordinates [[x1,y1], [x2,y2], ...] to guide segmentation towards specific locations.
- **point_labels** (array, optional): List of point labels corresponding to point_coords. Use 1 for foreground (object) and 0 for background (not object).
- **box** (array, optional): Bounding box coordinates [x1,y1,x2,y2] to constrain the segmentation region.

### Call Format
<tool_call>
{"name": "segment_image_tool", "arguments": {"image_path": "<image_path>", "point_coords": "<point_coords>", "point_labels": "<point_labels>", "box": "<box>"}}
</tool_call>

### Output
Returns segmentation masks and a visualization image with highlighted regions.
