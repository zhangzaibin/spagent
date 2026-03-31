---
name: moondream
title: Moondream Vision
summary: >
  Locate objects in images by pointing. Specify object names and get
  their pixel coordinates. Supports single or multiple objects.
tool_name: moondream_tool
---

## Moondream Vision Tool

Locate objects in an image by name, returning their pixel coordinates.

### When to Use
- Finding the exact position of named objects in an image
- Locating multiple objects at once (comma-separated names)
- Getting pixel coordinates for spatial reasoning

### Parameters
- **image_path** (string, required): The path to the input image.
- **task** (string, required): The task to perform. Currently only supports "point" (locate objects). Supports both single objects like 'car' and multiple objects like 'car, person, tree'.
- **object_name** (string, required): Name of the object(s) to locate. Can be a single object like 'car' or multiple objects separated by commas like 'car, person, tree'.

### Call Format
<tool_call>
{"name": "moondream_tool", "arguments": {"image_path": "<image_path>", "task": "<task>", "object_name": "<object_name>"}}
</tool_call>

### Output
Returns pixel coordinates for each detected object and a visualization image with marked points.
