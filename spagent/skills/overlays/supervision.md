# Curated guidance for supervision (harvested from PR #157; facts above are generated)

### When to use
- Quick object detection with bounding boxes (task='image_det')
- Instance segmentation with pixel-level masks (task='image_seg')
- General scene understanding when you need to identify all objects

### Tips
- This tool uses a fixed YOLO vocabulary; for open-vocabulary text queries
  prefer detect_objects_tool, for custom class lists prefer
  yoloe_detection_tool.
