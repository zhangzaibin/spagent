# Curated guidance for yoloe (harvested from PR #157; facts above are generated)

### When to use
- Detecting specific object categories you define (custom vocabulary via
  `class_names`, e.g. ['person', 'car', 'dog'])
- High-speed detection on images or video frames

### Tips
- Detection only (bounding boxes) — it does not return segmentation masks.
