# Curated guidance for segmentation (harvested from PR #157; facts above are generated)

### When to use
- Isolating specific objects or regions in a scene
- Counting distinct objects by their boundaries
- Analyzing object shapes, sizes, and spatial extent
- Comparing areas occupied by different objects

### Tips
- Guide the mask with `point_coords` + `point_labels` (1=foreground,
  0=background) or constrain it with a `box`; omit all three for automatic
  segmentation.
