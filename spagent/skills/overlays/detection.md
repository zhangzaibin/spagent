# Curated guidance for detection (harvested from PR #157; facts above are generated)

### When to use
- Locating specific objects mentioned in a question (open vocabulary —
  describe what to find in natural language)
- Counting instances of a particular object type
- Finding bounding-box coordinates for spatial reasoning
- Verifying whether certain objects exist in the scene

### Tips
- Use " . " to separate multiple object types in `text_prompt`
  (e.g. "person . car . dog").
- Raise `box_threshold` for fewer but more confident detections.
