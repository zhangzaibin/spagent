# Curated guidance for pi3x (harvested from PR #157; facts above are generated)

### When to use
- Same use cases as pi3, but when higher-quality 3D reconstruction is needed
  (smoother point clouds, approximate metric scale)
- Understanding 3D spatial relationships between objects
- Viewing a scene from novel angles not in the original images
- Questions about spatial arrangement, orientation, or relative positioning

### Key concepts
- Input images are at (azimuth=0, elevation=0). Do NOT request (0,0) again.
- azimuth: left-right rotation (-180 to 180). Negative=left, positive=right.
- elevation: up-down rotation (-90 to 90). Negative=down, positive=up.
- rotation_reference_camera: which camera to rotate around (1-based index)
- camera_view: True=first-person from that camera, False=bird's-eye view

### Recommended angles
- Left: azimuth=-45 or -90; Right: azimuth=45 or 90
- Top: elevation=30 to 60 (great for relative positioning)
- Back: azimuth=180 or +/-135; Diagonal: e.g. (45, 30)
