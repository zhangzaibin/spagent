# MapAnything Tool - Test & Usage

This folder contains a simple setup to test the **MapAnything** tool for multi-view 3D reconstruction from images.

## Folder Structure

```
test/map_anything/
├── output/           # Output folder for MapAnything inference results
└── README.md         # This file
```

## Prerequisites

* Python >= 3.8
* PyTorch with CUDA support
* MapAnything server code located at `spagent/external_experts/map_anything/map_anything_server.py`
* Required Python packages (can use your existing spagent environment)

## Starting the MapAnything Server

Before running inference, you must start the MapAnything server:

```bash
python spagent/external_experts/map_anything/map_anything_server.py
```

This will start a local server at `http://127.0.0.1:20033`. Leave it running while sending inference requests.

## Running Inference

1. Make sure the output folder exists:

```bash
mkdir -p output
```

2. Run MapAnything inference with example images (some times havd to use path, not relative path):

```bash
curl -X POST http://127.0.0.1:20033/infer \
-H "Content-Type: application/json" \
-d '{
    "image_paths": [
        "assets/dog.jpeg",
        "assets/example.png"
    ],
    "output_dir": "output"
}'
```

* `image_paths`: List of image paths relative to `test/map_anything/`
* `output_dir`: Relative directory to save inference results

## Output

After successful inference, the results are saved in the `output/` folder:

```
output/
├── view_000/
│   ├── pts3d.npy
│   ├── pts3d_cam.npy
│   ├── depth_along_ray.npy
│   ├── cam_trans.npy
│   ├── cam_quats.npy
│   ├── conf.npy
│   ├── mask.npy
│   ├── intrinsics.npy
│   └── metadata.json
└── view_001/
    └── ...
```

* `.npy` files: core data for 3D points, depth, camera poses, masks, etc.
* `metadata.json`: contains previews, metric scaling, and additional info

## Testing / Visualizing

You can quickly visualize 3D points using **Open3D**:

```python
import numpy as np
import open3d as o3d

pts = np.load("output/view_000/pts3d.npy")
pts = pts.reshape(-1, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
o3d.visualization.draw_geometries([pcd])
```

## Notes

* Make sure the MapAnything server is running before calling the inference API.
* You can change the `output_dir` to any relative folder you prefer.
* For multiple views, each view will have a separate `view_XXX/` folder with corresponding outputs.
