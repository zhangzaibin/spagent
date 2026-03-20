from __future__ import annotations

import os
import cv2
import json
import time
from pathlib import Path
from typing import List, Optional, Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="D4RT Mock/Wrapper Server")


class D4RTRequest(BaseModel):
    video_path: Optional[str] = None
    image_paths: Optional[List[str]] = None
    query_mode: str = "both"
    num_frames: int = 32
    save_visualization: bool = True
    output_dir: Optional[str] = None
    query_points: Optional[List[List[float]]] = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_frames_from_video(video_path: str, output_dir: Path, max_frames: int = 32) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_paths = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = max_frames

    step = max(1, total // max_frames)
    idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            frame_path = output_dir / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            saved += 1
            if saved >= max_frames:
                break

        idx += 1

    cap.release()
    return frame_paths


def run_fake_d4rt_pipeline(
    video_path: Optional[str],
    image_paths: Optional[List[str]],
    query_mode: str,
    num_frames: int,
    save_visualization: bool,
    output_dir: Path,
    query_points: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Mock pipeline.
    Later you can replace this with real D4RT inference.
    """

    frames_dir = ensure_dir(output_dir / "frames")

    if video_path:
        frames = extract_frames_from_video(video_path, frames_dir, max_frames=num_frames)
    else:
        frames = image_paths[:num_frames] if image_paths else []

    # 假装生成一些结果文件
    vis_path = output_dir / "vis.mp4"
    scene_path = output_dir / "scene.ply"
    camera_json = output_dir / "cameras.json"
    tracks_json = output_dir / "tracks.json"

    # 生成假的 camera 信息
    cam_data = {
        "num_frames": len(frames),
        "query_mode": query_mode,
        "cameras": [
            {
                "frame_idx": i,
                "pose": [[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, float(i) * 0.01],
                         [0, 0, 0, 1]]
            }
            for i in range(len(frames))
        ]
    }

    # 生成假的 tracking 信息
    track_data = {
        "query_points": query_points or [],
        "tracks": [
            {
                "point_id": i,
                "trajectory": [
                    {"frame_idx": t, "x": 100 + t * 2 + i, "y": 120 + t * 1.5 + i, "z": 1.0 + 0.01 * t}
                    for t in range(len(frames))
                ]
            }
            for i in range(len(query_points or []))
        ]
    }

    with open(camera_json, "w", encoding="utf-8") as f:
        json.dump(cam_data, f, indent=2, ensure_ascii=False)

    with open(tracks_json, "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=2, ensure_ascii=False)

    # 写一个假的点云文件
    with open(scene_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex 4\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        f.write("0 0 0\n")
        f.write("1 0 0\n")
        f.write("0 1 0\n")
        f.write("0 0 1\n")

    # 如果有帧，就简单合成一个视频
    if save_visualization and frames:
        first = cv2.imread(frames[0])
        if first is not None:
            h, w = first.shape[:2]
            writer = cv2.VideoWriter(
                str(vis_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10,
                (w, h)
            )
            for fp in frames:
                img = cv2.imread(fp)
                if img is None:
                    continue
                cv2.putText(
                    img,
                    f"D4RT MOCK | mode={query_mode}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                writer.write(img)
            writer.release()
        else:
            vis_path = None
    else:
        vis_path = None

    return {
        "success": True,
        "summary": (
            f"D4RT mock inference finished. "
            f"Processed {len(frames)} frames in mode={query_mode}."
        ),
        "output_dir": str(output_dir),
        "visualization_path": str(vis_path) if vis_path else None,
        "pointcloud_path": str(scene_path),
        "camera_json": str(camera_json),
        "tracks_json": str(tracks_json),
        "num_frames_processed": len(frames),
    }


# ===== 未来你把真实 D4RT 接到这里 =====
def run_real_d4rt_pipeline(
    video_path: Optional[str],
    image_paths: Optional[List[str]],
    query_mode: str,
    num_frames: int,
    save_visualization: bool,
    output_dir: Path,
    query_points: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    TODO:
    Replace this with the actual D4RT model inference code once your backend is ready.

    Example:
        1. load model
        2. read video / frames
        3. run D4RT
        4. save results
        5. return paths
    """
    raise NotImplementedError("Real D4RT backend is not connected yet.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer")
def infer(req: D4RTRequest):
    try:
        if not req.video_path and not req.image_paths:
            return {"success": False, "error": "No video_path or image_paths provided."}

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(req.output_dir) if req.output_dir else Path("outputs") / f"d4rt_{timestamp}"
        ensure_dir(out_dir)

        use_real = os.environ.get("D4RT_USE_REAL", "0") == "1"

        if use_real:
            result = run_real_d4rt_pipeline(
                video_path=req.video_path,
                image_paths=req.image_paths,
                query_mode=req.query_mode,
                num_frames=req.num_frames,
                save_visualization=req.save_visualization,
                output_dir=out_dir,
                query_points=req.query_points,
            )
        else:
            result = run_fake_d4rt_pipeline(
                video_path=req.video_path,
                image_paths=req.image_paths,
                query_mode=req.query_mode,
                num_frames=req.num_frames,
                save_visualization=req.save_visualization,
                output_dir=out_dir,
                query_points=req.query_points,
            )

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("d4rt_server:app", host="0.0.0.0", port=20034, reload=False)