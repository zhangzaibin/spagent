import argparse
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None
    jsonify = None
    request = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_REAL_FRAMES = 8


class _MissingFlaskApp:
    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def run(self, *args, **kwargs):
        raise ImportError("LingBot-Map server requires flask")


app = Flask(__name__) if Flask is not None else _MissingFlaskApp()
config: Dict[str, Any] = {}


def configure(
    repo_path: str,
    model_path: str,
    python_bin: Optional[str] = None,
    viewer_host: str = "127.0.0.1",
    viewer_port: int = 8080,
    work_dir: Optional[str] = None,
    use_sdpa: bool = True,
    camera_num_iterations: int = 1,
    runner_path: Optional[str] = None,
) -> None:
    global config
    config = {
        "repo_path": str(Path(repo_path).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "python_bin": python_bin or os.environ.get("LINGBOT_MAP_PYTHON", "python"),
        "viewer_host": viewer_host,
        "viewer_port": int(viewer_port),
        "work_dir": work_dir or str(Path(tempfile.gettempdir()) / "spagent_lingbot_map_server"),
        "use_sdpa": bool(use_sdpa),
        "camera_num_iterations": max(1, int(camera_num_iterations)),
        "runner_path": str(Path(runner_path).resolve()) if runner_path else str(_default_runner_path()),
    }
    Path(config["work_dir"]).mkdir(parents=True, exist_ok=True)


@app.route("/health", methods=["GET"])
def health_check():
    repo = Path(config.get("repo_path", ""))
    script = repo / "demo.py"
    model_path = Path(config.get("model_path", ""))
    return jsonify(
        {
            "status": "healthy" if script.exists() and model_path.exists() and _runner_path().exists() else "unhealthy",
            "repo_path": str(repo),
            "demo_exists": script.exists(),
            "model_path": str(model_path),
            "model_exists": model_path.exists(),
            "runner_exists": _runner_path().exists(),
            "viewer_url": _viewer_url(),
            "use_sdpa": config.get("use_sdpa", True),
            "camera_num_iterations": config.get("camera_num_iterations", 1),
        }
    )


@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json() or {}
        image_folder = data.get("image_folder")
        images = data.get("images") or []
        if bool(image_folder) == bool(images):
            return jsonify({"success": False, "error": "Provide exactly one of image_folder or images"}), 400

        mask_sky = bool(data.get("mask_sky", False))
        keyframe_interval = max(1, int(data.get("keyframe_interval", 1)))
        max_frames = max(1, int(data.get("max_frames", 128)))
        wait_for_completion = bool(data.get("wait_for_completion", True))

        output_dir = Path(tempfile.mkdtemp(prefix="run_", dir=config["work_dir"]))
        frame_dir = _prepare_frame_dir(
            image_folder=image_folder,
            images=images,
            output_dir=output_dir,
            keyframe_interval=keyframe_interval,
            max_frames=max_frames,
        )
        num_frames = len(_list_images(frame_dir))
        if num_frames < MIN_REAL_FRAMES:
            return jsonify({"success": False, "error": f"LingBot-Map requires at least {MIN_REAL_FRAMES} sampled frames"}), 400
        result = _run_lingbot_map(
            frame_dir=frame_dir,
            output_dir=output_dir,
            mask_sky=mask_sky,
            keyframe_interval=keyframe_interval,
            max_frames=max_frames,
            wait_for_completion=wait_for_completion,
        )
        result["num_frames"] = num_frames
        return jsonify(result)
    except Exception as e:
        logger.error("LingBot-Map inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


def _prepare_frame_dir(
    image_folder: Optional[str],
    images: List[Dict[str, str]],
    output_dir: Path,
    keyframe_interval: int,
    max_frames: int,
) -> Path:
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    if image_folder:
        source = Path(image_folder)
        if not source.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        frames = _list_images(source)[::keyframe_interval][:max_frames]
        if not frames:
            raise ValueError(f"No supported images found in: {image_folder}")
        for idx, frame in enumerate(frames):
            with Image.open(frame) as image:
                image.convert("RGB").save(frame_dir / f"{idx:06d}.png", format="PNG")
        return frame_dir

    for idx, item in enumerate(images[::keyframe_interval][:max_frames]):
        image = _decode_image(item["data"])
        image.save(frame_dir / f"{idx:06d}.png", format="PNG")
    if not _list_images(frame_dir):
        raise ValueError("No uploaded images were decoded")
    return frame_dir


def _run_lingbot_map(
    frame_dir: Path,
    output_dir: Path,
    mask_sky: bool,
    keyframe_interval: int = 1,
    max_frames: int = 128,
    wait_for_completion: bool = True,
) -> Dict[str, Any]:
    repo = Path(config["repo_path"])
    script = repo / "demo.py"
    if not script.exists():
        raise FileNotFoundError(f"LingBot-Map demo.py not found: {script}")
    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"LingBot-Map checkpoint not found: {model_path}")

    log_path = output_dir / "lingbot_map.log"
    if wait_for_completion:
        command = [
            config["python_bin"],
            str(_runner_path()),
            "--repo_path",
            str(repo),
            "--model_path",
            str(model_path),
            "--image_folder",
            str(frame_dir),
            "--output_dir",
            str(output_dir),
            "--camera_num_iterations",
            str(max(1, int(config.get("camera_num_iterations", 1)))),
        ]
        if config.get("use_sdpa", True):
            command.append("--use_sdpa")
        if mask_sky:
            command.append("--mask_sky")
        logger.info("Running LingBot-Map export command: %s", " ".join(command))
        completed = subprocess.run(
            command,
            cwd=repo,
            env={**os.environ, "LINGBOT_MAP_OUTPUT_DIR": str(output_dir)},
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        log_path.write_text((completed.stdout or "") + "\n" + (completed.stderr or ""), encoding="utf-8")
        if completed.returncode != 0:
            return {
                "success": False,
                "error": completed.stderr[-2000:] or completed.stdout[-2000:] or "LingBot-Map command failed",
                "command": command,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
        result = _collect_outputs(output_dir)
        if not result.get("point_cloud_path") or not result.get("trajectory_path"):
            return {
                "success": False,
                "error": "LingBot-Map completed without a point cloud and trajectory",
                "command": command,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
        if not isinstance(result.get("points_count"), int) or result["points_count"] <= 0:
            return {
                "success": False,
                "error": f"LingBot-Map returned invalid points_count: {result.get('points_count')!r}",
                "command": command,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
        result.update(
            {
                "success": True,
                "command": command,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "wait_for_completion": True,
                "log": _encode_file(log_path),
            }
        )
        return result

    command = [
        config["python_bin"],
        str(script),
        "--model_path",
        str(model_path),
        "--image_folder",
        str(frame_dir),
        "--port",
        str(config.get("viewer_port", 8080)),
        "--keyframe_interval",
        str(max(1, int(keyframe_interval))),
        "--camera_num_iterations",
        str(max(1, int(config.get("camera_num_iterations", 1)))),
    ]
    if config.get("use_sdpa", True):
        command.append("--use_sdpa")
    if mask_sky:
        command.append("--mask_sky")
    logger.info("Running LingBot-Map viewer command: %s", " ".join(command))
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env={**os.environ, "LINGBOT_MAP_OUTPUT_DIR": str(output_dir)},
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return {
        "success": True,
        "command": command,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "viewer_url": _viewer_url(),
        "process_id": process.pid,
        "wait_for_completion": False,
    }


def _collect_outputs(output_dir: Path) -> Dict[str, Any]:
    files = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        if "frames" in path.relative_to(output_dir).parts:
            continue
        files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    result: Dict[str, Any] = {}
    metadata_path = output_dir / "reconstruction_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result["metadata_json"] = _encode_file(metadata_path)
        result["metadata_path"] = str(metadata_path)
        result["points_count"] = metadata.get("points_count")
        result["checkpoint_path"] = metadata.get("checkpoint_path")
        result["checkpoint_bytes"] = metadata.get("checkpoint_bytes")
    for path in files:
        suffix = path.suffix.lower()
        name = path.name.lower()
        if "preview_path" not in result and suffix in {".png", ".jpg", ".jpeg"}:
            result["preview_image"] = _encode_file(path)
            result["preview_path"] = str(path)
        elif "trajectory_path" not in result and suffix == ".json" and ("traj" in name or "pose" in name):
            result["trajectory_json"] = _encode_file(path)
            result["trajectory_path"] = str(path)
        elif "point_cloud_path" not in result and suffix in {".ply", ".pcd"}:
            result["point_cloud"] = _encode_file(path)
            result["point_cloud_path"] = str(path)
        elif "video_path" not in result and suffix == ".mp4":
            result["video"] = _encode_file(path)
            result["video_path"] = str(path)
    if result.get("points_count") is None and result.get("point_cloud_path"):
        result["points_count"] = _points_count_from_ply(Path(result["point_cloud_path"]))
    return result


def _list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=_frame_sort_key,
    )


def _frame_sort_key(path: Path):
    if path.stem.isdigit():
        return 0, int(path.stem)
    return 1, path.name.lower()


def _decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64, validate=True))).convert("RGB")


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _viewer_url() -> str:
    return f"http://{config.get('viewer_host', '127.0.0.1')}:{config.get('viewer_port', 8080)}"


def _runner_path() -> Path:
    return Path(config.get("runner_path") or _default_runner_path())


def _default_runner_path() -> Path:
    return Path(__file__).with_name("lingbot_map_runner.py")


def _points_count_from_ply(path: Path) -> Optional[int]:
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.rsplit(" ", 1)[-1])
            if line == "end_header":
                break
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LingBot-Map Server")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the official LingBot-Map repository")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a LingBot-Map checkpoint")
    parser.add_argument("--port", type=int, default=20040, help="Port to run this SPAgent wrapper server on")
    parser.add_argument("--python_bin", type=str, default=None, help="Python executable for the LingBot-Map environment")
    parser.add_argument("--viewer_host", type=str, default="127.0.0.1", help="Host shown in returned viewer URL")
    parser.add_argument("--viewer_port", type=int, default=8080, help="Viser viewer port used by LingBot-Map demo.py")
    parser.add_argument("--work_dir", type=str, default=None, help="Directory for server-side run outputs")
    parser.add_argument("--camera_num_iterations", type=int, default=1, help="Camera optimization iterations for demo.py")
    parser.add_argument("--use_sdpa", action="store_true", default=True, help="Use PyTorch SDPA backend for demo.py")
    parser.add_argument("--no_use_sdpa", action="store_false", dest="use_sdpa", help="Use FlashInfer backend for demo.py")
    args = parser.parse_args()

    configure(
        repo_path=args.repo_path,
        model_path=args.model_path,
        python_bin=args.python_bin,
        viewer_host=args.viewer_host,
        viewer_port=args.viewer_port,
        work_dir=args.work_dir,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    app.run(host="0.0.0.0", port=args.port, debug=False)
