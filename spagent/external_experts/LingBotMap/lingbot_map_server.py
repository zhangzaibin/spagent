import argparse
import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import time
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
) -> None:
    global config
    config = {
        "repo_path": str(Path(repo_path).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "python_bin": python_bin or os.environ.get("LINGBOT_MAP_PYTHON", "python"),
        "viewer_host": viewer_host,
        "viewer_port": int(viewer_port),
        "work_dir": work_dir or str(Path(tempfile.gettempdir()) / "spagent_lingbot_map_server"),
    }
    Path(config["work_dir"]).mkdir(parents=True, exist_ok=True)


@app.route("/health", methods=["GET"])
def health_check():
    repo = Path(config.get("repo_path", ""))
    script = repo / "demo.py"
    model_path = Path(config.get("model_path", ""))
    return jsonify(
        {
            "status": "healthy" if script.exists() and model_path.exists() else "unhealthy",
            "repo_path": str(repo),
            "demo_exists": script.exists(),
            "model_path": str(model_path),
            "model_exists": model_path.exists(),
            "viewer_url": _viewer_url(),
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
        wait_for_completion = bool(data.get("wait_for_completion", False))

        output_dir = Path(data.get("output_dir") or Path(config["work_dir"]) / f"run_{int(time.time() * 1000)}")
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = _prepare_frame_dir(
            image_folder=image_folder,
            images=images,
            output_dir=output_dir,
            keyframe_interval=keyframe_interval,
            max_frames=max_frames,
        )
        result = _run_lingbot_map(
            frame_dir=frame_dir,
            output_dir=output_dir,
            mask_sky=mask_sky,
            wait_for_completion=wait_for_completion,
        )
        result["num_frames"] = len(_list_images(frame_dir))
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
            shutil.copy2(frame, frame_dir / f"{idx:06d}{frame.suffix.lower()}")
        return frame_dir

    for idx, item in enumerate(images[::keyframe_interval][:max_frames]):
        filename = Path(item.get("filename") or f"{idx:06d}.png").name
        suffix = Path(filename).suffix.lower() if Path(filename).suffix else ".png"
        image = _decode_image(item["data"])
        image.save(frame_dir / f"{idx:06d}{suffix}")
    if not _list_images(frame_dir):
        raise ValueError("No uploaded images were decoded")
    return frame_dir


def _run_lingbot_map(frame_dir: Path, output_dir: Path, mask_sky: bool, wait_for_completion: bool) -> Dict[str, Any]:
    repo = Path(config["repo_path"])
    script = repo / "demo.py"
    if not script.exists():
        raise FileNotFoundError(f"LingBot-Map demo.py not found: {script}")
    model_path = Path(config["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"LingBot-Map checkpoint not found: {model_path}")

    log_path = output_dir / "lingbot_map.log"
    command = [
        config["python_bin"],
        str(script),
        "--model_path",
        str(model_path),
        "--image_folder",
        str(frame_dir),
    ]
    if mask_sky:
        command.append("--mask_sky")

    logger.info("Running LingBot-Map command: %s", " ".join(command))
    if wait_for_completion:
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
        result.update(
            {
                "success": True,
                "command": command,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "viewer_url": _viewer_url(),
                "wait_for_completion": True,
            }
        )
        return result

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
    return result


def _list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def _decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _viewer_url() -> str:
    return f"http://{config.get('viewer_host', '127.0.0.1')}:{config.get('viewer_port', 8080)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LingBot-Map Server")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the official LingBot-Map repository")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a LingBot-Map checkpoint")
    parser.add_argument("--port", type=int, default=20038, help="Port to run this SPAgent wrapper server on")
    parser.add_argument("--python_bin", type=str, default=None, help="Python executable for the LingBot-Map environment")
    parser.add_argument("--viewer_host", type=str, default="127.0.0.1", help="Host shown in returned viewer URL")
    parser.add_argument("--viewer_port", type=int, default=8080, help="Viser viewer port used by LingBot-Map demo.py")
    parser.add_argument("--work_dir", type=str, default=None, help="Directory for server-side run outputs")
    args = parser.parse_args()

    configure(
        repo_path=args.repo_path,
        model_path=args.model_path,
        python_bin=args.python_bin,
        viewer_host=args.viewer_host,
        viewer_port=args.viewer_port,
        work_dir=args.work_dir,
    )
    app.run(host="0.0.0.0", port=args.port, debug=False)
