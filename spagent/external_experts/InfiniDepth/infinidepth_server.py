import argparse
import base64
import io
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None
    jsonify = None
    request = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _MissingFlaskApp:
    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def run(self, *args, **kwargs):
        raise ImportError("InfiniDepth server requires flask")


app = Flask(__name__) if Flask is not None else _MissingFlaskApp()

config: Dict[str, Any] = {}


def configure(
    repo_path: str,
    depth_model_path: str,
    moge2_model_path: Optional[str] = None,
    output_resolution_mode: str = "upsample",
    python_bin: Optional[str] = None,
) -> None:
    global config
    config = {
        "repo_path": str(Path(repo_path).resolve()),
        "depth_model_path": str(Path(depth_model_path).resolve()),
        "moge2_model_path": str(Path(moge2_model_path).resolve()) if moge2_model_path else None,
        "output_resolution_mode": output_resolution_mode,
        "python_bin": python_bin or os.environ.get("INFINIDEPTH_PYTHON", "python"),
    }


@app.route("/health", methods=["GET"])
def health_check():
    repo = Path(config.get("repo_path", ""))
    script = repo / "inference_depth.py"
    model_path = Path(config.get("depth_model_path", ""))
    moge2_model_path = Path(config["moge2_model_path"]) if config.get("moge2_model_path") else None
    return jsonify(
        {
            "status": "healthy" if script.exists() and model_path.exists() else "unhealthy",
            "repo_path": str(repo),
            "script_exists": script.exists(),
            "depth_model_path": str(model_path),
            "depth_model_exists": model_path.exists(),
            "moge2_model_path": str(moge2_model_path) if moge2_model_path else None,
            "moge2_model_exists": moge2_model_path.exists() if moge2_model_path else None,
        }
    )


@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json() or {}
        if "image" not in data:
            return jsonify({"success": False, "error": "Missing image data"}), 400

        image = _decode_image(data["image"])
        filename = data.get("filename") or "input.png"
        save_pcd = bool(data.get("save_pcd", False))
        upsample_ratio = float(data.get("upsample_ratio", 2))
        if upsample_ratio <= 0:
            return jsonify({"success": False, "error": "upsample_ratio must be positive"}), 400
        if upsample_ratio.is_integer():
            upsample_ratio = int(upsample_ratio)

        with tempfile.TemporaryDirectory(prefix="infinidepth_") as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / Path(filename).with_suffix(".png").name
            image.save(input_path)
            run_dir = tmp_dir / "run"
            run_dir.mkdir()
            result = _run_infinidepth(
                input_path=input_path,
                run_dir=run_dir,
                save_pcd=save_pcd,
                upsample_ratio=upsample_ratio,
            )
            return jsonify(result)
    except Exception as e:
        logger.error("InfiniDepth inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


def _run_infinidepth(input_path: Path, run_dir: Path, save_pcd: bool, upsample_ratio: float) -> Dict[str, Any]:
    repo = Path(config["repo_path"])
    script = repo / "inference_depth.py"
    if not script.exists():
        raise FileNotFoundError(f"InfiniDepth inference script not found: {script}")
    model_path = Path(config["depth_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"InfiniDepth checkpoint not found: {model_path}")

    command = [
        config["python_bin"],
        str(script),
        f"--input_image_path={input_path}",
        "--model_type=InfiniDepth",
        f"--depth_model_path={model_path}",
        f"--output_resolution_mode={config.get('output_resolution_mode', 'upsample')}",
        f"--upsample_ratio={upsample_ratio}",
        f"--depth_output_dir={run_dir / 'pred_depth'}",
        f"--pcd_output_dir={run_dir / 'pred_pcd'}",
        "--save-pcd" if save_pcd else "--no-save-pcd",
    ]
    if config.get("moge2_model_path"):
        command.append(f"--moge2_pretrained={config['moge2_model_path']}")

    logger.info("Running InfiniDepth command: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=repo,
        env={**os.environ, "INFINIDEPTH_OUTPUT_DIR": str(run_dir)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "success": False,
            "error": completed.stderr[-2000:] or completed.stdout[-2000:] or "InfiniDepth command failed",
            "command": command,
        }

    files = _collect_outputs(repo, input_path, run_dir)
    if files.get("depth") is None and files.get("colored_depth") is None:
        return {
            "success": False,
            "error": "InfiniDepth command completed but no depth output file was found",
            "command": command,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    response: Dict[str, Any] = {
        "success": True,
        "command": command,
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }
    if files.get("depth"):
        response["depth_image"] = _encode_file(files["depth"])
    if files.get("colored_depth"):
        response["colored_depth_image"] = _encode_file(files["colored_depth"])
    elif files.get("depth"):
        response["colored_depth_image"] = _encode_file(files["depth"])
    if files.get("point_cloud"):
        response["point_cloud"] = _encode_file(files["point_cloud"])
    return response


def _collect_outputs(repo: Path, input_path: Path, run_dir: Path) -> Dict[str, Optional[Path]]:
    candidates = []
    for root in [run_dir, repo / "example_data" / "pred_depth", repo / "example_data" / "pred_pcd", input_path.parent]:
        if root.exists():
            candidates.extend([p for p in root.rglob("*") if p.is_file()])
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    outputs: Dict[str, Optional[Path]] = {"depth": None, "colored_depth": None, "point_cloud": None}
    for path in candidates:
        suffix = path.suffix.lower()
        name = path.name.lower()
        if outputs["point_cloud"] is None and suffix in {".ply", ".pcd"}:
            outputs["point_cloud"] = path
        elif suffix in {".png", ".jpg", ".jpeg"}:
            if outputs["colored_depth"] is None and ("color" in name or "vis" in name or "depth" in name):
                outputs["colored_depth"] = path
            if outputs["depth"] is None and "depth" in name:
                outputs["depth"] = path
    if outputs["depth"] is None:
        outputs["depth"] = outputs["colored_depth"]
    return outputs


def _decode_image(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniDepth Depth Estimation Server")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the official InfiniDepth repository")
    parser.add_argument("--depth_model_path", type=str, required=True, help="Path to infinidepth.ckpt")
    parser.add_argument("--port", type=int, default=20037, help="Port to run the server on")
    parser.add_argument("--python_bin", type=str, default=None, help="Python executable for the InfiniDepth environment")
    parser.add_argument("--output_resolution_mode", type=str, default="upsample")
    parser.add_argument("--moge2_model_path", type=str, default=None, help="Path to MoGe-2 model.pt")
    args = parser.parse_args()

    configure(
        repo_path=args.repo_path,
        depth_model_path=args.depth_model_path,
        moge2_model_path=args.moge2_model_path,
        output_resolution_mode=args.output_resolution_mode,
        python_bin=args.python_bin,
    )
    app.run(host="0.0.0.0", port=args.port, debug=False)
