"""
Depth Anything 3 Server (per ADDING_NEW_TOOLS.md / Pi3X pattern).

Run the model in a separate process; Tool talks to this server via HTTP.
Start with: python -m spagent.external_experts.depth_anything3.depth_anything3_server --checkpoint_path <path_or_hf_id> --port 20032
"""

import argparse
import base64
import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify

logger = logging.getLogger(__name__)
app = Flask(__name__)

model = None
_backend = None  # "official" | "legacy"
_device = "cpu"


def load_model(checkpoint_path: str, device: str = "cuda", encoder: str = "vitl"):
    """Load Depth Anything V3 (official API or legacy dpt)."""
    global model, _backend, _device
    _device = device if device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    ckpt = Path(checkpoint_path)

    # Official depth_anything_3 API
    try:
        import torch
        from depth_anything_3.api import DepthAnything3

        is_hf = not ckpt.exists() and "/" in checkpoint_path and not checkpoint_path.startswith("/")
        if is_hf:
            model_obj = DepthAnything3.from_pretrained(checkpoint_path)
        elif ckpt.exists():
            model_dir = str(ckpt) if ckpt.is_dir() else str(ckpt.parent)
            model_obj = DepthAnything3.from_pretrained(model_dir)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = model_obj.to(torch.device(_device))
        _backend = "official"
        logger.info("Depth Anything 3 loaded (official API), device=%s", _device)
        return True
    except Exception as e:
        logger.debug("Official depth_anything_3 not used: %s", e)

    # Legacy depth_anything_v3.dpt
    if not ckpt.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return False
    try:
        import torch
        from depth_anything_v3.dpt import DepthAnythingV3

        configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        if encoder not in configs:
            encoder = "vitl"
        model = DepthAnythingV3(**configs[encoder])
        state_dict = torch.load(str(ckpt), map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(_device).eval()
        _backend = "legacy"
        logger.info("Depth Anything 3 loaded (legacy dpt), encoder=%s, device=%s", encoder, _device)
        return True
    except Exception as e:
        logger.error("Failed to load model: %s", e, exc_info=True)
        return False


def _run_inference(image_bgr: np.ndarray, input_size: int = 518) -> np.ndarray:
    """Run inference; image_bgr is HxWx3 BGR. Returns float32 depth HxW."""
    global model, _backend
    if _backend == "official":
        # Official API expects path or image list; we have array, save to temp or pass in memory
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            cv2.imwrite(f.name, image_bgr)
            try:
                pred = model.inference([f.name])
            finally:
                os.unlink(f.name)
        depth = pred.depth
        if hasattr(depth, "cpu"):
            depth = depth.cpu().numpy()
        if depth.ndim == 3:
            depth = depth[0]
        return depth.astype(np.float32)
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        depth = model.infer_image(image_rgb, input_size=input_size)
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth, dtype=np.float32)
        return depth.astype(np.float32)


@app.route("/health", methods=["GET"])
def health():
    try:
        status = {
            "status": "ok",
            "model_loaded": model is not None,
            "backend": _backend,
            "device": _device,
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/test", methods=["GET"])
def test():
    global model
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    try:
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        depth = _run_inference(dummy)
        return jsonify({"success": True, "shape": list(depth.shape)})
    except Exception as e:
        logger.exception("Test inference failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    global model
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' (base64)"}), 400
        raw = base64.b64decode(data["image"])
        image_bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400
        input_size = int(data.get("input_size", 518))
        depth = _run_inference(image_bgr, input_size=input_size)
        buf = io.BytesIO()
        np.save(buf, depth)
        depth_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return jsonify({
            "success": True,
            "depth_b64": depth_b64,
            "shape": list(depth.shape),
        })
    except Exception as e:
        logger.exception("Infer failed")
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Depth Anything 3 Server")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to .pth or HF model id (e.g. depth-anything/DA3MONO-LARGE)")
    parser.add_argument("--port", type=int, default=20032, help="Port (default 20032)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    args = parser.parse_args()
    if not args.checkpoint_path:
        logger.error("--checkpoint_path required (or use HF id e.g. depth-anything/DA3MONO-LARGE)")
        return 1
    if not load_model(args.checkpoint_path, device=args.device, encoder=args.encoder):
        return 1
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    exit(main() or 0)
