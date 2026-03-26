"""
Orient Anything V2 Flask Server (port 20034)

Wraps the Orient Anything V2 model inference behind a simple HTTP API.

Setup:
  1. Clone model repo:
       git clone https://github.com/SpatialVision/Orient-Anything-V2
  2. Download checkpoint:
       huggingface-cli download SpatialVision/Orient-Anything-V2 \
           --local-dir checkpoints/orient_anything_v2
  3. Start server:
       python spagent/external_experts/OrientAnythingV2/oa_v2_server.py \
           --checkpoint_path checkpoints/orient_anything_v2 \
           --port 20034
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import math

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model import — only required when running the real server
# ---------------------------------------------------------------------------
try:
    import torch
    from PIL import Image
except ImportError:
    torch = None  # type: ignore

app = Flask(__name__)
MODEL = None  # loaded once at startup


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "orient_anything_v2"})


@app.route("/test", methods=["GET"])
def test():
    """Quick self-test endpoint."""
    return jsonify({"status": "ok", "message": "Orient Anything V2 server is running"})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    image = _decode_image(data["image"])
    category = data.get("object_category", "object")
    task = data.get("task", "orientation")
    image2 = _decode_image(data["image2"]) if "image2" in data else None

    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        result = _run_inference(image, category, task, image2)
        return jsonify(result)
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------

def _run_inference(image, category: str, task: str, image2=None) -> dict:
    """Dispatch to the correct model forward pass."""
    if task == "orientation":
        yaw, pitch, roll, conf = MODEL.predict_orientation(image, category)
        yaw_r, pitch_r = math.radians(yaw), math.radians(pitch)
        front = [
            round(math.cos(pitch_r) * math.sin(yaw_r), 4),
            round(-math.sin(pitch_r), 4),
            round(math.cos(pitch_r) * math.cos(yaw_r), 4),
        ]
        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "confidence": float(conf),
            "front_vector": front,
        }
    elif task == "symmetry":
        sym_type, axis, conf = MODEL.predict_symmetry(image, category)
        return {
            "symmetry_type": sym_type,
            "axis": [float(v) for v in axis] if axis is not None else None,
            "confidence": float(conf),
        }
    elif task == "relative_rotation":
        if image2 is None:
            raise ValueError("image2 required for relative_rotation task")
        rot_mat, euler, dist = MODEL.predict_relative_rotation(image, image2, category)
        return {
            "rotation_matrix": rot_mat.tolist(),
            "euler_angles": [float(v) for v in euler],
            "angular_distance_deg": float(dist),
        }
    else:
        raise ValueError(f"Unknown task: {task}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(b64: str):
    """Decode a base64 image string to PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_model(checkpoint_path: str, device: str):
    global MODEL
    # Uncomment and adjust once Orient-Anything-V2 is cloned locally:
    # from orient_anything_v2 import OrientAnythingV2Model
    # MODEL = OrientAnythingV2Model.from_pretrained(checkpoint_path)
    # MODEL = MODEL.to(device).eval()
    raise NotImplementedError(
        "Please clone https://github.com/SpatialVision/Orient-Anything-V2 and "
        "update the import path above before running the real server. "
        "Use --use_mock for development without a GPU."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orient Anything V2 server")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--port", type=int, default=20034)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    _load_model(args.checkpoint_path, args.device)
    logger.info(f"Orient Anything V2 server running on port {args.port}")
    app.run(host="0.0.0.0", port=args.port)
