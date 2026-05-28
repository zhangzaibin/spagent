"""
OneFormer Flask inference server.

Wraps OneFormerLocalClient as an HTTP service so multiple agents can share
one model instance without reloading weights on every call.

Usage:
    python spagent/external_experts/OneFormer/oneformer_server.py --port 20035

    # With a specific model:
    python spagent/external_experts/OneFormer/oneformer_server.py \
        --model_id shi-labs/oneformer_ade20k_swin_large --port 20035 --device cuda
"""

import argparse
import base64
import logging
import os
import sys
from pathlib import Path

# Make external_experts importable regardless of working directory
_HERE = Path(__file__).resolve().parent
_SPAGENT = _HERE.parents[1]  # .../spagent/spagent/
if str(_SPAGENT) not in sys.path:
    sys.path.insert(0, str(_SPAGENT))

import numpy as np
from flask import Flask, jsonify, request
from external_experts.OneFormer.oneformer_local import OneFormerLocalClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_client = None  # OneFormerLocalClient, set at startup


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "healthy" if _client is not None else "not_ready",
        "model_loaded": _client is not None and _client._model is not None,
        "model_id": _client.model_id if _client is not None else None,
        "device": _client.device if _client is not None else None,
    }
    return jsonify(status)


@app.route("/infer", methods=["POST"])
def infer():
    if _client is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "Missing 'image' field"}), 400

    task = data.get("task", "panoptic")

    # Decode base64 image → temp file
    try:
        import cv2
        image_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Image decode error: {e}"}), 400

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        result = _client.segment(image_path=tmp_path, task=task)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not result.get("success"):
        return jsonify(result), 500

    # Encode annotated output image as base64
    annotated_b64 = None
    output_path = result.get("output_path")
    if output_path and Path(output_path).exists():
        with open(output_path, "rb") as f:
            annotated_b64 = base64.b64encode(f.read()).decode("utf-8")

    return jsonify({
        "success": True,
        "task": result["task"],
        "segments": result["segments"],
        "num_segments": result["num_segments"],
        "description": result["description"],
        "annotated_image": annotated_b64,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OneFormer inference server")
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace model ID (overrides ONEFORMER_MODEL_ID env var)")
    parser.add_argument("--port", type=int, default=20035, help="Port (default: 20035)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    logger.info("Loading OneFormer model...")
    try:
        _client = OneFormerLocalClient(model_id=args.model_id, device=args.device)
        _client._ensure_model_loaded()
        logger.info("Model loaded. Starting server on port %d", args.port)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    app.run(host="0.0.0.0", port=args.port, debug=False)
