"""
CountGD Flask inference server.

Wraps CountGDLocalClient as an HTTP service so multiple agents can share
one model instance without reloading weights on every call.

Usage:
    export COUNTGD_CHECKPOINT=/path/to/checkpoint.pth
    python spagent/external_experts/CountGD/countgd_server.py --port 20026

    # or with explicit checkpoint:
    python spagent/external_experts/CountGD/countgd_server.py \
        --checkpoint /path/to/checkpoint.pth --port 20026
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
from external_experts.CountGD.countgd_local import CountGDLocalClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_client = None  # CountGDLocalClient, set at startup


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    status = {
        "status": "healthy" if _client is not None else "not_ready",
        "model_loaded": _client is not None and _client._model is not None,
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

    text = data.get("text", "object")

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

    # Write to a temp file so CountGDLocalClient can open it
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img_bgr)

    try:
        result = _client.count(image_path=tmp_path, text=text)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not result.get("success"):
        return jsonify(result), 500

    # Encode the annotated output image back as base64 so the client can save it
    annotated_b64 = None
    output_path = result.get("output_path")
    if output_path and Path(output_path).exists():
        with open(output_path, "rb") as f:
            annotated_b64 = base64.b64encode(f.read()).decode("utf-8")

    return jsonify({
        "success": True,
        "count": result["count"],
        "boxes": result["boxes"],
        "description": result["description"],
        "annotated_image": annotated_b64,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CountGD inference server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to CountGD .pth checkpoint (overrides COUNTGD_CHECKPOINT env var)",
    )
    parser.add_argument("--port", type=int, default=20026, help="Port (default: 20026)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--confidence_thresh",
        type=float,
        default=0.23,
        help="Detection confidence threshold (default: 0.23)",
    )
    args = parser.parse_args()

    logger.info("Loading CountGD model...")
    try:
        _client = CountGDLocalClient(
            checkpoint=args.checkpoint,
            confidence_thresh=args.confidence_thresh,
            device=args.device,
        )
        _client._ensure_model_loaded()
        logger.info("Model loaded. Starting server on port %d", args.port)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    app.run(host="0.0.0.0", port=args.port, debug=False)
