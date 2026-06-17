"""
PaddleOCR-VL-1.5 Flask inference server (port 20037).

Wraps PaddleOCR-VL-1.5 behind a simple HTTP API so the heavy model stays in
one process while multiple callers share it.

Setup:
  pip install transformers accelerate pillow flask

  export PADDLEOCR_VL_CHECKPOINT=PaddlePaddle/PaddleOCR-VL-1.5   # or a local path

  python spagent/external_experts/PaddleOCRVL/paddleocr_vl_server.py \
      --port 20037 --device cuda

Endpoints:
  GET  /health  — liveness probe
  POST /infer   — JSON body: {"image": "<base64>", "task": "ocr"}
                  Returns:    {"success": true, "text": "...", "task": "ocr"}
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)
app = Flask(__name__)

_client = None  # PaddleOCRVLLocalClient, set at startup


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "paddleocr-vl-1.5"})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)

    if "image" not in data:
        return jsonify({"success": False, "error": "Missing 'image' field"}), 400

    task = data.get("task", "ocr")

    try:
        image_bytes = base64.b64decode(data["image"])
        pil_image = _bytes_to_pil(image_bytes)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Invalid image data: {exc}"}), 400

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path, format="JPEG")

    try:
        result = _client.recognize(tmp_path, task=task)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return jsonify(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bytes_to_pil(data: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_client(checkpoint: str | None, device: str) -> None:
    global _client
    _dir = Path(__file__).resolve().parent
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
    from paddleocr_vl_local import PaddleOCRVLLocalClient
    _client = PaddleOCRVLLocalClient(checkpoint=checkpoint, device=device)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR-VL-1.5 Flask server")
    parser.add_argument("--port", type=int, default=20037)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="HuggingFace model ID or local path (default: $PADDLEOCR_VL_CHECKPOINT or PaddlePaddle/PaddleOCR-VL-1.5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _load_client(args.checkpoint, args.device)

    logger.info("PaddleOCR-VL-1.5 server listening on 0.0.0.0:%d", args.port)
    app.run(host="0.0.0.0", port=args.port)
