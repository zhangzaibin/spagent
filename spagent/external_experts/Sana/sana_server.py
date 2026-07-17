"""
Sana image-generation server (OpenAI images API compatible).

The in-tree ``sana_client.py`` POSTs to ``/v1/images/generations`` and expects
an OpenAI-style response (``data: [{"b64_json": ...}]``), but no server
implementation existed in the repo — only the client and a mock. This server
fills that gap with diffusers' SanaSprintPipeline.

Usage:
    python spagent/external_experts/Sana/sana_server.py \
        --model Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers \
        --port 30000
"""

import argparse
import base64
import io
import logging

import torch
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipe = None


def load_model(model_id: str):
    global pipe
    from diffusers import SanaSprintPipeline
    logger.info("Loading Sana pipeline: %s", model_id)
    pipe = SanaSprintPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Sana pipeline loaded on %s", pipe.device)
    return True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "sana", "model_loaded": pipe is not None})


@app.route("/v1/images/generations", methods=["POST"])
def generate():
    if pipe is None:
        return jsonify({"error": "model not loaded"}), 503
    body = request.get_json(force=True) or {}
    prompt = body.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt required"}), 400
    try:
        width, height = (int(v) for v in body.get("size", "1024x1024").split("x"))
    except Exception:
        width = height = 1024
    steps = int(body.get("num_inference_steps", 2))
    n = max(1, int(body.get("n", 1)))
    seed = int(body.get("seed", 42))

    try:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        out = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            num_images_per_prompt=n,
            generator=generator,
        )
        data = []
        for img in out.images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode()})
        return jsonify({"created": 0, "data": data})
    except Exception as e:  # pragma: no cover - surface inference errors
        logger.exception("Sana generation failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sana image generation server")
    parser.add_argument("--model", type=str,
                        default="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    if not load_model(args.model):
        raise SystemExit(1)
    app.run(host="0.0.0.0", port=args.port, debug=False)
