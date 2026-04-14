"""
Flask server for Molmo2 local inference.
"""

import argparse
import base64
import io
import logging
import mimetypes
from typing import List

from flask import Flask, jsonify, request
from PIL import Image

from .molmo2_local import Molmo2LocalClient
from .point_utils import annotate_images_as_base64, extract_points_from_text, group_points_by_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
_client: Molmo2LocalClient = None


def _decode_image(data_url: str) -> Image.Image:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "model_loaded": _client is not None,
        "checkpoint": getattr(_client, "checkpoint", None),
    })


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"success": True, "message": "Molmo2 server is reachable"})


@app.route("/infer", methods=["POST"])
def infer():
    images = []
    try:
        data = request.get_json(force=True)
        images_data: List[str] = data.get("images", [])
        image_names: List[str] = data.get("image_names", [])
        task = data.get("task", "qa")
        prompt = data.get("prompt", "")
        max_new_tokens = int(data.get("max_new_tokens", 256))
        temperature = float(data.get("temperature", 0.0))
        save_annotated = bool(data.get("save_annotated", True))

        if not images_data:
            return jsonify({"success": False, "error": "No images provided"}), 400

        images = [_decode_image(item) for item in images_data]
        if not image_names:
            ext = mimetypes.guess_extension("image/jpeg") or ".jpg"
            image_names = [f"image_{idx+1}{ext}" for idx in range(len(images))]

        result = _client.generate_from_images(
            images=images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if not result.get("success"):
            return jsonify(result), 500

        response = {
            "success": True,
            "generated_text": result.get("generated_text", "").strip(),
            "task": task,
            "image_names": image_names,
        }

        if task == "point":
            image_sizes = [image.size for image in images]
            fake_paths = image_names
            points = extract_points_from_text(response["generated_text"], image_sizes)
            grouped_points = group_points_by_image(points, fake_paths)
            response["points_by_image"] = grouped_points
            response["num_points"] = sum(len(group["points"]) for group in grouped_points)
            if save_annotated:
                response["annotated_images"] = annotate_images_as_base64(images, grouped_points)
        return jsonify(response)
    except Exception as e:
        logger.exception("Molmo2 inference failed")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        for image in images:
            image.close()


def main():
    parser = argparse.ArgumentParser(description="Molmo2 inference server")
    parser.add_argument("--checkpoint", default="allenai/Molmo2-4B")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--port", type=int, default=20035)
    args = parser.parse_args()

    global _client
    _client = Molmo2LocalClient(
        checkpoint=args.checkpoint,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    logger.info("Starting Molmo2 server on port %s", args.port)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
