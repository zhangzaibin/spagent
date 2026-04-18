"""
Molmo2 HTTP service using Hugging Face Transformers checkpoints.

Environment:
  MOLMO2_MODEL   HF repo id or local directory (default: allenai/Molmo2-4B)
  HF_HOME        Hugging Face cache location (optional)

Optional video input uses the same message schema as allenai/molmo2 (URL or path string).

Install: `transformers` 4.57+ and `<5` (v5 `ProcessorMixin` rejects Molmo2 optional kwargs), `accelerate`, `torch`.
Video decoding may require `torchcodec` per upstream molmo2 docs.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

_model = None
_processor = None
_device: Optional[str] = None


def _load_hf():
    global _model, _processor, _device
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    checkpoint_dir = os.environ.get("MOLMO2_MODEL", "allenai/Molmo2-4B")
    logger.info("Loading Molmo2 from %s", checkpoint_dir)

    common_kw = {"trust_remote_code": True}
    proc_kw = {**common_kw, "padding_side": "left"}

    _processor = AutoProcessor.from_pretrained(checkpoint_dir, **proc_kw)

    if torch.cuda.is_available():
        try:
            _model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_dir,
                **common_kw,
                dtype="auto",
                device_map="auto",
            )
        except TypeError:
            _model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_dir,
                **common_kw,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        _device = "cuda"
    else:
        logger.warning("CUDA not available; loading on CPU (slow).")
        try:
            _model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_dir, **common_kw, dtype="auto"
            )
        except TypeError:
            _model = AutoModelForImageTextToText.from_pretrained(
                checkpoint_dir, **common_kw, torch_dtype=torch.float32
            )
        _model = _model.to("cpu")
        _device = "cpu"

    _model.eval()
    logger.info("Molmo2 loaded.")
    return True


def load_model() -> bool:
    try:
        _load_hf()
        return True
    except Exception as e:
        logger.error("Failed to load Molmo2: %s", e)
        logger.error(traceback.format_exc())
        return False


def _model_device(torch_module) -> "torch.device":
    import torch

    try:
        return next(torch_module.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _generate_from_messages(messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
    import torch

    assert _processor is not None and _model is not None

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    dev = _model_device(_model)
    inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

    use_cuda = dev.type == "cuda"
    with torch.inference_mode():
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = _model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            output = _model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    return _processor.decode(generated_tokens, skip_special_tokens=True)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": _model is not None,
            "checkpoint": os.environ.get("MOLMO2_MODEL", "allenai/Molmo2-4B"),
            "device": _device,
        }
    )


@app.route("/test", methods=["GET"])
def test():
    if _model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    try:
        img = Image.new("RGB", (128, 128), color=(120, 80, 200))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with exactly: ok"},
                    {"type": "image", "image": img},
                ],
            }
        ]
        text = _generate_from_messages(messages, max_new_tokens=16)
        return jsonify({"success": True, "sample_output": text.strip()})
    except Exception as e:
        logger.error("Test inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    if _model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt") or "Describe this image."
    max_new_tokens = int(data.get("max_new_tokens", 200))

    try:
        if "video" in data and data["video"]:
            video_ref = data["video"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video", "video": video_ref},
                    ],
                }
            ]
            text = _generate_from_messages(messages, max_new_tokens=max_new_tokens)
            return jsonify(
                {
                    "success": True,
                    "text": text.strip(),
                    "checkpoint": os.environ.get("MOLMO2_MODEL", "allenai/Molmo2-4B"),
                }
            )

        if "image" not in data:
            return jsonify({"success": False, "error": "Missing 'image' (base64) or 'video' (url/path)"}), 400

        raw = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            }
        ]
        text = _generate_from_messages(messages, max_new_tokens=max_new_tokens)
        return jsonify(
            {
                "success": True,
                "text": text.strip(),
                "checkpoint": os.environ.get("MOLMO2_MODEL", "allenai/Molmo2-4B"),
            }
        )
    except Exception as e:
        logger.error("Infer failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Molmo2 Flask server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20025)
    parser.add_argument(
        "--model",
        default=None,
        help="Override MOLMO2_MODEL (HF id or local directory).",
    )
    args = parser.parse_args()
    if args.model:
        os.environ["MOLMO2_MODEL"] = args.model

    if not load_model():
        logger.error("Aborting: model load failed.")
        raise SystemExit(1)

    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
