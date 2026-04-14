"""
Orient Anything V2 Flask Server (port 20034)

Wraps the Orient Anything V2 model inference behind a simple HTTP API.

The model outputs azimuth (0-360°), elevation (-90~90°), rotation (-180~180°),
and symmetry_alpha (0/1/2/4) for a reference image.  Providing a second image
also returns relative azimuth/elevation/rotation between the two views.

Setup:
  1. Clone the HF Space (contains vision_tower.py, inference.py, and the
     bundled vggt sub-package — no separate pip install needed):

       git clone https://huggingface.co/spaces/Viglong/Orient-Anything-V2 \\
           third_party/orient_anything_v2

  2. Install inference dependencies (into the active conda env):

       pip install rembg scipy torchvision timm einops transformers

  3. Checkpoint is already at:
       checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt

  4. Start server:

       python spagent/external_experts/OrientAnythingV2/oa_v2_server.py \\
           --checkpoint_path checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt \\
           --repo_path third_party/orient_anything_v2 \\
           --port 20034

     For development without a GPU:

       python spagent/external_experts/OrientAnythingV2/oa_v2_server.py \\
           --checkpoint_path checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt \\
           --repo_path third_party/orient_anything_v2 \\
           --use_mock --port 20034
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import sys
from typing import Optional

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

app = Flask(__name__)

MODEL = None       # VGGT_OriAny_Ref instance
DEVICE = "cuda"
DTYPE = None
USE_MOCK = False
_MOCK_SVC = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "orient_anything_v2"})


@app.route("/test", methods=["GET"])
def test():
    return jsonify({"status": "ok", "message": "Orient Anything V2 server is running"})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    pil_ref = _decode_image(data["image"])
    pil_tgt = _decode_image(data["image2"]) if "image2" in data else None
    do_rm_bkg = bool(data.get("remove_background", False))

    if not USE_MOCK and MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        result = _run_inference(pil_ref, pil_tgt, do_rm_bkg)
        return jsonify(result)
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(pil_ref: "Image.Image", pil_tgt: Optional["Image.Image"] = None,
                   do_rm_bkg: bool = False) -> dict:
    if USE_MOCK:
        return _run_mock_inference(pil_ref, pil_tgt)
    return _run_real_inference(pil_ref, pil_tgt, do_rm_bkg)


def _run_real_inference(pil_ref: "Image.Image", pil_tgt: Optional["Image.Image"],
                        do_rm_bkg: bool) -> dict:
    from inference import preprocess_images, val_fit_alpha  # from repo_path

    if do_rm_bkg:
        from app_utils import background_preprocess
        pil_ref = background_preprocess(pil_ref, True)
        if pil_tgt is not None:
            pil_tgt = background_preprocess(pil_tgt, True)

    image_list = [pil_ref] if pil_tgt is None else [pil_ref, pil_tgt]
    image_tensors = preprocess_images(image_list, mode="pad")  # (S, 3, H, W)
    batch = image_tensors.unsqueeze(0)                          # (1, S, 3, H, W)

    with torch.no_grad():
        batch_img = batch.to(device=DEVICE, dtype=DTYPE)
        B, S, C, H, W = batch_img.shape
        pose_enc = MODEL(batch_img)      # (B*S, 900)
        pose_enc = pose_enc.view(B * S, -1)

        angle_az = torch.argmax(pose_enc[:, 0:360], dim=-1)           # 0-359
        angle_el = torch.argmax(pose_enc[:, 360:540], dim=-1) - 90    # -90..89
        angle_ro = torch.argmax(pose_enc[:, 540:900], dim=-1) - 180   # -180..179

        distribute = F.sigmoid(pose_enc[:, 0:360]).cpu().float().numpy()
        alpha_pred = val_fit_alpha(distribute=distribute)

    if S > 1:
        ref_az = int(angle_az.reshape(B, S)[0, 0])
        ref_el = int(angle_el.reshape(B, S)[0, 0])
        ref_ro = int(angle_ro.reshape(B, S)[0, 0])
        ref_alpha = int(alpha_pred.reshape(B, S)[0, 0])
        rel_az = int(angle_az.reshape(B, S)[0, 1])
        rel_el = int(angle_el.reshape(B, S)[0, 1])
        rel_ro = int(angle_ro.reshape(B, S)[0, 1])
        return {
            "azimuth": ref_az,
            "elevation": ref_el,
            "rotation": ref_ro,
            "symmetry_alpha": ref_alpha,
            "rel_azimuth": rel_az,
            "rel_elevation": rel_el,
            "rel_rotation": rel_ro,
        }
    else:
        return {
            "azimuth": int(angle_az[0]),
            "elevation": int(angle_el[0]),
            "rotation": int(angle_ro[0]),
            "symmetry_alpha": int(alpha_pred[0]),
        }


def _run_mock_inference(pil_ref, pil_tgt) -> dict:
    import random, math
    rng = random.Random(42)
    az = round(rng.uniform(0, 360), 1)
    el = round(rng.uniform(-90, 90), 1)
    ro = round(rng.uniform(-180, 180), 1)
    alpha = rng.choice([0, 1, 2, 4])
    result = {"azimuth": az, "elevation": el, "rotation": ro, "symmetry_alpha": alpha}
    if pil_tgt is not None:
        result.update({
            "rel_azimuth": round(rng.uniform(0, 360), 1),
            "rel_elevation": round(rng.uniform(-90, 90), 1),
            "rel_rotation": round(rng.uniform(-180, 180), 1),
        })
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(b64: str) -> "Image.Image":
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_model(checkpoint_path: str, repo_path: str, device: str) -> None:
    global MODEL, DEVICE, DTYPE

    abs_repo = os.path.abspath(repo_path)
    if not os.path.isdir(abs_repo):
        raise FileNotFoundError(
            f"repo_path does not exist: {abs_repo}\n"
            "Clone it with:\n"
            "  git clone https://huggingface.co/spaces/Viglong/Orient-Anything-V2 "
            f"{repo_path}"
        )
    if abs_repo not in sys.path:
        sys.path.insert(0, abs_repo)

    from vision_tower import VGGT_OriAny_Ref  # noqa: E402

    if device == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0]
        DTYPE = torch.bfloat16 if capability >= 8 else torch.float16
        DEVICE = "cuda"
    else:
        if device == "cuda":
            logger.warning("CUDA not available, falling back to CPU")
        DTYPE = torch.float32
        DEVICE = "cpu"

    abs_ckpt = os.path.abspath(checkpoint_path)
    if not os.path.isfile(abs_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {abs_ckpt}")

    logger.info("Loading Orient Anything V2  ckpt=%s  device=%s  dtype=%s",
                abs_ckpt, DEVICE, DTYPE)
    MODEL = VGGT_OriAny_Ref(out_dim=900, dtype=DTYPE, nopretrain=True)
    state_dict = torch.load(abs_ckpt, map_location="cpu", weights_only=False)
    MODEL.load_state_dict(state_dict)
    MODEL.eval()
    MODEL.to(device=DEVICE, dtype=DTYPE)
    logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orient Anything V2 Flask server")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to rotmod_realrotaug_best.pt",
    )
    parser.add_argument(
        "--repo_path", type=str, required=True,
        help="Path to cloned HF Space: "
             "git clone https://huggingface.co/spaces/Viglong/Orient-Anything-V2",
    )
    parser.add_argument("--port", type=int, default=20034)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--use_mock", action="store_true",
        help="Return deterministic dummy outputs (no GPU required, for development)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.use_mock:
        logger.info("Mock mode enabled — skipping model load")
        USE_MOCK = True
    else:
        _load_model(args.checkpoint_path, args.repo_path, args.device)

    logger.info("Orient Anything V2 server listening on 0.0.0.0:%d", args.port)
    app.run(host="0.0.0.0", port=args.port)
