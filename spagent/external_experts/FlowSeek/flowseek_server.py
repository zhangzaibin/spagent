"""
FlowSeek Flask server (port 20036).

Usage:
  export FLOWSEEK_CHECKPOINT=/path/to/flowseek_M_TartanCT_TSKH.pth
  export FLOWSEEK_DAV2_CHECKPOINT=/path/to/depth_anything_v2_vitb.pth
  python flowseek_server.py [--checkpoint PATH] [--port 20036] [--device cuda] [--variant M]
"""

import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SPAGENT = _HERE.parents[1]
if str(_SPAGENT) not in sys.path:
    sys.path.insert(0, str(_SPAGENT))

from external_experts.FlowSeek.flowseek_local import FlowSeekLocalClient
from flask import Flask, jsonify, request

app = Flask(__name__)
_client: FlowSeekLocalClient = None  # set at startup


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": "FlowSeek"})


@app.route("/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"success": False, "error": "No JSON body"}), 400

    img1_b64 = data.get("image1")
    img2_b64 = data.get("image2")
    if not img1_b64 or not img2_b64:
        return jsonify({"success": False, "error": "Both 'image1' and 'image2' are required"}), 400

    ext1 = data.get("ext1", ".jpg")
    ext2 = data.get("ext2", ".jpg")

    try:
        with tempfile.NamedTemporaryFile(suffix=ext1, delete=False) as f1:
            f1.write(base64.b64decode(img1_b64))
            tmp1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=ext2, delete=False) as f2:
            f2.write(base64.b64decode(img2_b64))
            tmp2 = f2.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fo:
            out_path = fo.name

        result = _client.estimate_flow(
            image1_path=tmp1,
            image2_path=tmp2,
            output_path=out_path,
        )

        if result.get("success") and Path(out_path).exists():
            with open(out_path, "rb") as f:
                result["image_b64"] = base64.b64encode(f.read()).decode()

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        for p in [tmp1, tmp2, out_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowSeek inference server")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--port", type=int, default=20036)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variant", type=str, default="M", choices=["M", "T"])
    args = parser.parse_args()

    _client = FlowSeekLocalClient(
        checkpoint=args.checkpoint,
        variant=args.variant,
        device=args.device,
    )
    _client._ensure_model_loaded()
    app.run(host="0.0.0.0", port=args.port, debug=False)
