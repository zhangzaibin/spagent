import argparse
import base64
import io
import logging
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None
    jsonify = None
    request = None

try:
    import torch
except ImportError:
    torch = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class _MissingFlaskApp:
    def route(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def run(self, *args, **kwargs):
        raise ImportError("WildDet3D server requires flask")


app = Flask(__name__) if Flask is not None else _MissingFlaskApp()

model = None
preprocess_fn = None
draw_3d_boxes_fn = None
model_name = "wilddet3d"


def load_model(checkpoint_path: str, score_threshold: float = 0.3) -> bool:
    global model, preprocess_fn, draw_3d_boxes_fn, model_name
    try:
        if torch is None:
            raise ImportError("WildDet3D server requires torch in the model-serving environment")
        from wilddet3d import build_model, preprocess

        try:
            from wilddet3d.vis.visualize import draw_3d_boxes
        except Exception:
            draw_3d_boxes = None

        model = _build_model(build_model, checkpoint_path, score_threshold)
        if hasattr(model, "eval"):
            model.eval()
        preprocess_fn = preprocess
        draw_3d_boxes_fn = draw_3d_boxes
        model_name = checkpoint_path
        logger.info("WildDet3D model loaded: %s", checkpoint_path)
        return True
    except Exception as e:
        logger.error("Failed to load WildDet3D: %s", e)
        logger.error(traceback.format_exc())
        return False


def _build_model(build_model, checkpoint_path: str, score_threshold: float):
    attempts = [
        {"checkpoint": checkpoint_path, "score_threshold": score_threshold, "skip_pretrained": True},
        {"checkpoint_path": checkpoint_path, "score_threshold": score_threshold, "skip_pretrained": True},
        {"checkpoint": checkpoint_path, "score_threshold": score_threshold},
        {"checkpoint": checkpoint_path},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return build_model(**kwargs)
        except TypeError as e:
            last_error = e
    raise last_error


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy" if model is not None else "unhealthy",
            "model_name": model_name,
            "cuda_available": bool(torch is not None and torch.cuda.is_available()),
        }
    )


@app.route("/infer", methods=["POST"])
def infer():
    if model is None or preprocess_fn is None:
        return jsonify({"success": False, "error": "WildDet3D model is not loaded"}), 500

    try:
        data = request.get_json() or {}
        image = _decode_image(data.get("image"))
        text_prompt = _clean_prompt(data.get("text_prompt"))
        boxes = data.get("boxes") or None
        points = data.get("points") or None

        if not text_prompt and not boxes and not points:
            return jsonify({"success": False, "error": "Provide text_prompt, boxes, or points"}), 400

        outputs = _run_wilddet3d(
            image=image,
            text_prompt=text_prompt,
            boxes=boxes,
            points=points,
            score_threshold=float(data.get("score_threshold", 0.3)),
        )
        boxes_2d, boxes_3d, scores, scores_2d, class_ids, depth_maps, intrinsics = outputs
        class_names = _class_names(text_prompt, boxes, points, class_ids)

        response: Dict[str, Any] = {
            "success": True,
            "boxes_2d": _to_list(boxes_2d),
            "boxes_3d": _to_list(boxes_3d),
            "scores": _to_list(scores),
            "class_names": class_names,
        }

        if depth_maps is not None:
            response["depth_image"] = _encode_depth(depth_maps)

        if data.get("save_visualization", True):
            response["output_image"] = _visualize(
                image,
                boxes_2d,
                boxes_3d,
                scores,
                scores_2d,
                class_ids,
                class_names,
                intrinsics,
            )

        return jsonify(response)
    except Exception as e:
        logger.error("WildDet3D inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


def _run_wilddet3d(
    image: Image.Image,
    text_prompt: Optional[str],
    boxes: Optional[List[List[float]]],
    points: Optional[List[List[float]]],
    score_threshold: float,
):
    if torch is None:
        raise ImportError("WildDet3D inference requires torch")
    image_array = np.array(image).astype(np.float32)
    data = preprocess_fn(image_array)
    data = _move_to_device(data, _model_device())

    kwargs = {
        "images": data["images"],
        "intrinsics": data["intrinsics"][None],
        "input_hw": [data["input_hw"]],
        "original_hw": [data["original_hw"]],
        "padding": [data["padding"]],
    }
    if text_prompt:
        kwargs["input_texts"] = _split_prompt(text_prompt)
    elif boxes:
        kwargs["input_boxes"] = boxes
        kwargs["prompt_text"] = "geometric"
    elif points:
        kwargs["input_points"] = [points]
        kwargs["prompt_text"] = "geometric"

    with torch.no_grad():
        results = model(**kwargs)

    boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results
    boxes_2d = _first_batch(boxes_2d)
    boxes_3d = _first_batch(boxes_3d)
    scores = _first_batch(scores)
    scores_2d = _first_batch(scores_2d)
    scores_3d = _first_batch(scores_3d)
    class_ids = _first_batch(class_ids)
    scores_out = scores_3d if scores_3d is not None else scores
    boxes_2d, boxes_3d, scores_out, scores_2d, class_ids = _filter_by_score(
        boxes_2d,
        boxes_3d,
        scores_out,
        scores_2d,
        class_ids,
        score_threshold,
    )
    return boxes_2d, boxes_3d, scores_out, scores_2d, class_ids, depth_maps, data["intrinsics"]


def _first_batch(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else value
    arr = _to_numpy(value)
    if arr is not None and arr.ndim > 0:
        try:
            return value[0]
        except Exception:
            return arr[0]
    return value


def _filter_by_score(boxes_2d, boxes_3d, scores, scores_2d, class_ids, threshold: float):
    scores_np = _to_numpy(scores)
    if scores_np is None or scores_np.size == 0:
        return boxes_2d, boxes_3d, scores, scores_2d, class_ids
    flat = scores_np.reshape(-1)
    keep = np.where(flat >= threshold)[0]
    return (
        _take(boxes_2d, keep),
        _take(boxes_3d, keep),
        _take(scores, keep),
        _take(scores_2d, keep),
        _take(class_ids, keep),
    )


def _take(value, keep):
    if value is None:
        return value
    try:
        return value[keep]
    except Exception:
        arr = _to_numpy(value)
        return arr[keep] if arr is not None else value


def _decode_image(image_b64: Optional[str]) -> Image.Image:
    if not image_b64:
        raise ValueError("Missing image data")
    image_bytes = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _clean_prompt(prompt) -> Optional[str]:
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


def _split_prompt(prompt: str) -> List[str]:
    return [part.strip() for part in prompt.split(",") if part.strip()] or [prompt]


def _class_names(text_prompt, boxes, points, class_ids) -> List[str]:
    if text_prompt:
        names = _split_prompt(text_prompt)
        ids = _to_numpy(class_ids)
        if ids is not None and ids.size:
            return [names[int(i) % len(names)] for i in ids.reshape(-1)]
        return names
    count = len(boxes or points or [None])
    return ["object"] * count


def _model_device():
    try:
        return next(model.parameters()).device
    except Exception:
        if torch is None:
            return None
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_to_device(value, device):
    if device is None:
        return value
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "detach"):
        value = [v.detach().cpu().numpy() for v in value]
    return np.asarray(value)


def _to_list(value):
    arr = _to_numpy(value)
    if arr is None:
        return []
    return arr.tolist()


def _encode_depth(depth_maps) -> str:
    depth = _to_numpy(depth_maps)
    if depth is None or depth.size == 0:
        return ""
    depth = np.squeeze(depth)
    if depth.ndim > 2:
        depth = depth[0]
    depth = depth.astype(np.float32)
    depth = depth - np.nanmin(depth)
    depth = depth / max(float(np.nanmax(depth)), 1e-6)
    image = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    return _encode_png(image)


def _visualize(
    image: Image.Image,
    boxes_2d,
    boxes_3d,
    scores,
    scores_2d,
    class_ids,
    class_names: List[str],
    intrinsics,
) -> str:
    if draw_3d_boxes_fn is not None:
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png") as output:
                draw_3d_boxes_fn(
                    image=np.array(image).astype(np.uint8),
                    boxes3d=_to_numpy(boxes_3d),
                    intrinsics=_to_numpy(intrinsics),
                    scores_2d=_to_numpy(scores_2d),
                    scores_3d=_to_numpy(scores),
                    class_ids=_to_numpy(class_ids),
                    class_names=class_names,
                    save_path=output.name,
                    boxes_2d=_to_numpy(boxes_2d),
                    draw_predicted_2d_boxes=True,
                )
                output.seek(0)
                return base64.b64encode(output.read()).decode("utf-8")
        except Exception as e:
            logger.warning("Official WildDet3D visualization failed; using fallback: %s", e)

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    boxes = _to_numpy(boxes_2d)
    scores_np = _to_numpy(scores)
    if boxes is not None:
        boxes = boxes.reshape((-1, 4))
        for idx, box in enumerate(boxes):
            label = class_names[idx] if idx < len(class_names) else "object"
            score = float(scores_np.reshape(-1)[idx]) if scores_np is not None and scores_np.size > idx else 0.0
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((float(box[0]) + 3, max(0, float(box[1]) - 14)), f"{label} {score:.2f}", fill="red")
    return _encode_png(canvas)


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildDet3D Promptable 3D Detection Server")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to WildDet3D checkpoint")
    parser.add_argument("--port", type=int, default=20036, help="Port to run the server on")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Default score threshold")
    args = parser.parse_args()

    if not load_model(checkpoint_path=args.checkpoint_path, score_threshold=args.score_threshold):
        raise SystemExit(1)
    app.run(host="0.0.0.0", port=args.port, debug=False)
