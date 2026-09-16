import argparse
import base64
import io
import logging
import math
import os
import sys
import traceback
from pathlib import Path
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
default_score_threshold = 0.3


def load_model(
    checkpoint_path: str,
    score_threshold: float = 0.3,
    repo_path: Optional[str] = None,
) -> bool:
    global model, preprocess_fn, draw_3d_boxes_fn, model_name, default_score_threshold
    try:
        if torch is None:
            raise ImportError("WildDet3D server requires torch in the model-serving environment")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"WildDet3D checkpoint not found: {checkpoint}")
        if repo_path:
            repo = Path(repo_path).expanduser().resolve()
            if not (repo / "wilddet3d" / "__init__.py").is_file():
                raise FileNotFoundError(f"Invalid WildDet3D repository path: {repo}")
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
        from wilddet3d import build_model, preprocess

        try:
            from wilddet3d.vis.visualize import draw_3d_boxes
        except Exception:
            draw_3d_boxes = None

        default_score_threshold = _validate_threshold(score_threshold)
        # Keep model-side floors disabled so each request's threshold is
        # authoritative; filtering is applied after inference below.
        model = _build_model(build_model, str(checkpoint), 0.0)
        if hasattr(model, "eval"):
            model.eval()
        preprocess_fn = preprocess
        draw_3d_boxes_fn = draw_3d_boxes
        model_name = str(checkpoint)
        logger.info("WildDet3D model loaded: %s", checkpoint)
        return True
    except Exception as e:
        logger.error("Failed to load WildDet3D: %s", e)
        logger.error(traceback.format_exc())
        return False


def _build_model(build_model, checkpoint_path: str, score_threshold: float):
    attempts = [
        {
            "checkpoint": checkpoint_path,
            "score_threshold": score_threshold,
            "score_3d_threshold": 0.0,
            "skip_pretrained": True,
        },
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
        boxes = _validate_boxes(data.get("boxes"), image.size)
        points = _validate_points(data.get("points"), image.size)

        if sum(bool(value) for value in (text_prompt, boxes, points)) != 1:
            return jsonify({"success": False, "error": "Provide exactly one prompt mode: text_prompt, boxes, or points"}), 400

        threshold = _validate_threshold(
            data.get("score_threshold", default_score_threshold)
        )

        outputs = _run_wilddet3d(
            image=image,
            text_prompt=text_prompt,
            boxes=boxes,
            points=points,
            score_threshold=threshold,
        )
        (
            boxes_2d,
            boxes_3d,
            scores,
            scores_2d,
            scores_3d,
            class_ids,
            depth_maps,
            original_intrinsics,
        ) = outputs
        class_vocabulary = _class_vocabulary(text_prompt)
        class_names = _detection_class_names(
            class_vocabulary,
            class_ids,
            len(_to_list(boxes_2d)),
        )

        response: Dict[str, Any] = {
            "success": True,
            "boxes_2d": _to_list(boxes_2d),
            "boxes_3d": _to_list(boxes_3d),
            "scores": _to_list(scores),
            "scores_2d": _to_list(scores_2d),
            "scores_3d": _to_list(scores_3d),
            "class_names": class_names,
        }

        if depth_maps is not None:
            response["depth_image"] = _encode_depth(depth_maps)

        if data.get("save_visualization", True):
            response["output_image"] = _visualize(
                image,
                boxes_2d,
                boxes_3d,
                scores_2d,
                scores_3d,
                class_ids,
                class_vocabulary,
                original_intrinsics,
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

    base_kwargs = {
        "images": data["images"],
        "intrinsics": data["intrinsics"][None],
        "input_hw": [data["input_hw"]],
        "original_hw": [data["original_hw"]],
        "padding": [data["padding"]],
    }
    prompt_kwargs = []
    if text_prompt:
        prompt_kwargs.append({"input_texts": _split_prompt(text_prompt)})
    elif boxes:
        # The official predictor accepts one geometric box per image. Run
        # each box against the same preprocessed image, then aggregate the
        # aligned detections so the public API genuinely supports boxes=[...].
        prompt_kwargs.extend(
            {"input_boxes": [box], "prompt_text": "geometric"}
            for box in boxes
        )
    elif points:
        prompt_kwargs.append(
            {"input_points": [points], "prompt_text": "geometric"}
        )

    batches = []
    depth_maps = None
    device = _model_device()
    for prompt in prompt_kwargs:
        kwargs = dict(base_kwargs)
        kwargs.update(prompt)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if getattr(device, "type", None) == "cuda"
            else _NullContext()
        )
        with torch.inference_mode(), autocast:
            results = model(**kwargs)
        if not isinstance(results, (list, tuple)) or len(results) != 7:
            raise RuntimeError(
                f"Unexpected WildDet3D output: expected 7 values, got {type(results).__name__}"
            )
        if depth_maps is None:
            depth_maps = results[6]
        batches.append(tuple(_first_batch(value) for value in results[:6]))

    boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids = (
        _concat_batches([batch[index] for batch in batches])
        for index in range(6)
    )
    boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids = _filter_by_score(
        boxes_2d,
        boxes_3d,
        scores,
        scores_2d,
        scores_3d,
        class_ids,
        score_threshold,
    )
    return (
        boxes_2d,
        boxes_3d,
        scores,
        scores_2d,
        scores_3d,
        class_ids,
        depth_maps,
        data["original_intrinsics"],
    )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        return False


def _concat_batches(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    if torch is not None and all(isinstance(value, torch.Tensor) for value in values):
        return torch.cat(values, dim=0)
    return np.concatenate([_to_numpy(value) for value in values], axis=0)


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


def _filter_by_score(
    boxes_2d,
    boxes_3d,
    scores,
    scores_2d,
    scores_3d,
    class_ids,
    threshold: float,
):
    scores_np = _to_numpy(scores)
    if scores_np is None or scores_np.size == 0:
        return boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids
    flat = scores_np.reshape(-1)
    keep = np.where(flat >= threshold)[0]
    return (
        _take(boxes_2d, keep),
        _take(boxes_3d, keep),
        _take(scores, keep),
        _take(scores_2d, keep),
        _take(scores_3d, keep),
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
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid encoded image: {e}") from e


def _clean_prompt(prompt) -> Optional[str]:
    if isinstance(prompt, str) and prompt.strip():
        prompt = prompt.strip()
        if len(prompt) > 1000:
            raise ValueError("text_prompt must contain at most 1000 characters")
        return prompt
    return None


def _split_prompt(prompt: str) -> List[str]:
    prompts = [part.strip() for part in prompt.replace(".", ",").split(",") if part.strip()]
    if len(prompts) > 50:
        raise ValueError("text_prompt must contain at most 50 categories")
    return prompts or [prompt]


def _validate_boxes(boxes, image_size) -> Optional[List[List[float]]]:
    if boxes is None:
        return None
    if not isinstance(boxes, list) or not boxes:
        raise ValueError("boxes must be a non-empty list of [x1, y1, x2, y2] prompts")
    if len(boxes) > 20:
        raise ValueError("boxes must contain at most 20 prompts")
    width, height = image_size
    normalized = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Each box must have four values: [x1, y1, x2, y2]")
        values = [float(value) for value in box]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Box coordinates must be finite numbers")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(
                f"Each box must be a non-empty pixel xyxy region inside the {width}x{height} image"
            )
        normalized.append(values)
    return normalized


def _validate_points(points, image_size) -> Optional[List[List[float]]]:
    if points is None:
        return None
    if not isinstance(points, list) or not points:
        raise ValueError("points must be a non-empty list of [x, y, label] prompts")
    if len(points) > 100:
        raise ValueError("points must contain at most 100 prompts")
    width, height = image_size
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise ValueError("Each point must have three values: [x, y, label]")
        x, y, label = (float(value) for value in point)
        if not all(math.isfinite(value) for value in (x, y, label)):
            raise ValueError("Point coordinates and labels must be finite numbers")
        if label not in (0.0, 1.0):
            raise ValueError("Point labels must be 0 (background) or 1 (foreground)")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Each point must lie inside the {width}x{height} image")
        normalized.append([x, y, int(label)])
    if not any(point[2] == 1 for point in normalized):
        raise ValueError("At least one point must have foreground label 1")
    return normalized


def _validate_threshold(value) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("score_threshold must be a finite number greater than or equal to 0")
    return value


def _class_vocabulary(text_prompt: Optional[str]) -> List[str]:
    return _split_prompt(text_prompt) if text_prompt else ["object"]


def _detection_class_names(
    vocabulary: List[str],
    class_ids,
    detection_count: int,
) -> List[str]:
    ids = _to_numpy(class_ids)
    if ids is not None and ids.size:
        return [vocabulary[int(i) % len(vocabulary)] for i in ids.reshape(-1)]
    return [vocabulary[0]] * detection_count


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
        value = value.detach().cpu()
        if torch is not None and value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    if isinstance(value, (list, tuple)) and value and hasattr(value[0], "detach"):
        value = [_to_numpy(v) for v in value]
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
    finite = np.isfinite(depth)
    if not finite.any():
        return ""
    minimum = float(depth[finite].min())
    depth = np.where(finite, depth, minimum)
    depth = depth - minimum
    depth = depth / max(float(depth.max()), 1e-6)
    image = Image.fromarray((depth * 255).astype(np.uint8), mode="L")
    return _encode_png(image)


def _visualize(
    image: Image.Image,
    boxes_2d,
    boxes_3d,
    scores_2d,
    scores_3d,
    class_ids,
    class_vocabulary: List[str],
    intrinsics,
) -> str:
    boxes_2d_np = _to_numpy(boxes_2d)
    if boxes_2d_np is None or boxes_2d_np.size == 0:
        return _encode_png(image)
    if draw_3d_boxes_fn is not None:
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png") as output:
                draw_3d_boxes_fn(
                    image=np.array(image).astype(np.uint8),
                    boxes3d=_to_numpy(boxes_3d),
                    intrinsics=_to_numpy(intrinsics),
                    scores_2d=_to_numpy(scores_2d),
                    scores_3d=_to_numpy(scores_3d),
                    class_ids=_to_numpy(class_ids),
                    class_names=class_vocabulary,
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
    scores_np = _to_numpy(scores_3d)
    if boxes is not None:
        boxes = boxes.reshape((-1, 4))
        for idx, box in enumerate(boxes):
            class_ids_np = _to_numpy(class_ids)
            class_id = int(class_ids_np.reshape(-1)[idx]) if class_ids_np is not None and class_ids_np.size > idx else 0
            label = class_vocabulary[class_id % len(class_vocabulary)]
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
    parser.add_argument(
        "--repo_path",
        type=str,
        default=os.environ.get("WILDDET3D_REPO_PATH"),
        help="Path to the cloned allenai/WildDet3D repository",
    )
    parser.add_argument("--port", type=int, default=20027, help="Port to run the server on")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Default score threshold")
    args = parser.parse_args()

    if not load_model(
        checkpoint_path=args.checkpoint_path,
        score_threshold=args.score_threshold,
        repo_path=args.repo_path,
    ):
        raise SystemExit(1)
    app.run(host="0.0.0.0", port=args.port, debug=False)
