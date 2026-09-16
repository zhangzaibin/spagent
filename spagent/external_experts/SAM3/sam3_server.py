import argparse
import base64
import logging
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

image_model = None
image_processor = None
video_predictor = None
model_name = "sam3"


def load_model(checkpoint_path: Optional[str] = None, load_video: bool = True) -> bool:
    """Load SAM3 image processor and, optionally, the video predictor."""
    global image_model, image_processor, video_predictor, model_name
    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        image_model = _build_with_optional_checkpoint(build_sam3_image_model, checkpoint_path)
        image_processor = Sam3Processor(image_model)
        model_name = checkpoint_path or "sam3-default"
        logger.info("SAM3 image model loaded: %s", model_name)

        if load_video:
            from sam3.model_builder import build_sam3_video_predictor

            try:
                if not torch.cuda.is_available():
                    logger.warning("SAM3 video predictor requires CUDA; skipping video model load")
                else:
                    gpus_to_use = list(range(torch.cuda.device_count()))
                    video_predictor = build_sam3_video_predictor(
                        checkpoint_path=checkpoint_path,
                        gpus_to_use=gpus_to_use,
                    )
                    logger.info("SAM3 video predictor loaded")
            except Exception as e:
                logger.warning("SAM3 video predictor failed to load; image service remains available: %s", e)

        return True
    except Exception as e:
        logger.error("Failed to load SAM3: %s", e)
        logger.error(traceback.format_exc())
        return False


def _build_with_optional_checkpoint(builder, checkpoint_path: Optional[str]):
    if not checkpoint_path:
        return builder()

    attempts = [
        {"checkpoint_path": checkpoint_path},
        {"ckpt_path": checkpoint_path},
        {"checkpoint": checkpoint_path},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return builder(**kwargs)
        except TypeError as e:
            last_error = e
            continue
    if last_error is not None:
        logger.warning("SAM3 builder did not accept checkpoint path directly: %s", last_error)
    return builder()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy" if image_processor is not None else "unhealthy",
            "model_name": model_name,
            "image_model_loaded": image_processor is not None,
            "video_predictor_loaded": video_predictor is not None,
            "cuda_available": torch.cuda.is_available(),
        }
    )


@app.route("/test", methods=["GET"])
def test():
    try:
        test_image = Image.new("RGB", (256, 256), color=(40, 40, 40))
        state = image_processor.set_image(test_image)
        output = image_processor.set_text_prompt(state=state, prompt="square")
        return jsonify({"success": True, "keys": list(output.keys())})
    except Exception as e:
        logger.error("SAM3 test failed: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer", methods=["POST"])
def infer():
    if image_processor is None:
        return jsonify({"success": False, "error": "SAM3 image model is not loaded"}), 500

    try:
        data = request.get_json() or {}
        if "image" not in data:
            return jsonify({"success": False, "error": "Missing image data"}), 400
        text_prompt = _require_prompt(data)
        if isinstance(text_prompt, tuple):
            return text_prompt

        image_bytes = base64.b64decode(data["image"], validate=True)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "error": "Invalid image data"}), 400

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        state = image_processor.set_image(pil_image)
        output = image_processor.set_text_prompt(state=state, prompt=text_prompt)
        masks, boxes, scores = _extract_image_outputs(output)
        masks, boxes, scores = _filter_instances(
            masks=masks,
            boxes=boxes,
            scores=scores,
            score_threshold=float(data.get("score_threshold", 0.5)),
            max_instances=int(data.get("max_instances", 20)),
        )

        mask_records = []
        for idx, mask in enumerate(masks):
            mask_records.append(
                {
                    "id": idx,
                    "mask": _encode_mask(mask),
                    "bbox": boxes[idx].tolist() if idx < len(boxes) else None,
                    "score": float(scores[idx]) if idx < len(scores) else None,
                    "label": text_prompt,
                }
            )

        return jsonify(
            {
                "success": True,
                "task": "image",
                "text_prompt": text_prompt,
                "masks": mask_records,
                "boxes": [box.tolist() for box in boxes],
                "scores": [float(score) for score in scores],
                "shape": list(image.shape[:2]),
            }
        )
    except Exception as e:
        logger.error("SAM3 image inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/infer_video", methods=["POST"])
def infer_video():
    if video_predictor is None:
        return jsonify({"success": False, "error": "SAM3 video predictor is not loaded"}), 500

    temp_path = None
    output_path = None
    session_id = None
    try:
        data = request.get_json() or {}
        has_video = isinstance(data.get("video"), str) and bool(data.get("video"))
        has_frames = isinstance(data.get("frames"), list) and bool(data.get("frames"))
        if has_video == has_frames:
            return jsonify(
                {"success": False, "error": "Provide exactly one of video or non-empty frames"}
            ), 400
        text_prompt = _require_prompt(data)
        if isinstance(text_prompt, tuple):
            return text_prompt

        if has_frames:
            temp_path = _materialize_frame_directory(data["frames"])
        else:
            suffix = Path(data.get("filename") or "input.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(base64.b64decode(data["video"], validate=True))
                temp_path = f.name

        frame_index = int(data.get("frame_index", 0))
        if frame_index < 0:
            return jsonify({"success": False, "error": "frame_index must be non-negative"}), 400
        frame_count = _input_frame_count(temp_path)
        if frame_count and frame_index >= frame_count:
            return jsonify(
                {"success": False, "error": f"frame_index {frame_index} is outside {frame_count} frames"}
            ), 400
        score_threshold = float(data.get("score_threshold", 0.5))
        start_response = video_predictor.handle_request(
            request={"type": "start_session", "resource_path": temp_path}
        )
        session_id = start_response["session_id"]
        prompt_response = video_predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": text_prompt,
                "output_prob_thresh": score_threshold,
            }
        )
        outputs_per_frame = {frame_index: prompt_response.get("outputs", {})}
        if hasattr(video_predictor, "handle_stream_request"):
            for response in video_predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "output_prob_thresh": score_threshold,
                }
            ):
                outputs_per_frame[response["frame_index"]] = response.get("outputs", {})

        save_overlay = bool(data.get("save_overlay", True))
        output_path, stats, frame_masks = _render_video_overlay(
            temp_path,
            outputs_per_frame,
            text_prompt,
            max_instances=max(1, int(data.get("max_instances", 20))),
            save_overlay=save_overlay,
        )
        video_b64 = None
        if output_path:
            with open(output_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "task": "video",
                "text_prompt": text_prompt,
                "video": video_b64,
                "frames": stats["frames"],
                "fps": stats["fps"],
                "size": stats["size"],
                "frame_index": frame_index,
                "frame_masks": frame_masks,
            }
        )
    except Exception as e:
        logger.error("SAM3 video inference failed: %s", e)
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if session_id is not None:
            try:
                video_predictor.handle_request(
                    request={"type": "close_session", "session_id": session_id}
                )
            except Exception as e:
                logger.warning("Failed to close SAM3 video session %s: %s", session_id, e)
        if temp_path and os.path.isdir(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)
        elif temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass


def _require_prompt(data):
    text_prompt = data.get("text_prompt")
    if not isinstance(text_prompt, str) or not text_prompt.strip():
        return jsonify({"success": False, "error": "text_prompt must be a non-empty string"}), 400
    return text_prompt.strip()


def _extract_image_outputs(output: Dict):
    masks = _to_numpy(output.get("masks"))
    boxes = _to_numpy(output.get("boxes"))
    scores = _to_numpy(output.get("scores"))

    if masks is None:
        masks = np.zeros((0, 1, 1), dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim == 4:
        masks = masks[:, 0, :, :]
    masks = (masks > 0.5).astype(np.uint8) * 255

    if boxes is None:
        boxes = np.zeros((len(masks), 4), dtype=np.float32)
    boxes = boxes.reshape((-1, 4)) if boxes.size else np.zeros((0, 4), dtype=np.float32)

    if scores is None:
        scores = np.ones((len(masks),), dtype=np.float32)
    scores = scores.reshape((-1,))
    return masks, boxes, scores


def _filter_instances(masks, boxes, scores, score_threshold: float, max_instances: int):
    if len(scores) == 0:
        return masks[:0], boxes[:0], scores[:0]
    keep = np.where(scores >= score_threshold)[0]
    keep = keep[: max(1, max_instances)]
    return masks[keep], boxes[keep], scores[keep]


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, list):
        converted = [_to_numpy(v) for v in value]
        return np.asarray(converted)
    return np.asarray(value)


def _encode_mask(mask: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", mask.astype(np.uint8))
    if not ok:
        raise ValueError("Failed to encode mask")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _materialize_frame_directory(encoded_frames: List[str]) -> str:
    frame_dir = tempfile.mkdtemp(prefix="spagent_sam3_frames_")
    try:
        expected_size = None
        for index, encoded in enumerate(encoded_frames):
            if not isinstance(encoded, str) or not encoded:
                raise ValueError(f"Invalid frame data at index {index}")
            frame_bytes = base64.b64decode(encoded, validate=True)
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Invalid JPEG frame at index {index}")
            size = frame.shape[:2]
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise ValueError("All JPEG frames must have the same dimensions")
            frame_path = Path(frame_dir) / f"{index:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise OSError(f"Unable to save temporary frame {index}")
        return frame_dir
    except Exception:
        shutil.rmtree(frame_dir, ignore_errors=True)
        raise


def _input_frame_count(source_path: str) -> int:
    source = Path(source_path)
    if source.is_dir():
        return len(list(source.glob("*.jpg")))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _render_video_overlay(
    video_path: str,
    outputs_per_frame: Dict,
    text_prompt: str,
    max_instances: int = 20,
    save_overlay: bool = True,
):
    source = Path(video_path)
    cap = None
    frame_paths = None
    if source.is_dir():
        frame_paths = sorted(source.glob("*.jpg"), key=lambda path: int(path.stem))
        if not frame_paths:
            raise ValueError(f"No JPEG frames found in temporary directory: {video_path}")
        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Unable to read temporary frame: {frame_paths[0]}")
        height, width = first_frame.shape[:2]
        fps = 5.0
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open temporary video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = None
    writer = None
    if save_overlay:
        output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = output.name
        output.close()
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            if cap is not None:
                cap.release()
            raise ValueError("Unable to create output video")

    frame_idx = 0
    frame_mask_records = []
    while True:
        if frame_paths is not None:
            if frame_idx >= len(frame_paths):
                break
            frame = cv2.imread(str(frame_paths[frame_idx]))
            if frame is None:
                if writer is not None:
                    writer.release()
                raise ValueError(f"Unable to read temporary frame: {frame_paths[frame_idx]}")
        else:
            ret, frame = cap.read()
            if not ret:
                break
        if frame.shape[:2] != (height, width):
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            raise ValueError("All video frames must have the same dimensions")
        frame_outputs = outputs_per_frame.get(frame_idx, {})
        masks = _extract_video_masks(frame_outputs)[:max_instances]
        if save_overlay:
            frame = _overlay_frame(frame, masks)
        frame_mask_records.append(
            {
                "frame_index": frame_idx,
                "masks": [
                    {"instance_id": instance_idx, "mask": _encode_mask(mask)}
                    for instance_idx, mask in enumerate(masks)
                ],
            }
        )
        if writer is not None:
            cv2.putText(frame, text_prompt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
        frame_idx += 1

    if cap is not None:
        cap.release()
    if writer is not None:
        writer.release()
    return (
        output_path,
        {"frames": frame_idx, "fps": fps, "size": [width, height]},
        frame_mask_records,
    )


def _overlay_frame(frame: np.ndarray, masks: List[np.ndarray]):
    overlay = frame.copy()
    for idx, mask in enumerate(masks):
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = _color(idx)
        colored = np.zeros_like(frame)
        colored[mask > 0] = color
        indices = mask > 0
        overlay[indices] = cv2.addWeighted(overlay[indices], 0.6, colored[indices], 0.4, 0)
    return overlay


def _extract_video_masks(frame_outputs) -> List[np.ndarray]:
    if isinstance(frame_outputs, dict):
        for key in ["out_binary_masks", "masks", "pred_masks", "mask"]:
            if key in frame_outputs:
                value = _to_numpy(frame_outputs[key])
                if value is None:
                    continue
                if value.ndim == 2:
                    value = value[None, :, :]
                if value.ndim == 4:
                    value = value[:, 0, :, :]
                return [(mask > 0.5).astype(np.uint8) * 255 for mask in value]
        for value in frame_outputs.values():
            masks = _extract_video_masks(value)
            if masks:
                return masks
    elif isinstance(frame_outputs, Iterable) and not isinstance(frame_outputs, (str, bytes)):
        all_masks = []
        for value in frame_outputs:
            all_masks.extend(_extract_video_masks(value))
        return all_masks
    return []


def _color(idx: int):
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    return colors[idx % len(colors)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3 Image/Video Segmentation Server")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Optional SAM3 checkpoint path")
    parser.add_argument("--port", type=int, default=20035, help="Port to run the server on")
    parser.add_argument("--no_video", action="store_true", help="Only load the image model")
    args = parser.parse_args()

    if not load_model(checkpoint_path=args.checkpoint_path, load_video=not args.no_video):
        raise SystemExit(1)
    app.run(host="0.0.0.0", port=args.port, debug=False)
