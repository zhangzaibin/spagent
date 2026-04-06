"""
YOLO26 Tool

SPAgent tool wrapper for Ultralytics YOLO26 object detection.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class YOLO26Tool(Tool):
    """Tool for running YOLO26 object detection on an input image."""

    def __init__(
        self,
        model_path: str = "checkpoints/yolo26/yolo26n.pt",
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 100,
        save_annotated: bool = True,
        output_dir: str = "outputs/yolo26"
    ):
        super().__init__(
            name="yolo26_tool",
            description=(
                "Run YOLO26 object detection on an image. "
                "Use this when you need bounding boxes, class labels, and confidence scores "
                "for objects in an image."
            )
        )
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.save_annotated = save_annotated
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._init_model()

    def _init_model(self):
        """Initialize YOLO26 model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO26 model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO26 model: {e}")
            self._model = None

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image."
                },
                "conf": {
                    "type": "number",
                    "description": "Confidence threshold for detections.",
                    "default": 0.25,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "save_annotated": {
                    "type": "boolean",
                    "description": "Whether to save an annotated visualization image.",
                    "default": True
                }
            },
            "required": ["image_path"]
        }

    def call(
        self,
        image_path: str,
        conf: Optional[float] = None,
        save_annotated: Optional[bool] = None
    ) -> Dict[str, Any]:
        try:
            image_path = str(image_path)
            p = Path(image_path)

            if not p.exists():
                return {
                    "success": False,
                    "error": f"Image not found: {image_path}"
                }

            if self._model is None:
                return {
                    "success": False,
                    "error": "YOLO26 model is not initialized."
                }

            run_conf = self.conf if conf is None else conf
            run_save = self.save_annotated if save_annotated is None else save_annotated

            results = self._model.predict(
                source=image_path,
                conf=run_conf,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                verbose=False
            )

            if not results:
                return {
                    "success": False,
                    "error": "No prediction results returned by YOLO26."
                }

            r = results[0]
            detections: List[Dict[str, Any]] = []

            names = r.names if hasattr(r, "names") else {}

            if r.boxes is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().tolist()
                if r.boxes.conf is not None:
                    scores = r.boxes.conf.cpu().tolist()
                else:
                    scores = [0.0] * len(boxes_xyxy)
                cls_ids = r.boxes.cls.cpu().tolist() if r.boxes.cls is not None else []

                for i, box in enumerate(boxes_xyxy):
                    cls_id = int(cls_ids[i]) if i < len(cls_ids) else -1
                    score = float(scores[i]) if i < len(scores) else 0.0
                    label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                    detections.append({
                        "bbox_xyxy": [float(x) for x in box],
                        "class_id": cls_id,
                        "class_name": label,
                        "confidence": score
                    })

            output_path = None
            if run_save:
                annotated = r.plot()
                import cv2
                output_path = str(self.output_dir / f"{p.stem}_yolo26.jpg")
                cv2.imwrite(output_path, annotated)

            summary = f"Detected {len(detections)} object(s) in {p.name}."

            return {
                "success": True,
                "result": {
                    "image_path": image_path,
                    "num_detections": len(detections),
                    "detections": detections
                },
                "output_path": output_path,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"YOLO26Tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }