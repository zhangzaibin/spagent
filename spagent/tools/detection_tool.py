"""
Object Detection Tool

This module contains the ObjectDetectionTool that wraps
GroundingDINO functionality for the SPAgent system.
"""

import sys
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class ObjectDetectionTool(Tool):
    """Tool for object detection using GroundingDINO"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://10.8.131.51:30969", crop: bool = True):
        """
        Initialize object detection tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the GroundingDINO server
            crop: Whether to crop each detected region and save as separate images
        """
        super().__init__(
            name="detect_objects_tool",
            description=(
                "Detect and localize objects in an image using GroundingDINO open-vocabulary detection. "
                "Provide a text prompt naming the target object(s).\n\n"
                "When to use: find objects by name/description, get bounding boxes for named entities, "
                "or open-vocabulary counting/localization.\n"
                "When NOT to use: you need pixel masks (prefer segment_image_tool), depth maps, "
                "or fixed COCO-class detection without text (prefer yolo26_tool).\n"
                "Example: text_prompt='red backpack . person' on image_path='street.jpg'."
            )
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self.crop = crop
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the GroundingDINO client"""
        if self.use_mock:
            try:
                from external_experts.GroundingDINO.mock_gdino_service import MockGroundingDINOService
                self._client = MockGroundingDINOService()
                logger.info("Using mock GroundingDINO service")
            except ImportError:
                # Fallback to creating a simple mock
                class SimpleMockGDINO:
                    def infer_image(self, image_path, text_prompt, **kwargs):
                        return {
                            "success": True,
                            "boxes": [[100, 100, 200, 200]],
                            "labels": ["object"],
                            "confidence": [0.8],
                            "vis_path": f"outputs/gdino_mock_{Path(image_path).stem}.jpg"
                        }
                self._client = SimpleMockGDINO()
                logger.info("Using simple mock GroundingDINO service")
        else:
            try:
                from external_experts.GroundingDINO.grounding_dino_client import GroundingDINOClient
                self._client = GroundingDINOClient(server_url=self.server_url)
                logger.info(f"Using real GroundingDINO service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real GroundingDINO client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for object detection."
                },
                "text_prompt": {
                    "type": "string",
                    "description": "Text description of objects to detect."
                },
                "box_threshold": {
                    "type": "number",
                    "description": "Confidence threshold for box detection.",
                    "default": 0.35
                },
                "text_threshold": {
                    "type": "number",
                    "description": "Confidence threshold for text matching.",
                    "default": 0.25
                }
            },
            "required": ["image_path", "text_prompt"]
        }

    def _normalize_detections(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize client detections into a common list of bbox dicts."""
        detections = result.get("detections")
        if detections:
            return detections

        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        normalized: List[Dict[str, Any]] = []
        for i, bbox in enumerate(boxes):
            normalized.append(
                {
                    "id": i,
                    "bbox": bbox,
                    "label": labels[i] if i < len(labels) else "obj",
                }
            )
        return normalized

    def _bbox_to_pixel_xyxy(
        self,
        bbox: List[float],
        img_h: int,
        img_w: int,
    ) -> tuple:
        """Convert a bbox to pixel (x1, y1, x2, y2).

        GroundingDINO server returns raw ``predict()`` output which is in
        **normalized cxcywh** format (all values in [0, 1]).  Values > 2.0 are
        assumed to already be in pixel xyxy format.
        """
        a, b, c, d = bbox
        if max(abs(a), abs(b), abs(c), abs(d)) <= 2.0:
            # Normalized cxcywh → pixel xyxy
            cx, cy, w, h = a * img_w, b * img_h, c * img_w, d * img_h
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
        else:
            x1, y1, x2, y2 = a, b, c, d
        return (
            max(0, int(round(x1))),
            max(0, int(round(y1))),
            min(img_w, int(round(x2))),
            min(img_h, int(round(y2))),
        )

    def _crop_detections(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
    ) -> List[str]:
        """Crop each detection bbox from the source image and save to outputs/."""
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python is required for crop. pip install opencv-python")
            return []

        orig = cv2.imread(image_path)
        if orig is None:
            logger.error(f"Failed to read image for cropping: {image_path}")
            return []

        img_h, img_w = orig.shape[:2]
        stem = Path(image_path).stem
        output_dir = Path(image_path).parent  # same directory as the source image
        output_dir.mkdir(exist_ok=True)

        crop_paths: List[str] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = self._bbox_to_pixel_xyxy(bbox, img_h, img_w)
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Skipping degenerate bbox after conversion: {bbox} → ({x1},{y1},{x2},{y2})")
                continue

            crop_img = orig[y1:y2, x1:x2]
            det_id = det.get("id", len(crop_paths))
            label = re.sub(r"[^\w\-]+", "_", str(det.get("label", "obj"))).strip("_") or "obj"
            crop_path = output_dir / f"crop_{det_id}_{label}_{stem}.jpg"
            cv2.imwrite(str(crop_path), crop_img)
            crop_paths.append(str(crop_path))

        logger.info(f"Saved {len(crop_paths)} cropped detection regions")
        return crop_paths
    
    def call(
        self, 
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Execute object detection
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for box detection
            text_threshold: Confidence threshold for text matching
            
        Returns:
            Object detection result dictionary
        """
        try:
            logger.info(f"Running object detection on: {image_path} with prompt: {text_prompt}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Call object detection
            if hasattr(self._client, 'infer_image'):
                result = self._client.infer_image(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            elif hasattr(self._client, 'infer'):
                result = self._client.infer(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            else:
                # Fallback for different client interfaces
                result = self._client.detect(
                    image_path=image_path,
                    text_prompt=text_prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )
            
            if result and result.get('success'):
                logger.info("Object detection completed successfully")
                detections = self._normalize_detections(result)
                crop_paths: List[str] = []
                if self.crop and detections:
                    crop_paths = self._crop_detections(image_path, detections)

                return {
                    "success": True,
                    "result": result,
                    "boxes": result.get('boxes', []),
                    "labels": result.get('labels', []),
                    "confidence": result.get('confidence', []),
                    "vis_path": result.get('vis_path'),
                    "output_path": result.get('output_path'),
                    "crop_paths": crop_paths,
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Object detection failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Object detection failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Object detection tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 