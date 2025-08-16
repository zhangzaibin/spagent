"""
YOLO-E Tool

This module contains the YOLOETool that wraps
YOLO-World Enhanced functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class YOLOETool(Tool):
    """Tool for object detection using YOLO-World Enhanced"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://0.0.0.0:8000"):
        """
        Initialize YOLO-E tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the YOLO-E server
        """
        super().__init__(
            name="yoloe_detection_tool",
            description="Perform advanced object detection using YOLO-E model. This tool can detect objects with custom class names specified by the user. It supports both image and video processing with high accuracy object localization and bounding box detection. Note: This tool only performs detection (bounding boxes), not segmentation."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the YOLO-E client"""
        if self.use_mock:
            try:
                from external_experts.supervision.mock_yoloe_service import MockYOLOEService
                self._client = MockYOLOEService()
                logger.info("Using mock YOLO-E service")
            except ImportError:
                # Fallback to creating a simple mock
                class SimpleMockYOLOE:
                    def infer(self, image_path, class_names, **kwargs):
                        return {
                            "success": True,
                            "boxes": [[100, 100, 200, 200]],
                            "labels": class_names[:1] if class_names else ["object"],
                            "confidence": [0.8],
                            "vis_path": f"outputs/yoloe_mock_{Path(image_path).stem}.jpg",
                            "class_names": class_names
                        }
                self._client = SimpleMockYOLOE()
                logger.info("Using simple mock YOLO-E service")
        else:
            try:
                from external_experts.supervision.sv_yoloe_client import AnnotationClient
                self._client = AnnotationClient(server_url=self.server_url)
                logger.info(f"Using real YOLO-E service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real YOLO-E client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for YOLO-E processing."
                },
                "task": {
                    "type": "string",
                    "description": "The processing task type: 'image' for single image object detection, or 'video' for video frame processing.",
                    "enum": ["image", "video"]
                },
                "class_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of object class names to detect (e.g., ['person', 'car', 'dog', 'cat']). YOLO-E can detect custom objects based on text descriptions."
                }
            },
            "required": ["image_path", "task", "class_names"]
        }
    
    def call(
        self, 
        image_path: str,
        task: str,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Execute YOLO-E detection
        
        Args:
            image_path: Path to input image
            task: Task type ('image' or 'video')
            class_names: List of class names to detect
            
        Returns:
            YOLO-E detection result dictionary
        """
        try:
            logger.info(f"Running YOLO-E {task} detection on: {image_path} for classes: {class_names}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Validate task type
            if task not in ["image", "video"]:
                return {
                    "success": False,
                    "error": f"Invalid task type: {task}. Must be 'image' or 'video'"
                }
            
            # Validate class names
            if not class_names or not isinstance(class_names, list):
                return {
                    "success": False,
                    "error": "class_names must be a non-empty list of strings"
                }
            
            # Call YOLO-E detection
            if task == "image":
                if hasattr(self._client, 'infer_image'):
                    result = self._client.infer_image(image_path, class_names)
                elif hasattr(self._client, 'infer'):
                    result = self._client.infer(image_path, class_names)
                else:
                    result = self._client.detect_image(image_path, class_names)
            else:  # video
                if hasattr(self._client, 'infer_video'):
                    result = self._client.infer_video(image_path, class_names)
                elif hasattr(self._client, 'infer'):
                    result = self._client.infer(image_path, class_names, task="video")
                else:
                    result = self._client.detect_video(image_path, class_names)
            
            if result and result.get('success'):
                logger.info("YOLO-E detection completed successfully")
                return {
                    "success": True,
                    "result": result,
                    "boxes": result.get('boxes', []),
                    "labels": result.get('labels', []),
                    "confidence": result.get('confidence', []),
                    "class_names": result.get('class_names', class_names),
                    "vis_path": result.get('vis_path')
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"YOLO-E detection failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"YOLO-E detection failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"YOLO-E tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 