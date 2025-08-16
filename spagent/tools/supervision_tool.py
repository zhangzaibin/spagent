"""
Supervision Tool

This module contains the SupervisionTool that wraps
Supervision functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class SupervisionTool(Tool):
    """Tool for object detection and segmentation using Supervision"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://0.0.0.0:8000"):
        """
        Initialize supervision tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the Supervision server
        """
        super().__init__(
            name="supervision_tool",
            description="Perform object detection or segmentation on the input image to identify and analyze objects in the scene. Use 'image_det' for object detection with bounding boxes, or 'image_seg' for instance segmentation with masks."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Supervision client"""
        if self.use_mock:
            try:
                from external_experts.supervision.mock_supervision_service import MockSupervisionService
                self._client = MockSupervisionService()
                logger.info("Using mock Supervision service")
            except ImportError:
                # Fallback to creating a simple mock
                class SimpleMockSupervision:
                    def infer(self, image_path, task, **kwargs):
                        return {
                            "success": True,
                            "boxes": [[100, 100, 200, 200]],
                            "labels": ["object"],
                            "confidence": [0.8],
                            "vis_path": f"outputs/supervision_mock_{Path(image_path).stem}.jpg"
                        }
                self._client = SimpleMockSupervision()
                logger.info("Using simple mock Supervision service")
        else:
            try:
                from external_experts.supervision.supervision_client import AnnotationClient
                self._client = AnnotationClient(server_url=self.server_url)
                logger.info(f"Using real Supervision service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real Supervision client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for processing."
                },
                "task": {
                    "type": "string",
                    "description": "The task type: 'image_det' for object detection or 'image_seg' for segmentation.",
                    "enum": ["image_det", "image_seg"]
                }
            },
            "required": ["image_path", "task"]
        }
    
    def call(
        self, 
        image_path: str,
        task: str
    ) -> Dict[str, Any]:
        """
        Execute supervision task
        
        Args:
            image_path: Path to input image
            task: Task type ('image_det' or 'image_seg')
            
        Returns:
            Supervision result dictionary
        """
        try:
            logger.info(f"Running supervision {task} on: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Validate task type
            if task not in ["image_det", "image_seg"]:
                return {
                    "success": False,
                    "error": f"Invalid task type: {task}. Must be 'image_det' or 'image_seg'"
                }
            
            # Call supervision
            if hasattr(self._client, 'infer'):
                result = self._client.infer(image_path, task)
            else:
                # Fallback for different client interfaces
                result = self._client.process(image_path, task)
            
            if result and result.get('success'):
                logger.info("Supervision task completed successfully")
                return {
                    "success": True,
                    "result": result,
                    "boxes": result.get('boxes', []),
                    "labels": result.get('labels', []),
                    "confidence": result.get('confidence', []),
                    "masks": result.get('masks', []),
                    "vis_path": result.get('vis_path')
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Supervision task failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Supervision task failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Supervision tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 