"""
Moondream Tool

This module contains the MoondreamTool that wraps
Moondream functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class MoondreamTool(Tool):
    """Tool for vision-language tasks using Moondream"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20024"):
        """
        Initialize Moondream tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the Moondream server
        """
        super().__init__(
            name="moondream_tool",
            description="Perform vision-language tasks including image captioning, visual question answering, object detection, and object pointing using Moondream."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Moondream client"""
        if self.use_mock:
            try:
                from external_experts.moondream.mock_md_service import MockMoondream
                self._client = MockMoondream()
                logger.info("Using mock Moondream service")
            except ImportError:
                # Create a simple mock client
                class SimpleMockMoondream:
                    def caption(self, image_path):
                        return {
                            "success": True,
                            "caption": "A scene with various objects in a natural setting."
                        }
                    
                    def query(self, image_path, question):
                        return {
                            "success": True,
                            "answer": "Based on the image, I can see the requested elements."
                        }
                    
                    def detect(self, image_path, object_name):
                        return {
                            "success": True,
                            "detections": [
                                {
                                    "x_min": 0.2,
                                    "y_min": 0.2,
                                    "x_max": 0.8,
                                    "y_max": 0.8,
                                    "confidence": 0.85
                                }
                            ],
                            "output_path": f"outputs/detected_{Path(image_path).stem}.jpg"
                        }
                    
                    def point(self, image_path, object_name):
                        return {
                            "success": True,
                            "points": [
                                {
                                    "x": 0.5,
                                    "y": 0.5,
                                    "confidence": 0.9
                                }
                            ],
                            "output_path": f"outputs/pointing_{Path(image_path).stem}.jpg"
                        }
                
                self._client = SimpleMockMoondream()
                logger.info("Using mock Moondream service")
        else:
            try:
                from external_experts.moondream.md_client import MoondreamClient
                self._client = MoondreamClient(server_url=self.server_url)
                logger.info(f"Using real Moondream service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real Moondream client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image."
                },
                "task": {
                    "type": "string",
                    "enum": ["point"],
                    "description": "The task to perform: point (locate objects)"
                },
                "question": {
                    "type": "string",
                    "description": "Question to ask about the image (required for query task)"
                },
                "object_name": {
                    "type": "string",
                    "description": "Name of the object to locate (required for point tasks)"
                }
            },
            "required": ["image_path", "task", "object_name"]
        }
    
    def call(
        self, 
        image_path: str,
        task: str,
        question: Optional[str] = None,
        object_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute vision-language task
        
        Args:
            image_path: Path to input image
            task: Task type ("point")
            question: Question for query task
            object_name: Object name for detect/point tasks
            
        Returns:
            Task result dictionary
        """
        try:
            logger.info(f"Running Moondream {task} task on: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Validate task parameters
            if task == "query" and not question:
                return {
                    "success": False,
                    "error": "Question is required for query task"
                }
            
            if task in ["detect", "point"] and not object_name:
                return {
                    "success": False,
                    "error": f"Object name is required for {task} task"
                }
            
            # Execute the appropriate task
            result = None
            if task == "caption":
                result = self._client.caption(image_path)
            elif task == "query":
                result = self._client.query(image_path, question)
            elif task == "detect":
                result = self._client.detect(image_path, object_name)
            elif task == "point":
                result = self._client.point(image_path, object_name)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task: {task}"
                }
            
            if result and result.get('success'):
                logger.info(f"Moondream {task} task completed successfully")
                return {
                    "success": True,
                    "result": result,
                    "task": task,
                    "output_path": result.get('output_path')
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Moondream {task} task failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Moondream {task} task failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Moondream tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
