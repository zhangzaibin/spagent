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
                        # 解析对象输入，支持单个或多个对象
                        object_names = [name.strip() for name in object_name.split(',')]
                        is_multi_object = len(object_names) > 1
                        
                        if is_multi_object:
                            # 多对象模拟结果
                            mock_points = {}
                            color_mapping = {}
                            colors = ["RGB(255,0,0)", "RGB(0,255,0)", "RGB(0,0,255)", "RGB(255,255,0)", "RGB(255,0,255)"]
                            
                            for i, obj_name in enumerate(object_names):
                                mock_points[obj_name] = [
                                    {
                                        "x": 0.3 + i * 0.15,
                                        "y": 0.4 + i * 0.1,
                                        "confidence": 0.85
                                    }
                                ]
                                color_mapping[obj_name] = colors[i % len(colors)]
                            
                            return {
                                "success": True,
                                "is_multi_object": True,
                                "object": object_name,
                                "objects": object_names,
                                "all_points": mock_points,
                                "color_mapping": color_mapping,
                                "total_points": sum(len(pts) for pts in mock_points.values()),
                                "output_path": f"outputs/multi_pointing_{Path(image_path).stem}.jpg"
                            }
                        else:
                            # 单对象模拟结果
                            return {
                                "success": True,
                                "is_multi_object": False,
                                "object": object_names[0],
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
                    "description": "The task to perform: point (locate objects - supports both single objects like 'car' and multiple objects like 'car, person, tree')"
                },
                "object_name": {
                    "type": "string",
                    "description": "Name of the object(s) to locate. Can be a single object like 'car' or multiple objects separated by commas like 'car, person, tree'"
                }
            },
            "required": ["image_path", "task", "object_name"]
        }
    
    def call(
        self, 
        image_path: str,
        task: str,
        object_name: str
    ) -> Dict[str, Any]:
        """
        Execute vision-language task
        
        Args:
            image_path: Path to input image
            task: Task type ("point")
            object_name: Object name(s) - can be single like "car" or multiple like "car, person, tree"
            
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
            if task == "point" and not object_name:
                return {
                    "success": False,
                    "error": "Object name is required for point task"
                }
            
            # Parse object_name to detect if it's multiple objects
            object_names = [name.strip() for name in object_name.split(',')]
            is_multi_object = len(object_names) > 1
            
            logger.info(f"Detected objects: {object_names} (multi-object: {is_multi_object})")
            
            # Execute the appropriate task
            result = None
            if task == "point":
                # 统一使用 point 方法，它会自动处理单个或多个对象
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
                    "is_multi_object": is_multi_object,
                    "objects": object_names,
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
