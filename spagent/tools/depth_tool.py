"""
Depth Estimation Tool

This module contains the DepthEstimationTool that wraps
Depth-AnythingV2 functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class DepthEstimationTool(Tool):
    """Tool for depth estimation using Depth-AnythingV2"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://10.8.131.51:20019"):
        """
        Initialize depth estimation tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the depth estimation server
        """
        super().__init__(
            name="depth_estimation_tool",
            description="Generate a depth map for the input image to analyze the 3D spatial relationships and depth distribution of objects in the scene."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the depth estimation client"""
        if self.use_mock:
            try:
                from external_experts.Depth_AnythingV2.mock_depth_service import MockDepthService
                self._client = MockDepthService()
                logger.info("Using mock depth estimation service")
            except ImportError as e:
                logger.error(f"Failed to import mock depth service: {e}")
                raise
        else:
            try:
                from external_experts.Depth_AnythingV2.depth_client import DepthClient
                self._client = DepthClient(server_url=self.server_url)
                logger.info(f"Using real depth estimation service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real depth client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for depth estimation."
                }
            },
            "required": ["image_path"]
        }
    
    def call(self, image_path: str) -> Dict[str, Any]:
        """
        Execute depth estimation
        
        Args:
            image_path: Path to input image
            
        Returns:
            Depth estimation result dictionary
        """
        try:
            logger.info(f"Running depth estimation on: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Call depth estimation
            if hasattr(self._client, 'infer'):
                result = self._client.infer(image_path)
            else:
                # Fallback for different client interfaces
                result = self._client.process_image(image_path)
            
            if result and result.get('success'):
                logger.info("Depth estimation completed successfully")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get('output_path'),
                    "shape": result.get('shape'),
                    "depth_data": result.get('depth_data')
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Depth estimation failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Depth estimation failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Depth estimation tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 