"""
Segmentation Tool

This module contains the SegmentationTool that wraps
SAM2 functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class SegmentationTool(Tool):
    """Tool for image segmentation using SAM2"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://10.8.131.51:20020"):
        """
        Initialize segmentation tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the SAM2 server
        """
        super().__init__(
            name="segment_image_tool",
            description="Segment objects in the image based on user's request. Can use points, boxes to guide segmentation."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the SAM2 client"""
        if self.use_mock:
            try:
                from external_experts.SAM2.mock_sam2_service import MockSAM2Service
                self._client = MockSAM2Service()
                logger.info("Using mock SAM2 service")
            except ImportError:
                # Fallback to creating a simple mock
                class SimpleMockSAM2:
                    def infer(self, image_path, **kwargs):
                        stem = Path(image_path).stem
                        
                        # 模拟多个掩码（2-4个随机数量的对象）
                        import random
                        num_objects = random.randint(2, 4)
                        masks_data = []
                        
                        for i in range(num_objects):
                            masks_data.append({
                                'mask': f'mock_mask_data_{i}',
                                'id': i
                            })
                        
                        return {
                            "success": True,
                            "output_path": f"outputs/sam2_combined_{stem}.jpg",
                            "overlay_path": f"outputs/sam2_overlay_{stem}.jpg",
                            "mask_path": f"outputs/sam2_mask_{stem}.png",
                            "vis_path": f"outputs/sam2_mock_{stem}.jpg",  # Backward compatibility
                            "masks": masks_data,  # 多个掩码支持随机颜色
                            "shape": [1024, 1024]
                        }
                self._client = SimpleMockSAM2()
                logger.info("Using simple mock SAM2 service")
        else:
            try:
                from external_experts.SAM2.sam2_client import SAM2Client
                self._client = SAM2Client(server_url=self.server_url)
                logger.info(f"Using real SAM2 service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real SAM2 client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The path to the input image for segmentation."
                },
                "point_coords": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "description": "Optional list of point coordinates [[x1,y1], [x2,y2], ...]"
                },
                "point_labels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional list of point labels (1 for foreground, 0 for background)"
                },
                "box": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional bounding box coordinates [x1,y1,x2,y2]"
                }
            },
            "required": ["image_path"]
        }
    
    def call(
        self, 
        image_path: str,
        point_coords: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Execute image segmentation
        
        Args:
            image_path: Path to input image
            point_coords: Optional point coordinates for guided segmentation
            point_labels: Optional point labels (1=foreground, 0=background)
            box: Optional bounding box coordinates
            
        Returns:
            Segmentation result dictionary
        """
        try:
            logger.info(f"Running segmentation on: {image_path}")
            
            # Check if image exists
            if not Path(image_path).exists():
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}"
                }
            
            # Prepare arguments for segmentation
            seg_args = {"image_path": image_path}
            
            if point_coords is not None:
                seg_args["point_coords"] = point_coords
            if point_labels is not None:
                seg_args["point_labels"] = point_labels
            if box is not None:
                seg_args["box"] = box
            
            # Call segmentation
            if hasattr(self._client, 'infer'):
                result = self._client.infer(**seg_args)
            else:
                # Fallback for different client interfaces
                result = self._client.segment(**seg_args)
            
            if result and result.get('success'):
                logger.info("Segmentation completed successfully")
                return {
                    "success": True,
                    "result": result,
                    "output_path": result.get('output_path'),  # Combined image
                    "overlay_path": result.get('overlay_path'),  # Mask visualization
                    "mask_path": result.get('mask_path'),  # Original mask
                    "vis_path": result.get('vis_path'),  # Backward compatibility
                    "shape": result.get('shape'),
                    "masks": result.get('masks', [])
                }
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Segmentation failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Segmentation failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Segmentation tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            } 