"""
Pi3 3D Reconstruction Tool

This module contains the Pi3Tool that wraps
Pi3 3D reconstruction functionality for the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class Pi3Tool(Tool):
    """Tool for 3D reconstruction from single image using Pi3"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20030"):
        """
        Initialize Pi3 tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the Pi3 server
        """
        super().__init__(
            name="pi3_tool",
            description="This tool is suitable for motion and spatial reasoning tasks that involve camera movement, object rotation, or directional motion analysis," \
                        "perform 3D reconstruction from images to generate point clouds and visualizations from CUSTOM viewing angles. " \
                        "Note that the 0° azimuth angle and 0° elevation angle corresponds to the first input image viewpoint (cam1). Do not use this angle." \
                        "You can specify **azimuth_angle** (-180° to 180°, integer only; controls left-right rotation) and **elevation_angle** (-90° to 90°, integer only; controls up-down rotation) " \
                        "to view the reconstructed 3D scene from any angle.  By convention, (azimuth=0, elevation=0) corresponds EXACTLY to the first input " \
                        "image viewpoint (cam1). All rotations are defined in the INPUT CAMERA coordinate frame: azimuth rotates left/right around the camera's " \
                        "vertical axis; elevation rotates up/down around the camera's right axis. \n "\
                        "You can call this tool MULTIPLE times with DIFFERENT angles to analyze the 3D structure comprehensively; the MLLM is encouraged " \
                        "to autonomously explore angles (coarse-to-fine) until sufficient evidence is gathered. The generated visualization uses cone-shaped markers " \
                        "to indicate camera positions, numbered from 1 (cam1, cam2, etc.)."
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Pi3 client"""
        if self.use_mock:
            try:
                from external_experts.Pi3.mock_pi3_service import MockPi3Service
                self._client = MockPi3Service()
                logger.info("Using mock Pi3 service")
            except ImportError:
                # Create a simple mock client
                class SimpleMockPi3:
                    def infer_from_images(self, image_paths, azimuth_angle=None, elevation_angle=None, **kwargs):
                        """Mock 3D reconstruction"""
                        import os
                        import base64
                        from PIL import Image, ImageDraw
                        import io
                        import numpy as np
                        
                        # Generate mock result based on input
                        # Use first image from the list for naming
                        first_image = image_paths[0] if image_paths else "mock"
                        image_name = Path(first_image).stem
                        
                        # Create a mock rendered image with higher resolution
                        mock_img = Image.new('RGB', (1024, 1024), color='lightgray')
                        draw = ImageDraw.Draw(mock_img)
                        
                        # Draw some simple 3D-like shapes to simulate rendering
                        draw.ellipse([200, 200, 800, 800], fill='blue', outline='darkblue', width=3)
                        draw.rectangle([400, 300, 700, 600], fill='red', outline='darkred', width=3)
                        draw.polygon([(300, 150), (500, 100), (700, 150), (600, 250), (400, 250)], 
                                   fill='green', outline='darkgreen')
                        
                        # Create result first
                        result = {
                            "success": True,
                            "ply_filename": f"result_{image_name}.ply",
                            "points_count": 50000,
                            "camera_views": []
                        }
                        
                        # Add angle information as text with larger font
                        if azimuth_angle is not None and elevation_angle is not None:
                            # Draw text background
                            draw.rectangle([20, 20, 280, 120], fill='white', outline='black', width=2)
                            draw.text((30, 30), f"Azimuth: {azimuth_angle}°", fill='black')
                            draw.text((30, 50), f"Elevation: {elevation_angle}°", fill='black')
                            draw.text((30, 70), "Pi3 Mock Render", fill='gray')
                            draw.text((30, 90), f"Points: {result['points_count']}", fill='blue')
                        
                        # Add some visual noise to make it look more realistic
                        for i in range(100):
                            x, y = np.random.randint(50, 974, 2)
                            draw.ellipse([x-2, y-2, x+2, y+2], fill=(
                                np.random.randint(100, 255),
                                np.random.randint(100, 255), 
                                np.random.randint(100, 255)
                            ))
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        mock_img.save(img_buffer, format='PNG')
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                        
                        # Generate mock camera views with the provided angles
                        if azimuth_angle is not None and elevation_angle is not None:
                            result["camera_views"] = [{
                                "camera": 1,
                                "view": f"custom_azim_{azimuth_angle}_elev_{elevation_angle}",
                                "azimuth_angle": azimuth_angle,
                                "elevation_angle": elevation_angle,
                                "image": img_base64
                            }]
                        else:
                            # This should not happen as angles are now required
                            result["camera_views"] = [{
                                "camera": 1,
                                "view": "default_view",
                                "azimuth_angle": 0,
                                "elevation_angle": 0,
                                "image": img_base64
                            }]
                        
                        return result
                    
                    def health_check(self):
                        return {
                            "status": "健康",
                            "model_loaded": True,
                            "device": "mock_device"
                        }
                
                self._client = SimpleMockPi3()
                logger.info("Using simple mock Pi3 service")
        else:
            try:
                from external_experts.Pi3.pi3_client import Pi3Client
                self._client = Pi3Client(server_url=self.server_url)
                logger.info(f"Using real Pi3 service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real Pi3 client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "list",
                    "description": "The list of the path to the input images for 3D reconstruction."
                },
                "azimuth_angle": {
                    "type": "number",
                    "description": "Azimuth angle (left-right rotation) in degrees for custom viewpoint generation. Range: -180 to 180. Default is 0 (front view). Negative values rotate left, positive values rotate right."
                },
                "elevation_angle": {
                    "type": "number", 
                    "description": "Elevation angle (up-down rotation) in degrees for custom viewpoint generation. Range: -90 to 90. Default is 0 (horizontal). Negative values look down, positive values look up."
                }
            },
            "required": ["image_path"]
        }
    
    def call(
        self, 
        image_path: List[str],
        azimuth_angle: float = 0,
        elevation_angle: float = 0
    ) -> Dict[str, Any]:
        """
        Execute 3D reconstruction
        
        Args:
            image_path: List of paths to the input images for 3D reconstruction
            azimuth_angle: Azimuth angle for custom viewpoint (default: 0)
            elevation_angle: Elevation angle for custom viewpoint (default: 0)
            
        Returns:
            3D reconstruction result dictionary
        """
        try:
            # Validate image path list
            if not image_path or len(image_path) == 0:
                return {
                    "success": False,
                    "error": "image_path list is required and cannot be empty"
                }
            
            logger.info(f"Running Pi3 3D reconstruction on images: {image_path}")
            
            # Check if all images exist
            for img_path in image_path:
                if not Path(img_path).exists():
                    return {
                        "success": False,
                        "error": f"Image file not found: {img_path}"
                    }
            
            # Convert angles to float and validate
            try:
                azimuth_angle = float(azimuth_angle)
                elevation_angle = float(elevation_angle)
            except (ValueError, TypeError) as e:
                return {
                    "success": False,
                    "error": f"Invalid angle values: {e}"
                }
            
            # Validate angle parameters (both are now required)
            if not -180 <= azimuth_angle <= 180:
                return {
                    "success": False,
                    "error": "azimuth_angle must be between -180 and 180 degrees"
                }
            
            if not -90 <= elevation_angle <= 90:
                return {
                    "success": False,
                    "error": "elevation_angle must be between -90 and 90 degrees"
                }
            
            logger.info(f"Using angles: azimuth={azimuth_angle}°, elevation={elevation_angle}°")
            
            # Check if cached result already exists
            cached_result = self._check_cache(image_path[0], azimuth_angle, elevation_angle)
            if cached_result:
                logger.info(f"Using cached result for azimuth={azimuth_angle}°, elevation={elevation_angle}°")
                return cached_result
            
            # Execute 3D reconstruction with image list
            result = self._client.infer_from_images(
                image_paths=image_path,  # Pass the list directly
                conf_threshold=0.06,
                rtol=0.02,
                generate_views=True,
                use_filename=True,
                azimuth_angle=azimuth_angle,
                elevation_angle=elevation_angle
            )
            
            if result and result.get('success'):
                logger.info("Pi3 3D reconstruction completed successfully")
                
                # Extract key information
                points_count = result.get('points_count', 0)
                ply_filename = result.get('ply_filename', 'result.ply')
                camera_views = result.get('camera_views', [])
                
                # Save generated images and get output path
                # Use first image path for naming consistency
                output_path = self._save_generated_images(result, image_path[0])
                
                response = {
                    "success": True,
                    "result": result,
                    "points_count": points_count,
                    "ply_filename": ply_filename,
                    "view_count": len(camera_views),
                    "azimuth_angle": azimuth_angle,
                    "elevation_angle": elevation_angle,
                    "view_type": "custom_angle",
                    "input_images_count": len(image_path),
                    "output_path": output_path,  # This is what SPAgent looks for
                    "description": (
                        f"Pi3 tool has completed 3D reconstruction, generating a point cloud visualization "
                        f"with {len(image_path)} input images producing {len(image_path)} camera viewpoints. In the point cloud visualization, "
                        f"camera positions are indicated by cone-shaped markers, with their shooting direction "
                        f"pointing from the base towards the apex. The first input image corresponds to cam1 "
                        f"camera view, the second image corresponds to cam2 camera view, and so on. The generated "
                        f"multi-view renderings showcase the reconstructed 3D scene from different viewing angles."
                    )
                }
                
                return response
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.error(f"Pi3 3D reconstruction failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Pi3 3D reconstruction failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Pi3 tool error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_cache(self, image_path: str, azimuth_angle: float, elevation_angle: float) -> Optional[Dict[str, Any]]:
        """
        Check if a cached result already exists for the given angles
        
        Args:
            image_path: Original input image path for naming
            azimuth_angle: Azimuth angle
            elevation_angle: Elevation angle
            
        Returns:
            Cached result dictionary if found, None otherwise
        """
        try:
            import base64
            import os
            
            # Generate expected cache filename
            output_dir = "outputs"
            input_name = Path(image_path).stem
            cache_filename = f"pi3_{input_name}_azim{int(azimuth_angle)}_elev{int(elevation_angle)}.png"
            cache_path = os.path.join(output_dir, cache_filename)
            
            # Check if cache file exists
            if not os.path.exists(cache_path):
                return None
            
            logger.info(f"Found cached result: {cache_path}")
            
            # Read the cached image and convert to base64
            with open(cache_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Construct result in the same format as normal execution
            result = {
                "success": True,
                "result": {
                    "success": True,
                    "ply_filename": f"cached_result_{input_name}.ply",
                    "points_count": 50000,  # Default value for cached results
                    "camera_views": [{
                        "camera": 1,
                        "view": f"custom_azim_{int(azimuth_angle)}_elev_{int(elevation_angle)}",
                        "azimuth_angle": int(azimuth_angle),
                        "elevation_angle": int(elevation_angle),
                        "image": img_base64
                    }]
                },
                "points_count": 50000,
                "ply_filename": f"cached_result_{input_name}.ply",
                "view_count": 1,
                "azimuth_angle": azimuth_angle,
                "elevation_angle": elevation_angle,
                "view_type": "custom_angle",
                "input_images_count": 1,
                "output_path": cache_path,
                "cached": True,  # Mark as cached result
                "description": (
                    f"Using cached Pi3 visualization from previous reconstruction "
                    f"(azimuth={int(azimuth_angle)}°, elevation={int(elevation_angle)}°). "
                    f"The cached result shows the reconstructed 3D scene from the requested viewing angle."
                )
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None
    
    def _save_generated_images(self, result: Dict[str, Any], image_path: str) -> Optional[str]:
        """
        Save generated images from Pi3 result and return the path to the first saved image
        
        Args:
            result: Pi3 reconstruction result
            image_path: Original input image path for naming
            
        Returns:
            Path to the first saved image, or None if no images were saved
        """
        try:
            import base64
            import os
            
            camera_views = result.get('camera_views', [])
            if not camera_views:
                logger.warning("No camera views found in Pi3 result")
                return None
            
            # Create output directory
            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate base name from input image
            input_name = Path(image_path).stem
            
            saved_images = []
            
            # Save all generated images
            for i, view_data in enumerate(camera_views):
                try:
                    camera = view_data.get("camera", 1)
                    view_name = view_data.get("view", f"view_{i}")
                    azimuth = view_data.get("azimuth_angle", 0)
                    elevation = view_data.get("elevation_angle", 0)
                    
                    # Debug: log view data info
                    # Create filename with input image name, angles
                    img_filename = f"pi3_{input_name}_azim{azimuth}_elev{elevation}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    # Decode and save image
                    if "image" in view_data:
                        img_data = base64.b64decode(view_data["image"])
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        saved_images.append(img_path)
                        logger.info(f"Saved Pi3 generated image: {img_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to save camera view {i}: {e}")
                    continue
            
            # Return the first saved image path (SPAgent expects this)
            if saved_images:
                return saved_images[0]
            else:
                logger.warning("No images were successfully saved")
                return None
                
        except Exception as e:
            logger.error(f"Error saving Pi3 generated images: {e}")
            return None