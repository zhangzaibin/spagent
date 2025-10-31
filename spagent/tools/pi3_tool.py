"""
Pi3 3D Reconstruction Tool

This module contains the Pi3Tool that wraps
Pi3 3D reconstruction functionality for the SPAgent system.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


def extract_scene_id(image_path: str) -> str:
    """
    从图片路径中提取scene ID，适配多种数据集格式
    
    Args:
        image_path: 图片路径，支持多种格式:
            - VLM-3R/scannet: "VLM-3R/scannet_frames_25k/scene0296_01/color/000000.jpg" -> "scene0296_01"
            - VLM-3R/arkitscenes: "VLM-3R/scannet_frames_25k/arkitscenes_47333899/frame_0.jpg" -> "arkitscenes_47333899_frame_0"
            - 其他数据集: "dataset/images/file.jpg" -> "file" (仅文件名)
        
    Returns:
        scene ID字符串，VLM-3R返回scene ID，其他数据集返回文件名
    """
    # For VLM-3R datasets only, extract scene ID
    if 'vlm-3r' in image_path.lower():
        # 1. Try to extract scene ID for scannet format first
        parts = image_path.split('/')
        for part in parts:
            if part.startswith('scene') and '_' in part:
                return part
        
        # 2. For arkitscenes or other VLM-3R subdatasets (not scannet)
        path_parts = image_path.split('/')
        for part in reversed(path_parts[:-1]):  # From back to front, skip filename
            if any(c.isdigit() for c in part) or part.lower() in ['scene', 'view', 'camera']:
                filename = os.path.splitext(os.path.basename(image_path))[0]
                if filename and filename != part:
                    return f"{part}_{filename}"
                return part
    
    # For other datasets, just return the filename (original logic)
    return os.path.splitext(os.path.basename(image_path))[0]


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
            description=(
                "This tool is suitable for motion and spatial reasoning tasks that involve camera movement, "
                "object rotation, or directional motion analysis. It performs 3D reconstruction from images "
                "to generate point clouds and visualizations from CUSTOM viewing angles.\n\n"
                
                "**Important Note**: The 0° azimuth angle and 0° elevation angle corresponds to the first "
                "input image viewpoint (cam1). Do not use this angle.\n\n"
                
                "**Angle Parameters**:\n"
                "- **azimuth_angle** (-180° to 180°, integer only): Controls left-right rotation.\n"
                "- **elevation_angle** (-90° to 90°, integer only): Controls up-down rotation.\n"
                "By convention, (azimuth=0, elevation=0) corresponds EXACTLY to the first input image "
                "viewpoint (cam1). All rotations are defined in the INPUT CAMERA coordinate frame: "
                "azimuth rotates left/right around the camera's vertical axis; elevation rotates up/down "
                "around the camera's right axis.\n\n"
                
                "**rotation_reference_camera** (must be output, 1-based):This parameter is used to rotate around a specific input image's "
                "camera. By picking an image you pick its camera (e.g., set rotation_reference_camera=3 for "
                "the third image's viewpoint; defaults to 1).\n\n"
                
                "**camera_view** (must be output, boolean): This parameter is used to generate first-person perspective from "
                "the selected camera position (as if standing at that camera looking at the scene), "
                "instead of the default global bird's-eye view. This is especially useful for understanding "
                "what each camera can see and analyzing spatial relationships from specific viewpoints. "
                "Combine with rotation_reference_camera to experience the scene from different camera positions.\n\n"
                "Note that default camera_view is false. You must output camera_view = true if you want to set ego-view. If you want to set global-view, you must output camera_view = false."
                
                "**Usage Strategy**: You can call this tool MULTIPLE times with DIFFERENT angles and "
                "different camera views to analyze the 3D structure comprehensively. The MLLM is encouraged "
                "to autonomously explore angles (coarse-to-fine) until sufficient evidence is gathered. "
                "The generated visualization uses cone-shaped markers to indicate camera positions, "
                "numbered from 1 (cam1, cam2, etc.).\n"



                

            )
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
                },
                "rotation_reference_camera": {
                    "type": "integer",
                    "description": "Reference camera index (1-based) to define rotation center and axes when generating viewpoints. When you have multiple input images, try DIFFERENT values (1, 2, 3, etc.) to rotate around different camera positions for better analysis. Default is 1 (uses the first input camera)."
                },
                "camera_view": {
                    "type": "boolean",
                    "description": "Whether to use first-person camera view mode. When True, generates point cloud visualization from the selected camera's first-person perspective (as if you are standing at that camera position looking at the scene). When False (default), uses global bird's-eye view. Combine with rotation_reference_camera to view from different camera positions."
                }
            },
            "required": ["image_path"]
        }
    
    def call(
        self, 
        image_path: List[str],
        azimuth_angle: float = 0,
        elevation_angle: float = 0,
        rotation_reference_camera: int = 1,
        camera_view: bool = False
    ) -> Dict[str, Any]:
        """
        Execute 3D reconstruction
        
        Args:
            image_path: List of paths to the input images for 3D reconstruction
            azimuth_angle: Azimuth angle for custom viewpoint (default: 0)
            elevation_angle: Elevation angle for custom viewpoint (default: 0)
            rotation_reference_camera: Reference camera index for rotation (default: 1)
            camera_view: Whether to use first-person camera view mode (default: False)
            
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
            cached_result = self._check_cache(image_path[0], azimuth_angle, elevation_angle, 
                                              rotation_reference_camera, camera_view)
            if cached_result:
                logger.info(f"Using cached result for azimuth={azimuth_angle}°, elevation={elevation_angle}°, "
                           f"refcam={rotation_reference_camera}, camera_view={camera_view}")
                return cached_result
            
            # Execute 3D reconstruction with image list
            result = self._client.infer_from_images(
                image_paths=image_path,  # Pass the list directly
                conf_threshold=0.06,
                rtol=0.02,
                generate_views=True,
                use_filename=True,
                azimuth_angle=azimuth_angle,
                elevation_angle=elevation_angle,
                rotation_reference_camera=rotation_reference_camera,
                camera_view=camera_view
            )
            
            if result and result.get('success'):
                logger.info("Pi3 3D reconstruction completed successfully")
                
                # Extract key information
                points_count = result.get('points_count', 0)
                ply_filename = result.get('ply_filename', 'result.ply')
                camera_views = result.get('camera_views', [])
                
                # Save generated images and get output path
                # Use first image path for naming consistency
                output_path = self._save_generated_images(result, image_path[0], 
                                                          rotation_reference_camera, camera_view)
                
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
    
    def _check_cache(self, image_path: str, azimuth_angle: float, elevation_angle: float, 
                     rotation_reference_camera: int = 1, camera_view: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check if a cached result already exists for the given angles
        
        Args:
            image_path: Original input image path for naming
            azimuth_angle: Azimuth angle
            elevation_angle: Elevation angle
            rotation_reference_camera: Reference camera index (1-based)
            camera_view: Whether using camera view mode
            
        Returns:
            Cached result dictionary if found, None otherwise
        """
        try:
            import base64
            import os
            
            # Generate expected cache filename
            tool_file = Path(__file__).resolve()
            project_root = tool_file.parent.parent.parent
            output_dir = project_root / "outputs"
            scene_id = extract_scene_id(image_path)
            
            # Build filename with suffixes for camera_view and rotation_reference_camera
            suffix = ""
            if rotation_reference_camera != 1:
                suffix += f"_refcam{rotation_reference_camera}"
            if camera_view:
                suffix += "_camview"
            
            cache_filename = f"pi3_{scene_id}_azim{azimuth_angle:.1f}_elev{elevation_angle:.1f}{suffix}.png"
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
                    "ply_filename": f"cached_result_{scene_id}.ply",
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
                "ply_filename": f"cached_result_{scene_id}.ply",
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
    
    def _save_generated_images(self, result: Dict[str, Any], image_path: str, 
                               rotation_reference_camera: int = 1, camera_view: bool = False) -> Optional[str]:
        """
        Save generated images from Pi3 result and return the path to the first saved image
        
        Args:
            result: Pi3 reconstruction result
            image_path: Original input image path for naming
            rotation_reference_camera: Reference camera index (1-based)
            camera_view: Whether using camera view mode
            
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
            
            # Generate scene ID from input image path
            scene_id = extract_scene_id(image_path)
            
            # Build suffix for filename based on parameters
            suffix = ""
            if rotation_reference_camera != 1:
                suffix += f"_refcam{rotation_reference_camera}"
            if camera_view:
                suffix += "_camview"
            
            saved_images = []
            
            # Save all generated images
            for i, view_data in enumerate(camera_views):
                try:
                    camera = view_data.get("camera", 1)
                    view_name = view_data.get("view", f"view_{i}")
                    azimuth = view_data.get("azimuth_angle", 0)
                    elevation = view_data.get("elevation_angle", 0)
                    
                    # Debug: log view data info
                    # Create filename with scene_id, angles, and optional suffixes
                    # Using format: pi3_{scene_id}_azim{azimuth}_elev{elevation}[_refcam{N}][_camview].png
                    img_filename = f"pi3_{scene_id}_azim{azimuth:.1f}_elev{elevation:.1f}{suffix}.png"
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