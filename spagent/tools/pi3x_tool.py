"""
Pi3X 3D Reconstruction Tool

This module contains the Pi3XTool that wraps
Pi3X 3D reconstruction functionality for the SPAgent system.
Pi3X is an upgraded version of Pi3 with smoother point clouds,
flexible conditioning, and approximate metric scale reconstruction.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import PointCloudPayload, ToolResult

logger = logging.getLogger(__name__)


# Default (inference) tool description — dynamically generated schema used by the
# online eval / SPAgent path.
_PI3X_DEFAULT_DESCRIPTION = (
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


# RL-trained tool description — MUST stay byte-for-byte identical to the
# `<tools>` block baked into train/system_prompt/system_prompt_grpo.txt so that
# models trained with GRPO see exactly the same tool schema at eval time.
# Using this avoids the (azimuth=0, elevation=0) fallback that appears when the
# eval schema advertises 0 as an optional default.
_PI3X_RL_DESCRIPTION = (
    "3D reconstruction and novel-view synthesis tool for spatial reasoning. "
    "Given one or more images of a scene, renders the scene from a new viewpoint "
    "specified by azimuth and elevation angles.\n\n"
    "IMPORTANT: azimuth=0 AND elevation=0 repeats the first input image \u2014 never use this combination.\n\n"
    "Angle guide:\n"
    "- Top-down layout (best for left/right/behind reasoning): elevation=60, azimuth=0\n"
    "- Left side view: azimuth=-90, elevation=0\n"
    "- Right side view: azimuth=90, elevation=0\n"
    "- Back view: azimuth=180, elevation=0\n"
    "- Diagonal views: azimuth=\u00b145, elevation=0 or 30\n\n"
    "For 'from camera N / image N viewpoint' questions: set rotation_reference_camera=N, camera_view=true, elevation=30~45.\n\n"
    "Camera positions in the rendered image are shown as numbered cone frustums (cam1, cam2, \u2026)."
)


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
    elif 'mindcube' in image_path.lower():
        # 获取文件名（无扩展名）
        filename = os.path.splitext(os.path.basename(image_path))[0]  
        # 获取上一级目录名
        parent = os.path.basename(os.path.dirname(image_path))        
        # 拼接
        return f"{parent}_{filename}"
    
    # For other datasets, just return the filename (original logic)
    return os.path.splitext(os.path.basename(image_path))[0]


class Pi3XTool(Tool):
    """Tool for 3D reconstruction from images using Pi3X (upgraded version with smoother point clouds)"""
    
    def __init__(self, use_mock: bool = True, server_url: str = "http://localhost:20031", mode='inference'):
        """
        Initialize Pi3 tool
        
        Args:
            use_mock: Whether to use mock client for testing
            server_url: URL of the Pi3 server
            mode: Schema/description variant to expose.
                - 'inference' (default): dynamically generated eval schema
                  (angles optional, description mentions "Default is 0").
                - 'train': minimal schema (image_path/azimuth/elevation only).
                - 'rl': schema identical to train/system_prompt/system_prompt_grpo.txt,
                  used to eval GRPO-trained models with the exact prompt they saw.
        """
        self.mode = mode
        super().__init__(
            name="pi3x_tool",
            description=_PI3X_RL_DESCRIPTION if mode == 'rl' else _PI3X_DEFAULT_DESCRIPTION,
        )
        
        self.use_mock = use_mock
        self.server_url = server_url
        self._client = None
        
        # Initialize client
        self._init_client()
    
    def _init_client(self):
        """Initialize the Pi3X client"""
        if self.use_mock:
            try:
                from external_experts.Pi3.mock_pi3_service import MockPi3Service
                self._client = MockPi3Service()
                logger.info("Using mock Pi3X service")
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
                logger.info("Using simple mock Pi3X service")
        else:
            try:
                from external_experts.Pi3.pi3x_client import Pi3XClient
                self._client = Pi3XClient(server_url=self.server_url)
                logger.info(f"Using real Pi3X service at {self.server_url}")
            except ImportError as e:
                logger.error(f"Failed to import real Pi3X client: {e}")
                raise
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameter schema"""
        if self.mode == 'inference':
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
        elif self.mode == 'train':
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
                },
                "required": ["image_path"]
            }
        elif self.mode == 'rl':
            # Byte-for-byte identical to the schema in
            # train/system_prompt/system_prompt_grpo.txt so GRPO-trained models
            # see the exact same tool signature they were optimized against
            # (angles REQUIRED, no "Default is 0" wording).
            return {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of input image file paths"
                    },
                    "azimuth_angle": {
                        "type": "number",
                        "description": "Horizontal rotation in degrees. 0=front, 90=right, -90=left, 180=back."
                    },
                    "elevation_angle": {
                        "type": "number",
                        "description": "Vertical rotation in degrees. 0=horizontal, 60=top-down, negative=below."
                    },
                    "camera_view": {
                        "type": "boolean",
                        "description": "If true, render from the reference camera's ego perspective. Default false."
                    },
                    "rotation_reference_camera": {
                        "type": "integer",
                        "description": "1-indexed camera to use as rotation origin. Default 1."
                    }
                },
                "required": ["image_path", "azimuth_angle", "elevation_angle"]
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
            
            logger.info(f"Running Pi3X 3D reconstruction on images: {image_path}")
            
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
            cached_result = self._check_cache(
                image_path[0], azimuth_angle, elevation_angle,
                rotation_reference_camera=rotation_reference_camera,
                camera_view=camera_view
            )
            if cached_result:
                logger.info(f"Using cached result for azimuth={azimuth_angle}°, elevation={elevation_angle}°, "
                           )
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
                logger.info("Pi3X 3D reconstruction completed successfully")
                
                # Extract key information
                points_count = result.get('points_count', 0)
                ply_filename = result.get('ply_filename', 'result.ply')
                camera_views = result.get('camera_views', [])
                
                # Save generated images and get output path
                # Use first image path for naming consistency
                if self.mode == 'train':
                    output_path = self._save_generated_images(result, image_path[0])
                else:
                    # 'inference', 'rl', or any other mode: the schema exposes
                    # rotation_reference_camera + camera_view, so save with the full signature.
                    output_path = self._save_generated_images(result, image_path[0], rotation_reference_camera, camera_view)
                
                description = (
                    f"Pi3X tool has completed 3D reconstruction, generating a point cloud visualization "
                    f"with {len(image_path)} input images producing {len(image_path)} camera viewpoints. In the point cloud visualization, "
                    f"camera positions are indicated by cone-shaped markers, with their shooting direction "
                    f"pointing from the base towards the apex. The first input image corresponds to cam1 "
                    f"camera view, the second image corresponds to cam2 camera view, and so on. The generated "
                    f"multi-view renderings showcase the reconstructed 3D scene from different viewing angles."
                )

                # Standardized output: ToolResult is dict-compatible, and every
                # legacy key is kept as an extra so existing consumers (e.g. the
                # agent loop reading azimuth_angle/elevation_angle into memory
                # metadata) see the same shape.
                payload = PointCloudPayload(
                    ply_filename=ply_filename,
                    points_count=points_count,
                    camera_views=camera_views,
                )
                return ToolResult(
                    success=True,
                    payload=payload,
                    description=description,
                    output_path=output_path,  # This is what SPAgent looks for
                    result=result,
                    view_count=len(camera_views),
                    azimuth_angle=azimuth_angle,
                    elevation_angle=elevation_angle,
                    view_type="custom_angle",
                    input_images_count=len(image_path),
                )
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                logger.info(f"Not found azimuth{azimuth_angle} and elevation{elevation_angle} in {str(image_path)}.")
                logger.error(f"Pi3X 3D reconstruction failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Pi3X 3D reconstruction failed: {error_msg}"
                }
                
        except Exception as e:
            logger.error(f"Pi3X tool error: {e}")
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
            
            cache_filename = f"pi3x_{scene_id}_azim{azimuth_angle:.1f}_elev{elevation_angle:.1f}{suffix}.png"
            cache_path = os.path.join(output_dir, cache_filename)
            
            # Check if cache file exists
            if not os.path.exists(cache_path):
                print("no_cache_found: ", cache_path)
                return None
            
            logger.info(f"Found cached result: {cache_path}")
            
            # Read the cached image and convert to base64
            with open(cache_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            # Construct result in the same format as normal execution
            # (standardized ToolResult; legacy keys preserved as extras)
            camera_views = [{
                "camera": 1,
                "view": f"custom_azim_{int(azimuth_angle)}_elev_{int(elevation_angle)}",
                "azimuth_angle": int(azimuth_angle),
                "elevation_angle": int(elevation_angle),
                "image": img_base64
            }]
            raw_result = {
                "success": True,
                "ply_filename": f"cached_result_{scene_id}.ply",
                "points_count": 50000,  # Default value for cached results
                "camera_views": camera_views
            }
            payload = PointCloudPayload(
                ply_filename=raw_result["ply_filename"],
                points_count=raw_result["points_count"],
                camera_views=camera_views,
            )
            return ToolResult(
                success=True,
                payload=payload,
                description=(
                    f"Using cached Pi3X visualization from previous reconstruction "
                    f"(azimuth={int(azimuth_angle)}°, elevation={int(elevation_angle)}°). "
                    f"The cached result shows the reconstructed 3D scene from the requested viewing angle."
                ),
                output_path=cache_path,
                result=raw_result,
                view_count=1,
                azimuth_angle=azimuth_angle,
                elevation_angle=elevation_angle,
                view_type="custom_angle",
                input_images_count=1,
                cached=True,  # Mark as cached result
            )
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    def _save_generated_images(self, result: Dict[str, Any], image_path: str, rotation_reference_camera: int = 1, camera_view: bool = False) -> Optional[str]:
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
                    
                    # Create filename with scene_id, angles, and optional suffixes
                    img_filename = f"pi3x_{scene_id}_azim{azimuth:.1f}_elev{elevation:.1f}{suffix}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    # Decode and save image
                    if "image" in view_data:
                        img_data = base64.b64decode(view_data["image"])
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                        saved_images.append(img_path)
                        logger.info(f"Saved Pi3X generated image: {img_path}")
                    
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
            logger.error(f"Error saving Pi3X generated images: {e}")
            return None