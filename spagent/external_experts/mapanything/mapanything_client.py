# MapAnything Client - Python client for MapAnything inference service
# Designed to match Pi3Client interface for easy comparison

import base64
import cv2
import requests
import numpy as np
import os
import logging
import time
from typing import List, Optional, Dict, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MapAnythingClient:
    """MapAnything 3D reconstruction client"""
    
    def __init__(self, server_url: str = "http://localhost:20033"):
        """
        Initialize client
        
        Args:
            server_url: Server address, e.g., 'http://localhost:20033'
        """
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 300  # 5 minute timeout for 3D reconstruction
        
    def health_check(self) -> Optional[Dict[str, Any]]:
        """Check server health status"""
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Health check failed, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None
    
    def test_infer(self) -> Optional[Dict[str, Any]]:
        """Test inference with built-in test data"""
        try:
            response = self.session.get(f"{self.server_url}/test")
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Test inference failed, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Test inference failed: {e}")
            return None
    
    def _encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 string"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
                
            # Read image using cv2 (BGR format)
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', image_bgr)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None
    
    def _encode_video_frames(self, video_path: str, interval: int = 10, 
                            save_frames: bool = False, 
                            output_dir: str = "outputs/video_frames") -> Optional[tuple]:
        """Extract frames from video and encode as base64"""
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
            
            if not video_path.lower().endswith('.mp4'):
                logger.error(f"Only MP4 format is supported: {video_path}")
                return None
            
            logger.info(f"Extracting frames from video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return None
            
            if save_frames:
                os.makedirs(output_dir, exist_ok=True)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_output_dir = os.path.join(output_dir, video_name)
                os.makedirs(frame_output_dir, exist_ok=True)
                logger.info(f"Saving frames to: {frame_output_dir}")
            
            encoded_frames = []
            frame_idx = 0
            extracted_frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Ensure size is divisible by patch size (14)
                    h, w = rgb_frame.shape[:2]
                    patch_size = 14
                    new_h = ((h + patch_size - 1) // patch_size) * patch_size
                    new_w = ((w + patch_size - 1) // patch_size) * patch_size
                    
                    if new_h != h or new_w != w:
                        rgb_frame = cv2.resize(rgb_frame, (new_w, new_h))
                    
                    if save_frames:
                        frame_filename = f"frame_{extracted_frame_count:04d}_idx_{frame_idx:06d}.jpg"
                        frame_path = os.path.join(frame_output_dir, frame_filename)
                        cv2.imwrite(frame_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                    
                    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    encoded_frames.append(frame_b64)
                    extracted_frame_count += 1
                
                frame_idx += 1
            
            cap.release()
            
            if not encoded_frames:
                logger.error("No frames extracted from video")
                return None
            
            logger.info(f"Extracted {len(encoded_frames)} frames from video")
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            return (encoded_frames, video_name)
            
        except Exception as e:
            logger.error(f"Failed to process video: {e}")
            return None
    
    def infer_from_images(self, 
                         image_paths: List[str], 
                         conf_threshold: float = 0.08,
                         generate_views: bool = True,
                         use_filename: bool = True,
                         azimuth_angle: Optional[float] = None,
                         elevation_angle: Optional[float] = None,
                         rotation_reference_camera: int = 1,
                         camera_view: bool = False,
                         remove_outliers: bool = True,
                         apply_mask: bool = True) -> Optional[Dict[str, Any]]:
        """
        Perform 3D reconstruction from image list
        
        Args:
            image_paths: List of image file paths
            conf_threshold: Confidence threshold
            generate_views: Whether to generate multi-view images
            use_filename: Whether to use filename for output naming
            azimuth_angle: Custom azimuth angle (horizontal rotation) in degrees
            elevation_angle: Custom elevation angle (vertical rotation) in degrees
            rotation_reference_camera: Reference camera index (1-based)
            camera_view: Whether to use camera view mode
            remove_outliers: Whether to remove outlier points
            apply_mask: Whether to apply mask during inference
            
        Returns:
            Reconstruction result dictionary with PLY file and view images
        """
        try:
            # Encode all images
            encoded_images = []
            image_names = []
            for img_path in image_paths:
                encoded = self._encode_image(img_path)
                if encoded:
                    encoded_images.append(encoded)
                    if use_filename:
                        image_names.append(img_path)
                else:
                    logger.error(f"Failed to encode image: {img_path}")
                    return None
            
            if not encoded_images:
                logger.error("No images were successfully encoded")
                return None
            
            # Build request data
            request_data = {
                "images": encoded_images,
                "conf_threshold": conf_threshold,
                "generate_views": generate_views,
                "rotation_reference_camera": rotation_reference_camera,
                "camera_view": camera_view,
                "remove_outliers": remove_outliers,
                "apply_mask": apply_mask,
                "conf_percentile": 30
            }
            
            if use_filename and image_names:
                request_data["image_names"] = image_names
            
            if azimuth_angle is not None and elevation_angle is not None:
                request_data["azimuth_angle"] = azimuth_angle
                request_data["elevation_angle"] = elevation_angle
            
            # Send request
            logger.info(f"Sending {len(encoded_images)} images for 3D reconstruction...")
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):                    
                    return result
                else:
                    logger.error(f"3D reconstruction failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                logger.error(f"Request failed, status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"3D reconstruction request failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def infer_from_video(self, 
                        video_path: str, 
                        interval: int = 10,
                        conf_threshold: float = 0.08,
                        generate_views: bool = True,
                        max_views_per_camera: int = 7,
                        azimuth_angle: Optional[float] = None,
                        elevation_angle: Optional[float] = None,
                        rotation_reference_camera: int = 1,
                        camera_view: bool = False,
                        save_frames: bool = False,
                        frames_output_dir: str = "outputs/video_frames",
                        remove_outliers: bool = True,
                        apply_mask: bool = True) -> Optional[Dict[str, Any]]:
        """
        Perform 3D reconstruction from video file
        
        Args:
            video_path: Path to video file (MP4 format)
            interval: Frame sampling interval
            conf_threshold: Confidence threshold
            generate_views: Whether to generate multi-view images
            max_views_per_camera: Maximum number of view images per camera
            azimuth_angle: Custom azimuth angle in degrees
            elevation_angle: Custom elevation angle in degrees
            rotation_reference_camera: Reference camera index (1-based)
            camera_view: Whether to use camera view mode
            save_frames: Whether to save extracted frames locally
            frames_output_dir: Output directory for saved frames
            remove_outliers: Whether to remove outlier points
            apply_mask: Whether to apply mask during inference
            
        Returns:
            Reconstruction result dictionary with PLY file and view images
        """
        try:
            # Extract frames from video
            result = self._encode_video_frames(video_path, interval, 
                                              save_frames=save_frames, 
                                              output_dir=frames_output_dir)
            if not result:
                return None
            
            encoded_frames, video_name = result
            
            # Build request data
            request_data = {
                "images": encoded_frames,
                "conf_threshold": conf_threshold,
                "generate_views": generate_views,
                "image_names": [video_name],
                "rotation_reference_camera": rotation_reference_camera,
                "camera_view": camera_view,
                "remove_outliers": remove_outliers,
                "apply_mask": apply_mask
            }
            
            if azimuth_angle is not None and elevation_angle is not None:
                request_data["azimuth_angle"] = azimuth_angle
                request_data["elevation_angle"] = elevation_angle
                view_mode = "camera view" if camera_view else "global view"
                logger.info(f"Using custom angles: azim={azimuth_angle}°, elev={elevation_angle}°, mode={view_mode}")
            else:
                request_data["max_views_per_camera"] = max_views_per_camera
            
            # Send request
            logger.info(f"Sending {len(encoded_frames)} frames for 3D reconstruction...")
            response = self.session.post(
                f"{self.server_url}/infer",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info("Video 3D reconstruction successful!")
                    
                    # Limit views per camera
                    if "camera_views" in result and result["camera_views"]:
                        original_views = result["camera_views"]
                        limited_views = []
                        
                        camera_groups = {}
                        for view in original_views:
                            camera_id = view.get("camera", 1)
                            if camera_id not in camera_groups:
                                camera_groups[camera_id] = []
                            camera_groups[camera_id].append(view)
                        
                        for camera_id, views in camera_groups.items():
                            limited_camera_views = views[:max_views_per_camera]
                            limited_views.extend(limited_camera_views)
                        
                        result["camera_views"] = limited_views
                    
                    return result
                else:
                    logger.error(f"Video 3D reconstruction failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                logger.error(f"Request failed, status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Video 3D reconstruction request failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def save_results(self, result: Dict[str, Any], output_dir: str = "outputs", 
                    rotation_reference_camera: int = 1, camera_view: bool = False) -> bool:
        """
        Save reconstruction results to local files
        
        Args:
            result: Reconstruction result dictionary
            output_dir: Output directory
            rotation_reference_camera: Reference camera index (1-based)
            camera_view: Whether using camera view mode
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Build suffix for filename
            suffix = ""
            if rotation_reference_camera != 1:
                suffix += f"_refcam{rotation_reference_camera}"
            if camera_view:
                suffix += "_camview"
            
            # Save PLY file
            if "ply_file" in result:
                ply_filename = result.get("ply_filename", "mapanything_result.ply")
                ply_path = os.path.join(output_dir, ply_filename)
                
                ply_data = base64.b64decode(result["ply_file"])
                with open(ply_path, 'wb') as f:
                    f.write(ply_data)
                logger.info(f"PLY file saved: {ply_path}")
            
            # Save view images
            if "camera_views" in result and result["camera_views"]:
                views_dir = os.path.join(output_dir, "camera_views")
                os.makedirs(views_dir, exist_ok=True)
                
                for view_data in result["camera_views"]:
                    camera = view_data.get("camera", 1)
                    view_name = view_data.get("view", "unknown")
                    azimuth = view_data.get("azimuth_angle", 0)
                    elevation = view_data.get("elevation_angle", 0)
                    
                    img_filename = f"mapanything_cam{camera:02d}_azim{azimuth:.1f}_elev{elevation:.1f}{suffix}.png"
                    img_path = os.path.join(views_dir, img_filename)
                    
                    img_data = base64.b64decode(view_data["image"])
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                logger.info(f"View images saved to: {views_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False


if __name__ == "__main__":
    # Test MapAnything client
    client = MapAnythingClient()
    
    # 1. Health check
    logger.info("\n=== Running health check ===")
    health = client.health_check()
    if health:
        logger.info(f"Server status: {health}")
    else:
        logger.error("Server health check failed, exiting")
        exit(1)
    
    # 2. Test inference
    logger.info("\n=== Running test inference ===")
    test_result = client.test_infer()
    if test_result:
        logger.info(f"Test inference successful: {test_result}")
    else:
        logger.error("Test inference failed, exiting")
        exit(1)
    
    # 3. Process images
    logger.info("\n=== Processing images ===")
    
    test_images = [
        "dataset/BLINK_images/Multi-view_Reasoning_val_000100_img1.jpg",
        "dataset/BLINK_images/Multi-view_Reasoning_val_000100_img2.jpg"
    ]
    
    # Check if test images exist
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if existing_images:
        logger.info(f"Using test images: {existing_images}")
        
        # Test with multiple images
        multi_result = client.infer_from_images(
            image_paths=existing_images,
            conf_threshold=0.08,
            generate_views=True,
            azimuth_angle=0,
            elevation_angle=30,
            rotation_reference_camera=1,
            camera_view=False
        )
        
        if multi_result:
            logger.info("Image 3D reconstruction successful!")
            logger.info(f"- Points count: {multi_result.get('points_count', 'unknown')}")
            logger.info(f"- PLY filename: {multi_result.get('ply_filename', 'unknown')}")
            logger.info(f"- View count: {len(multi_result.get('camera_views', []))}")
            
            # Save results
            if client.save_results(multi_result, "outputs/mapanything_test"):
                logger.info("Results saved successfully!")
            else:
                logger.error("Failed to save results")
        else:
            logger.error("Image 3D reconstruction failed")
    else:
        logger.warning("No test images found")
        logger.info("Expected test image paths:")
        for img in test_images:
            logger.info(f"  - {img}")
    
    # 4. Process video (if available)
    logger.info("\n=== Processing video ===")
    
    test_videos = ["dataset/VSI_videos/arkitscenes_41069048.mp4"]
    
    video_path = None
    for test_video in test_videos:
        if os.path.exists(test_video):
            video_path = test_video
            break
    
    if video_path:
        logger.info(f"Using test video: {video_path}")
        video_result = client.infer_from_video(
            video_path=video_path,
            interval=200,
            conf_threshold=0.08,
            generate_views=True,
            max_views_per_camera=7,
            azimuth_angle=0,
            elevation_angle=-45,
            rotation_reference_camera=1,
            camera_view=False,
            save_frames=False
        )
        
        if video_result:
            logger.info("Video 3D reconstruction successful!")
            logger.info(f"- Points count: {video_result.get('points_count', 'unknown')}")
            logger.info(f"- PLY filename: {video_result.get('ply_filename', 'unknown')}")
            logger.info(f"- View count: {len(video_result.get('camera_views', []))}")
            
            if client.save_results(video_result, "outputs/mapanything_video"):
                logger.info("Video results saved successfully!")
            else:
                logger.error("Failed to save video results")
        else:
            logger.error("Video 3D reconstruction failed")
    else:
        logger.warning("No test video found")
