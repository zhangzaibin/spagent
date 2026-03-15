"""
Tool Direct Test Module

Directly test external expert tools without going through LLM Agent.
This is useful for verifying tool functionality, debugging, and development.

Usage:
    # Test Pi3 tool
    python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 45 --elevation -30

    # Test Pi3X tool (upgraded version with smoother point clouds)
    python test/test_tool.py --tool pi3x --image assets/dog.jpeg --azimuth 45 --elevation -30

    # Test Pi3 with camera view mode
    python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 90 --elevation 0 --camera_view

    # Test Pi3 with multiple images
    python test/test_tool.py --tool pi3 --image img1.jpg img2.jpg --azimuth 45 --elevation -30

    # Test Pi3 with custom server url
    python test/test_tool.py --tool pi3 --image assets/dog.jpeg --azimuth 45 --elevation -30 --server_url http://10.7.8.94:20030

    # Test Veo video generation (text-to-video)
    python test/test_tool.py --tool veo --image dummy --prompt "A dog running on a beach" --duration 8

    # Test Veo video generation (image-to-video)
    python test/test_tool.py --tool veo --image assets/dog.jpeg --prompt "The dog starts running" --duration 8

    # Test Sora video generation (text-to-video)
    python test/test_tool.py --tool sora --image dummy --prompt "A cat playing with yarn" --duration 10 --resolution 1080p

    # Test Sora video generation (image-to-video)
    python test/test_tool.py --tool sora --image assets/dog.jpeg --prompt "The dog starts running" --duration 10

    # Test with mock service (no API keys needed)
    python test/test_tool.py --tool veo --image dummy --prompt "test video" --use_mock
    python test/test_tool.py --tool sora --image dummy --prompt "test video" --use_mock
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Pi3 Tool Test
# ============================================================

def test_pi3(
    image_paths: List[str],
    azimuth_angle: float = 45,
    elevation_angle: float = 0,
    server_url: str = "http://localhost:20030",
    rotation_reference_camera: int = 1,
    camera_view: bool = False,
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Pi3 3D reconstruction tool.

    Given input image(s) and a target viewing angle, call Pi3 to perform
    3D reconstruction and render a point-cloud visualization from the
    specified angle.  The rendered image is saved to *output_dir*.

    Args:
        image_paths: List of input image paths.
        azimuth_angle: Horizontal rotation angle (-180 ~ 180).
        elevation_angle: Vertical rotation angle (-90 ~ 90).
        server_url: Pi3 server URL.
        rotation_reference_camera: Reference camera index (1-based).
        camera_view: If True, use first-person camera perspective.
        output_dir: Directory to save the rendered output image.

    Returns:
        Path to the saved output image, or None on failure.
    """
    from spagent.tools import Pi3Tool

    # --- validate inputs ---
    for p in image_paths:
        if not os.path.exists(p):
            logger.error(f"Image not found: {p}")
            return None

    logger.info("=" * 60)
    logger.info("Pi3 Tool Test")
    logger.info("=" * 60)
    logger.info(f"  Input images     : {image_paths}")
    logger.info(f"  Azimuth angle    : {azimuth_angle}°")
    logger.info(f"  Elevation angle  : {elevation_angle}°")
    logger.info(f"  Server URL       : {server_url}")
    logger.info(f"  Ref camera       : {rotation_reference_camera}")
    logger.info(f"  Camera view      : {camera_view}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info("-" * 60)

    # --- create tool (real mode, no mock) ---
    tool = Pi3Tool(
        use_mock=False,
        server_url=server_url,
        mode="inference",
    )

    # --- call ---
    result = tool.call(
        image_path=image_paths,
        azimuth_angle=azimuth_angle,
        elevation_angle=elevation_angle,
        rotation_reference_camera=rotation_reference_camera,
        camera_view=camera_view,
    )

    # --- handle result ---
    if not result.get("success"):
        logger.error(f"Pi3 tool failed: {result.get('error', 'unknown error')}")
        return None

    logger.info("Pi3 reconstruction succeeded!")
    logger.info(f"  Points count : {result.get('points_count', 'N/A')}")
    logger.info(f"  View count   : {result.get('view_count', 'N/A')}")
    logger.info(f"  PLY file     : {result.get('ply_filename', 'N/A')}")

    # The tool already saves the image via _save_generated_images and returns
    # its path in result["output_path"].  We copy it to the requested
    # output_dir so all tool-test outputs live in one place.
    src_path = result.get("output_path")
    if src_path and os.path.exists(src_path):
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        logger.info(f"  Output saved : {dst_path}")
        return dst_path
    else:
        logger.warning("No output image path in result.")
        return None


# ============================================================
# Pi3X Tool Test
# ============================================================

def test_pi3x(
    image_paths: List[str],
    azimuth_angle: float = 45,
    elevation_angle: float = 0,
    server_url: str = "http://localhost:20031",
    rotation_reference_camera: int = 1,
    camera_view: bool = False,
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Pi3X 3D reconstruction tool (upgraded version with smoother point clouds).

    Args:
        image_paths: List of input image paths.
        azimuth_angle: Horizontal rotation angle (-180 ~ 180).
        elevation_angle: Vertical rotation angle (-90 ~ 90).
        server_url: Pi3X server URL.
        rotation_reference_camera: Reference camera index (1-based).
        camera_view: If True, use first-person camera perspective.
        output_dir: Directory to save the rendered output image.

    Returns:
        Path to the saved output image, or None on failure.
    """
    from spagent.tools import Pi3XTool

    # --- validate inputs ---
    for p in image_paths:
        if not os.path.exists(p):
            logger.error(f"Image not found: {p}")
            return None

    logger.info("=" * 60)
    logger.info("Pi3X Tool Test")
    logger.info("=" * 60)
    logger.info(f"  Input images     : {image_paths}")
    logger.info(f"  Azimuth angle    : {azimuth_angle}°")
    logger.info(f"  Elevation angle  : {elevation_angle}°")
    logger.info(f"  Server URL       : {server_url}")
    logger.info(f"  Ref camera       : {rotation_reference_camera}")
    logger.info(f"  Camera view      : {camera_view}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info("-" * 60)

    # --- create tool (real mode, no mock) ---
    tool = Pi3XTool(
        use_mock=False,
        server_url=server_url,
        mode="inference",
    )

    # --- call ---
    result = tool.call(
        image_path=image_paths,
        azimuth_angle=azimuth_angle,
        elevation_angle=elevation_angle,
        rotation_reference_camera=rotation_reference_camera,
        camera_view=camera_view,
    )

    # --- handle result ---
    if not result.get("success"):
        logger.error(f"Pi3X tool failed: {result.get('error', 'unknown error')}")
        return None

    logger.info("Pi3X reconstruction succeeded!")
    logger.info(f"  Points count : {result.get('points_count', 'N/A')}")
    logger.info(f"  View count   : {result.get('view_count', 'N/A')}")
    logger.info(f"  PLY file     : {result.get('ply_filename', 'N/A')}")

    src_path = result.get("output_path")
    if src_path and os.path.exists(src_path):
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        logger.info(f"  Output saved : {dst_path}")
        return dst_path
    else:
        logger.warning("No output image path in result.")
        return None


# ============================================================
# Depth Tool Test  (placeholder for future)
# ============================================================

def test_depth(
    image_path: str,
    server_url: str = "http://localhost:20019",
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Depth Estimation tool.  (TODO: implement)
    """
    logger.info("Depth tool test is not implemented yet.")
    return None


# ============================================================
# Segmentation Tool Test  (placeholder for future)
# ============================================================

def test_segmentation(
    image_path: str,
    server_url: str = "http://localhost:20020",
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Segmentation tool.  (TODO: implement)
    """
    logger.info("Segmentation tool test is not implemented yet.")
    return None


# ============================================================
# Detection Tool Test  (placeholder for future)
# ============================================================

def test_detection(
    image_path: str,
    prompt: str = "object",
    server_url: str = "http://localhost:20022",
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Object Detection tool.  (TODO: implement)
    """
    logger.info("Detection tool test is not implemented yet.")
    return None


# ============================================================
# Veo Video Generation Tool Test
# ============================================================

def test_veo(
    prompt: str,
    image_path: str = None,
    duration: int = 8,
    aspect_ratio: str = "16:9",
    use_mock: bool = False,
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Veo video generation tool.

    Args:
        prompt: Text description of the video to generate.
        image_path: Optional reference image for image-to-video.
        duration: Video duration in seconds (5 or 8).
        aspect_ratio: '16:9' or '9:16'.
        use_mock: If True, use mock service instead of real API.
        output_dir: Directory to save output video.

    Returns:
        Path to the saved output video, or None on failure.
    """
    from spagent.tools import VeoTool

    if image_path and not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None

    logger.info("=" * 60)
    logger.info("Veo Video Generation Tool Test")
    logger.info("=" * 60)
    logger.info(f"  Prompt           : {prompt[:80]}...")
    logger.info(f"  Image path       : {image_path or 'None (text-to-video)'}")
    logger.info(f"  Duration         : {duration}s")
    logger.info(f"  Aspect ratio     : {aspect_ratio}")
    logger.info(f"  Use mock         : {use_mock}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info("-" * 60)

    tool = VeoTool(use_mock=use_mock)

    call_kwargs = {
        "prompt": prompt,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
    }
    if image_path:
        call_kwargs["image_path"] = image_path

    result = tool.call(**call_kwargs)

    if not result.get("success"):
        logger.error(f"Veo tool failed: {result.get('error', 'unknown error')}")
        return None

    logger.info("Veo video generation succeeded!")

    src_path = result.get("output_path")
    if src_path and os.path.exists(src_path):
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        logger.info(f"  Output saved : {dst_path}")
        return dst_path
    else:
        logger.warning("No output video path in result.")
        return None


# ============================================================
# Sora Video Generation Tool Test
# ============================================================

def test_sora(
    prompt: str,
    image_path: str = None,
    duration: int = 10,
    resolution: str = "1080p",
    aspect_ratio: str = "16:9",
    use_mock: bool = False,
    output_dir: str = "outputs/tool_test",
) -> Optional[str]:
    """
    Directly test Sora video generation tool.

    Args:
        prompt: Text description of the video to generate.
        image_path: Optional reference image for image-to-video.
        duration: Video duration in seconds (5-20).
        resolution: '480p', '720p', or '1080p'.
        aspect_ratio: '16:9', '9:16', or '1:1'.
        use_mock: If True, use mock service instead of real API.
        output_dir: Directory to save output video.

    Returns:
        Path to the saved output video, or None on failure.
    """
    from spagent.tools import SoraTool

    if image_path and not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None

    logger.info("=" * 60)
    logger.info("Sora Video Generation Tool Test")
    logger.info("=" * 60)
    logger.info(f"  Prompt           : {prompt[:80]}...")
    logger.info(f"  Image path       : {image_path or 'None (text-to-video)'}")
    logger.info(f"  Duration         : {duration}s")
    logger.info(f"  Resolution       : {resolution}")
    logger.info(f"  Aspect ratio     : {aspect_ratio}")
    logger.info(f"  Use mock         : {use_mock}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info("-" * 60)

    tool = SoraTool(use_mock=use_mock)

    call_kwargs = {
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }
    if image_path:
        call_kwargs["image_path"] = image_path

    result = tool.call(**call_kwargs)

    if not result.get("success"):
        logger.error(f"Sora tool failed: {result.get('error', 'unknown error')}")
        return None

    logger.info("Sora video generation succeeded!")

    src_path = result.get("output_path")
    if src_path and os.path.exists(src_path):
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst_path)
        logger.info(f"  Output saved : {dst_path}")
        return dst_path
    else:
        logger.warning("No output video path in result.")
        return None


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Directly test SPAgent external expert tools (without LLM Agent)."
    )

    parser.add_argument(
        "--tool",
        type=str,
        required=True,
        choices=["pi3", "pi3x", "depth", "segmentation", "detection", "veo", "sora"],
        help="Which tool to test.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=None,
        help="Input image path(s). Required for pi3/pi3x/depth/segmentation/detection. Optional for veo/sora.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/tool_test",
        help="Directory to save output images (default: outputs/tool_test).",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default=None,
        help="Server URL for the tool (uses tool default if not specified).",
    )

    # --- Pi3 specific ---
    pi3_group = parser.add_argument_group("Pi3 options")
    pi3_group.add_argument(
        "--azimuth",
        type=float,
        default=45,
        help="Azimuth angle in degrees, -180~180 (default: 45).",
    )
    pi3_group.add_argument(
        "--elevation",
        type=float,
        default=0,
        help="Elevation angle in degrees, -90~90 (default: 0).",
    )
    pi3_group.add_argument(
        "--ref_camera",
        type=int,
        default=1,
        help="Rotation reference camera index, 1-based (default: 1).",
    )
    pi3_group.add_argument(
        "--camera_view",
        action="store_true",
        help="Use first-person camera view mode.",
    )

    # --- Detection specific ---
    det_group = parser.add_argument_group("Detection options")
    det_group.add_argument(
        "--prompt",
        type=str,
        default="object",
        help="Text prompt for object detection (default: 'object'). Also used as the video prompt for veo/sora.",
    )

    # --- Video generation (Veo / Sora) specific ---
    vid_group = parser.add_argument_group("Video generation options (Veo / Sora)")
    vid_group.add_argument(
        "--duration",
        type=int,
        default=8,
        help="Video duration in seconds (default: 8).",
    )
    vid_group.add_argument(
        "--aspect_ratio",
        type=str,
        default="16:9",
        help="Aspect ratio, e.g. '16:9', '9:16', '1:1' (default: '16:9').",
    )
    vid_group.add_argument(
        "--resolution",
        type=str,
        default="1080p",
        choices=["480p", "720p", "1080p"],
        help="Video resolution for Sora (default: '1080p').",
    )
    vid_group.add_argument(
        "--use_mock",
        action="store_true",
        help="Use mock service instead of real API (for testing without keys).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_required_tools = {"pi3", "pi3x", "depth", "segmentation", "detection"}
    if args.tool in image_required_tools and not args.image:
        print(f"Error: --image is required for tool '{args.tool}'")
        sys.exit(1)

    if args.tool == "pi3":
        server = args.server_url or "http://localhost:20030"
        result_path = test_pi3(
            image_paths=args.image,
            azimuth_angle=args.azimuth,
            elevation_angle=args.elevation,
            server_url=server,
            rotation_reference_camera=args.ref_camera,
            camera_view=args.camera_view,
            output_dir=args.output_dir,
        )

    elif args.tool == "pi3x":
        server = args.server_url or "http://localhost:20031"
        result_path = test_pi3x(
            image_paths=args.image,
            azimuth_angle=args.azimuth,
            elevation_angle=args.elevation,
            server_url=server,
            rotation_reference_camera=args.ref_camera,
            camera_view=args.camera_view,
            output_dir=args.output_dir,
        )

    elif args.tool == "depth":
        server = args.server_url or "http://localhost:20019"
        result_path = test_depth(
            image_path=args.image[0],
            server_url=server,
            output_dir=args.output_dir,
        )

    elif args.tool == "segmentation":
        server = args.server_url or "http://localhost:20020"
        result_path = test_segmentation(
            image_path=args.image[0],
            server_url=server,
            output_dir=args.output_dir,
        )

    elif args.tool == "detection":
        server = args.server_url or "http://localhost:20022"
        result_path = test_detection(
            image_path=args.image[0],
            prompt=args.prompt,
            server_url=server,
            output_dir=args.output_dir,
        )

    elif args.tool == "veo":
        result_path = test_veo(
            prompt=args.prompt,
            image_path=args.image[0] if args.image else None,
            duration=args.duration,
            aspect_ratio=args.aspect_ratio,
            use_mock=args.use_mock,
            output_dir=args.output_dir,
        )

    elif args.tool == "sora":
        result_path = test_sora(
            prompt=args.prompt,
            image_path=args.image[0] if args.image else None,
            duration=args.duration,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
            use_mock=args.use_mock,
            output_dir=args.output_dir,
        )

    # --- summary ---
    print()
    if result_path:
        print(f"✅ Test passed!  Output: {result_path}")
    else:
        print("❌ Test failed.  See logs above for details.")


if __name__ == "__main__":
    main()

