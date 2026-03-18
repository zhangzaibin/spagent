import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Configure logging once: console + file for RoboTracer
if not logger.handlers:
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler (stderr)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (always place logs under RoboTracer module folder)
    try:
        base_dir = Path(__file__).resolve().parent
        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "robotracer_client.log"

        # Avoid duplicating the same FileHandler if the module is imported multiple times
        existing_log_paths = {
            getattr(h, "baseFilename", "") for h in logger.handlers if hasattr(h, "baseFilename")
        }
        if str(log_path) not in existing_log_paths:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    except Exception:
        # If file logging fails, we still keep console logging
        pass


ImageInput = Union[str, np.ndarray, Image.Image]


class RoboTracerClient:
    """
    Client for communicating with the RoboTracer server.

    Expected server endpoints:
        GET  /health
        GET  /test
        POST /trace

    Expected /trace request JSON:
        {
            "image": "<base64 RGB image>",
            "instruction": "pick up the red cup and place it on the shelf",
            "depth": "<optional base64 depth image>"
        }

    Expected /trace response JSON:
        {
            "success": true,
            "waypoints": [
                {"x": 0.1, "y": 0.2, "z": 0.3},
                {"x": 0.2, "y": 0.3, "z": 0.4}
            ],
            "num_steps": 2,
            "meta": {...},
            "output_path": "..."
        }
    """

    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 60):
        """
        Initialize RoboTracer client.

        Args:
            server_url: RoboTracer server base URL.
            timeout: Default timeout in seconds for POST inference requests.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _encode_image(self, image: ImageInput) -> str:
        """
        Encode an image into a base64 string.

        Supported inputs:
            - file path (str)
            - numpy ndarray
            - PIL.Image.Image

        Args:
            image: input image

        Returns:
            Base64-encoded image string.

        Raises:
            ValueError: if input type is unsupported
            FileNotFoundError: if input path does not exist
        """
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            with open(image, "rb") as f:
                image_bytes = f.read()

        elif isinstance(image, np.ndarray):
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                raise ValueError("Failed to encode numpy image to JPEG")
            image_bytes = buffer.tobytes()

        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()

        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        return base64.b64encode(image_bytes).decode("utf-8")

    def _make_output_name(self, input_path: Optional[str], prefix: str = "trace") -> str:
        """
        Create a deterministic output JSON path.

        Args:
            input_path: original input file path if available
            prefix: filename prefix

        Returns:
            Output file path string.
        """
        os.makedirs("outputs", exist_ok=True)

        if input_path and isinstance(input_path, str):
            input_filename = os.path.basename(input_path)
            name, _ = os.path.splitext(input_filename)
        else:
            name = "image"

        return os.path.join("outputs", f"{prefix}_{name}.json")

    def _save_trace_json(self, result: Dict[str, Any], input_path: Optional[str] = None) -> Optional[str]:
        """
        Save trace result as a JSON file under outputs/.

        Args:
            result: result dict to save
            input_path: original input image path if available

        Returns:
            Output file path if saved successfully, else None
        """
        try:
            output_path = self._make_output_name(input_path=input_path, prefix="trace")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(f"Trace JSON saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to save trace JSON: {e}")
            return None

    def _normalize_trace_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize server response fields into a stable schema.

        This protects the rest of the tool stack if the server returns
        slightly different key names.

        Args:
            result: raw server response dict

        Returns:
            Normalized dict
        """
        if not isinstance(result, dict):
            return {
                "success": False,
                "error": f"Invalid response type from server: {type(result)}"
            }

        # Standardize success
        success = bool(result.get("success", False))

        # Standardize waypoints
        waypoints = result.get("waypoints")
        if waypoints is None and "trajectory" in result:
            waypoints = result.get("trajectory")
        if waypoints is None:
            waypoints = []

        # Standardize num_steps
        num_steps = result.get("num_steps")
        if num_steps is None:
            num_steps = result.get("steps", len(waypoints))

        normalized = {
            "success": success,
            "waypoints": waypoints,
            "num_steps": num_steps,
        }

        if "meta" in result:
            normalized["meta"] = result["meta"]

        if "output_path" in result:
            normalized["output_path"] = result["output_path"]

        if not success:
            normalized["error"] = result.get("error", "Unknown RoboTracer server error")

        return normalized

    # -----------------------------
    # Public server utilities
    # -----------------------------
    def health_check(self) -> Dict[str, Any]:
        """
        Check server health.

        Returns:
            JSON dict from server or a standardized error dict.
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=15)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Health check result: {result}")
            return result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def test(self) -> Dict[str, Any]:
        """
        Call lightweight server test endpoint.

        Returns:
            JSON dict from server or error dict.
        """
        try:
            response = requests.get(f"{self.server_url}/test", timeout=20)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Test endpoint result: {result}")
            return result
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return {"success": False, "error": str(e)}

    def trace(
        self,
        image: ImageInput,
        instruction: str,
        depth_path: str = "",
        save_output: bool = True
    ) -> Dict[str, Any]:
        """
        Generate 3D manipulation waypoints using the RoboTracer server.

        Args:
            image: RGB image path / numpy array / PIL image
            instruction: manipulation instruction text
            depth_path: optional depth map path
            save_output: whether to save the result JSON under outputs/

        Returns:
            Normalized result dict:
                {
                    "success": True/False,
                    "waypoints": [...],
                    "num_steps": int,
                    "meta": {...},
                    "output_path": "outputs/trace_xxx.json",
                    "error": "..."
                }
        """
        try:
            # Validate instruction
            if not isinstance(instruction, str) or not instruction.strip():
                return {"success": False, "error": "Instruction cannot be empty"}

            # Encode main image
            image_b64 = self._encode_image(image)

            # Build payload
            data: Dict[str, Any] = {
                "image": image_b64,
                "instruction": instruction.strip()
            }

            # Optional depth
            if depth_path:
                if not os.path.exists(depth_path):
                    return {"success": False, "error": f"Depth file not found: {depth_path}"}
                data["depth"] = self._encode_image(depth_path)

            logger.info("Sending RoboTracer trace request...")
            response = requests.post(
                f"{self.server_url}/trace",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()

            raw_result = response.json()
            logger.info(f"Raw RoboTracer response: {raw_result}")

            result = self._normalize_trace_result(raw_result)

            if result.get("success") and save_output:
                input_path = image if isinstance(image, str) else None
                output_path = self._save_trace_json(result, input_path=input_path)
                if output_path:
                    result["output_path"] = output_path

            return result

        except requests.exceptions.Timeout:
            logger.error("Trace request timed out")
            return {"success": False, "error": "Trace request timed out"}

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to RoboTracer server: {e}")
            return {"success": False, "error": f"Could not connect to RoboTracer server: {e}"}

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during trace request: {e}")
            try:
                error_body = response.json()
            except Exception:
                error_body = {"error": str(e)}
            return {
                "success": False,
                "error": f"HTTP error during trace request: {error_body}"
            }

        except Exception as e:
            logger.error(f"Trace generation failed: {e}")
            return {"success": False, "error": str(e)}


def main():
    """
    Standalone test runner for RoboTracerClient.
    """
    SERVER_URL = "http://localhost:8000"
    TEST_IMAGE_CANDIDATES = [
        "assets/example.png",
        "assets/example.jpg",
        "test_data/robot_scene.jpg",
        "test_data/example.png",
    ]
    TEST_INSTRUCTION = "Pick up the red cup and place it on the shelf."

    client = RoboTracerClient(server_url=SERVER_URL, timeout=60)

    print("\n1 === Health check ===")
    health = client.health_check()
    print(health)

    print("\n2 === Test endpoint ===")
    test_result = client.test()
    print(test_result)

    # Find test image
    image_path = None
    for p in TEST_IMAGE_CANDIDATES:
        if os.path.exists(p):
            image_path = p
            break

    if image_path is None:
        print("\n3 === Trace test skipped ===")
        print("No test image found. Please place an image in one of these locations:")
        for p in TEST_IMAGE_CANDIDATES:
            print(f"  - {p}")
        return False

    print("\n3 === Trace generation ===")
    print(f"Using image: {image_path}")
    print(f"Instruction: {TEST_INSTRUCTION}")

    result = client.trace(
        image=image_path,
        instruction=TEST_INSTRUCTION,
        depth_path="",
        save_output=True
    )

    print("\n=== TRACE RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("success"):
        print("\n✅ Trace request succeeded")
        print(f"Waypoints: {result.get('waypoints')}")
        print(f"Num steps: {result.get('num_steps')}")
        print(f"Output path: {result.get('output_path')}")
        return True

    print("\n❌ Trace request failed")
    print(f"Error: {result.get('error')}")
    return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ RoboTracerClient test completed successfully.")
        else:
            print("\n⚠️ RoboTracerClient test did not fully succeed.")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
