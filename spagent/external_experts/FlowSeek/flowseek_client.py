"""
HTTP client for the FlowSeek inference server.
"""

import base64
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import requests


class FlowSeekClient:
    """HTTP client that talks to a running FlowSeek server."""

    def __init__(self, server_url: str, timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def estimate_flow(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Send two images to the server and retrieve the colorized flow result.

        Args:
            image1_path: Path to the first (source) image.
            image2_path: Path to the second (target) image.
            output_path: Where to save the returned flow image locally. Auto-generated if None.

        Returns:
            dict with keys: success, output_path, description, flow_magnitude_mean
        """
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        if img1 is None:
            return {"success": False, "error": f"Could not read: {image1_path}"}
        if img2 is None:
            return {"success": False, "error": f"Could not read: {image2_path}"}

        ext1 = Path(image1_path).suffix or ".jpg"
        ext2 = Path(image2_path).suffix or ".jpg"

        _, buf1 = cv2.imencode(ext1, img1)
        _, buf2 = cv2.imencode(ext2, img2)

        payload = {
            "image1": base64.b64encode(buf1.tobytes()).decode(),
            "image2": base64.b64encode(buf2.tobytes()).decode(),
            "ext1": ext1,
            "ext2": ext2,
        }

        try:
            resp = requests.post(
                f"{self.server_url}/infer",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

        if result.get("success") and result.get("image_b64"):
            if output_path is None:
                stem1 = Path(image1_path).stem
                stem2 = Path(image2_path).stem
                os.makedirs("outputs", exist_ok=True)
                output_path = f"outputs/flowseek_{stem1}_{stem2}.png"

            os.makedirs(Path(output_path).parent, exist_ok=True)
            img_bytes = base64.b64decode(result["image_b64"])
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            result["output_path"] = output_path

        return result
