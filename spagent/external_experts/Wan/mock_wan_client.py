import os
import time
import base64
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WanClient:
    """Client for Alibaba Wan (万相) video generation via DashScope API."""

    # Supported text-to-video and image-to-video model names
    T2V_MODEL = "wan2.6-t2v"
    I2V_MODEL = "wan2.6-i2v"

    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY env variable "
                "or pass api_key to WanClient."
            )
        # Default to international (Singapore) endpoint; switch to Chinese mainland
        # endpoint if needed: https://dashscope.aliyuncs.com/api/v1
        self.base_url = base_url or "https://dashscope-intl.aliyuncs.com/api/v1"

    def generate_video(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 5,
        size: str = "1280*720",
    ) -> dict:
        """
        Generate a video from a text prompt, optionally conditioned on an image.

        Args:
            prompt: Text description of the video to generate.
            image_path: Optional path to a reference image for image-to-video (first frame).
            duration: Desired video duration in seconds (2-15 for wan2.6-t2v).
            size: Resolution in 'width*height' format (e.g. '1280*720').

        Returns:
            dict with keys: success, output_path, error.
        """
        try:
            # Determine model and build input payload
            if image_path:
                model = self.I2V_MODEL
                if not os.path.exists(image_path):
                    return {"success": False, "error": f"Image not found: {image_path}"}
                # Upload image and get URL, or use base64
                # For DashScope i2v, we need a publicly accessible image URL.
                # In a real deployment you would upload to OSS first.
                # Here we use a placeholder approach.
                input_payload = {
                    "prompt": prompt,
                    "img_url": self._get_image_url(image_path),
                }
            else:
                model = self.T2V_MODEL
                input_payload = {
                    "prompt": prompt,
                }

            payload = {
                "model": model,
                "input": input_payload,
                "parameters": {
                    "size": size,
                    "duration": duration,
                },
            }

            url = f"{self.base_url}/services/aigc/video-generation/video-synthesis"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-DashScope-Async": "enable",
            }

            logger.info(f"Sending video generation request to Wan (model={model})...")
            resp = requests.post(url, json=payload, headers=headers, timeout=60)

            if not resp.ok:
                error_body = resp.text
                logger.error(f"Wan API error ({resp.status_code}): {error_body}")
                return {"success": False, "error": f"Wan API {resp.status_code}: {error_body}"}

            result = resp.json()
            task_id = result.get("output", {}).get("task_id")
            if not task_id:
                return {"success": False, "error": f"No task_id in Wan response: {result}"}

            logger.info(f"Wan task created: {task_id}")
            return self._poll_task(task_id)

        except requests.exceptions.RequestException as e:
            logger.error(f"Wan API request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Wan generation error: {e}")
            return {"success": False, "error": str(e)}

    def _get_image_url(self, image_path: str) -> str:
        """
        Convert a local image to a URL for the DashScope API.
        In production, upload to Alibaba Cloud OSS and return the URL.
        For now, return a placeholder or use base64 data URI.
        """
        # NOTE: DashScope requires a publicly accessible URL.
        # In a real deployment, you would upload to OSS here.
        # This is a placeholder that will need to be replaced with
        # actual upload logic for production use.
        logger.warning(
            "Image-to-video requires a publicly accessible image URL. "
            "In production, upload the image to OSS first."
        )
        return f"file://{os.path.abspath(image_path)}"

    def _poll_task(self, task_id: str, timeout: int = 600, interval: int = 10) -> dict:
        """Poll a DashScope async task until completion."""
        url = f"{self.base_url}/tasks/{task_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        start = time.time()

        while time.time() - start < timeout:
            resp = requests.get(url, headers=headers, timeout=30)
            if not resp.ok:
                logger.warning(f"Poll request returned {resp.status_code}: {resp.text}")
                time.sleep(interval)
                continue

            result = resp.json()
            task_status = result.get("output", {}).get("task_status", "")

            if task_status == "SUCCEEDED":
                return self._save_video(result)
            elif task_status == "FAILED":
                error_msg = result.get("output", {}).get("message", "Task failed")
                return {"success": False, "error": error_msg}

            logger.info(f"Wan generation in progress... ({int(time.time() - start)}s elapsed, status={task_status})")
            time.sleep(interval)

        return {"success": False, "error": f"Wan generation timed out after {timeout}s"}

    def _save_video(self, result: dict) -> dict:
        """Download and save the generated video from a completed task."""
        try:
            output = result.get("output", {})
            video_url = output.get("video_url")

            if not video_url:
                return {"success": False, "error": f"No video_url in Wan response: {output}"}

            logger.info(f"Downloading Wan video from: {video_url[:80]}...")
            dl_resp = requests.get(video_url, timeout=120)
            dl_resp.raise_for_status()
            video_bytes = dl_resp.content

            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/wan_{int(time.time())}.mp4"
            with open(output_path, "wb") as f:
                f.write(video_bytes)

            logger.info(f"Wan video saved to: {output_path} ({len(video_bytes)} bytes)")
            return {
                "success": True,
                "output_path": output_path,
                "file_size_bytes": len(video_bytes),
            }
        except Exception as e:
            logger.error(f"Failed to save Wan video: {e}")
            return {"success": False, "error": str(e)}
