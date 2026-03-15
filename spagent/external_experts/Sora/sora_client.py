import os
import time
import base64
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIZES_SORA2 = {
    "16:9": "1280x720",
    "9:16": "720x1280",
}

SIZES_SORA2_PRO = {
    ("16:9", "720p"): "1280x720",
    ("16:9", "1080p"): "1792x1024",
    ("9:16", "720p"): "720x1280",
    ("9:16", "1080p"): "1024x1792",
}

ALLOWED_SECONDS = ("4", "8", "12")


class SoraClient:
    """Client for OpenAI Sora video generation via the OpenAI API."""

    def __init__(self, api_key: str = None, model: str = "sora-2"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY env variable "
                "or pass api_key to SoraClient."
            )
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_video(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 8,
        resolution: str = "1080p",
        aspect_ratio: str = "16:9",
    ) -> dict:
        """
        Generate a video from a text prompt, optionally conditioned on an image.

        Args:
            prompt: Text description of the video to generate.
            image_path: Optional path to a reference image for image-to-video.
            duration: Desired video duration in seconds. Snapped to nearest
                      allowed value (4, 8, or 12).
            resolution: "720p" or "1080p".
            aspect_ratio: "16:9" or "9:16".

        Returns:
            dict with keys: success, output_path, error.
        """
        try:
            seconds = str(min(ALLOWED_SECONDS, key=lambda s: abs(int(s) - duration)))
            if "pro" in self.model:
                size = SIZES_SORA2_PRO.get((aspect_ratio, resolution), "1280x720")
            else:
                size = SIZES_SORA2.get(aspect_ratio, "1280x720")

            body: dict = {
                "model": self.model,
                "prompt": prompt,
                "seconds": seconds,
                "size": size,
            }

            if image_path:
                if not os.path.exists(image_path):
                    return {"success": False, "error": f"Image not found: {image_path}"}
                with open(image_path, "rb") as f:
                    image_bytes = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lstrip(".").lower()
                mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/jpeg"
                data_url = f"data:{mime};base64,{image_bytes}"
                body["image_reference"] = {"image_url": data_url}

            logger.info("Sending video generation request to Sora...")
            resp = requests.post(
                f"{self.base_url}/videos",
                headers=self.headers,
                json=body,
                timeout=60,
            )
            if not resp.ok:
                logger.error(f"Sora API error ({resp.status_code}): {resp.text}")
                return {"success": False, "error": f"Sora API {resp.status_code}: {resp.text}"}
            result = resp.json()

            video_id = result.get("id")
            if not video_id:
                return {"success": False, "error": "No video id returned from Sora API."}

            logger.info(f"Sora job created: {video_id}, status={result.get('status')}")
            return self._poll_video(video_id)

        except requests.exceptions.RequestException as e:
            logger.error(f"Sora API request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Sora generation error: {e}")
            return {"success": False, "error": str(e)}

    def _poll_video(self, video_id: str, timeout: int = 600, interval: int = 15) -> dict:
        """Poll the Sora API until the video is ready."""
        url = f"{self.base_url}/videos/{video_id}"
        start = time.time()

        while time.time() - start < timeout:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status")
            progress = data.get("progress", 0)

            if status == "completed":
                return self._download_video(video_id)
            elif status == "failed":
                err = data.get("error", {})
                msg = err.get("message", "Sora generation failed.") if isinstance(err, dict) else str(err)
                return {"success": False, "error": msg}

            logger.info(
                f"Sora generation: status={status}, progress={progress}% "
                f"({int(time.time() - start)}s elapsed)"
            )
            time.sleep(interval)

        return {"success": False, "error": f"Sora generation timed out after {timeout}s"}

    def _download_video(self, video_id: str) -> dict:
        """Download the completed video via GET /videos/{id}/content."""
        try:
            url = f"{self.base_url}/videos/{video_id}/content"
            dl_resp = requests.get(url, headers=self.headers, timeout=120, stream=True)
            dl_resp.raise_for_status()

            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/sora_{int(time.time())}.mp4"
            with open(output_path, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = os.path.getsize(output_path)
            logger.info(f"Sora video saved to: {output_path} ({file_size} bytes)")
            return {
                "success": True,
                "output_path": output_path,
                "file_size_bytes": file_size,
            }
        except Exception as e:
            logger.error(f"Failed to download Sora video: {e}")
            return {"success": False, "error": str(e)}
