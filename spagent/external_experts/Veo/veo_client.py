import os
import time
import base64
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VeoClient:
    """Client for Google Veo video generation via Gemini API."""

    def __init__(self, api_key: str = None, model: str = "veo-3.0-fast-generate-001"):
        self.api_key = api_key or os.environ.get("GCP_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY env variable "
                "or pass api_key to VeoClient."
            )
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = model

    def generate_video(
        self,
        prompt: str,
        image_path: str = None,
        duration: int = 8,
        aspect_ratio: str = "16:9",
    ) -> dict:
        """
        Generate a video from a text prompt, optionally conditioned on an image.

        Args:
            prompt: Text description of the video to generate.
            image_path: Optional path to a reference image for image-to-video.
            duration: Desired video duration in seconds (5 or 8).
            aspect_ratio: "16:9" or "9:16".

        Returns:
            dict with keys: success, output_path, error.
        """
        try:
            instance: dict = {"prompt": prompt}

            if image_path:
                if not os.path.exists(image_path):
                    return {"success": False, "error": f"Image not found: {image_path}"}
                with open(image_path, "rb") as f:
                    image_bytes = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lstrip(".").lower()
                mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/jpeg"
                instance["image"] = {
                    "bytesBase64Encoded": image_bytes,
                    "mimeType": mime,
                }

            clamped_duration = 4 if duration <= 6 else 8

            payload = {
                "instances": [instance],
                "parameters": {
                    "aspectRatio": aspect_ratio,
                    "durationSeconds": clamped_duration,
                    "sampleCount": 1,
                },
            }

            url = (
                f"{self.base_url}/models/{self.model}:predictLongRunning"
                f"?key={self.api_key}"
            )
            logger.info(f"Sending video generation request to Veo (model={self.model})...")
            resp = requests.post(url, json=payload, timeout=60)

            if not resp.ok:
                error_body = resp.text
                logger.error(f"Veo API error ({resp.status_code}): {error_body}")
                return {"success": False, "error": f"Veo API {resp.status_code}: {error_body}"}

            operation = resp.json()
            operation_name = operation.get("name")
            if not operation_name:
                return {"success": False, "error": f"No operation name in Veo response: {operation}"}

            logger.info(f"Veo operation started: {operation_name}")
            return self._poll_operation(operation_name)

        except requests.exceptions.RequestException as e:
            logger.error(f"Veo API request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Veo generation error: {e}")
            return {"success": False, "error": str(e)}

    def _poll_operation(self, operation_name: str, timeout: int = 300, interval: int = 10) -> dict:
        """Poll a long-running operation until completion."""
        url = f"{self.base_url}/{operation_name}?key={self.api_key}"
        start = time.time()

        while time.time() - start < timeout:
            resp = requests.get(url, timeout=30)
            if not resp.ok:
                logger.warning(f"Poll request returned {resp.status_code}: {resp.text}")
                time.sleep(interval)
                continue

            op = resp.json()

            if op.get("done"):
                if "error" in op:
                    err = op["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    return {"success": False, "error": msg}
                return self._save_video(op)

            logger.info(f"Veo generation in progress... ({int(time.time() - start)}s elapsed)")
            time.sleep(interval)

        return {"success": False, "error": f"Veo generation timed out after {timeout}s"}

    def _save_video(self, operation: dict) -> dict:
        """Download and save the generated video from a completed operation."""
        try:
            response_data = operation.get("response", {})
            videos = response_data.get("generateVideoResponse", {}).get("generatedSamples", [])
            if not videos:
                return {"success": False, "error": f"No video samples in Veo response: {response_data}"}

            video_info = videos[0].get("video", {})
            video_uri = video_info.get("uri")
            video_b64 = video_info.get("bytesBase64Encoded")

            if video_b64:
                video_bytes = base64.b64decode(video_b64)
            elif video_uri:
                sep = "&" if "?" in video_uri else "?"
                dl_url = f"{video_uri}{sep}key={self.api_key}"
                dl_resp = requests.get(dl_url, timeout=120)
                dl_resp.raise_for_status()
                video_bytes = dl_resp.content
            else:
                return {"success": False, "error": f"No video URI or data in Veo response: {video_info}"}

            os.makedirs("outputs", exist_ok=True)
            output_path = f"outputs/veo_{int(time.time())}.mp4"
            with open(output_path, "wb") as f:
                f.write(video_bytes)

            logger.info(f"Veo video saved to: {output_path} ({len(video_bytes)} bytes)")
            return {
                "success": True,
                "output_path": output_path,
                "file_size_bytes": len(video_bytes),
            }
        except Exception as e:
            logger.error(f"Failed to save Veo video: {e}")
            return {"success": False, "error": str(e)}
