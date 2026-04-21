import base64
import logging
import os
import time
from typing import Any, Dict, List

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SanaClient:
    """Client for Sana image generation via an OpenAI-compatible SGLang server."""

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:30000",
        model: str = "default",
        timeout: int = 300,
        output_dir: str = "outputs/sana_client",
    ):
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.output_dir = output_dir

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        num_inference_steps: int = 2,
        guidance_scale: float = 4.5,
        seed: int = 42,
        n: int = 1,
        response_format: str = "b64_json",
        negative_prompt: str = None,
        extra_body: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate image(s) from a text prompt using Sana.

        Args:
            prompt: Text prompt to generate from.
            size: Output image size, e.g. "1024x1024".
            num_inference_steps: Diffusion sampling steps.
            guidance_scale: CFG guidance scale.
            seed: Random seed for reproducibility.
            n: Number of images to request.
            response_format: Expected response format. Prefer "b64_json".
            negative_prompt: Optional negative prompt.
            extra_body: Optional extra request fields to merge into the payload.

        Returns:
            A dictionary with keys such as:
            - success
            - output_path
            - image_paths
            - file_size_bytes
            - error
        """
        try:
            if not prompt or not prompt.strip():
                return {"success": False, "error": "Prompt must be a non-empty string."}

            payload: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
                "size": size,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "seed": int(seed),
                "n": max(1, int(n)),
                "response_format": response_format,
            }
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
            if extra_body:
                payload.update(extra_body)

            logger.info(
                "Sending image generation request to Sana: size=%s, steps=%s, seed=%s, n=%s",
                size,
                num_inference_steps,
                seed,
                payload["n"],
            )
            resp = requests.post(
                f"{self.server_url}/v1/images/generations",
                json=payload,
                timeout=self.timeout,
            )
            if not resp.ok:
                logger.error("Sana API error (%s): %s", resp.status_code, resp.text)
                return {
                    "success": False,
                    "error": f"Sana API {resp.status_code}: {resp.text}",
                }

            result = resp.json()
            image_paths = self._save_images(result, seed=seed)
            if not image_paths:
                return {
                    "success": False,
                    "error": f"No images were returned by Sana: {result}",
                }

            output_path = image_paths[0]
            file_size_bytes = os.path.getsize(output_path)
            logger.info("Sana image saved to: %s (%s bytes)", output_path, file_size_bytes)
            return {
                "success": True,
                "output_path": output_path,
                "image_paths": image_paths,
                "file_size_bytes": file_size_bytes,
                "model": self.model,
                "size": size,
                "seed": seed,
                "raw_response": result,
            }

        except requests.exceptions.RequestException as e:
            logger.error("Sana API request failed: %s", e)
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error("Sana generation error: %s", e)
            return {"success": False, "error": str(e)}

    def _save_images(self, response_data: Dict[str, Any], seed: int) -> List[str]:
        """Save generated images from a Sana/OpenAI-compatible response."""
        data = response_data.get("data", [])
        if not data:
            return []

        os.makedirs(self.output_dir, exist_ok=True)
        image_paths: List[str] = []
        timestamp = int(time.time())

        for idx, item in enumerate(data):
            image_bytes = self._extract_image_bytes(item)
            if image_bytes is None:
                logger.warning("Skipping image %s because no decodable payload was found.", idx)
                continue

            output_path = os.path.join(
                self.output_dir,
                f"sana_{timestamp}_{seed}_{idx + 1}.png",
            )
            with open(output_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(output_path)

        return image_paths

    def _extract_image_bytes(self, item: Dict[str, Any]) -> bytes | None:
        """Extract image bytes from a single data item."""
        b64_json = item.get("b64_json")
        if b64_json:
            return base64.b64decode(b64_json)

        url = item.get("url")
        if url:
            dl_resp = requests.get(url, timeout=self.timeout)
            dl_resp.raise_for_status()
            return dl_resp.content

        return None
