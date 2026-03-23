import os
import re
import base64
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DASHSCOPE_CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class QwenVLClient:
    """Client for Qwen VL 2.5 via DashScope OpenAI-compatible API.

    Supports referring detection and reasoning detection tasks.
    The model returns bounding boxes in <ref>label</ref><box>(x1,y1),(x2,y2)</box> format,
    with coordinates normalized to [0, 1000].
    """

    def __init__(self, api_key: str = None, model: str = "qwen-vl-max-latest"):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY env variable "
                "or pass api_key to QwenVLClient."
            )
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def detect(self, image_path: str, text_prompt: str, task: str = "ref_detection") -> dict:
        """Run detection on an image with the given text prompt.

        Args:
            image_path: Local file path or public URL of the image.
            text_prompt: Object description (ref_detection) or reasoning question (reasoning_detection).
            task: "ref_detection" or "reasoning_detection".

        Returns:
            dict with keys: success, boxes, labels, raw_response, error.
        """
        try:
            image_url = self._resolve_image(image_path)
            if image_url is None:
                return {"success": False, "error": f"Image not found: {image_path}"}

            if task == "ref_detection":
                query = f"Detect all <ref>{text_prompt}</ref> in this image and output their bounding boxes."
            else:
                query = (
                    f"Based on the following question, detect the relevant objects in the image "
                    f"and output their bounding boxes.\nQuestion: {text_prompt}"
                )

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": query},
                        ],
                    }
                ],
            }

            logger.info(f"Sending {task} request to Qwen VL (model={self.model})...")
            resp = requests.post(
                DASHSCOPE_CHAT_URL, headers=self.headers, json=payload, timeout=60
            )

            if not resp.ok:
                logger.error(f"Qwen VL API error ({resp.status_code}): {resp.text}")
                return {"success": False, "error": f"Qwen VL API {resp.status_code}: {resp.text}"}

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            boxes, labels = self._parse_boxes(content)

            logger.info(f"Qwen VL detected {len(boxes)} objects")
            return {
                "success": True,
                "boxes": boxes,
                "labels": labels,
                "raw_response": content,
            }

        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Qwen VL response: {e}")
            return {"success": False, "error": f"Response parse error: {e}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Qwen VL API request failed: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Qwen VL error: {e}")
            return {"success": False, "error": str(e)}

    def _resolve_image(self, image_path: str):
        """Convert local path to base64 data-URL, or return URL as-is."""
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return image_path

        if not os.path.exists(image_path):
            return None

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/jpeg"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _parse_boxes(content: str):
        """Parse Qwen VL bounding box output.

        Expected format: <ref>label</ref><box>(x1,y1),(x2,y2)</box>
        Coordinates are in [0, 1000] and normalized to [0, 1] on output.
        """
        boxes = []
        labels = []

        pattern = r"<ref>(.*?)</ref><box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
        for match in re.finditer(pattern, content):
            label = match.group(1)
            x1 = int(match.group(2)) / 1000.0
            y1 = int(match.group(3)) / 1000.0
            x2 = int(match.group(4)) / 1000.0
            y2 = int(match.group(5)) / 1000.0
            labels.append(label)
            boxes.append([x1, y1, x2, y2])

        return boxes, labels
