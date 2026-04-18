import logging
import os
import time
from typing import Any, Dict

from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockSanaService:
    """Mock Sana service for local testing without a running generation server."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        seed: int = 42,
        n: int = 1,
        response_format: str = "b64_json",
        negative_prompt: str = None,
        extra_body: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        try:
            width, height = self._parse_size(size)
            n = max(1, int(n))
            image_paths = []
            timestamp = int(time.time())

            for idx in range(n):
                output_path = os.path.join(
                    self.output_dir,
                    f"sana_mock_{timestamp}_{seed}_{idx + 1}.png",
                )
                self._create_mock_image(
                    output_path=output_path,
                    width=width,
                    height=height,
                    prompt=prompt,
                    seed=seed,
                    steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )
                image_paths.append(output_path)

            primary_path = image_paths[0]
            file_size_bytes = os.path.getsize(primary_path)
            logger.info("Mock Sana image saved to: %s", primary_path)
            return {
                "success": True,
                "output_path": primary_path,
                "image_paths": image_paths,
                "file_size_bytes": file_size_bytes,
                "model": "mock-sana",
                "size": size,
                "seed": seed,
                "raw_response": {
                    "mock": True,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "n": n,
                    "response_format": response_format,
                    "extra_body": extra_body or {},
                },
            }
        except Exception as e:
            logger.error("Mock Sana generation error: %s", e)
            return {"success": False, "error": str(e)}

    def _parse_size(self, size: str) -> tuple[int, int]:
        try:
            width_str, height_str = size.lower().split("x")
            width = max(64, int(width_str))
            height = max(64, int(height_str))
            return width, height
        except Exception as e:
            raise ValueError(f"Invalid size format: {size}") from e

    def _create_mock_image(
        self,
        output_path: str,
        width: int,
        height: int,
        prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        negative_prompt: str = None,
    ) -> None:
        bg_color = (242, 236, 223)
        accent_color = (58, 88, 121)
        frame_color = (204, 190, 163)
        text_color = (32, 37, 43)

        image = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(image)

        margin = max(20, width // 32)
        draw.rectangle(
            [margin, margin, width - margin, height - margin],
            outline=frame_color,
            width=max(2, width // 256),
        )
        draw.rectangle(
            [margin * 2, margin * 2, width - margin * 2, height // 3],
            fill=accent_color,
        )

        lines = [
            "Sana Mock Render",
            f"Seed: {seed}",
            f"Steps: {steps}",
            f"CFG: {guidance_scale}",
            "",
            "Prompt:",
        ]
        lines.extend(self._wrap_text(prompt, width=48))
        if negative_prompt:
            lines.append("")
            lines.append("Negative:")
            lines.extend(self._wrap_text(negative_prompt, width=48))

        x = margin * 2
        y = height // 3 + margin
        line_height = max(18, height // 28)
        for line in lines:
            draw.text((x, y), line, fill=text_color)
            y += line_height
            if y > height - margin * 2:
                break

        image.save(output_path)

    def _wrap_text(self, text: str, width: int = 48) -> list[str]:
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = []
        current_len = 0
        for word in words:
            projected = current_len + len(word) + (1 if current else 0)
            if projected > width and current:
                lines.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len = projected
        if current:
            lines.append(" ".join(current))
        return lines
