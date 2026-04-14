"""Mock Molmo2 service compatible with the Molmo2 client/server flow."""

from pathlib import Path
from typing import Dict, List

from .point_utils import (
    annotate_images_as_base64,
    extract_points_from_text,
    group_points_by_image,
    save_annotated_images,
)


class MockMolmo2Service:
    """Lightweight Molmo2 mock used by tests and development."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir

    def infer(
        self,
        image_paths: List[str],
        task: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        save_annotated: bool = True,
    ) -> Dict:
        _ = max_new_tokens
        _ = temperature

        normalized = [str(Path(p)) for p in image_paths]
        prompt_lower = prompt.lower()

        if task == "point" or "point" in prompt_lower or "locat" in prompt_lower:
            if len(normalized) == 1:
                generated_text = '<points label="mock target" coords="1 1 500 500"/>'
            else:
                generated_text = '<points label="mock target" coords="1 1 420 420;2 2 580 580"/>'
        elif "compare" in prompt_lower and len(normalized) > 1:
            generated_text = (
                "The first image appears more centered and closer to the camera, "
                "while the second image shows a wider surrounding context."
            )
        else:
            generated_text = (
                "This is a mock Molmo2 response describing the main visual content "
                "and answering the request at a high level."
            )

        result = {
            "success": True,
            "generated_text": generated_text,
            "task": task,
            "image_paths": normalized,
        }
        if task == "point":
            from PIL import Image

            sizes = []
            for path in normalized:
                with Image.open(path) as image:
                    sizes.append(image.size)
            points = extract_points_from_text(generated_text, sizes)
            grouped_points = group_points_by_image(points, normalized)
            result["points_by_image"] = grouped_points
            result["num_points"] = sum(len(group["points"]) for group in grouped_points)
            if save_annotated:
                annotated_images = annotate_images_as_base64(normalized, grouped_points)
                output_paths = save_annotated_images(annotated_images, output_dir=self.output_dir)
                result["annotated_images"] = annotated_images
                result["output_paths"] = output_paths
                result["output_path"] = output_paths[0] if output_paths else None
        return result
