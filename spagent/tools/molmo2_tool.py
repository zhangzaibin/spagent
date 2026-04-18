"""
Molmo2 Tool — vision-language inference via the Molmo2 HTTP expert (or mock).
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool

logger = logging.getLogger(__name__)


class Molmo2Tool(Tool):
    """Vision-language tasks (qa, caption, point) using a Molmo2 Transformers server."""

    def __init__(
        self,
        use_mock: bool = True,
        server_url: str = "http://localhost:20025",
        output_dir: Optional[str] = None,
    ):
        super().__init__(
            name="molmo2_tool",
            description=(
                "Molmo2: image QA, captioning, or pointing (point parses model output and can save overlays). "
                "Requires Molmo2 server unless use_mock=True."
            ),
        )
        self.use_mock = use_mock
        self.server_url = server_url
        self.output_dir = output_dir
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.use_mock:
            self._client = None
            logger.info("Using mock Molmo2 (no HTTP)")
            return
        try:
            from external_experts.Molmo2.molmo2_client import Molmo2Client

            self._client = Molmo2Client(server_url=self.server_url)
            logger.info("Using Molmo2 service at %s", self.server_url)
        except ImportError as e:
            logger.error("Failed to import Molmo2 client: %s", e)
            raise

    @staticmethod
    def _normalize_paths(image_path: Union[str, List[str]]) -> List[str]:
        if isinstance(image_path, (list, tuple)):
            return [str(Path(p)) for p in image_path]
        return [str(Path(image_path))]

    def _generate_text(self, primary_path: str, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        if self.use_mock:
            return {
                "success": True,
                "text": f"[mock] {prompt[:120]} on {Path(primary_path).name}",
            }
        assert self._client is not None
        return self._client.infer_path(primary_path, prompt=prompt, max_new_tokens=max_new_tokens)

    def _run_point(
        self,
        paths: List[str],
        prompt: str,
        save_annotated: bool,
        max_new_tokens: int,
        out_base: Optional[Path],
    ) -> Dict[str, Any]:
        from PIL import Image

        from external_experts.Molmo2.point_utils import (
            annotate_images_as_base64,
            default_output_dir,
            extract_points_from_text,
            group_points_by_image,
            save_annotated_images,
        )

        if self.use_mock:
            # Legacy point regex requires decimals, e.g. Click(50.0, 50.0)
            raw_text = "Click(50.0, 50.0)"
        else:
            gen = self._generate_text(paths[0], prompt, max_new_tokens)
            if not gen.get("success"):
                return {"success": False, "error": gen.get("error", "generation failed")}
            raw_text = gen.get("text", "")
        sizes: List[tuple] = []
        for p in paths:
            with Image.open(p) as im:
                sizes.append(im.size)
        points = extract_points_from_text(raw_text, sizes)
        grouped = group_points_by_image(points, paths)

        result: Dict[str, Any] = {
            "task": "point",
            "raw_text": raw_text,
            "num_points": len(points),
            "points_by_image": grouped,
        }

        output_path: Optional[str] = None
        if save_annotated and points:
            annotated = annotate_images_as_base64(paths, grouped)
            dest = out_base if out_base is not None else default_output_dir()
            saved = save_annotated_images(annotated, output_dir=dest)
            if saved:
                output_path = saved[0]
            result["saved_paths"] = saved
        elif save_annotated and not points:
            logger.warning("point task: no parseable points in model output")

        return {"success": True, "result": result, "output_path": output_path, "response_text": raw_text[:500]}

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the input image (or use list in code for multi-image).",
                },
                "task": {
                    "type": "string",
                    "enum": ["qa", "caption", "point"],
                    "description": "qa: answer a question; caption: describe; point: grounding (parses coords).",
                },
                "prompt": {"type": "string", "description": "Question, caption hint, or pointing instruction."},
                "save_annotated": {
                    "type": "boolean",
                    "description": "For task=point, save JPEG overlay(s) with marked points.",
                    "default": True,
                },
                "max_new_tokens": {"type": "integer", "default": 200},
            },
            "required": ["image_path"],
        }

    def call(
        self,
        image_path: Union[str, List[str]],
        task: str = "qa",
        prompt: Optional[str] = None,
        save_annotated: bool = True,
        max_new_tokens: int = 200,
    ) -> Dict[str, Any]:
        paths = self._normalize_paths(image_path)
        for p in paths:
            if not Path(p).exists():
                return {"success": False, "error": f"Image file not found: {p}"}

        out_base = Path(self.output_dir) if self.output_dir else None

        try:
            if task == "caption":
                use_prompt = prompt or "Describe this image."
                eff_task = "caption"
            elif task == "qa":
                use_prompt = prompt or "What do you see in this image? Answer briefly."
                eff_task = "qa"
            elif task == "point":
                use_prompt = prompt or (
                    "Point to the requested location. Output pointing coordinates in the model's standard format."
                )
                return self._run_point(paths, use_prompt, save_annotated, max_new_tokens, out_base)
            else:
                return {"success": False, "error": f"Unknown task: {task}"}

            if self.use_mock:
                gen_text = f"[mock] {use_prompt[:120]} on {Path(paths[0]).name}"
                return {
                    "success": True,
                    "result": {"task": eff_task, "generated_text": gen_text, "prompt": use_prompt},
                    "response_text": gen_text,
                }

            assert self._client is not None
            remote = self._client.infer_path(paths[0], prompt=use_prompt, max_new_tokens=max_new_tokens)
            if not remote.get("success"):
                return {"success": False, "error": remote.get("error", "Unknown error")}

            gen_text = remote.get("text", "")
            return {
                "success": True,
                "result": {"task": eff_task, "generated_text": gen_text, "prompt": use_prompt},
                "response_text": gen_text,
                "raw": remote,
            }
        except Exception as e:
            logger.exception("Molmo2 tool error")
            return {"success": False, "error": str(e)}
