"""
PaddleOCR-VL-1.5 local inference client.

Runs PaddlePaddle/PaddleOCR-VL-1.5 directly in-process via HuggingFace
Transformers. No server required.

Env vars:
  PADDLEOCR_VL_CHECKPOINT  — HuggingFace model ID or local path
                              (default: PaddlePaddle/PaddleOCR-VL-1.5)

Usage:
  from paddleocr_vl_local import PaddleOCRVLLocalClient
  client = PaddleOCRVLLocalClient(device="cuda")
  result = client.recognize("image.jpg", task="ocr")
  print(result["text"])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

TASK_PROMPTS: Dict[str, str] = {
    "ocr": "OCR:",
    "table": "Table:",
    "chart": "Chart:",
    "formula": "Formula:",
    "spotting": "Spotting:",
    "seal": "Seal:",
}

DEFAULT_CHECKPOINT = "PaddlePaddle/PaddleOCR-VL-1.5"


class PaddleOCRVLLocalClient:
    """Load PaddleOCR-VL-1.5 locally and run recognition tasks."""

    def __init__(
        self,
        checkpoint: str | None = None,
        device: str = "cuda",
        max_new_tokens: int = 1024,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        self.checkpoint = checkpoint or os.environ.get(
            "PADDLEOCR_VL_CHECKPOINT", DEFAULT_CHECKPOINT
        )
        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info("Loading PaddleOCR-VL-1.5 from %s on %s ...", self.checkpoint, device)
        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        logger.info("PaddleOCR-VL-1.5 loaded.")
        self._patch_create_causal_mask()

    @staticmethod
    def _patch_create_causal_mask() -> None:
        # transformers>=4.50 passes `inputs_embeds` (plural) but the model's
        # custom create_causal_mask expects `input_embeds` (no trailing s).
        # Rename the kwarg so the signatures match.
        import sys
        for name, mod in list(sys.modules.items()):
            if "paddleocr_vl" in name.lower() and hasattr(mod, "create_causal_mask"):
                _orig = mod.create_causal_mask
                def _patched(*args, _fn=_orig, **kwargs):
                    if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
                    return _fn(*args, **kwargs)
                mod.create_causal_mask = _patched
                logger.debug("Patched create_causal_mask in %s", name)
                break

    def recognize(self, image_path: str, task: str = "ocr") -> Dict[str, Any]:
        """Run a recognition task on a single image.

        Args:
            image_path: Path to the input image.
            task: One of 'ocr', 'table', 'chart', 'formula', 'spotting', 'seal'.

        Returns:
            Dict with keys: success, text, task, error (on failure).
        """
        import torch
        from PIL import Image

        if task not in TASK_PROMPTS:
            return {
                "success": False,
                "error": f"Unknown task '{task}'. Valid tasks: {list(TASK_PROMPTS)}",
            }

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            image = Image.open(image_path).convert("RGB")
            prompt_text = TASK_PROMPTS[task]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            prompt_len = inputs["input_ids"].shape[-1]
            generated = output_ids[0][prompt_len:-1]
            text = self.processor.decode(generated, skip_special_tokens=True).strip()

            return {"success": True, "text": text, "task": task}

        except Exception as exc:
            logger.exception("PaddleOCR-VL-1.5 inference error")
            return {"success": False, "error": str(exc)}
