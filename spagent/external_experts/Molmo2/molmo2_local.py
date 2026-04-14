"""
Local Molmo2 inference client built on top of Hugging Face Transformers.
"""

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class Molmo2LocalClient:
    """Lazy-loading local Molmo2 client."""

    def __init__(
        self,
        checkpoint: str = "allenai/Molmo2-4B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.checkpoint = checkpoint
        self.device_map = device_map
        self.torch_dtype = torch_dtype

        self._torch = None
        self._processor = None
        self._model = None

    def _ensure_model_loaded(self):
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "Molmo2 real mode requires transformers and torch. "
                "If Molmo2 remote-code dependencies are missing, install the official project "
                "from https://github.com/allenai/molmo2 or the ai2-molmo2 package."
            ) from e

        model_kwargs = {
            "trust_remote_code": True,
            "device_map": self.device_map,
        }
        if self.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            dtype = getattr(torch, self.torch_dtype, None)
            if dtype is None:
                raise ValueError(f"Unsupported torch dtype: {self.torch_dtype}")
            model_kwargs["torch_dtype"] = dtype

        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            padding_side="left",
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint,
            **model_kwargs,
        )
        self._torch = torch
        logger.info("Loaded Molmo2 checkpoint: %s", self.checkpoint)

    def _get_input_device(self):
        if self._model is None:
            raise RuntimeError("Molmo2 model is not loaded.")
        try:
            return next(self._model.parameters()).device
        except StopIteration as e:
            raise RuntimeError("Molmo2 model has no parameters.") from e

    @staticmethod
    def _load_images(image_paths: List[str]):
        from PIL import Image

        return [Image.open(path).convert("RGB") for path in image_paths]

    def generate(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict:
        self._ensure_model_loaded()

        torch = self._torch
        images = self._load_images(image_paths)
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + [
                    {"type": "image", "image": image} for image in images
                ],
            }]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )

            input_device = self._get_input_device()
            model_inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature

            with torch.inference_mode():
                if input_device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = self._model.generate(**model_inputs, **generation_kwargs)
                else:
                    output = self._model.generate(**model_inputs, **generation_kwargs)

            generated_tokens = output[:, model_inputs["input_ids"].size(1):]
            if hasattr(self._processor, "post_process_image_text_to_text"):
                generated_text = self._processor.post_process_image_text_to_text(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            else:
                generated_text = self._processor.decode(
                    generated_tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            return {
                "success": True,
                "generated_text": generated_text,
                "prompt": prompt,
                "image_paths": [str(Path(path)) for path in image_paths],
            }
        finally:
            for image in images:
                image.close()

    def generate_from_images(
        self,
        images: List,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> Dict:
        self._ensure_model_loaded()

        torch = self._torch
        try:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + [
                    {"type": "image", "image": image} for image in images
                ],
            }]

            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )

            input_device = self._get_input_device()
            model_inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature

            with torch.inference_mode():
                if input_device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = self._model.generate(**model_inputs, **generation_kwargs)
                else:
                    output = self._model.generate(**model_inputs, **generation_kwargs)

            generated_tokens = output[:, model_inputs["input_ids"].size(1):]
            if hasattr(self._processor, "post_process_image_text_to_text"):
                generated_text = self._processor.post_process_image_text_to_text(
                    generated_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            else:
                generated_text = self._processor.decode(
                    generated_tokens[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            return {
                "success": True,
                "generated_text": generated_text,
                "prompt": prompt,
            }
        except Exception as e:
            logger.error("Molmo2 generate_from_images failed: %s", e)
            return {"success": False, "error": str(e)}
