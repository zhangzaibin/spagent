import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)


class OrientAnythingClient:
    """
    Local Python client for Orient Anything.

    This client:
    1. Imports the official Orient-Anything repo from a local clone.
    2. Prefers local checkpoints under repo_root first.
    3. Falls back to Hugging Face download only if the local checkpoint is missing.
    4. Uses local processor cache only, to avoid accidental online fetches.
    """

    MODEL_CONFIGS = {
        "small": {
            "filename": "cropsmallEX2/dino_weight.pt",
            "dino_mode": "small",
            "in_dim": 384,
            "out_dim": 360 + 180 + 180 + 2,
        },
        "base": {
            "filename": "cropbaseEX2/dino_weight.pt",
            "dino_mode": "base",
            "in_dim": 768,
            "out_dim": 360 + 180 + 180 + 2,
        },
        "large": {
            "filename": "croplargeEX2/dino_weight.pt",
            "dino_mode": "large",
            "in_dim": 1024,
            "out_dim": 360 + 180 + 180 + 2,
        },
    }

    def __init__(
        self,
        repo_root: str,
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.device = device if torch.cuda.is_available() else "cpu"
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            xdg = os.getenv("XDG_CACHE_HOME")
            base = Path(xdg) if xdg else (Path.home() / ".cache")
            self.cache_dir = (base / "spagent" / "orient_anything_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.repo_root.exists():
            raise FileNotFoundError(f"Orient-Anything repo not found: {self.repo_root}")

        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from paths import DINO_SMALL, DINO_BASE, DINO_LARGE
        from vision_tower import DINOv2_MLP
        from inference import get_3angle, get_3angle_infer_aug
        from utils import background_preprocess

        self.DINO_SMALL = DINO_SMALL
        self.DINO_BASE = DINO_BASE
        self.DINO_LARGE = DINO_LARGE
        self.DINOv2_MLP = DINOv2_MLP
        self.get_3angle = get_3angle
        self.get_3angle_infer_aug = get_3angle_infer_aug
        self.background_preprocess = background_preprocess

        self._models: Dict[str, Any] = {}
        self._preprocessors: Dict[str, Any] = {}

    def _get_processor_name(self, model_size: str) -> str:
        if model_size == "small":
            return self.DINO_SMALL
        if model_size == "base":
            return self.DINO_BASE
        if model_size == "large":
            return self.DINO_LARGE
        raise ValueError(f"Unsupported model_size: {model_size}")

    def _load_model(self, model_size: str) -> Tuple[Any, Any]:
        if model_size in self._models:
            return self._models[model_size], self._preprocessors[model_size]

        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model_size: {model_size}")

        cfg = self.MODEL_CONFIGS[model_size]

        local_ckpt = self.repo_root / cfg["filename"]
        if local_ckpt.exists():
            ckpt_path = str(local_ckpt)
            logger.info("Using local checkpoint: %s", ckpt_path)
        else:
            logger.info(
                "Local checkpoint not found at %s, falling back to Hugging Face download.",
                local_ckpt,
            )
            ckpt_path = hf_hub_download(
                repo_id="Viglong/Orient-Anything",
                filename=cfg["filename"],
                repo_type="model",
                cache_dir=str(self.cache_dir),
            )

        model = self.DINOv2_MLP(
            dino_mode=cfg["dino_mode"],
            in_dim=cfg["in_dim"],
            out_dim=cfg["out_dim"],
            evaluate=True,
            mask_dino=False,
            frozen_back=False,
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        model = model.to(self.device)

        processor = AutoImageProcessor.from_pretrained(
            self._get_processor_name(model_size),
            cache_dir=str(self.cache_dir),
            local_files_only=True,
        )

        self._models[model_size] = model
        self._preprocessors[model_size] = processor
        return model, processor

    def predict(
        self,
        image_path: str,
        model_size: str = "large",
        use_tta: bool = False,
        remove_background: bool = True,
        device: Optional[str] = None,
        save_vis: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        del save_vis, output_dir  # currently unused

        image_path = str(Path(image_path).resolve())
        img_path = Path(image_path)

        if not img_path.exists():
            return {
                "success": False,
                "error": f"Image not found: {image_path}",
            }

        run_device = device or self.device
        if "cuda" in str(run_device) and not torch.cuda.is_available():
            run_device = "cpu"
        self.device = run_device

        try:
            model, processor = self._load_model(model_size)
            origin_image = Image.open(img_path).convert("RGB")

            if use_tta:
                rm_bkg_img = self.background_preprocess(origin_image, True)
                angles = self.get_3angle_infer_aug(
                    origin_image,
                    rm_bkg_img,
                    model,
                    processor,
                    self.device,
                )
            else:
                input_img = self.background_preprocess(
                    origin_image,
                    remove_background,
                )
                angles = self.get_3angle(
                    input_img,
                    model,
                    processor,
                    self.device,
                )

            azimuth = float(angles[0])
            polar = float(angles[1])
            rotation = float(angles[2])
            confidence = float(angles[3])

            return {
                "success": True,
                "azimuth": azimuth,
                "polar": polar,
                "rotation": rotation,
                "confidence": confidence,
                "model_size": model_size,
                "use_tta": use_tta,
                "remove_background": remove_background,
                "visualization_path": None,
            }

        except Exception as e:
            logger.exception("OrientAnything prediction failed")
            return {
                "success": False,
                "error": str(e),
            }