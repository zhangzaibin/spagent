import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import importlib.util

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
        repo_root: Optional[str] = None,
        device: str = "cuda:0",
        cache_dir: Optional[str] = None,
        use_mock: bool = True,
    ):
        # 优先从参数或环境变量获取 repo_root
        repo_root = repo_root or os.getenv("ORIENT_ANYTHING_REPO_ROOT")
        if not use_mock and not repo_root:
            raise ValueError(
                "repo_root must be specified when use_mock=False "
                "or set environment variable ORIENT_ANYTHING_REPO_ROOT"
            )
        self.repo_root = Path(repo_root).resolve() if repo_root else None

        self.device = device if torch.cuda.is_available() else "cpu"

        # 用户可写缓存路径
        default_cache = Path.home() / ".cache" / "spagent" / "orient_anything"
        self.cache_dir = Path(cache_dir or default_cache).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 插入 repo_root 到 sys.path
        if self.repo_root and str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))

        # 动态导入 Orient-Anything 模块
        self._import_repo_modules()

        self._models: Dict[str, Any] = {}
        self._preprocessors: Dict[str, Any] = {}

    def _import_repo_modules(self):
        """Thread-safe import of Orient-Anything modules without changing cwd"""
        if self.repo_root is None:
            return

        def import_from_file(name: str, path: Path):
            spec = importlib.util.spec_from_file_location(name, str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        self.paths_module = import_from_file("paths", self.repo_root / "paths.py")
        self.vision_tower_module = import_from_file("vision_tower", self.repo_root / "vision_tower.py")
        self.inference_module = import_from_file("inference", self.repo_root / "inference.py")
        self.utils_module = import_from_file("utils", self.repo_root / "utils.py")

        # 保存方法和常量
        self.DINO_SMALL = self.paths_module.DINO_SMALL
        self.DINO_BASE = self.paths_module.DINO_BASE
        self.DINO_LARGE = self.paths_module.DINO_LARGE
        self.DINOv2_MLP = self.vision_tower_module.DINOv2_MLP
        self.get_3angle = self.inference_module.get_3angle
        self.get_3angle_infer_aug = self.inference_module.get_3angle_infer_aug
        self.background_preprocess = self.utils_module.background_preprocess

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
                cache_dir=self.cache_dir,
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
            cache_dir=self.cache_dir,
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
            return {"success": False, "error": f"Image not found: {image_path}"}

        run_device = device or self.device
        if "cuda" in str(run_device) and not torch.cuda.is_available():
            run_device = "cpu"
        self.device = run_device

        try:
            model, processor = self._load_model(model_size)
            origin_image = Image.open(img_path).convert("RGB")

            # 使用绝对路径调用 repo 模块
            if use_tta:
                rm_bkg_img = self.background_preprocess(origin_image, True)
                angles = self.get_3angle_infer_aug(
                    origin_image, rm_bkg_img, model, processor, self.device
                )
            else:
                input_img = self.background_preprocess(origin_image, remove_background)
                angles = self.get_3angle(input_img, model, processor, self.device)

            azimuth, polar, rotation, confidence = map(float, angles)
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
            return {"success": False, "error": str(e)}
