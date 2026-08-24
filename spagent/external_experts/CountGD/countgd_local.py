"""
Local CountGD inference client.

CountGD source is vendored into src/ within this package — no cloning needed.

BERT weights are loaded via HuggingFace transformers (uses HF_HOME cache if set).

Required:
  COUNTGD_CHECKPOINT  path to the CountGD .pth checkpoint file (~1.2 GB)
  Download: see https://github.com/niki-amini-naieni/CountGD README for Google Drive link

Dependencies (pip install):
  addict yapf timm scipy pycocotools
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SRC = Path(__file__).parent / "src"


def _ensure_src_on_path():
    src = str(_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def _ensure_bert() -> str:
    """Return the bert-base-uncased model name/path for transformers.

    Uses HF cache (HF_HOME env var) automatically when set.
    Falls back to downloading from HuggingFace Hub on first run.
    """
    return "bert-base-uncased"


class CountGDLocalClient:
    """Lazy-loading local CountGD client for text-prompted object counting."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        confidence_thresh: float = 0.23,
        device: str = "cuda",
    ):
        self.checkpoint = checkpoint or os.environ.get("COUNTGD_CHECKPOINT")
        if not self.checkpoint:
            raise EnvironmentError(
                "No checkpoint path provided. Pass checkpoint= or set COUNTGD_CHECKPOINT.\n"
                "Download the CountGD checkpoint from the link in the CountGD README."
            )
        self.confidence_thresh = confidence_thresh
        self.device = device
        self._model = None
        self._transform = None

    def _ensure_model_loaded(self):
        if self._model is not None:
            return

        _ensure_src_on_path()
        bert_path = _ensure_bert()

        import argparse
        import random

        import numpy as np
        import torch

        from util.slconfig import SLConfig

        config_path = _SRC / "config" / "cfg_fsc147_vit_b.py"
        cfg = SLConfig.fromfile(str(config_path))
        cfg.merge_from_dict({"text_encoder_type": str(bert_path)})
        cfg_dict = cfg._cfg_dict.to_dict()

        args = argparse.Namespace()
        args.device = self.device
        args.pretrain_model_path = self.checkpoint
        args.config = str(config_path)
        args.options = None
        args.remove_difficult = False
        args.fix_size = False
        args.note = ""
        args.resume = ""
        args.finetune_ignore = None
        args.start_epoch = 0
        args.eval = True
        args.num_workers = 0
        args.test = False
        args.debug = False
        args.find_unused_params = False
        args.save_results = False
        args.save_log = False
        args.world_size = 1
        args.dist_url = "env://"
        args.rank = 0
        args.local_rank = 0
        args.amp = False

        for k, v in cfg_dict.items():
            if not hasattr(args, k):
                setattr(args, k, v)

        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        from models.registry import MODULE_BUILD_FUNCS

        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _, _ = build_func(args)
        model.to(torch.device(self.device))

        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)["model"]
        model.load_state_dict(ckpt, strict=False)
        model.eval()

        import datasets_inference.transforms as T

        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        self._transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
        self._model = model
        logger.info("CountGD loaded on %s", self.device)

    def count(self, image_path: str, text: str) -> Dict:
        """
        Count objects in an image described by text.

        Args:
            image_path: Path to input image.
            text: Text description of object to count (e.g. 'car', 'person').

        Returns:
            dict with keys: success, count, boxes, output_path, description
        """
        import torch
        from PIL import Image

        self._ensure_model_loaded()

        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        try:
            pil_image = Image.open(image_path).convert("RGB")
            input_image, target = self._transform(
                pil_image, {"exemplars": torch.tensor([])}
            )
            input_image = input_image.to(self.device)
            input_exemplar = target["exemplars"].to(self.device)

            with torch.no_grad():
                model_output = self._model(
                    input_image.unsqueeze(0),
                    [input_exemplar],
                    [torch.tensor([0]).to(self.device)],
                    captions=[text + " ."],
                )

            logits = model_output["pred_logits"][0].sigmoid()
            boxes = model_output["pred_boxes"][0]

            mask = logits.max(dim=-1).values > self.confidence_thresh
            boxes = boxes[mask]
            count = boxes.shape[0]

            w, h = pil_image.size
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = ((cx - bw / 2) * w).cpu().tolist()
            y1 = ((cy - bh / 2) * h).cpu().tolist()
            x2 = ((cx + bw / 2) * w).cpu().tolist()
            y2 = ((cy + bh / 2) * h).cpu().tolist()
            boxes_xyxy = [[x1[i], y1[i], x2[i], y2[i]] for i in range(count)]

            output_path = self._visualize(image_path, pil_image, boxes_xyxy, text, count)

            return {
                "success": True,
                "count": count,
                "boxes": boxes_xyxy,
                "output_path": output_path,
                "description": f"CountGD counted {count} '{text}' object(s) in the image.",
            }

        except Exception as e:
            logger.exception("CountGD inference error")
            return {"success": False, "error": str(e)}

    def _visualize(self, image_path: str, pil_image, boxes_xyxy, text: str, count: int) -> str:
        import cv2
        import numpy as np

        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        for box in boxes_xyxy:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)

        label = f"{text}: {count}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

        os.makedirs("outputs", exist_ok=True)
        stem = Path(image_path).stem
        output_path = f"outputs/countgd_{stem}.png"
        cv2.imwrite(output_path, img)
        return output_path
