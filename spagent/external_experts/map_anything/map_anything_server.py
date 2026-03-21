from __future__ import annotations

import json
import os
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

# Optional memory fragmentation mitigation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


DEFAULT_MODEL_NAME = os.environ.get("MAPANYTHING_MODEL_NAME", "facebook/map-anything")
DEFAULT_DEVICE = os.environ.get(
    "MAPANYTHING_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("MAPANYTHING_OUTPUT_ROOT", "/tmp/mapanything_outputs"))
DEFAULT_PORT = int(os.environ.get("MAPANYTHING_PORT", "20033"))

# Large tensors should be saved to disk, not returned inline in JSON.
SAVE_KEYS = {
    "pts3d",
    "pts3d_cam",
    "ray_directions",
    "depth_along_ray",
    "depth_z",
    "cam_trans",
    "cam_quats",
    "intrinsics",
    "camera_poses",
    "conf",
    "mask",
    "non_ambiguous_mask",
}


class InferRequest(BaseModel):
    image_paths: List[str] = Field(..., min_length=1, description="List of image paths.")
    output_dir: Optional[str] = Field(
        default=None,
        description="If omitted, server creates a unique output directory under MAPANYTHING_OUTPUT_ROOT.",
    )
    save_outputs: bool = True
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    memory_efficient_inference: bool = True
    minibatch_size: Optional[int] = None
    use_amp: bool = True
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    apply_mask: bool = True
    mask_edges: bool = True
    apply_confidence_mask: bool = False
    confidence_percentile: int = 10

    @model_validator(mode="after")
    def validate_fields(self) -> "InferRequest":
        if self.minibatch_size is not None and self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if not (0 <= self.confidence_percentile <= 100):
            raise ValueError("confidence_percentile must be in [0, 100]")
        return self


class InferResponse(BaseModel):
    success: bool
    message: str
    run_id: str
    model_name: str
    device: str
    output_dir: str
    num_views: int
    summaries: List[Dict[str, Any]]
    warnings: List[str] = Field(default_factory=list)


@dataclass
class LoadedModel:
    name: str
    device: str
    model: Any


class ModelStore:
    def __init__(self) -> None:
        self._cache: Dict[tuple[str, str], LoadedModel] = {}

    def get(self, model_name: str, device: str) -> LoadedModel:
        key = (model_name, device)
        if key not in self._cache:
            model = MapAnything.from_pretrained(model_name).to(device)
            model.eval()
            self._cache[key] = LoadedModel(name=model_name, device=device, model=model)
        return self._cache[key]


model_store = ModelStore()
app = FastAPI(title="MapAnything Server", version="1.0.0")


def _resolve_image_paths(image_paths: List[str]) -> List[str]:
    resolved: List[str] = []
    for p in image_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        resolved.append(str(path))
    return resolved


def _choose_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _small_preview(arr: np.ndarray, max_items: int = 8) -> List[Any]:
    flat = arr.reshape(-1)
    items = flat[:max_items].tolist()
    preview: List[Any] = []
    for x in items:
        if isinstance(x, np.generic):
            preview.append(x.item())
        else:
            preview.append(x)
    return preview


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _save_prediction_dict(pred: Dict[str, Any], view_dir: Path, save_outputs: bool) -> Dict[str, Any]:
    view_dir.mkdir(parents=True, exist_ok=True)

    files: Dict[str, str] = {}
    shapes: Dict[str, List[int]] = {}
    previews: Dict[str, List[Any]] = {}
    metadata: Dict[str, Any] = {}

    for key, value in pred.items():
        arr = _to_numpy(value)

        if key in SAVE_KEYS and arr is not None:
            shapes[key] = list(arr.shape)
            previews[key] = _small_preview(arr)

            if save_outputs:
                file_path = view_dir / f"{key}.npy"
                np.save(file_path, arr)
                files[key] = str(file_path)
        else:
            metadata[key] = _json_safe_scalar(value)

    meta_path = view_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    idx_val = pred.get("idx", -1)
    if isinstance(idx_val, np.generic):
        idx_val = idx_val.item()
    if not isinstance(idx_val, int):
        idx_val = -1

    return {
        "view_index": idx_val,
        "files": files,
        "shapes": shapes,
        "previews": previews,
        "metadata": metadata,
        "metadata_path": str(meta_path),
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "default_model_name": DEFAULT_MODEL_NAME,
        "default_device": DEFAULT_DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "output_root": str(DEFAULT_OUTPUT_ROOT),
        "port": DEFAULT_PORT,
    }


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    try:
        device = request.device or DEFAULT_DEVICE
        image_paths = _resolve_image_paths(request.image_paths)
        loaded = model_store.get(request.model_name, device)

        run_id = uuid.uuid4().hex[:12]
        output_dir = Path(request.output_dir) if request.output_dir else DEFAULT_OUTPUT_ROOT / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        views = load_images(image_paths)
        amp_dtype = _choose_dtype(request.amp_dtype)

        with torch.inference_mode():
            predictions = loaded.model.infer(
                views,
                memory_efficient_inference=request.memory_efficient_inference,
                minibatch_size=request.minibatch_size,
                use_amp=request.use_amp,
                amp_dtype=amp_dtype,
                apply_mask=request.apply_mask,
                mask_edges=request.mask_edges,
                apply_confidence_mask=request.apply_confidence_mask,
                confidence_percentile=request.confidence_percentile,
            )

        if not isinstance(predictions, list):
            raise RuntimeError(f"Expected model.infer(...) to return list, got {type(predictions)!r}")

        summaries: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for i, pred in enumerate(predictions):
            if not isinstance(pred, dict):
                warnings.append(f"Prediction {i} is not a dict: {type(pred)!r}")
                continue

            view_dir = output_dir / f"view_{i:03d}"
            summaries.append(_save_prediction_dict(pred, view_dir, request.save_outputs))

        return InferResponse(
            success=True,
            message="MapAnything inference completed successfully.",
            run_id=run_id,
            model_name=loaded.name,
            device=loaded.device,
            output_dir=str(output_dir.resolve()),
            num_views=len(summaries),
            summaries=summaries,
            warnings=warnings,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e}\n\n{tb}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, reload=False)