from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from spagent.external_experts.map_anything.map_anything_client import MapAnythingClient
except Exception:
    from external_experts.map_anything.map_anything_client import MapAnythingClient  # type: ignore


class MapAnythingTool:
    def __init__(
        self,
        use_mock: bool = False,
        server_url: str = "http://127.0.0.1:20033",
        timeout: int = 1800,
    ) -> None:
        self.use_mock = use_mock
        self.server_url = server_url
        self.timeout = timeout
        self.client = MapAnythingClient(server_url=server_url, timeout=timeout)

    def call(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        save_outputs: bool = True,
        model_name: str = "facebook/map-anything",
        device: Optional[str] = None,
        memory_efficient_inference: bool = True,
        minibatch_size: Optional[int] = None,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        apply_mask: bool = True,
        mask_edges: bool = True,
        apply_confidence_mask: bool = False,
        confidence_percentile: int = 10,
    ) -> Dict[str, Any]:
        if self.use_mock:
            return {
                "success": True,
                "message": "Mock MapAnything result.",
                "run_id": "mock_run",
                "model_name": model_name,
                "device": device or "cpu",
                "output_dir": str(Path(output_dir or "/tmp/mapanything_mock_output").resolve()),
                "num_views": len(image_paths),
                "summaries": [],
                "warnings": [],
            }

        return self.client.infer(
            image_paths=image_paths,
            output_dir=output_dir,
            save_outputs=save_outputs,
            model_name=model_name,
            device=device,
            memory_efficient_inference=memory_efficient_inference,
            minibatch_size=minibatch_size,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
        )