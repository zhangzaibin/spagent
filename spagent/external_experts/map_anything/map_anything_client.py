from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class MapAnythingClient:
    def __init__(self, server_url: str = "http://127.0.0.1:20033", timeout: int = 1800) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

        # Don't inherit shell proxy settings when talking to local server.
        self.session = requests.Session()
        self.session.trust_env = False

    def health(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.server_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def infer(
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
        payload = {
            "image_paths": image_paths,
            "output_dir": output_dir,
            "save_outputs": save_outputs,
            "model_name": model_name,
            "device": device,
            "memory_efficient_inference": memory_efficient_inference,
            "minibatch_size": minibatch_size,
            "use_amp": use_amp,
            "amp_dtype": amp_dtype,
            "apply_mask": apply_mask,
            "mask_edges": mask_edges,
            "apply_confidence_mask": apply_confidence_mask,
            "confidence_percentile": confidence_percentile,
        }

        response = self.session.post(
            f"{self.server_url}/infer",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()