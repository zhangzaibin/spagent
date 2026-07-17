"""
FlowSeek Tool — optical flow estimation between two images.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.tool import Tool
from core.tool_result import FlowPayload, ToolResult

logger = logging.getLogger(__name__)


class FlowSeekTool(Tool):
    """Optical flow estimation using FlowSeek."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        variant: str = "M",
        device: str = "cuda",
        server_url: Optional[str] = None,
        use_mock: bool = False,
    ):
        super().__init__(
            name="flowseek_tool",
            description=(
                "FlowSeek: optical flow estimation between two images. "
                "Given a pair of images (e.g. consecutive video frames or a before/after pair), "
                "estimates the per-pixel motion field and returns a colorized visualization. "
                "The M variant uses ResNet-34 + ViT-B (higher accuracy); "
                "the T variant uses ResNet-18 + ViT-S (faster). "
                "Useful for motion analysis, video understanding, and dynamic scene tasks."
            ),
        )
        self.use_mock = use_mock
        self._server_url = server_url
        self._client = None
        self._client_kwargs = dict(checkpoint=checkpoint, variant=variant, device=device)

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_mock:
            self._client = _MockFlowSeekClient()
        elif self._server_url:
            from external_experts.FlowSeek.flowseek_client import FlowSeekClient
            self._client = FlowSeekClient(self._server_url)
        else:
            from external_experts.FlowSeek.flowseek_local import FlowSeekLocalClient
            self._client = FlowSeekLocalClient(**self._client_kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image1_path": {
                    "type": "string",
                    "description": "Path to the first (source) image.",
                },
                "image2_path": {
                    "type": "string",
                    "description": "Path to the second (target) image.",
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Optional path to save the colorized flow image. "
                        "Auto-generated under outputs/ if not specified."
                    ),
                },
            },
            "required": ["image1_path", "image2_path"],
        }

    def call(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not Path(image1_path).exists():
            return {"success": False, "error": f"Image not found: {image1_path}"}
        if not Path(image2_path).exists():
            return {"success": False, "error": f"Image not found: {image2_path}"}

        try:
            self._ensure_client()
            raw = self._client.estimate_flow(
                image1_path=image1_path,
                image2_path=image2_path,
                output_path=output_path,
            )
            if not raw.get("success"):
                return raw

            legacy_result = {
                "flow_magnitude_mean": raw.get("flow_magnitude_mean", 0.0),
                "output_path": raw.get("output_path", ""),
                "flow_path": raw.get("flow_path"),
                "flow_shape": raw.get("flow_shape"),
            }
            # Standardized output; server mode may not transport the raw
            # field yet, so fall back to a legacy dict when flow_path is
            # absent instead of failing the optical-flow contract.
            if raw.get("flow_path"):
                return ToolResult(
                    success=True,
                    payload=FlowPayload(
                        flow_path=raw["flow_path"],
                        flow_shape=raw.get("flow_shape"),
                        flow_magnitude_mean=raw.get("flow_magnitude_mean"),
                    ),
                    description=raw.get("description", ""),
                    output_path=raw.get("output_path"),
                    result=legacy_result,
                )
            raw["result"] = legacy_result
            return raw
        except Exception as e:
            logger.exception("FlowSeekTool error")
            return {"success": False, "error": str(e)}


class _MockFlowSeekClient:
    def estimate_flow(
        self,
        image1_path: str,
        image2_path: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        out = output_path or image1_path
        flow_path = str(Path(out).with_suffix(".npy"))
        # Write a real (tiny) zero-flow array so flow_path is actually loadable
        # in mock mode, mirroring the real client's .npy contract.
        flow_shape = [2, 2, 2]
        try:
            import numpy as np
            np.save(flow_path, np.zeros(flow_shape, dtype=np.float32))
        except Exception:
            flow_path = None
        return {
            "success": True,
            "result": {
                "flow_magnitude_mean": 5.0,
                "output_path": out,
                "flow_path": flow_path,
                "flow_shape": flow_shape,
            },
            "output_path": out,
            "flow_path": flow_path,
            "flow_shape": flow_shape,
            "flow_magnitude_mean": 5.0,
            "description": f"[mock] FlowSeek estimated flow from {Path(image1_path).name} to {Path(image2_path).name}.",
        }
