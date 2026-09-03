"""
Runtime requirements per catalog tool: what a skill needs to actually run.

This is the ONLY skill-side knowledge that is not derivable from
``tools/catalog.py`` (which knows classes, categories, and default server
URLs but not deployment: launch commands, checkpoints, API keys). Launch
commands mirror docs/Tool/TOOL_USING.md and are repo-root relative.

Runtime classes (the INDEX column):

- ``local``     — runs in-process; no server (may auto-download weights)
- ``server``    — needs a local expert server (health: ``GET /health``)
- ``cloud-API`` — wraps a third-party provider; needs an API key
- ``mock-only`` — real backend not vendored in this repo; only ``--use-mock``
                  works out of the box (SKILL.md explains how to enable it)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

RUNTIME_LOCAL = "local"
RUNTIME_SERVER = "server"
RUNTIME_CLOUD = "cloud-API"
RUNTIME_MOCK_ONLY = "mock-only"


@dataclass(frozen=True)
class RuntimeSpec:
    """Deployment requirements for one catalog tool (keyed by catalog key)."""

    runtime: str
    launch_command: Optional[str] = None   # for `server` runtimes
    checkpoint: Optional[str] = None       # weights the backend loads
    requires: Tuple[str, ...] = field(default_factory=tuple)
    notes: Optional[str] = None


_GDINO = RuntimeSpec(
    runtime=RUNTIME_SERVER,
    launch_command=(
        "python spagent/external_experts/GroundingDINO/grounding_dino_server.py "
        "--checkpoint_path checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth "
        "--port 20022"
    ),
    checkpoint="checkpoints/grounding_dino/groundingdino_swinb_cogcoor.pth",
)

RUNTIME_SPECS: Dict[str, RuntimeSpec] = {
    "depth": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/Depth_AnythingV2/depth_server.py "
            "--checkpoint_path checkpoints/depth_anything/depth_anything_v2_vitb.pth "
            "--port 20019"
        ),
        checkpoint="checkpoints/depth_anything/depth_anything_v2_vitb.pth",
    ),
    "segmentation": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/SAM2/sam2_server.py "
            "--checkpoint_path checkpoints/sam2/sam2.1_b.pt --port 20020"
        ),
        checkpoint="checkpoints/sam2/sam2.1_b.pt",
    ),
    "detection": _GDINO,
    "zoom": _GDINO,
    "localize": _GDINO,
    "supervision": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/supervision/supervision_server.py"
        ),
        notes="YOLO/Supervision annotation server on port 8000.",
    ),
    "yoloe": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/supervision/sv_yoloe_server.py"
        ),
        notes="YOLO-E server on port 8000 (weights per "
              "spagent/external_experts/supervision/download_weights.py).",
    ),
    "yolo26": RuntimeSpec(
        runtime=RUNTIME_LOCAL,
        checkpoint="checkpoints/yolo26/yolo26n.pt",
        requires=("ultralytics",),
        notes="Runs in-process (CPU or GPU); weights auto-download on first use. "
              "No mock mode.",
    ),
    "qwenvl": RuntimeSpec(
        runtime=RUNTIME_CLOUD,
        requires=("DashScope API key (constructor arg `api_key`)",),
        notes="Third-party VLM detection API; without a key only --use-mock works.",
    ),
    "moondream": RuntimeSpec(
        runtime=RUNTIME_CLOUD,
        requires=("Moondream provider API key",),
        notes="Served via spagent/external_experts/moondream/md_server.py "
              "(port 20024), which itself needs a provider API key; without "
              "one only --use-mock works.",
    ),
    "molmo2": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/Molmo2/molmo2_server.py --port 20025"
        ),
        notes="HF weights download on first launch.",
    ),
    "pi3": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/Pi3/pi3_server.py "
            "--checkpoint_path checkpoints/pi3/model.safetensors --port 20030"
        ),
        checkpoint="checkpoints/pi3/model.safetensors",
    ),
    "pi3x": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/Pi3/pi3x_server.py "
            "--checkpoint_path checkpoints/pi3x/model.safetensors --port 20031"
        ),
        checkpoint="checkpoints/pi3x/model.safetensors",
    ),
    "vggt": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/VGGT/vggt_server.py --port 20032"
        ),
        notes="facebook/VGGT-1B auto-downloads from HF on first launch.",
    ),
    "mapanything": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/mapanything/mapanything_server.py "
            "--port 20033"
        ),
        notes="facebook/map-anything auto-downloads from HF on first launch.",
    ),
    "orient_anything_v2": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/OrientAnythingV2/oa_v2_server.py "
            "--checkpoint_path checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt "
            "--repo_path third_party/orient_anything_v2 --port 20034"
        ),
        checkpoint="checkpoints/orient_anything_v2/rotmod_realrotaug_best.pt",
        requires=("third_party/orient_anything_v2 (upstream repo checkout)",),
    ),
    "sana": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/Sana/sana_server.py --port 30000"
        ),
        notes="If port 30000 is taken, launch on another port and pass "
              "--server-url to spagent.skills.run.",
    ),
    "veo": RuntimeSpec(
        runtime=RUNTIME_CLOUD,
        requires=("Google Veo API key (constructor arg `api_key`)",),
        notes="Without a key only --use-mock works.",
    ),
    "sora": RuntimeSpec(
        runtime=RUNTIME_CLOUD,
        requires=("OpenAI Sora API key (constructor arg `api_key`)",),
        notes="Without a key only --use-mock works.",
    ),
    "wan": RuntimeSpec(
        runtime=RUNTIME_CLOUD,
        requires=("DashScope Wan API key (constructor arg `api_key`)",),
        notes="Without a key only --use-mock works.",
    ),
    "vace": RuntimeSpec(
        runtime=RUNTIME_SERVER,
        launch_command=(
            "python spagent/external_experts/vace/vace_server.py --port 20034"
        ),
        checkpoint="spagent/external_experts/vace/models/Wan2.1-VACE-1.3B",
        notes="Heavy Wan2.1-VACE stack (large download, big GPU); see "
              "spagent/external_experts/vace/README.md.",
    ),
    "flowseek": RuntimeSpec(
        runtime=RUNTIME_LOCAL,
        requires=(
            "FLOWSEEK_CHECKPOINT env var (FlowSeek weights)",
            "FLOWSEEK_DAV2_CHECKPOINT env var (Depth-AnythingV2 weights)",
        ),
        notes="Runs in-process on GPU when both checkpoints are set; "
              "otherwise use --use-mock.",
    ),
    "paddleocr_vl": RuntimeSpec(
        runtime=RUNTIME_LOCAL,
        notes="PaddleOCR-VL weights auto-download from HF on first use.",
    ),
    "wilddet3d": RuntimeSpec(
        runtime=RUNTIME_MOCK_ONLY,
        requires=(
            "WILDDET3D_ROOT env var (upstream WildDet3D repo checkout)",
            "optional WILDDET3D_CHECKPOINT env var",
        ),
        notes="The real model is not vendored in this repo; set WILDDET3D_ROOT "
              "to enable it, otherwise only --use-mock works.",
    ),
}


def get_runtime_spec(key: str) -> RuntimeSpec:
    """Runtime spec for a catalog key (safe default for future tools)."""
    return RUNTIME_SPECS.get(key, RuntimeSpec(runtime=RUNTIME_LOCAL))
