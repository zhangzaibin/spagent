"""
Molmo2 vision-language expert (Allen AI).

- molmo2_server: Flask HTTP service loading HF checkpoints (e.g. allenai/Molmo2-4B).
- molmo2_client: HTTP client for the service.
- download_weights: snapshot_download helper for offline/air-gapped setups.
"""

from .molmo2_client import Molmo2Client

__all__ = ["Molmo2Client"]
