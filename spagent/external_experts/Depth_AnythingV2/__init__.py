"""
Depth-AnythingV2 Expert Module

This module provides depth estimation capabilities using the Depth-AnythingV2 model.
"""

from .depth_client import OpenPIClient
from .mock_depth_service import MockDepthService, MockOpenPIClient

__all__ = ['OpenPIClient', 'DepthServer', 'MockDepthService', 'MockOpenPIClient'] 