"""
Depth-AnythingV2 Expert Module

This module provides depth estimation capabilities using the Depth-AnythingV2 model.
"""

from .depth_client import DepthClient
from .mock_depth_service import MockDepthService

__all__ = ['DepthClient', 'MockDepthService'] 