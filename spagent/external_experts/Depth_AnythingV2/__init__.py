"""
Depth-AnythingV2 Expert Module

This module provides depth estimation capabilities using the Depth-AnythingV2 model.
"""

from .depth_client import OpenPIClient

__all__ = ['OpenPIClient', 'DepthServer', 'MockDepthService', 'MockOpenPIClient'] 