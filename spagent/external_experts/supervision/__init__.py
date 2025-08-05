"""
Supervision Expert Module

This module provides Object Detection, Object Segmentation, Tracking(id only/with track/with keypoints) 
capabilities using the Supervision model.
"""

from .supervision_client import AnnotationClient
from .mock_supervision_service import MockSupervisionService

__all__ = ['AnnotationClient', 'SupervisionServer', 'MockSupervisionService'] 