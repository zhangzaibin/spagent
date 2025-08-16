"""
Model Base Class

This module defines the base Model interface for VLLM model wrappers
in the SPAgent system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class Model(ABC):
    """Abstract base class for VLLM model wrappers"""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: Optional[int] = None):
        """
        Initialize a model wrapper
        
        Args:
            model_name: Name/identifier of the model
            temperature: Sampling temperature for generation
            max_tokens: Maximum number of tokens to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def single_image_inference(
        self, 
        image_path: str, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Perform inference with a single image
        
        Args:
            image_path: Path to the input image
            prompt: Text prompt for the model
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Model response text
        """
        pass
    
    @abstractmethod
    def multiple_images_inference(
        self, 
        image_paths: List[str], 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Perform inference with multiple images
        
        Args:
            image_paths: List of paths to input images
            prompt: Text prompt for the model
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Model response text
        """
        pass
    
    @abstractmethod
    def text_only_inference(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Perform text-only inference
        
        Args:
            prompt: Text prompt for the model
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Model response text
        """
        pass
    
    def _get_temperature(self, override: Optional[float]) -> float:
        """Get temperature with optional override"""
        return override if override is not None else self.temperature
    
    def _get_max_tokens(self, override: Optional[int]) -> Optional[int]:
        """Get max_tokens with optional override"""
        return override if override is not None else self.max_tokens 