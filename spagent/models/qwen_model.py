"""
Qwen Model Wrapper

This module contains the QwenModel wrapper that integrates
existing Qwen functionality with the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.model import Model

logger = logging.getLogger(__name__)


class QwenModel(Model):
    """Qwen model wrapper for SPAgent"""
    
    def __init__(
        self, 
        model_name: str = "qwen2.5-vl-7b-instruct",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize Qwen model wrapper
        
        Args:
            model_name: Qwen model name (e.g., "qwen2.5-vl-7b-instruct")
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        super().__init__(model_name, temperature, max_tokens)
        
        # Import Qwen functions
        try:
            from vllm_models.qwen import (
                qwen_single_image_inference,
                qwen_multiple_images_inference,
                qwen_text_only_inference
            )
            self._single_image_func = qwen_single_image_inference
            self._multiple_images_func = qwen_multiple_images_inference
            self._text_only_func = qwen_text_only_inference
            logger.info(f"Initialized Qwen model: {model_name}")
        except ImportError as e:
            logger.error(f"Failed to import Qwen functions: {e}")
            raise
    
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
        try:
            temp = self._get_temperature(temperature)
            tokens = self._get_max_tokens(max_tokens)
            
            result = self._single_image_func(
                image_path=image_path,
                prompt=prompt,
                model=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Qwen single image inference failed: {e}")
            raise
    
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
        try:
            temp = self._get_temperature(temperature)
            tokens = self._get_max_tokens(max_tokens)
            
            result = self._multiple_images_func(
                image_paths=image_paths,
                prompt=prompt,
                model=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Qwen multiple images inference failed: {e}")
            raise
    
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
        try:
            temp = self._get_temperature(temperature)
            tokens = self._get_max_tokens(max_tokens)
            
            result = self._text_only_func(
                prompt=prompt,
                model=self.model_name,
                temperature=temp,
                max_tokens=tokens
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Qwen text-only inference failed: {e}")
            raise 