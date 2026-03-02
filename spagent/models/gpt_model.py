"""
GPT Model Wrapper

This module contains the GPTModel wrapper that integrates
existing GPT functionality with the SPAgent system.
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.model import Model

logger = logging.getLogger(__name__)


class GPTModel(Model):
    """GPT model wrapper for SPAgent"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        """
        Initialize GPT model wrapper

        Args:
            model_name: GPT model name (e.g., "gpt-4o", "gpt-4o-mini")
            temperature: Sampling temperature. Use 0 for greedy / deterministic output.
            max_tokens: Maximum tokens to generate
            seed: Optional integer seed passed to the OpenAI API for reproducibility
                  (best-effort; combine with temperature=0 for strongest guarantee).
            top_p: Nucleus sampling probability mass (e.g. 1.0 = no truncation).
                   Cannot be used together with temperature != 0 per OpenAI guidelines.
        """
        super().__init__(model_name, temperature, max_tokens)
        self.seed = seed
        self.top_p = top_p
        
        # Import GPT functions
        try:
            from vllm_models.gpt import (
                gpt_single_image_inference,
                gpt_multiple_images_inference,
                gpt_text_only_inference
            )
            self._single_image_func = gpt_single_image_inference
            self._multiple_images_func = gpt_multiple_images_inference
            self._text_only_func = gpt_text_only_inference
            logger.info(f"Initialized GPT model: {model_name}")
        except ImportError as e:
            logger.error(f"Failed to import GPT functions: {e}")
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
                max_tokens=tokens,
                seed=self.seed,
                top_p=self.top_p,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"GPT single image inference failed: {e}")
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
                max_tokens=tokens,
                seed=self.seed,
                top_p=self.top_p,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"GPT multiple images inference failed: {e}")
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
                max_tokens=tokens,
                seed=self.seed,
                top_p=self.top_p,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"GPT text-only inference failed: {e}")
            raise 