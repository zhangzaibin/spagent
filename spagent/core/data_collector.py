"""
Data Collection Module for SPAgent Training Data

This module provides functionality to collect training data from SPAgent inference sessions,
including prompts, responses, images, and context for multimodal model training.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class InferenceSample:
    """Single inference sample containing prompt, response, and context"""
    
    def __init__(
        self,
        sample_id: str,
        iteration: int,
        images: List[str],
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.sample_id = sample_id
        self.iteration = iteration
        self.images = images  # List of image paths used in this inference
        self.prompt = prompt
        self.response = response
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary"""
        return {
            "sample_id": self.sample_id,
            "iteration": self.iteration,
            "images": self.images,
            "prompt": self.prompt,
            "response": self.response,
            "context": self.context,
            "timestamp": self.timestamp
        }


class SessionData:
    """Data for a complete inference session (potentially multi-turn)"""
    
    def __init__(
        self,
        session_id: str,
        question: str,
        original_images: List[str]
    ):
        self.session_id = session_id
        self.question = question
        self.original_images = original_images
        self.samples: List[InferenceSample] = []
        self.success = False
        self.final_answer = None
        self.error_message = None
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.metadata = {}
    
    def add_sample(self, sample: InferenceSample):
        """Add an inference sample to this session"""
        self.samples.append(sample)
    
    def mark_success(self, final_answer: str):
        """Mark session as successful with final answer"""
        self.success = True
        self.final_answer = final_answer
        self.end_time = datetime.now().isoformat()
    
    def mark_failure(self, error_message: str):
        """Mark session as failed"""
        self.success = False
        self.error_message = error_message
        self.end_time = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "original_images": self.original_images,
            "success": self.success,
            "final_answer": self.final_answer,
            "error_message": self.error_message,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "num_iterations": len(self.samples),
            "samples": [sample.to_dict() for sample in self.samples],
            "metadata": self.metadata
        }


class DataCollector:
    """
    Collects training data from SPAgent inference sessions
    
    Only saves successful sessions. For multi-turn sessions, all samples
    from successful sessions are saved as positive examples.
    """
    
    def __init__(
        self,
        output_dir: str = "training_data",
        save_images: bool = True,
        auto_save: bool = True
    ):
        """
        Initialize DataCollector
        
        Args:
            output_dir: Directory to save collected data
            save_images: Whether to copy images to output directory
            auto_save: Whether to automatically save successful sessions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_images = save_images
        self.auto_save = auto_save
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.sessions_dir = self.output_dir / "sessions"
        self.images_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)
        
        # In-memory storage for current session
        self.current_session: Optional[SessionData] = None
        
        # Statistics
        self.total_sessions = 0
        self.successful_sessions = 0
        self.total_samples = 0
        
        logger.info(f"DataCollector initialized with output_dir: {output_dir}")
    
    def start_session(
        self,
        question: str,
        image_paths: List[str],
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a new data collection session
        
        Args:
            question: The question being asked
            image_paths: Original input images
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.current_session = SessionData(
            session_id=session_id,
            question=question,
            original_images=image_paths
        )
        
        self.total_sessions += 1
        logger.info(f"Started data collection session: {session_id}")
        
        return session_id
    
    def record_inference(
        self,
        iteration: int,
        images: List[str],
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record a single inference step
        
        Args:
            iteration: Iteration number
            images: Images used in this inference
            prompt: Input prompt
            response: Model response
            context: Additional context (tool calls, results, etc.)
        """
        if self.current_session is None:
            logger.warning("No active session. Call start_session() first.")
            return
        
        sample_id = f"{self.current_session.session_id}_iter_{iteration}"
        
        sample = InferenceSample(
            sample_id=sample_id,
            iteration=iteration,
            images=images,
            prompt=prompt,
            response=response,
            context=context
        )
        
        self.current_session.add_sample(sample)
        logger.debug(f"Recorded inference sample: {sample_id}")
    
    def end_session(
        self,
        success: bool,
        final_answer: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        End current session and optionally save if successful
        
        Args:
            success: Whether the session completed successfully
            final_answer: Final answer (required if success=True)
            error_message: Error message (if success=False)
            metadata: Additional metadata to save
            
        Returns:
            Session data dict if saved, None otherwise
        """
        if self.current_session is None:
            logger.warning("No active session to end.")
            return None
        
        # Update session metadata
        if metadata:
            self.current_session.metadata.update(metadata)
        
        # Mark session status
        if success:
            if final_answer is None:
                logger.warning("Success=True but no final_answer provided")
                final_answer = ""
            self.current_session.mark_success(final_answer)
            self.successful_sessions += 1
            self.total_samples += len(self.current_session.samples)
        else:
            self.current_session.mark_failure(error_message or "Unknown error")
        
        # Save if successful and auto_save is enabled
        session_data = None
        if success and self.auto_save:
            session_data = self._save_session(self.current_session)
            logger.info(f"Session {self.current_session.session_id} saved successfully "
                       f"({len(self.current_session.samples)} samples)")
        elif not success:
            logger.info(f"Session {self.current_session.session_id} failed, not saving")
        
        # Clear current session
        self.current_session = None
        
        return session_data
    
    def _save_session(self, session: SessionData) -> Dict[str, Any]:
        """
        Save session data to disk
        
        Args:
            session: SessionData to save
            
        Returns:
            Session data dictionary
        """
        # Create session directory
        session_dir = self.sessions_dir / session.session_id
        session_dir.mkdir(exist_ok=True)
        
        # Copy images if enabled
        if self.save_images:
            session_images_dir = session_dir / "images"
            session_images_dir.mkdir(exist_ok=True)
            
            # Copy original images
            for img_path in session.original_images:
                if Path(img_path).exists():
                    dest = session_images_dir / Path(img_path).name
                    shutil.copy2(img_path, dest)
            
            # Copy images from each sample
            for sample in session.samples:
                for img_path in sample.images:
                    if Path(img_path).exists():
                        dest = session_images_dir / Path(img_path).name
                        if not dest.exists():  # Avoid duplicate copies
                            shutil.copy2(img_path, dest)
        
        # Save session metadata as JSON
        session_data = session.to_dict()
        metadata_path = session_dir / "session_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Save each sample separately (easier for training data loading)
        samples_dir = session_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(session.samples):
            sample_path = samples_dir / f"sample_{i+1}.json"
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(sample.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved session to {session_dir}")
        return session_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "total_sessions": self.total_sessions,
            "successful_sessions": self.successful_sessions,
            "failed_sessions": self.total_sessions - self.successful_sessions,
            "total_samples": self.total_samples,
            "success_rate": self.successful_sessions / self.total_sessions if self.total_sessions > 0 else 0,
            "avg_samples_per_success": self.total_samples / self.successful_sessions if self.successful_sessions > 0 else 0
        }
    
    def save_statistics(self):
        """Save statistics to file"""
        stats = self.get_statistics()
        stats_path = self.output_dir / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")
    
    def export_for_training(
        self,
        output_file: str,
        format: str = "jsonl",
        simple_format: bool = False
    ):
        """
        Export collected data in training-ready format
        
        Args:
            output_file: Output file path
            format: Export format ('jsonl', 'json', 'sharegpt', or 'simple')
            simple_format: If True, export only essential Q&A without system prompts
        """
        output_path = Path(output_file)
        
        # Collect all samples from successful sessions
        all_samples = []
        
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                metadata_path = session_dir / "session_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    # Only include successful sessions
                    if session_data.get('success'):
                        all_samples.extend(session_data.get('samples', []))
        
        # Export based on format
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in all_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        elif format == "sharegpt":
            # Convert to ShareGPT format for multimodal training
            sharegpt_data = []
            for sample in all_samples:
                if simple_format:
                    # Extract only the essential question from prompt
                    prompt_value = self._extract_question_from_prompt(sample['prompt'])
                    response_value = sample['response']
                else:
                    prompt_value = sample['prompt']
                    response_value = sample['response']
                
                conversation = {
                    "id": sample['sample_id'],
                    "images": sample['images'],
                    "conversations": [
                        {
                            "from": "human",
                            "value": prompt_value
                        },
                        {
                            "from": "gpt",
                            "value": response_value
                        }
                    ]
                }
                sharegpt_data.append(conversation)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
        
        elif format == "simple":
            # Simple format: only images, question, and answer
            simple_data = []
            for sample in all_samples:
                simple_sample = {
                    "id": sample['sample_id'],
                    "images": sample['images'],
                    "question": self._extract_question_from_prompt(sample['prompt']),
                    "answer": sample['response'],
                    "iteration": sample.get('iteration', 1),
                    "context": sample.get('context', {})
                }
                simple_data.append(simple_sample)
            
            if output_path.suffix == '.jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for item in simple_data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_data, f, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(all_samples)} samples to {output_path} in {format} format")
    
    def _extract_question_from_prompt(self, prompt: str) -> str:
        """
        Extract the core question from a complex prompt
        
        Args:
            prompt: Full prompt text
            
        Returns:
            Extracted question or simplified prompt
        """
        import re
        
        # Try to extract from "Question:" or "Original Question:" section
        patterns = [
            r'Original Question:\s*(.+?)(?:\n\n|$)',
            r'Question:\s*(.+?)(?:\n\n|\nThink step|$)',
            r'Please analyze the following.+?Question:\s*(.+?)(?:\n\n|\nThink step|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                question = match.group(1).strip()
                # Also extract images info if present
                images_match = re.search(r'Images? to analyze:\s*(.+?)(?:\n\nQuestion|$)', prompt, re.DOTALL)
                if images_match:
                    images_info = images_match.group(1).strip()
                    return f"{images_info}\n\nQuestion:\n{question}"
                return question
        
        # If no pattern matches, try to extract after "Please analyze"
        if "Please analyze" in prompt:
            parts = prompt.split("Please analyze the following", 1)
            if len(parts) > 1:
                # Get everything after "Please analyze"
                after_analyze = parts[1].strip()
                # Remove tool definitions and system prompts
                if "Question:" in after_analyze:
                    question_part = after_analyze.split("Question:", 1)[1]
                    question_part = question_part.split("\n\nThink step")[0].strip()
                    question_part = question_part.split("\n\nImportant Notes:")[0].strip()
                    return f"Question:\n{question_part}"
        
        # For continuation prompts, extract the original question and context
        if "=== Multi-Step Analysis:" in prompt:
            # Extract original question
            orig_q_match = re.search(r'Original Question:\s*(.+?)(?:\n\nYour Previous Response:|$)', prompt, re.DOTALL)
            if orig_q_match:
                original_question = orig_q_match.group(1).strip()
                
                # Extract tool results summary
                tool_summary_match = re.search(r'Tool Execution Summary:\s*(.+?)(?:\n\nGenerated Images|$)', prompt, re.DOTALL)
                tool_summary = tool_summary_match.group(1).strip() if tool_summary_match else ""
                
                # Extract generated images
                images_match = re.search(r'Generated Images Available for Analysis:\s*(.+?)(?:\n\n===|$)', prompt, re.DOTALL)
                images_info = images_match.group(1).strip() if images_match else ""
                
                return f"Original Question:\n{original_question}\n\nPrevious Tool Results:\n{tool_summary}\n\nAvailable Images:\n{images_info}\n\nPlease continue your analysis."
        
        # Fallback: return a truncated version
        if len(prompt) > 500:
            return prompt[:500] + "...[truncated]"
        return prompt

