"""
SPAgent - Spatial Intelligence Agent

This module contains the main SPAgent class that orchestrates problem solving
using external expert tools and VLLM models.
"""

import json
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .tool import Tool, ToolRegistry
from .model import Model
from .memory import AgentMemory, StepResult
from .prompts import (
    create_system_prompt, create_follow_up_prompt, create_user_prompt, create_fallback_prompt,
    SPATIAL_3D_CONTINUATION_HINT, GENERAL_VISION_CONTINUATION_HINT,
    build_general_vision_continuation_hint,
    SPATIAL_3D_SYSTEM_PROMPT, GENERAL_VISION_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT, GENERATION_CONTINUATION_HINT,
    TOOL_CALLING_BLOCK, build_system_prompt, create_all_tools_system_prompt,
    SPATIAL_3D_ROLE, GENERAL_VISION_ROLE, GENERATION_ROLE,
    SPATIAL_3D_WORKFLOW, GENERAL_VISION_WORKFLOW, GENERATION_WORKFLOW,
    ALL_TOOLS_CONTINUATION_HINT,
)
import json as _json
from .data_collector import DataCollector

logger = logging.getLogger(__name__)


class SPAgent:
    """
    Spatial Intelligence Agent
    
    An agent that can solve spatial intelligence problems by combining
    VLLM models with external expert tools.
    """
    
    def __init__(
        self,
        model: Model,
        tools: Optional[List[Tool]] = None,
        max_workers: int = 4,
        data_collector: Optional[DataCollector] = None,
        system_prompt: Optional[str] = None,
        continuation_hint: Optional[str] = None,
        workflow_mode: str = "default",
    ):
        """
        Initialize SPAgent

        Args:
            model: VLLM model wrapper to use
            tools: List of external expert tools (optional)
            max_workers: Maximum number of parallel tool executions
            data_collector: Optional DataCollector for training data collection
            system_prompt: Optional *role* prompt that describes what the agent
                is and what it should do.  The tool-calling block (tool list +
                ``<tool_call>`` wire format) is **always appended automatically**,
                so you do not need to include it here.  Example::

                    agent = SPAgent(model=..., system_prompt="You are a robot navigation assistant.")

                For full control over all three layers (role + tools + workflow),
                call ``build_system_prompt()`` from ``spagent.core.prompts`` and
                pass the result here.  Legacy strings that already contain a
                ``{tools_json}`` placeholder are treated as full templates and
                substituted directly (backward-compatible).
            continuation_hint: Optional next-step instructions injected into
                every multi-step continuation prompt (iteration 2+).
                If None, auto-selects: GENERAL_VISION_CONTINUATION_HINT when
                system_prompt is set, SPATIAL_3D_CONTINUATION_HINT otherwise.
                Use constants from ``spagent.core.prompts``.
            workflow_mode: Workflow routing mode.
                - "default": preserve existing behavior
                - "auto": automatically choose between spatial_3d, general_vision,
                  and generation workflows based on tools, images, and task text
                - "all_tools": use the all-tools role, selection guide, and workflow
        """
        self.model = model
        self.tool_registry = ToolRegistry()
        self.max_workers = max_workers
        self.data_collector = data_collector
        self.system_prompt_template = system_prompt
        self.user_continuation_hint = continuation_hint
        self.workflow_mode = workflow_mode
        self.continuation_hint = continuation_hint
        
        # Register provided tools
        if tools:
            for tool in tools:
                self.add_tool(tool)
        
        logger.info(f"Initialized SPAgent with model: {model.model_name}")
        if data_collector:
            logger.info("Data collection enabled")
    
    def add_tool(self, tool: Tool):
        """
        Add a tool to the agent
        
        Args:
            tool: Tool instance to add
        """
        self.tool_registry.register(tool)
    
    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the agent
        
        Args:
            tool_name: Name of tool to remove
        """
        self.tool_registry.unregister(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        Get list of available tool names
        
        Returns:
            List of tool names
        """
        return self.tool_registry.list_tools()
    
    def set_model(self, model: Model):
        """
        Set the VLLM model
        
        Args:
            model: Model instance to use
        """
        self.model = model
        logger.info(f"Updated model to: {model.model_name}")
    
    def step(
        self,
        content: str,
        images: Optional[Union[str, List[str]]] = None,
        memory: Optional[AgentMemory] = None,
        system_prompt: Optional[str] = None,
        max_tool_iterations: int = 3,
        max_images_in_context: int = 6,
        video_path: Optional[str] = None,
        pi3_num_frames: int = 7,
        video_num_frames: int = 4,
        use_baseline_comparison: bool = False,
        **model_kwargs,
    ) -> StepResult:
        """
        Execute a single agent step: perceive → reason → act → update memory.

        This is the primary entry point for all agent interactions.  It can be
        called once for a self-contained query (stateless) or repeatedly with
        the same ``memory`` object to build a multi-turn conversation (stateful).

        Args:
            content:               Text instruction or question for this step.
            images:                Input image path(s) for this step.  May be
                                   ``None`` for text-only tasks.
            memory:                An existing :class:`AgentMemory` to append to.
                                   When ``None`` a fresh memory is created, giving
                                   stateless (one-shot) behavior.
            system_prompt:         Optional *role* prompt that overrides the
                                   agent-level default for this step only.
                                   Provide just the role / task description —
                                   the tool-calling block and workflow are appended
                                   automatically.  Use ``build_system_prompt()``
                                   from ``spagent.core.prompts`` when you need
                                   full control over all three layers.
            max_tool_iterations:   Maximum number of tool-call iterations within
                                   this single step (default: 3).
            max_images_in_context: Maximum number of images sent to the model per
                                   inference call. Original input images are always
                                   kept; only the most recent tool outputs fill the
                                   remaining budget (default: 6).
            video_path:            Optional path to original video (used to
                                   re-sample frames for the pi3 tool).
            pi3_num_frames:        Number of frames to uniformly sample for the
                                   pi3 tool when ``video_path`` is provided.
            video_num_frames:      Number of frames to extract from tool-generated
                                   video outputs before feeding them back to the
                                   model (default: 4).
            use_baseline_comparison: When ``True``, a naive (no-tool) baseline
                                   answer is obtained the first time tool calls
                                   are detected and later synthesized with the
                                   tool-enhanced answer.
            **model_kwargs:        Extra keyword arguments forwarded to every
                                   model inference call.

        Returns:
            A :class:`StepResult` containing the final answer, the updated
            memory, and full trace information (tool calls, results, images,
            iteration count, prompts).
        """
        if memory is None:
            memory = AgentMemory()

        # Normalize image input
        if images is None:
            image_paths: List[str] = []
        elif isinstance(images, str):
            image_paths = [images]
        else:
            image_paths = list(images)

        for path in image_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")

        logger.info(
            f"Starting step: content={content!r}, images={image_paths}, "
            f"max_tool_iterations={max_tool_iterations}"
        )

        # Start data collection session if enabled
        if self.data_collector:
            self.data_collector.start_session(content, image_paths)

        # Resolve system prompt, continuation hint, and workflow label
        tool_schemas = self.tool_registry.get_function_schemas()
        system_prompt_str, active_continuation_hint, active_workflow = self._resolve_workflow_prompts(
            question=content,
            image_paths=image_paths,
            tool_schemas=tool_schemas,
            role_prompt_override=system_prompt,
        )
        system_prompt = system_prompt_str  # rebind to local for rest of step
        user_prompt = create_user_prompt(content, image_paths, tool_schemas)
        logger.info(
            "System prompt size: %d chars (~%d tokens), tools=%d, workflow=%s",
            len(system_prompt),
            len(system_prompt) // 4,
            len(tool_schemas),
            active_workflow,
        )

        # Record system and user turns in memory
        memory.add_system(system_prompt)
        memory.add_user_turn(content, images=image_paths)

        # Per-step tracking (not stored in memory directly, used to build StepResult)
        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: Dict[str, Any] = {}
        all_additional_images: List[str] = []
        all_successful_tools: List[str] = []
        current_images = image_paths if image_paths else []
        iteration = 0
        baseline_answer = None
        baseline_triggered = False

        # ----------------------------------------------------------------
        # Tool-call iteration loop
        # ----------------------------------------------------------------
        while iteration < max_tool_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{max_tool_iterations} ===")

            if iteration == 1:
                prompt = system_prompt + "\n\n" + user_prompt
                logger.info("Getting initial model response...")
            else:
                prompt = memory.build_prompt_context(
                    current_iteration=iteration,
                    max_iterations=max_tool_iterations,
                    continuation_hint=active_continuation_hint,
                )
                logger.info(f"Getting continuation response for iteration {iteration}...")

            current_response = self._run_model_inference(
                current_images, prompt, **model_kwargs
            )
            logger.info(f"Response: {current_response}")

            # Record assistant turn in memory
            memory.add_assistant_turn(current_response, metadata={"iteration": iteration})

            # Record inference in data collector if enabled
            if self.data_collector:
                self.data_collector.record_inference(
                    iteration=iteration,
                    images=current_images,
                    prompt=prompt,
                    response=current_response,
                    context={
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results,
                        "additional_images_history": all_additional_images,
                    },
                )

            # Parse tool calls from the response
            tool_calls = self._parse_tool_calls(current_response)

            # Optionally trigger naive baseline on first tool use
            if use_baseline_comparison and tool_calls and not baseline_triggered:
                baseline_triggered = True
                logger.info("Tool calls detected — triggering naive baseline agent...")
                baseline_answer = self._get_naive_baseline_answer(
                    image_paths, content, **model_kwargs
                )
                logger.info(f"Naive baseline answer: {baseline_answer}")

            has_answer = self._has_answer_tags(current_response)

            if not tool_calls:
                logger.info(f"No tool calls found in iteration {iteration}")
                if has_answer or iteration == max_tool_iterations:
                    logger.info("Ending workflow: final answer provided or max iterations reached")
                    break
                continue

            # Execute tools
            logger.info(f"Executing {len(tool_calls)} tool calls in iteration {iteration}...")
            tool_results = self._execute_tools(
                tool_calls, video_path=video_path, pi3_num_frames=pi3_num_frames
            )

            # Collect output images and record everything in memory
            iteration_additional_images: List[str] = []
            for tool_name, result in tool_results.items():
                output_images: List[str] = []
                if result.get("success"):
                    all_successful_tools.append(f"{tool_name}_iter{iteration}")
                    if result.get("output_path") is not None:
                        out_path = result["output_path"]
                        if Path(out_path).exists():
                            if Path(out_path).suffix.lower() == ".mp4":
                                frame_paths = self._extract_video_frames(out_path, video_num_frames)
                                logger.info(
                                    f"Extracted {len(frame_paths)} frames from generated video: {out_path}"
                                )
                                iteration_additional_images.extend(frame_paths)
                                output_images.extend(frame_paths)
                            else:
                                iteration_additional_images.append(out_path)
                                output_images.append(out_path)
                    if result.get("vis_path") is not None and Path(result["vis_path"]).exists():
                        iteration_additional_images.append(result["vis_path"])
                        output_images.append(result["vis_path"])
                    if result.get("crop_paths"):
                        for crop_path in result["crop_paths"]:
                            if Path(crop_path).exists():
                                iteration_additional_images.append(crop_path)
                                output_images.append(crop_path)

                # Record each tool call + result pair in memory
                for tc in tool_calls:
                    if tc["name"] == tool_name:
                        memory.add_tool_call(
                            tool_name=tool_name,
                            arguments=tc["arguments"],
                            iteration=iteration,
                        )
                        break
                memory.add_tool_result(
                    tool_name=tool_name,
                    result=result,
                    output_images=output_images,
                    iteration=iteration,
                )

            # Update step-level accumulators
            all_tool_calls.extend(tool_calls)
            all_tool_results.update(
                {f"{k}_iter{iteration}": v for k, v in tool_results.items()}
            )
            all_additional_images.extend(iteration_additional_images)

            # Update current_images for next iteration (respect image budget)
            if iteration_additional_images:
                valid_all_images = self._sort_additional_images_by_input_order(
                    image_paths, all_additional_images
                )
                if valid_all_images:
                    current_images = self._apply_image_budget(
                        image_paths,
                        valid_all_images,
                        max_images_in_context,
                    )

            if has_answer and iteration < max_tool_iterations:
                logger.info(
                    f"Answer provided in iteration {iteration}, "
                    "but continuing workflow if more tool calls exist"
                )

        # ----------------------------------------------------------------
        # Post-loop: generate final response if needed
        # ----------------------------------------------------------------
        last_response = memory.get_last_assistant_text() or ""

        if not self._has_answer_tags(last_response) and all_successful_tools:
            logger.info("Generating final response...")
            tool_description = None
            if all_tool_results:
                last_result = list(all_tool_results.values())[-1]
                if last_result.get("description"):
                    tool_description = last_result["description"]

            initial_response = memory.get_first_assistant_text() or last_response
            follow_up_prompt = create_follow_up_prompt(
                content,
                initial_response,
                all_tool_results,
                image_paths,
                all_additional_images,
                tool_description,
                continuation_hint=active_continuation_hint,
            )
            valid_additional_images = self._sort_additional_images_by_input_order(
                image_paths, all_additional_images
            )
            final_images = valid_additional_images if valid_additional_images else (image_paths or current_images)

            final_response = self._run_model_inference(final_images, follow_up_prompt, **model_kwargs)
            memory.add_assistant_turn(final_response, metadata={"type": "final_synthesis"})

            if self.data_collector:
                self.data_collector.record_inference(
                    iteration=iteration + 1,
                    images=final_images,
                    prompt=follow_up_prompt,
                    response=final_response,
                    context={
                        "type": "final_synthesis",
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results,
                        "additional_images_history": all_additional_images,
                    },
                )

        elif not self._has_answer_tags(last_response):
            logger.warning("No answer tags found, generating fallback response")
            fallback_prompt = create_fallback_prompt(content, last_response)
            final_response = self._run_model_inference(
                current_images, fallback_prompt, **model_kwargs
            )
            memory.add_assistant_turn(final_response, metadata={"type": "fallback"})

            if self.data_collector:
                self.data_collector.record_inference(
                    iteration=iteration + 1,
                    images=current_images,
                    prompt=fallback_prompt,
                    response=final_response,
                    context={
                        "type": "fallback",
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results,
                    },
                )
        else:
            final_response = last_response

        # Optional baseline synthesis
        if use_baseline_comparison and baseline_answer is not None:
            logger.info("Synthesizing final answer from tool-based and baseline responses...")
            pre_synthesis_response = final_response
            final_response = self._synthesize_with_baseline(
                content,
                final_response,
                baseline_answer,
                image_paths,
                all_additional_images,
                **model_kwargs,
            )
            memory.add_assistant_turn(final_response, metadata={"type": "baseline_synthesis"})

            if self.data_collector:
                valid_additional_images = self._sort_additional_images_by_input_order(
                    image_paths, all_additional_images
                )
                final_images = valid_additional_images if valid_additional_images else image_paths
                self.data_collector.record_inference(
                    iteration=iteration + 2,
                    images=final_images,
                    prompt=f"Synthesizing tool-based and baseline answers for: {content}",
                    response=final_response,
                    context={
                        "type": "baseline_synthesis",
                        "tool_based_answer": pre_synthesis_response,
                        "baseline_answer": baseline_answer,
                        "tool_calls_history": all_tool_calls,
                        "tool_results_history": all_tool_results,
                    },
                )

        # Clean up temporary pi3 frames
        self._cleanup_pi3_frames()

        # End data collection session
        if self.data_collector:
            extracted_answer = self._extract_answer(final_response)
            success = extracted_answer is not None
            self.data_collector.end_session(
                success=success,
                final_answer=extracted_answer or final_response,
                error_message=None if success else "No answer tags found",
                metadata={
                    "iterations": iteration,
                    "num_tool_calls": len(all_tool_calls),
                    "used_tools": all_successful_tools,
                    "num_additional_images": len(all_additional_images),
                },
            )

        return StepResult(
            answer=final_response,
            memory=memory,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            used_tools=all_successful_tools,
            additional_images=all_additional_images,
            iterations=iteration,
            prompts={
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "follow_up_prompt": None,
                "workflow": active_workflow,
            },
        )

    def solve_problem(
        self,
        image_path: Union[str, List[str]],
        question: str,
        max_iterations: int = 3,
        video_path: Optional[str] = None,
        pi3_num_frames: int = 7,
        use_baseline_comparison: bool = False,
        video_num_frames: int = 4,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        Solve a spatial intelligence problem.

        This is a backward-compatible wrapper around :meth:`step`.  All existing
        callers (evaluation scripts, examples) continue to receive the same
        dictionary-shaped return value.

        Args:
            image_path:            Path to image or list of image paths.
            question:              User's question about the image(s).
            max_iterations:        Maximum number of tool-call iterations (default: 3).
            video_path:            Optional path to original video (for pi3 re-sampling).
            pi3_num_frames:        Frames to uniformly sample for the pi3 tool.
            use_baseline_comparison: Run a naive baseline and synthesize with the
                                   tool-enhanced answer (default: False).
            video_num_frames:      Frames to extract from tool-generated videos.
            **model_kwargs:        Additional arguments forwarded to model inference.

        Returns:
            Dictionary containing:
            - answer:           Final answer text.
            - initial_response: Model's first response.
            - tool_calls:       List of tool calls made.
            - tool_results:     Results from tool execution.
            - used_tools:       List of tools that succeeded.
            - additional_images:List of images produced by tools.
            - iterations:       Number of iterations performed.
            - baseline_answer:  Naive baseline answer (if use_baseline_comparison=True).
            - prompts:          Dict of key prompts used.
        """
        result = self.step(
            content=question,
            images=image_path,
            max_tool_iterations=max_iterations,
            video_path=video_path,
            pi3_num_frames=pi3_num_frames,
            use_baseline_comparison=use_baseline_comparison,
            video_num_frames=video_num_frames,
            **model_kwargs,
        )

        initial_response = result.memory.get_first_assistant_text() or result.answer
        baseline_answer = None
        for entry in reversed(result.memory.get_by_role("assistant")):
            if entry.metadata.get("type") == "baseline_synthesis":
                # baseline_answer is stored in the tool result metadata, not directly in memory;
                # recover it from the StepResult via the pre-synthesis path — we expose it
                # through the data_collector context but not in StepResult fields.  To keep
                # backward compatibility we leave it as None when not accessible.
                break

        return {
            "answer": result.answer,
            "initial_response": initial_response,
            "tool_calls": result.tool_calls,
            "tool_results": result.tool_results,
            "used_tools": result.used_tools,
            "additional_images": result.additional_images,
            "iterations": result.iterations,
            "baseline_answer": baseline_answer,
            "prompts": result.prompts,
            "memory_entries": [e.to_dict() for e in result.memory.entries],
        }

    def _run_model_inference(
        self,
        images: List[str],
        prompt: str,
        **model_kwargs
    ) -> str:
        """
        Run model inference for text-only, single-image, or multi-image inputs.

        Args:
            images: Input image paths. May be empty for text-only tasks.
            prompt: Prompt sent to the model.
            **model_kwargs: Additional generation arguments.

        Returns:
            Model response text.
        """
        if not images:
            logger.info("Running text-only inference")
            return self.model.text_only_inference(prompt, **model_kwargs)
        if len(images) == 1:
            return self.model.single_image_inference(images[0], prompt, **model_kwargs)
        return self.model.multiple_images_inference(images, prompt, **model_kwargs)

    def _resolve_workflow_prompts(
        self,
        question: str,
        image_paths: List[str],
        tool_schemas: List[Dict[str, Any]],
        role_prompt_override: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """
        Resolve the active system prompt, continuation hint, and workflow label.

        Priority order for the role prompt:
          1. ``role_prompt_override``  — per-step argument to ``step()``
          2. ``self.system_prompt_template``  — agent-level default
          3. Built-in preset chosen by ``workflow_mode``
        """
        tools_json = _json.dumps(tool_schemas, indent=2)
        tool_names = {s["function"]["name"] for s in tool_schemas}

        # ── all_tools workflow ────────────────────────────────────────────────
        if self.workflow_mode == "all_tools":
            system_prompt = create_all_tools_system_prompt(tool_schemas)
            continuation_hint = (
                self.user_continuation_hint
                if self.user_continuation_hint is not None
                else ALL_TOOLS_CONTINUATION_HINT
            )
            return system_prompt, continuation_hint, "all_tools"

        # ── per-step or agent-level custom role prompt ───────────────────────
        role_prompt = role_prompt_override or self.system_prompt_template
        if role_prompt is not None:
            system_prompt = self._render_role_prompt(role_prompt, tools_json)
            continuation_hint = (
                self.user_continuation_hint
                if self.user_continuation_hint is not None
                else build_general_vision_continuation_hint(tool_names)
            )
            return system_prompt, continuation_hint, "custom"

        # ── auto workflow selection ───────────────────────────────────────────
        if self.workflow_mode == "auto":
            workflow = self._select_workflow(question, image_paths)
            if workflow == "generation":
                role, wf_block, hint = GENERATION_ROLE, GENERATION_WORKFLOW, GENERATION_CONTINUATION_HINT
            elif workflow == "general_vision":
                role, wf_block, hint = (
                    GENERAL_VISION_ROLE,
                    GENERAL_VISION_WORKFLOW,
                    build_general_vision_continuation_hint(tool_names),
                )
            else:
                role, wf_block, hint = SPATIAL_3D_ROLE, SPATIAL_3D_WORKFLOW, SPATIAL_3D_CONTINUATION_HINT
            system_prompt = build_system_prompt(role, tools_json, workflow=wf_block)
            continuation_hint = self.user_continuation_hint if self.user_continuation_hint is not None else hint
            return system_prompt, continuation_hint, workflow

        # ── default: spatial_3d ───────────────────────────────────────────────
        system_prompt = create_system_prompt(tool_schemas)
        continuation_hint = (
            self.user_continuation_hint
            if self.user_continuation_hint is not None
            else SPATIAL_3D_CONTINUATION_HINT
        )
        return system_prompt, continuation_hint, "spatial_3d"

    def _render_role_prompt(
        self,
        role_prompt: str,
        tools_json: str,
    ) -> str:
        """
        Render a user-supplied role prompt into a complete system prompt.

        Two cases:
        * The prompt already contains ``{tools_json}`` — it is a legacy
          full-template; substitute the placeholder and return as-is.
        * The prompt is a plain role description — automatically append the
          full tool-calling block (tool list + ``<tool_call>`` wire format)
          so the model always knows how to invoke tools.
        """
        if "{tools_json}" in role_prompt:
            # Legacy full-template: trust the user's structure
            return role_prompt.replace("{tools_json}", tools_json)
        # Pure role description: append tool-calling instructions
        return build_system_prompt(role_prompt, tools_json, workflow=None)

    def _select_workflow(self, question: str, image_paths: List[str]) -> str:
        """
        Heuristically select the workflow family for the current task.
        """
        tool_names = set(self.tool_registry.list_tools())
        question_lc = (question or "").lower()

        generation_tools = {
            "image_generation_sana_tool",
            "video_generation_veo_tool",
            "video_generation_sora_tool",
            "video_generation_wan_tool",
        }
        spatial_tools = {
            "pi3_tool",
            "pi3x_tool",
            "vggt_tool",
            "mapanything_tool",
        }

        generation_keywords = [
            "generate", "create", "visualize", "imagine", "render", "synthesize",
            "draw", "make an image", "make a video", "produce an image", "produce a video",
        ]
        spatial_keywords = [
            "3d", "viewpoint", "azimuth", "elevation", "camera", "orientation",
            "relative position", "spatial relationship", "bird's-eye", "top view",
            "novel view", "camera view", "depth relationship",
        ]

        has_generation_tools = bool(tool_names & generation_tools)
        has_spatial_tools = bool(tool_names & spatial_tools)
        generation_only = bool(tool_names) and tool_names.issubset(generation_tools)

        if generation_only:
            return "generation"
        if not image_paths and has_generation_tools:
            return "generation"
        if any(keyword in question_lc for keyword in generation_keywords) and has_generation_tools:
            return "generation"

        if has_spatial_tools and (len(image_paths) > 1 or any(keyword in question_lc for keyword in spatial_keywords)):
            return "spatial_3d"

        return "general_vision"

    def _apply_image_budget(
        self,
        image_paths: List[str],
        additional_images: List[str],
        max_images_in_context: int,
    ) -> List[str]:
        """
        Keep all original input images and only the most recent tool outputs
        that fit within the image budget.
        """
        if max_images_in_context <= 0:
            return image_paths

        remaining = max(0, max_images_in_context - len(image_paths))
        if remaining <= 0:
            logger.info(
                "Image budget reached by input images alone (%d); dropping tool images",
                len(image_paths),
            )
            return image_paths

        if len(additional_images) <= remaining:
            return image_paths + additional_images

        trimmed = additional_images[-remaining:]
        logger.info(
            "Image budget applied: kept %d/%d tool images (max=%d total)",
            len(trimmed),
            len(additional_images),
            max_images_in_context,
        )
        return image_paths + trimmed
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from model response
        
        Args:
            response: Model response text
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        # Find all tool_call blocks
        pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                if 'name' in tool_call and 'arguments' in tool_call:
                    tool_calls.append(tool_call)
                else:
                    logger.warning(f"Invalid tool call format: {match}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool call JSON: {match}, error: {e}")
        
        return tool_calls
    
    def _execute_tools(self, tool_calls: List[Dict[str, Any]], video_path: Optional[str] = None, pi3_num_frames: int = 10) -> Dict[str, Any]:
        """
        Execute tool calls in parallel when possible (except Pi3 and VACE: sequential / single-flight)
        
        Args:
            tool_calls: List of tool call dictionaries
            video_path: Optional path to original video (for pi3 tool re-sampling)
            pi3_num_frames: Number of frames to uniformly sample for pi3 tool
            
        Returns:
            Dictionary of tool_name -> result
        """
        tool_results = {}
        
        # Group tool calls by tool name to handle multiple calls to same tool
        tool_groups = {}
        for i, call in enumerate(tool_calls):
            tool_name = call['name']
            if tool_name not in tool_groups:
                tool_groups[tool_name] = []
            tool_groups[tool_name].append((i, call))
        
        # Execute tools in parallel, but Pi3Tool and Pi3MultiimgTool sequentially to avoid server issues;
        # video_generation_vace_tool runs at most once per iteration (GPU memory).
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {}
            
            # Handle Pi3Tool and Pi3MultiimgTool calls sequentially first
            pi3_calls = []
            vace_calls = []
            other_calls = {}
            
            for tool_name, calls in tool_groups.items():
                if tool_name in ['pi3_tool', 'pi3_multiimg_tool']:
                    pi3_calls.extend(calls)
                elif tool_name == 'video_generation_vace_tool':
                    vace_calls.extend(calls)
                else:
                    other_calls[tool_name] = calls
            
            # Execute Pi3Tool and Pi3MultiimgTool calls sequentially
            if pi3_calls:
                logger.info(f"Executing {len(pi3_calls)} Pi3 tool calls sequentially...")
                
                # Extract more frames for pi3 if video_path is provided
                pi3_frame_paths = []
                if video_path and Path(video_path).exists():
                    logger.info(f"Extracting {pi3_num_frames} frames for pi3 tool from video: {video_path}")
                    pi3_frame_paths = self._extract_frames_for_pi3(video_path, pi3_num_frames)
                
                for call_idx, call in pi3_calls:
                    tool_name = call['name']
                    tool = self.tool_registry.get(tool_name)
                    if tool:
                        # If we extracted frames for pi3, update the arguments
                        arguments = call['arguments'].copy()
                        if pi3_frame_paths:
                            # Update image_path in arguments
                            if 'image_path' in arguments:
                                logger.info(f"Updating pi3 tool arguments with {len(pi3_frame_paths)} newly extracted frames")
                                arguments['image_path'] = pi3_frame_paths
                        
                        result = self._safe_tool_call(tool, arguments)
                        result_key = tool_name if len(pi3_calls) == 1 else f"{tool_name}_{call_idx}"
                        tool_results[result_key] = result
                        # Add small delay between Pi3 tool calls
                        import time
                        time.sleep(1)
                    else:
                        logger.error(f"{tool_name} not found")
                        result_key = tool_name if len(pi3_calls) == 1 else f"{tool_name}_{call_idx}"
                        tool_results[result_key] = {
                            "success": False,
                            "error": f"{tool_name} not found"
                        }
            
            if vace_calls:
                tool_name = 'video_generation_vace_tool'
                tool = self.tool_registry.get(tool_name)
                n_vace = len(vace_calls)
                if n_vace > 1:
                    logger.warning(
                        "video_generation_vace_tool called %d times in one iteration; "
                        "only the first runs (GPU memory).",
                        n_vace,
                    )
                for j, (call_idx, call) in enumerate(vace_calls):
                    result_key = tool_name if n_vace == 1 else f"{tool_name}_{call_idx}"
                    if tool is None:
                        logger.error("%s not found", tool_name)
                        tool_results[result_key] = {"success": False, "error": f"{tool_name} not found"}
                        continue
                    if j == 0:
                        tool_results[result_key] = self._safe_tool_call(tool, call['arguments'])
                    else:
                        tool_results[result_key] = {
                            "success": False,
                            "error": (
                                "Skipped: only one VACE video_generation call per iteration (GPU memory)."
                            ),
                        }
            
            # Execute other tools in parallel as before
            for tool_name, calls in other_calls.items():
                tool = self.tool_registry.get(tool_name)
                if tool is None:
                    logger.error(f"Tool not found: {tool_name}")
                    for _, call in calls:
                        tool_results[f"{tool_name}_{_}"] = {
                            "success": False,
                            "error": f"Tool not found: {tool_name}"
                        }
                    continue
                
                # Submit tool execution for each call
                for call_idx, call in calls:
                    future = executor.submit(self._safe_tool_call, tool, call['arguments'])
                    future_to_tool[future] = (tool_name, call_idx)
            
            # Collect results from parallel execution
            for future in as_completed(future_to_tool):
                tool_name, call_idx = future_to_tool[future]
                try:
                    result = future.result()
                    # Use unique key for multiple calls to same tool
                    result_key = tool_name if len([t for t in other_calls.get(tool_name, [])]) == 1 else f"{tool_name}_{call_idx}"
                    tool_results[result_key] = result
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    result_key = tool_name if len([t for t in other_calls.get(tool_name, [])]) == 1 else f"{tool_name}_{call_idx}"
                    tool_results[result_key] = {
                        "success": False,
                        "error": str(e)
                    }

        return tool_results
    
    def _safe_tool_call(self, tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute a tool call with error handling
        
        Args:
            tool: Tool instance
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"Executing tool: {tool.name} with args: {arguments}")
            result = tool.call(**arguments)
            logger.info(f"Tool {tool.name} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool {tool.name} execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _sort_additional_images_by_input_order(self, image_paths: List[str], additional_images: List[str]) -> List[str]:
        """
        Sort additional images to match the order of input image_paths
        
        Args:
            image_paths: Original input image paths in order
            additional_images: Generated additional images (may be in different order)
            
        Returns:
            List of valid additional images sorted by input order
        """
        # Filter out None and invalid paths first
        valid_additional_images = [img for img in additional_images if img is not None and Path(img).exists()]
        
        if not valid_additional_images:
            return []
        
        # Create a sorted list of additional images based on input order
        sorted_additional_images = []
        
        for input_path in image_paths:
            input_stem = Path(input_path).stem
            
            # Find all additional images that correspond to this input image
            matching_images = []
            
            for additional_img in valid_additional_images:
                if additional_img in sorted_additional_images:
                    continue  # Already matched
                
                additional_stem = Path(additional_img).stem
                
                # Check if this additional image corresponds to the current input image
                if self._is_image_match(input_stem, additional_stem):
                    matching_images.append(additional_img)
            
            # Add matching images in a consistent order (by filename)
            matching_images.sort()
            sorted_additional_images.extend(matching_images)
        
        # Add any remaining unmatched additional images at the end
        for additional_img in valid_additional_images:
            if additional_img not in sorted_additional_images:
                sorted_additional_images.append(additional_img)
        
        logger.info(f"Sorted additional images: {[Path(img).name for img in sorted_additional_images]}")
        return sorted_additional_images
    
    def _is_image_match(self, input_stem: str, additional_stem: str) -> bool:
        """
        Check if an additional image corresponds to an input image
        
        Args:
            input_stem: Stem of input image filename
            additional_stem: Stem of additional image filename
            
        Returns:
            True if they match, False otherwise
        """
        # Since the suffix of saved images is always the same as the original image name,
        # we check if the additional image name ends with the input image name
        # This handles any prefix automatically without needing to maintain a list
        return additional_stem.endswith(input_stem)
    
    def _has_answer_tags(self, response: str) -> bool:
        """
        Check if response contains <answer> tags
        
        Args:
            response: Response text to check
            
        Returns:
            True if response contains <answer> tags, False otherwise
        """
        return '<answer>' in response and '</answer>' in response
    
    def _extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer content from <answer> tags
        
        Args:
            response: Response text to extract from
            
        Returns:
            Extracted answer text or None if no answer tags found
        """
        pattern = r'<answer>(.*?)</answer>'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_video_frames(self, video_path: str, num_frames: int = 4) -> List[str]:
        """Uniformly sample frames from a video file and save them as JPEG images.

        Args:
            video_path: Path to the input video file.
            num_frames: Number of frames to extract uniformly.

        Returns:
            List of paths to the extracted JPEG frame images.
        """
        try:
            import cv2
        except ImportError:
            logger.error("opencv-python is required for video frame extraction. pip install opencv-python")
            return []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            logger.error(f"Video has no frames: {video_path}")
            return []

        frame_interval = max(total_frames / num_frames, 1)
        temp_dir = Path("temp_veo_frames")
        temp_dir.mkdir(exist_ok=True)
        video_stem = Path(video_path).stem

        frame_paths: List[str] = []
        for i in range(num_frames):
            frame_idx = int(i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = temp_dir / f"{video_stem}_frame{i:02d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))

        cap.release()
        return frame_paths

    def _extract_frames_for_pi3(self, video_path: str, num_frames: int = 10) -> List[str]:
        """
        Extract frames from video for pi3 tool by uniformly sampling
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to uniformly sample from video
            
        Returns:
            List of paths to extracted frame images
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("cv2 is required for video frame extraction. Please install opencv-python.")
            return []
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / original_fps
        
        # Use the specified number of frames directly
        frame_interval = total_frames / num_frames
        
        frame_paths = []
        temp_dir = Path("temp_frames_pi3")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract video filename (without extension)
        video_filename = Path(video_path).stem
        
        # Extract frames evenly
        for i in range(num_frames):
            frame_idx = int(i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = temp_dir / f"{video_filename}_pi3_frame_{i}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
        
        cap.release()
        logger.info(f"Extracted {len(frame_paths)} frames from video for pi3 tool (duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, uniformly sampled {num_frames} frames)")
        return frame_paths
    
    def _cleanup_pi3_frames(self):
        """
        Clean up temporary frames extracted for pi3 tool
        """
        import os
        import shutil
        
        temp_dir = Path("temp_frames_pi3")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary pi3 frames")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary pi3 frames: {e}")
    
    def _get_naive_baseline_answer(
        self, 
        image_paths: List[str], 
        question: str,
        **model_kwargs
    ) -> str:
        """
        Get answer from naive baseline agent (no tools)
        
        Args:
            image_paths: List of image paths
            question: User's question
            **model_kwargs: Additional arguments for model inference
            
        Returns:
            Baseline answer string
        """
        # Create simple prompt without tool information
        naive_prompt = f"""You are a helpful AI assistant specialized in spatial intelligence and visual reasoning.

Please analyze the image(s) and answer the following question directly, using only what you can see in the image(s).

Question: {question}

Please provide your answer directly in <answer></answer> tags."""
        
        try:
            response = self._run_model_inference(
                image_paths,
                naive_prompt,
                **model_kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Failed to get naive baseline answer: {e}")
            return f"Error: {str(e)}"
    
    def _synthesize_with_baseline(
        self, 
        question: str,
        tool_based_answer: str,
        baseline_answer: str,
        image_paths: List[str],
        additional_images: List[str],
        **model_kwargs
    ) -> str:
        """
        Synthesize final answer by comparing tool-based and baseline answers
        
        Args:
            question: Original question
            tool_based_answer: Answer from tool-enhanced reasoning
            baseline_answer: Answer from naive baseline (no tools)
            image_paths: Original image paths
            additional_images: Additional images generated by tools
            **model_kwargs: Additional arguments for model inference
            
        Returns:
            Synthesized final answer
        """
        synthesis_prompt = f"""You are a helpful AI assistant tasked with providing the most accurate answer by synthesizing information from two different approaches.

Original Question: {question}

Approach 1 - Tool-Enhanced Analysis:
{tool_based_answer}

Approach 2 - Direct Visual Analysis (Baseline):
{baseline_answer}

=== Your Task ===

Compare the two approaches and provide the MOST ACCURATE answer to the original question.

Instructions:
1. Carefully analyze both answers
2. Consider which approach provides more reliable information:
   - Tool-enhanced analysis may have additional insights from specialized tools (depth, 3D, segmentation, etc.)
   - Direct visual analysis relies purely on what's visible in the image
3. Identify where they agree or disagree
4. When they disagree, determine which is more likely to be correct and why
5. Synthesize the information to provide the best possible answer

Important:
- If both approaches agree, confirm the answer with confidence
- If they disagree, explain your reasoning for choosing one over the other
- Use information from the tool-enhanced approach when it provides clear additional insights
- Use information from the baseline when the tool-enhanced approach seems uncertain or incorrect
- Your goal is to provide the MOST ACCURATE answer, not just combine both answers

Please provide your final synthesized answer in <answer></answer> tags."""
        
        try:
            # Use additional images if available, otherwise use original images
            valid_additional_images = self._sort_additional_images_by_input_order(image_paths, additional_images)
            final_images = valid_additional_images if valid_additional_images else image_paths
            
            response = self._run_model_inference(
                final_images,
                synthesis_prompt,
                **model_kwargs
            )
            
            logger.info(f"Synthesized answer: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            # Fallback to tool-based answer if synthesis fails
            return tool_based_answer
