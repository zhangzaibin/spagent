"""
R4 orchestrator: SkillAgent — the skills-mode counterpart of SPAgent.

Mirrors ``SPAgent.step()``'s surface (question + images in, StepResult out)
but drives tools through the skill layer with progressive disclosure:

- The system prompt carries only ``skills/INDEX.md`` (one line per skill).
- The model reads a skill's full SKILL.md on demand with
  ``<skill_read>skill_name</skill_read>`` (activated docs are re-injected
  into every later iteration's context).
- The ONE invocation format is a structured block::

      <skill_run>{"skill": "<name>", "args": {...}}</skill_run>

- Execution goes through the R3 backend in-process
  (``skills.run.run_skill``); every result is projected through
  ``core.render.render`` — the same projection the tool-call path uses.

SPAgent and the ``<tool_call>`` path are untouched; users construct
SkillAgent explicitly. Model wrappers are reused as-is (any ``core.model.
Model`` subclass works).

Known limitation vs SPAgent.step(): skills run sequentially (no thread
pool) and ``.mp4`` outputs are attached as paths without frame extraction.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

from core.memory import AgentMemory, StepResult  # noqa: E402
from core.model import Model  # noqa: E402
from core.render import render as render_tool_result  # noqa: E402
from core.tool_result import CATEGORY_CONTRACTS  # noqa: E402

from .registry import SkillRegistry  # noqa: E402
from .run import SkillRunError, run_skill  # noqa: E402

logger = logging.getLogger(__name__)

_SKILL_READ_RE = re.compile(r"<skill_read>\s*([\w.-]+)\s*</skill_read>")
_SKILL_RUN_RE = re.compile(r"<skill_run>\s*({.*?})\s*</skill_run>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


SKILL_SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about images by orchestrating specialized skills.

# Skills
You have the skills below. The index shows one line per skill; the full
documentation (arguments, output contract, runtime requirements) lives in
each skill's SKILL.md, which you can request at any time.

<skills_index>
{skills_index}
</skills_index>

# How to work
1. READ a skill before its first use. Output the skill name in a read tag:
<skill_read>skill_name</skill_read>
You will receive the full SKILL.md in the next turn. You may read several
skills at once (one tag each).
2. RUN a skill by outputting one JSON object per invocation inside run tags:
<skill_run>{{"skill": "<skill_name>", "args": {{"param1": "value1"}}}}</skill_run>
Arguments must follow the skill's argument table. You may issue multiple
<skill_run> blocks in one turn.
3. Each result comes back as text (and images, when the skill produced any).
You can then read or run more skills.
4. When you have enough evidence, give your final answer inside answer tags:
<answer>your final answer</answer>

Rules:
- Use only skill names that appear in the index; never invent skills or arguments.
- <skill_run> content must be valid JSON with exactly the keys "skill" and "args".
- Reading a skill costs an iteration, so batch your reads when possible."""


SKILL_CONTINUATION_HINT = """1. If you still need information, read further skills with <skill_read>name</skill_read> or run them with <skill_run>{"skill": ..., "args": {...}}</skill_run>.
2. If you already have enough evidence, respond with <answer>your final answer</answer>.
3. Do NOT repeat a skill run that already succeeded with the same arguments."""


_FINAL_ANSWER_HINT = ("You have used all skill iterations. Based on the "
                      "evidence above, respond now with your final answer "
                      "inside <answer></answer> tags.")


class SkillAgent:
    """Skills-mode orchestrator (opt-in alternative to SPAgent).

    Args:
        model:          Any ``core.model.Model`` wrapper (QwenVLLMModel, ...).
        skills_dir:     Generated skills directory (default: ``<repo>/skills``).
        use_mock:       Build tools with mock backends (no GPU/server).
        tool_overrides: Per-catalog-key constructor overrides, same shape as
                        ``tools.catalog.build_tools(overrides=...)`` — e.g.
                        ``{"sana": {"server_url": "http://127.0.0.1:30010"}}``.
        render_config:  Default projection config for ``core.render.render``.
        check_server:   Health-check server-backed tools before each build.
    """

    def __init__(
        self,
        model: Model,
        skills_dir: Optional[Union[str, Path]] = None,
        use_mock: bool = False,
        tool_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        render_config: Optional[Dict[str, Any]] = None,
        check_server: bool = True,
    ):
        self.model = model
        self.registry = SkillRegistry(Path(skills_dir) if skills_dir else None)
        self.use_mock = use_mock
        self.tool_overrides = tool_overrides or {}
        self.render_config = render_config
        self.check_server = check_server
        self._read_skills: set = set()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_skill_reads(response: str) -> List[str]:
        return _SKILL_READ_RE.findall(response)

    @staticmethod
    def _parse_skill_runs(response: str) -> List[Dict[str, Any]]:
        """Parse ``<skill_run>{"skill":..., "args":...}</skill_run>`` blocks."""
        runs: List[Dict[str, Any]] = []
        for raw in _SKILL_RUN_RE.findall(response):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning("Unparseable skill_run block: %s (%s)", raw, e)
                runs.append({"skill": None, "args": {},
                             "parse_error": f"invalid JSON: {e}", "raw": raw})
                continue
            if not isinstance(obj, dict) or "skill" not in obj:
                logger.warning("skill_run missing 'skill' key: %s", raw)
                runs.append({"skill": None, "args": {},
                             "parse_error": "missing 'skill' key", "raw": raw})
                continue
            args = obj.get("args", {})
            if not isinstance(args, dict):
                runs.append({"skill": obj["skill"], "args": {},
                             "parse_error": "'args' must be a JSON object",
                             "raw": raw})
                continue
            runs.append({"skill": obj["skill"], "args": args})
        return runs

    @staticmethod
    def _has_answer(response: str) -> bool:
        return bool(_ANSWER_RE.search(response or ""))

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_run(self, skill_name: str, args: Dict[str, Any]):
        """Run one skill through the R3 backend; never raises."""
        skill = self.registry.get(skill_name)
        catalog_key = skill.catalog_key if skill else skill_name
        try:
            result, _ = run_skill(
                skill_name,
                args,
                use_mock=self.use_mock,
                check_server=self.check_server,
                extra_overrides=self.tool_overrides.get(catalog_key),
            )
            return result
        except SkillRunError as e:
            payload = dict(e.payload)
            payload.setdefault("description", payload.get("error", ""))
            return payload
        except Exception as e:  # defensive: an orchestrator must not die
            logger.exception("skill run %s failed unexpectedly", skill_name)
            return {"success": False,
                    "description": f"skill {skill_name} failed",
                    "error": f"{type(e).__name__}: {e}"}

    def _run_model_inference(self, images: List[str], prompt: str,
                             **model_kwargs) -> str:
        if not images:
            return self.model.text_only_inference(prompt, **model_kwargs)
        if len(images) == 1:
            return self.model.single_image_inference(images[0], prompt,
                                                     **model_kwargs)
        return self.model.multiple_images_inference(images, prompt,
                                                    **model_kwargs)

    def _continuation_hint(self) -> str:
        """Base hint + full docs of every skill activated so far."""
        if not self._read_skills:
            return SKILL_CONTINUATION_HINT
        blocks = []
        for name in sorted(self._read_skills):
            skill = self.registry.get(name)
            if skill:
                blocks.append(f"## Activated skill: {name}\n\n{skill.body}")
        docs = "\n\n".join(blocks)
        return (f"{SKILL_CONTINUATION_HINT}\n\n"
                f"# Activated skill documentation\n\n{docs}")

    def build_system_prompt(self) -> str:
        return SKILL_SYSTEM_PROMPT_TEMPLATE.format(
            skills_index=self.registry.index_text.strip())

    # ------------------------------------------------------------------
    # Main entry point — mirrors SPAgent.step()
    # ------------------------------------------------------------------

    def step(
        self,
        content: str,
        images: Optional[Union[str, List[str]]] = None,
        memory: Optional[AgentMemory] = None,
        system_prompt: Optional[str] = None,
        max_tool_iterations: int = 3,
        max_images_in_context: int = 6,
        render_config: Optional[Dict[str, Any]] = None,
        **model_kwargs,
    ) -> StepResult:
        """One skills-mode step: perceive → (read/run skills)* → answer.

        Same surface as ``SPAgent.step``: returns a ``StepResult`` whose
        memory can be threaded into later calls for multi-turn use.
        """
        if memory is None:
            memory = AgentMemory()
        self._read_skills = set()
        effective_render_config = (render_config if render_config is not None
                                   else self.render_config)

        if images is None:
            image_paths: List[str] = []
        elif isinstance(images, str):
            image_paths = [images]
        else:
            image_paths = list(images)
        for path in image_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")

        system_prompt = system_prompt or self.build_system_prompt()
        user_prompt = f"Question: {content}"
        memory.add_system(system_prompt)
        memory.add_user_turn(content, images=image_paths)

        all_tool_calls: List[Dict[str, Any]] = []
        all_tool_results: Dict[str, Any] = {}
        all_successful: List[str] = []
        all_additional_images: List[str] = []
        current_images = list(image_paths)
        iteration = 0

        while iteration < max_tool_iterations:
            iteration += 1
            if iteration == 1:
                prompt = system_prompt + "\n\n" + user_prompt
            else:
                prompt = memory.build_prompt_context(
                    current_iteration=iteration,
                    max_iterations=max_tool_iterations,
                    continuation_hint=self._continuation_hint(),
                )

            response = self._run_model_inference(current_images, prompt,
                                                 **model_kwargs)
            memory.add_assistant_turn(response, metadata={"iteration": iteration})

            reads = self._parse_skill_reads(response)
            runs = self._parse_skill_runs(response)
            has_answer = self._has_answer(response)

            # -- skill reads (progressive disclosure phase 2) --------------
            for name in reads:
                skill = self.registry.get(name)
                if skill is None:
                    memory.add_tool_result(
                        tool_name=f"skill_read:{name}",
                        result={"success": False,
                                "error": f"unknown skill {name!r}",
                                "description": f"No skill named {name!r} — "
                                               "use a name from the index."},
                        iteration=iteration,
                    )
                    continue
                self._read_skills.add(name)
                memory.add_tool_result(
                    tool_name=f"skill_read:{name}",
                    result={"success": True,
                            "description": f"SKILL.md for {name} activated "
                                           "(full documentation follows in "
                                           "the next context)."},
                    iteration=iteration,
                )

            # -- skill runs -------------------------------------------------
            for run in runs:
                skill_name = run.get("skill") or "invalid_skill_run"
                if run.get("parse_error"):
                    result: Dict[str, Any] = {
                        "success": False,
                        "error": run["parse_error"],
                        "description": f"skill_run rejected: {run['parse_error']}",
                    }
                else:
                    if skill_name not in self._read_skills:
                        logger.info("skill %s run before being read "
                                    "(allowed, but the prompt asks to read "
                                    "first)", skill_name)
                    result = self._execute_run(skill_name, run["args"])

                rendered = render_tool_result(
                    result, config=effective_render_config,
                    tool_name=skill_name,
                )
                is_standardized = result.get("category") in CATEGORY_CONTRACTS

                result_key = skill_name
                n = 2
                while f"{result_key}_iter{iteration}" in all_tool_results:
                    result_key = f"{skill_name}_{n}"
                    n += 1

                output_images = list(rendered.images)
                if result.get("success"):
                    all_successful.append(f"{result_key}_iter{iteration}")
                    all_additional_images.extend(output_images)

                memory.add_tool_call(tool_name=skill_name,
                                     arguments=run.get("args", {}),
                                     iteration=iteration)
                memory.add_tool_result(
                    tool_name=skill_name,
                    result=result,
                    output_images=output_images,
                    iteration=iteration,
                    rendered_text=rendered.text if is_standardized else None,
                )
                all_tool_calls.append({"name": skill_name,
                                       "arguments": run.get("args", {})})
                all_tool_results[f"{result_key}_iter{iteration}"] = result

            # -- loop control (mirrors SPAgent) -----------------------------
            if not reads and not runs:
                if has_answer or iteration == max_tool_iterations:
                    break
                continue

            if all_additional_images:
                budget = max(1, max_images_in_context - len(image_paths))
                current_images = image_paths + [
                    p for p in all_additional_images if Path(p).exists()
                ][-budget:]

        # -- post-loop: force a final answer when tags are missing ----------
        last_response = memory.get_last_assistant_text() or ""
        if not self._has_answer(last_response):
            final_prompt = memory.build_prompt_context(
                current_iteration=iteration,
                max_iterations=max_tool_iterations,
                continuation_hint=_FINAL_ANSWER_HINT,
            )
            final_response = self._run_model_inference(
                current_images, final_prompt, **model_kwargs)
            memory.add_assistant_turn(final_response,
                                      metadata={"type": "final_synthesis"})
        else:
            final_response = last_response

        return StepResult(
            answer=final_response,
            memory=memory,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            used_tools=all_successful,
            additional_images=all_additional_images,
            iterations=iteration,
            prompts={
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "workflow": "skills",
            },
        )
