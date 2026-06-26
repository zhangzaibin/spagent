"""
AgentMemory — multimodal, typed memory for SPAgent.

Each interaction (user turn, assistant turn, tool call, tool result)
is recorded as a MemoryEntry. The memory can be passed across multiple
`step()` calls to maintain multi-turn conversation state, serialized
to disk for persistence, and queried by role or entry type.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# MemoryEntry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """
    A single typed event in agent memory.

    Attributes:
        id:          Unique entry identifier (uuid4 string).
        timestamp:   Unix timestamp of when the entry was created.
        role:        Who produced this entry: "system" | "user" | "assistant" | "tool".
        entry_type:  Nature of the content: "text" | "image" | "tool_call" |
                     "tool_result" | "multimodal".
        text:        Textual content (prompt, response, description, …).
        images:      Ordered list of image file paths associated with this entry.
                     VLM backends always consume images as file paths, so memory
                     stores them the same way rather than embedding raw bytes.
        metadata:    Arbitrary key/value bag — tool name, iteration number,
                     success flag, viewing angles, etc.
    """
    id: str
    timestamp: float
    role: str
    entry_type: str
    text: Optional[str]
    images: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "role": self.role,
            "entry_type": self.entry_type,
            "text": self.text,
            "images": self.images,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            id=d["id"],
            timestamp=d["timestamp"],
            role=d["role"],
            entry_type=d["entry_type"],
            text=d.get("text"),
            images=d.get("images", []),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """
    Structured return value from SPAgent.step().

    Attributes:
        answer:           Final answer text (may contain <answer> tags).
        memory:           The AgentMemory instance after this step (includes
                          all new entries appended during the step).
        tool_calls:       Flat list of all tool-call dicts made this step.
        tool_results:     Mapping of ``{tool_name_iterN: result_dict}``.
        used_tools:       Names of tools that succeeded (with iteration suffix).
        additional_images:All image paths produced by tools this step.
        iterations:       Number of tool-call iterations performed.
        prompts:          Dict of key prompts (system, user, workflow label).
    """
    answer: str
    memory: "AgentMemory"
    tool_calls: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    used_tools: List[str]
    additional_images: List[str]
    iterations: int
    prompts: Dict[str, str]


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

class AgentMemory:
    """
    Multimodal memory for SPAgent.

    Stores a chronological list of MemoryEntry objects representing every
    event in an agent session: system messages, user turns, assistant turns,
    tool calls, and tool results (with their output images).

    Usage — stateless (one-shot):
        result = agent.step("What is the depth of the scene?", images="photo.jpg")

    Usage — stateful (multi-turn):
        memory = AgentMemory()
        r1 = agent.step("Describe the scene.", images="photo.jpg", memory=memory)
        r2 = agent.step("Now estimate depth.", memory=memory)   # memory carries context
    """

    def __init__(self, max_entries: Optional[int] = None):
        """
        Args:
            max_entries: Optional cap on the number of entries kept in memory.
                         When the limit is reached, the oldest non-system entries
                         are pruned to make room.  None means unlimited.
        """
        self._entries: List[MemoryEntry] = []
        self.max_entries = max_entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_entry(
        self,
        role: str,
        entry_type: str,
        text: Optional[str] = None,
        images: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            role=role,
            entry_type=entry_type,
            text=text,
            images=images or [],
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._maybe_prune()
        return entry

    def _maybe_prune(self):
        """Remove oldest non-system entries when max_entries is exceeded."""
        if self.max_entries is None:
            return
        while len(self._entries) > self.max_entries:
            for i, e in enumerate(self._entries):
                if e.role != "system":
                    self._entries.pop(i)
                    break
            else:
                # All remaining entries are system messages — nothing to prune.
                break

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def add_system(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> MemoryEntry:
        """Record the system prompt."""
        return self._make_entry("system", "text", text=text, metadata=metadata)

    def add_user_turn(
        self,
        text: str,
        images: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record a user message (text + optional images)."""
        entry_type = "multimodal" if images else "text"
        return self._make_entry("user", entry_type, text=text, images=images, metadata=metadata)

    def add_assistant_turn(
        self,
        text: str,
        images: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record an assistant (model) response."""
        entry_type = "multimodal" if images else "text"
        return self._make_entry("assistant", entry_type, text=text, images=images, metadata=metadata)

    def add_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Record a tool invocation."""
        meta = {"tool_name": tool_name, "arguments": arguments}
        if iteration is not None:
            meta["iteration"] = iteration
        if metadata:
            meta.update(metadata)
        text = f"Tool call: {tool_name}({json.dumps(arguments)})"
        return self._make_entry("tool", "tool_call", text=text, metadata=meta)

    def add_tool_result(
        self,
        tool_name: str,
        result: Dict[str, Any],
        output_images: Optional[List[str]] = None,
        iteration: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Record the result returned by a tool, including any output images
        (depth maps, segmentation overlays, point-cloud renders, etc.).
        """
        meta: Dict[str, Any] = {
            "tool_name": tool_name,
            "success": result.get("success", False),
        }
        if iteration is not None:
            meta["iteration"] = iteration
        if "azimuth_angle" in result:
            meta["azimuth_angle"] = result["azimuth_angle"]
        if "elevation_angle" in result:
            meta["elevation_angle"] = result["elevation_angle"]
        if "error" in result:
            meta["error"] = result["error"]
        if metadata:
            meta.update(metadata)

        description = result.get("description") or ""
        text = f"Tool result [{tool_name}]: {description}" if description else f"Tool result [{tool_name}]"
        return self._make_entry(
            "tool", "tool_result",
            text=text,
            images=output_images or [],
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[MemoryEntry]:
        """Read-only view of all memory entries in chronological order."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get_recent(self, n: int) -> List[MemoryEntry]:
        """Return the *n* most recent entries."""
        return self._entries[-n:]

    def get_by_role(self, role: str) -> List[MemoryEntry]:
        """Return all entries with the given role."""
        return [e for e in self._entries if e.role == role]

    def get_by_type(self, entry_type: str) -> List[MemoryEntry]:
        """Return all entries with the given entry_type."""
        return [e for e in self._entries if e.entry_type == entry_type]

    def get_tool_results(self) -> List[MemoryEntry]:
        """Return all tool-result entries."""
        return self.get_by_type("tool_result")

    def get_tool_calls(self) -> List[MemoryEntry]:
        """Return all tool-call entries."""
        return self.get_by_type("tool_call")

    def get_all_images(self) -> List[str]:
        """
        Return all image file paths recorded across every entry,
        in the order they were produced.
        """
        seen: set = set()
        result: List[str] = []
        for entry in self._entries:
            for img in entry.images:
                if img not in seen:
                    seen.add(img)
                    result.append(img)
        return result

    def get_last_assistant_text(self) -> Optional[str]:
        """Return the text of the most recent assistant entry, or None."""
        for entry in reversed(self._entries):
            if entry.role == "assistant":
                return entry.text
        return None

    def get_first_assistant_text(self) -> Optional[str]:
        """Return the text of the first assistant entry, or None."""
        for entry in self._entries:
            if entry.role == "assistant":
                return entry.text
        return None

    # ------------------------------------------------------------------
    # Context builder (replaces _create_continuation_prompt)
    # ------------------------------------------------------------------

    def build_prompt_context(
        self,
        current_iteration: int,
        max_iterations: int,
        continuation_hint: str = "",
    ) -> str:
        """
        Serialize memory into a human-readable continuation prompt.

        This replaces ``SPAgent._create_continuation_prompt`` so that the
        prompt construction stays co-located with the memory data.

        Args:
            current_iteration: The iteration that just completed (1-based).
            max_iterations:    The total allowed iterations.
            continuation_hint: Workflow-specific guidance injected at the end.

        Returns:
            A formatted string ready to be sent to the model.
        """
        # Recover original question from the first user entry
        user_entries = self.get_by_role("user")
        question = user_entries[0].text if user_entries else ""

        # Recover last assistant response
        last_response = self.get_last_assistant_text() or ""

        # Build tool summary from tool_result entries
        tool_summary_lines: List[str] = []
        for entry in self.get_tool_results():
            tool_name = entry.metadata.get("tool_name", "unknown_tool")
            if entry.metadata.get("success"):
                tool_summary_lines.append(f"- {tool_name}: Successfully executed")
                azim = entry.metadata.get("azimuth_angle")
                elev = entry.metadata.get("elevation_angle")
                if azim is not None and elev is not None:
                    tool_summary_lines.append(
                        f"  └─ Viewing angle: azimuth={azim}°, elevation={elev}°"
                    )
                if entry.text:
                    # Strip "Tool result [name]: " prefix for readability
                    desc = entry.text.split(": ", 1)[-1] if ": " in entry.text else entry.text
                    tool_summary_lines.append(f"  Description: {desc}")
            else:
                err = entry.metadata.get("error", "Unknown error")
                tool_summary_lines.append(f"- {tool_name}: Failed - {err}")

        tool_summary_text = "\n".join(tool_summary_lines) if tool_summary_lines else "None yet"

        # Collect original images from user entries
        original_image_lines: List[str] = []
        for entry in user_entries:
            for img in entry.images:
                original_image_lines.append(f"- {img}")
        original_images_info = "\n".join(original_image_lines) if original_image_lines else "None"

        # Collect additional images from tool results
        additional_image_lines: List[str] = []
        for entry in self.get_tool_results():
            for img in entry.images:
                additional_image_lines.append(f"- {img}")
        additional_images_info = (
            "\n".join(additional_image_lines) if additional_image_lines else "None yet"
        )

        remaining = max_iterations - current_iteration

        prompt = f"""=== Multi-Step Analysis: Iteration {current_iteration}/{max_iterations} ===

Original Question: {question}

Your Previous Response: 
{last_response}

Tool Execution Summary:
{tool_summary_text}

Original Images:
{original_images_info}

Generated Images Available for Analysis:
{additional_images_info}

=== Next Steps ===

You have {remaining} more iteration(s) available.

{continuation_hint}

Please continue:"""
        return prompt

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Serialize memory to a JSON file.

        Images are stored as their file paths (not raw bytes).
        The file can be reloaded with ``AgentMemory.load()``.
        """
        data = {
            "max_entries": self.max_entries,
            "entries": [e.to_dict() for e in self._entries],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AgentMemory":
        """
        Deserialize memory from a JSON file previously written by ``save()``.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        memory = cls(max_entries=data.get("max_entries"))
        for entry_dict in data.get("entries", []):
            memory._entries.append(MemoryEntry.from_dict(entry_dict))
        return memory

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Wipe all entries."""
        self._entries.clear()

    def __repr__(self) -> str:
        return (
            f"AgentMemory(entries={len(self._entries)}, "
            f"roles={[e.role for e in self._entries]})"
        )
