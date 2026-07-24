# Skills Mode

A second, **opt-in** way to drive SPAgent's 24 expert tools. The existing
tool-call path (`SPAgent` parsing `<tool_call>` blocks) is untouched; skills
mode is a parallel door into the *same* tools, returning the *same*
`ToolResult` envelopes projected by the *same* `core.render.render()`.

```
  tool-call path (unchanged)          skills path (this doc)
  ─────────────────────────           ──────────────────────
  SPAgent.step()                      SkillAgent.step()
    └─ <tool_call> JSON                 └─ <skill_run> JSON
       └─ ToolRegistry                     └─ spagent.skills.run backend
          └─ Tool.call() ──────┬────────────── Tool.call()
                               ▼
                     ToolResult (core.tool_result)
                               ▼
                     core.render.render() → text + images
```

## Why catalog-generated

PR #157 (the first skills attempt) hand-wrote 8 skill docs; they drifted
from the code immediately and the catalog has since grown to 24 tools.
Here every skill doc is **generated** from the single source of truth:

- name / description / argument schema → the live `Tool` instances built
  via `spagent/tools/catalog.py`
- output contract → `core.tool_result.CATEGORY_CONTRACTS`
- deployment facts (runtime class, launch command, checkpoints, API keys)
  → `spagent/skills/runtime.py`
- optional judgment ("when to use", recommended pi3 angles, …) →
  curated overlays in `spagent/skills/overlays/<key>.md`, harvested from
  PR #157 and appended verbatim under *Guidance (curated)*

Generation is deterministic and idempotent; `sync` repairs and reports
drift. Facts regenerate; only the overlays are edited by hand.

## Layers

### R1 — packaging (generator)

```bash
python -m spagent.skills.generate            # write skills/<tool>/SKILL.md + skills/INDEX.md
python -m spagent.skills.generate --check    # report drift, exit 1 if stale
```

Each `skills/<tool_name>/SKILL.md` has YAML frontmatter (name, one-line
description, category, group, runtime class, catalog key) and body
sections: *When to use* (the tool's full description), *Arguments* (table
from the JSON schema), *Output contract* (required carrier groups +
optional fields from the category contract), *Invocation* (a runnable
`spagent.skills.run` command), *Runtime requirements* (server? port?
launch command? checkpoint? mock?), and optionally *Guidance (curated)*.

`supervision_tool` resolves its category per call (`task` argument); its
SKILL.md documents both the detection and segmentation contracts.

### R2 — registry / index

```bash
python -m spagent.skills list          # INDEX view (one line per skill)
python -m spagent.skills show <name>   # print one full SKILL.md
python -m spagent.skills sync          # regenerate from catalog + report drift
```

`skills/INDEX.md` is one line per skill: name — one-liner — category —
runtime class (`local` / `server` / `cloud-API` / `mock-only`). Only the
INDEX is loaded into the orchestrator's context; full SKILL.md files are
read on demand (progressive disclosure — 24 full docs never get dumped
into the prompt).

### R3 — execution backend

```bash
python -m spagent.skills.run zoom_object_tool \
    --args '{"image_path": "assets/dog.jpeg", "text_prompt": "dog"}' --use-mock
```

Builds the tool through `tools.catalog.build_tools` (catalog defaults +
CLI overrides `--server-url` / `--output-dir`), calls it, and prints the
ToolResult as **single-line JSON on stdout** (tool prints are diverted to
stderr). Values that are not JSON-safe — e.g. the depth tool's numpy
`depth_data` — are summarized as `"<array shape=(H, W) dtype=...>"`.

Server-backed tools are health-checked (`GET /health`) before the call;
on failure the CLI prints a structured error that includes the launch
command from the skill's runtime spec.

Exit codes: `0` success · `1` tool returned `success: false` (or raised)
· `2` usage/config error · `3` server unreachable.

### R4 — orchestrator (`SkillAgent`)

```python
from spagent.skills import SkillAgent
from spagent.models import QwenVLLMModel   # any core.model.Model wrapper

agent = SkillAgent(model=QwenVLLMModel(...), use_mock=False)
result = agent.step("How many dogs are in the image?", images="assets/dog.jpeg")
print(result.answer)
```

`SkillAgent.step()` mirrors `SPAgent.step()`'s surface (question + images
in, `StepResult` out; pass `memory=` for multi-turn). Its system prompt
contains only the skills INDEX plus two instructions:

1. **Read before first use** — `<skill_read>skill_name</skill_read>`
   activates a skill; its full SKILL.md is injected into every subsequent
   iteration's context.
2. **Invoke via ONE structured format**:

   ```
   <skill_run>{"skill": "<skill_name>", "args": {...}}</skill_run>
   ```

Each run executes through the R3 backend in-process, the raw ToolResult is
projected with `core.render.render()` (same config precedence: per-tool >
per-category > preset), and the rendered text + images are appended to the
shared `AgentMemory`. `<answer>...</answer>` ends the loop; if the model
never produces answer tags, a final synthesis turn forces one.

Constructor knobs: `skills_dir`, `use_mock`, `render_config`,
`check_server`, and `tool_overrides` (per catalog key, e.g.
`{"sana": {"server_url": "http://127.0.0.1:30010"}}`).

## Relation to the tool-call path

| | tool-call path | skills path |
|---|---|---|
| entry | `SPAgent.step()` | `SkillAgent.step()` |
| tool discovery | all schemas in system prompt | INDEX one-liners; SKILL.md on demand |
| wire format | `<tool_call>{name, arguments}</tool_call>` | `<skill_run>{skill, args}</skill_run>` |
| execution | `ToolRegistry` in-process | `spagent.skills.run` backend (also a standalone CLI) |
| result | `ToolResult` → `render()` | same `ToolResult` → same `render()` |
| opt-in | default | explicit `SkillAgent` construction |

Nothing in `spagent/core/` or `spagent/tools/` changed; skills mode is
additive (`spagent/skills/`, `skills/`, `docs/Skills/`, `test/test_skills*`).

## Tests

```bash
python3 test/test_skills.py                 # zero GPU: generator idempotence,
                                            # INDEX↔catalog, backend mock runs,
                                            # SkillAgent with a scripted model
CUDA_VISIBLE_DEVICES="" python test/test_skills_real_yolo26.py
                                            # real local yolo26: CLI + agent turn
```

## Limitations

- `SkillAgent` runs skills sequentially (no thread pool) and attaches
  `.mp4` outputs as paths without frame extraction — use SPAgent when
  those matter.
- Reading a skill consumes a loop iteration (the model is told to batch
  reads); budget `max_tool_iterations` accordingly.
- The runtime classes in `skills/runtime.py` describe what each backend
  needs, not what is currently running; `cloud-API` tools need provider
  keys, `mock-only` (wilddet3d) needs an external repo checkout to run
  for real.
- Skill folders for tools removed from the catalog are reported by
  `sync` as orphans but not auto-deleted.
