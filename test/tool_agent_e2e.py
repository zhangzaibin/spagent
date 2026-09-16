"""Full-agent e2e test for a tool (with-compute, lane 3).

Runs a real SPAgent.step() episode where a live VLM decides to invoke the
tool under test against its live backend — verifying the whole chain:
system-prompt tool schema -> model tool call -> backend -> ToolResult ->
render() projection -> final answer.

Providers (API keys via environment variables ONLY — never CLI args):

    --provider gemini      $GEMINI_API_KEY      (Gemini Developer API)
    --provider openai      $OPENAI_API_KEY
    --provider anthropic   $ANTHROPIC_API_KEY
    --provider local       OpenAI-compatible endpoint (vLLM/llama.cpp/...):
                           --local-base-url http://host:port/v1  [$LOCAL_BASE_URL]
                           optional $LOCAL_API_KEY (defaults to "EMPTY")

Usage:

    # backend up first (server-backed tools), then:
    python test/tool_agent_e2e.py --tool detection --provider gemini \
        --url detection=http://localhost:20122

    python test/tool_agent_e2e.py --tool yolo26 --provider anthropic
    python test/tool_agent_e2e.py --tool depth --provider local \
        --local-base-url http://localhost:8000/v1 --model Qwen2.5-VL-7B

Pass criteria per episode: the agent loop completes, the tool under test was
actually invoked, at least one invocation succeeded, and a non-empty final
answer came back. Markdown report to $GITHUB_STEP_SUMMARY / --markdown-out;
exit 1 on failure.
"""
import argparse
import base64
import logging
import os
import re
import sys
from pathlib import Path

logging.disable(logging.WARNING)
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "spagent"))
sys.path.insert(0, str(REPO / "test"))
os.chdir(str(REPO))

from core import SPAgent                                       # noqa: E402
from core.model import Model                                   # noqa: E402
from tools.catalog import TOOL_CATALOG, DEFAULT_SERVER_URLS, build_tools  # noqa: E402
from tool_real_smoke import _probe                              # noqa: E402

IMG = "assets/dog.jpeg"
PASS, FAIL = "✅", "❌"

DEFAULT_MODELS = {
    "gemini": "gemini-flash-latest",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5",
    "local": None,  # must be passed via --model
}

# Category-appropriate episode prompts. Deliberately task-oriented with a
# gentle nudge to verify with a tool — this lane smokes the agent<->tool
# wiring, it is not a tool-selection benchmark.
EPISODE_PROMPTS = {
    "detection": "What objects are in this image and where are they? Verify with the available tool before answering.",
    "segmentation": "Segment the main subject of this image and tell me roughly how much of the frame it covers. Use the available tool.",
    "depth": "Which parts of this image are closest to the camera? Verify with the available tool.",
    "3d_reconstruction": "Reconstruct this scene in 3D from a different viewing angle and describe what the reconstruction shows. Use the available tool.",
    "orientation": "Which direction is the main subject of this image facing? Verify with the available tool.",
    "point_grounding": "Point to the main subject in this image using the available tool and report the location.",
    "optical_flow": "Estimate the motion between the two provided images using the available tool and summarize it.",
    "ocr": "Read and transcribe any text in this image using the available tool.",
    "image_generation": "Generate an image of a dog running on a beach using the available tool and report where it was saved.",
}

TOOL_EPISODE_PROMPTS = {
    "qwen_image_edit": (
        "Edit this image so the dog wears blue sunglasses. Preserve the dog, "
        "pose, and background, and use the available tool."
    ),
    "crop": "Crop the main subject tightly from this image using the available tool and report where the crop was saved.",
}


# ---------------------------------------------------------------------------
# Provider adapters (all implement the SPAgent Model interface)
# ---------------------------------------------------------------------------

class GeminiModel(Model):
    def __init__(self, model_name, max_tokens=8192):
        super().__init__(model_name, temperature=0.0, max_tokens=max_tokens)
        from google import genai
        self._genai_types = __import__("google.genai.types", fromlist=["types"])
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _img(self, path):
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            return self._genai_types.Part.from_bytes(data=f.read(), mime_type=mime)

    def _gen(self, parts):
        cfg = self._genai_types.GenerateContentConfig(
            temperature=0.0, max_output_tokens=self.max_tokens)
        resp = self._client.models.generate_content(
            model=self.model_name, contents=parts, config=cfg)
        return resp.text or ""

    def single_image_inference(self, image_path, prompt, temperature=None, max_tokens=None):
        return self._gen([self._img(image_path), prompt])

    def multiple_images_inference(self, image_paths, prompt, temperature=None, max_tokens=None):
        return self._gen([self._img(p) for p in image_paths] + [prompt])

    def text_only_inference(self, prompt, temperature=None, max_tokens=None):
        return self._gen([prompt])


class OpenAICompatModel(Model):
    """OpenAI API, or any OpenAI-compatible local endpoint (vLLM, llama.cpp)."""

    def __init__(self, model_name, base_url=None, api_key_env="OPENAI_API_KEY",
                 max_tokens=8192):
        super().__init__(model_name, temperature=0.0, max_tokens=max_tokens)
        from openai import OpenAI
        key = os.environ.get(api_key_env) or ("EMPTY" if base_url else None)
        if key is None:
            raise EnvironmentError(f"{api_key_env} is not set")
        self._client = OpenAI(api_key=key, base_url=base_url)

    @staticmethod
    def _img(path):
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

    def _gen(self, content):
        resp = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content or ""

    def single_image_inference(self, image_path, prompt, temperature=None, max_tokens=None):
        return self._gen([self._img(image_path), {"type": "text", "text": prompt}])

    def multiple_images_inference(self, image_paths, prompt, temperature=None, max_tokens=None):
        return self._gen([self._img(p) for p in image_paths]
                         + [{"type": "text", "text": prompt}])

    def text_only_inference(self, prompt, temperature=None, max_tokens=None):
        return self._gen(prompt)


class AnthropicModel(Model):
    def __init__(self, model_name, max_tokens=8192):
        super().__init__(model_name, temperature=0.0, max_tokens=max_tokens)
        import anthropic
        self._client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    @staticmethod
    def _img(path):
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}

    def _gen(self, content):
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")

    def single_image_inference(self, image_path, prompt, temperature=None, max_tokens=None):
        return self._gen([self._img(image_path), {"type": "text", "text": prompt}])

    def multiple_images_inference(self, image_paths, prompt, temperature=None, max_tokens=None):
        return self._gen([self._img(p) for p in image_paths]
                         + [{"type": "text", "text": prompt}])

    def text_only_inference(self, prompt, temperature=None, max_tokens=None):
        return self._gen(prompt)


def build_model(provider, model_name, local_base_url):
    if provider == "gemini":
        return GeminiModel(model_name)
    if provider == "openai":
        return OpenAICompatModel(model_name)
    if provider == "anthropic":
        return AnthropicModel(model_name)
    if provider == "local":
        base = local_base_url or os.environ.get("LOCAL_BASE_URL")
        if not base:
            raise EnvironmentError("local provider needs --local-base-url or $LOCAL_BASE_URL")
        return OpenAICompatModel(model_name, base_url=base, api_key_env="LOCAL_API_KEY")
    raise ValueError(provider)


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def run_episode(entry, model, url, prompt):
    """One SPAgent episode with only the tool under test. Returns (ok, notes)."""
    notes = []

    effective_url = url or DEFAULT_SERVER_URLS.get(entry.key)
    if effective_url and not _probe(effective_url):
        return False, [f"backend unreachable: {effective_url} — start the server first"]

    tools, errs = build_tools([entry.key], use_mock=False,
                              overrides={entry.key: {"server_url": url}} if url else None)
    if not tools:
        return False, [f"tool build failed: {'; '.join(errs)}"[:150]]
    tool = tools[0]

    prompt = prompt or TOOL_EPISODE_PROMPTS.get(entry.key) or EPISODE_PROMPTS.get(entry.category)
    if not prompt:
        return False, [f"no default episode prompt for category {entry.category!r} "
                       "(e.g. video_generation is paid) — pass --prompt to run anyway"]

    images = IMG
    if entry.category == "optical_flow":
        images = [IMG, IMG]
    if entry.category == "image_generation":
        images = None

    agent = SPAgent(model=model, tools=[tool])
    try:
        res = agent.step(content=prompt, images=images)
    except Exception as e:
        return False, [f"agent loop raised {type(e).__name__}: {e}"[:200]]

    # 1. tool under test was actually invoked
    tool_calls = [c for c in (res.tool_calls or []) if c.get("name") == tool.name]
    if not tool_calls:
        return False, [f"the model never called {tool.name} "
                       f"(used: {res.used_tools or 'nothing'}) — check the tool "
                       "description; the VLM chooses tools based on it"]
    notes.append(f"{len(tool_calls)} call(s) to {tool.name} over {res.iterations} iteration(s)")

    # 2. at least one invocation succeeded
    successes = [k for k, v in (res.tool_results or {}).items()
                 if isinstance(v, dict) and v.get("success")]
    if not successes:
        errs = {k: str((v or {}).get("error"))[:80]
                for k, v in (res.tool_results or {}).items() if isinstance(v, dict)}
        return False, [f"all tool invocations failed: {errs}"]
    notes.append(f"tool succeeded: {successes}")

    # 3. non-empty final answer
    m = re.search(r"<answer>(.*?)</answer>", res.answer or "", re.S)
    answer = (m.group(1) if m else (res.answer or "")).strip()
    if not answer:
        return False, notes + ["empty final answer — model produced no conclusion "
                               "after tool use (often max_tokens too small)"]
    notes.append(f"answer: {answer[:120]!r}")
    return True, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tool", required=True, help="comma-separated catalog keys")
    ap.add_argument("--provider", required=True,
                    choices=["gemini", "openai", "anthropic", "local"])
    ap.add_argument("--model", help="model name (default per provider; required for local)")
    ap.add_argument("--url", action="append", default=[],
                    help="key=http://host:port backend override (repeatable)")
    ap.add_argument("--local-base-url", help="OpenAI-compatible base URL for --provider local")
    ap.add_argument("--prompt", help="override the category default episode prompt")
    ap.add_argument("--markdown-out")
    args = ap.parse_args()

    model_name = args.model or DEFAULT_MODELS[args.provider]
    if not model_name:
        print("❌ --provider local requires --model")
        return 1
    try:
        model = build_model(args.provider, model_name, args.local_base_url)
    except Exception as e:
        print(f"❌ model init failed: {type(e).__name__}: {e}")
        return 1

    urls = dict(u.split("=", 1) for u in args.url)
    entries = {e.key: e for e in TOOL_CATALOG}
    keys = [k.strip() for k in args.tool.split(",") if k.strip()]
    unknown = [k for k in keys if k not in entries]
    if unknown:
        print(f"❌ unknown tool key(s): {unknown}")
        return 1

    md = ["# Tool full-agent e2e report (with-compute)", "",
          f"Provider: **{args.provider}** / model `{model_name}`. One real "
          "SPAgent episode per tool: the VLM must choose to invoke the tool "
          "against its live backend and produce a grounded answer.", "",
          "| tool | backend | status | episode |", "|---|---|---|---|"]
    failed = []
    for k in keys:
        e = entries[k]
        ok, notes = run_episode(e, model, urls.get(k), args.prompt)
        if not ok:
            failed.append(k)
        shown = urls.get(k) or DEFAULT_SERVER_URLS.get(k, "(local)")
        md.append(f"| {k} | {shown} | {PASS if ok else FAIL} | "
                  + "; ".join(notes)[:250] + " |")
    md.append("")
    md.append("Pass criteria: agent loop completes, the tool under test is "
              "invoked by the model, at least one invocation succeeds, and a "
              "non-empty final answer is produced.")
    md.append("")
    md.append(f"## {'❌ failed: ' + ', '.join(failed) if failed else '✅ all episodes passed'}")
    report = "\n".join(md)

    print(report)
    for path in filter(None, [args.markdown_out, os.environ.get("GITHUB_STEP_SUMMARY")]):
        with open(path, "a") as f:
            f.write(report + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
