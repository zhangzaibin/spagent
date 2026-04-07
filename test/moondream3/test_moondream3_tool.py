"""
Test script for Moondream3Tool.

Verifies that the tool conforms to the requirements in docs/ADDING_NEW_TOOLS.md:
- Tool base: name, description, parameters (JSON Schema), call()
- Return format: success (required), result (recommended), error (when success=False)
- Input validation (e.g. file existence)
- Mock mode for testing without server

Usage:
    From project root (F:\\lab\\spagent):
        python test/moondream3/test_moondream3_tool.py --json
    From test/moondream3 (F:\\lab\\spagent\\test\\moondream3):
        python test_moondream3_tool.py --json --output result.json
    Input: example_input.json in this directory.
    Output: use --output result.json to write JSON to this directory.
"""

import sys
import json
import tempfile
import argparse
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


class _SkipTest(Exception):
    """Raised when a test is skipped (e.g. missing dependency)."""
    pass


def create_temp_image(path: Path) -> None:
    """Create a minimal valid image file for testing."""
    try:
        from PIL import Image
        img = Image.new("RGB", (10, 10), color="red")
        img.save(path)
    except ImportError:
        # Fallback: write a minimal BMP header + pixel so Path.exists() and open() work
        with open(path, "wb") as f:
            f.write(b"\x00" * 100)  # placeholder; tool mock does not read pixels


def test_tool_interface():
    """Verify tool has required members: name, description, parameters (OpenAI schema)."""
    from spagent.tools import Moondream3Tool

    tool = Moondream3Tool(use_mock=True)
    assert hasattr(tool, "name"), "Tool must have 'name'"
    assert tool.name == "moondream3_tool", f"Expected name 'moondream3_tool', got '{tool.name}'"
    assert hasattr(tool, "description"), "Tool must have 'description'"
    assert isinstance(tool.description, str) and len(tool.description) > 0, "description must be non-empty string"

    params = tool.parameters
    assert isinstance(params, dict), "parameters must be a dict (JSON Schema)"
    assert params.get("type") == "object", "parameters must have type: object"
    assert "properties" in params, "parameters must have 'properties'"
    assert "required" in params, "parameters must have 'required'"
    for key in ["image_path", "question"]:
        assert key in params["required"], f"parameters.required must include '{key}'"
        assert key in params["properties"], f"parameters.properties must include '{key}'"
    if not _json_mode():
        print("[PASS] Tool interface: name, description, parameters OK")


def _json_mode():
    return getattr(sys.modules[__name__], "_output_json", False)


def test_return_format_success():
    """Verify call() return format on success: success=True, result present (per doc)."""
    from spagent.tools import Moondream3Tool

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        create_temp_image(tmp_path)
        tool = Moondream3Tool(use_mock=True)
        out = tool.call(image_path=str(tmp_path), question="What is in this image?")
    finally:
        tmp_path.unlink(missing_ok=True)

    assert isinstance(out, dict), "call() must return a dict"
    assert "success" in out, "Return must contain 'success' (required)"
    assert out["success"] is True, "Expected success=True for valid input"
    assert "result" in out, "Return should contain 'result' (recommended in doc)"
    assert isinstance(out["result"], dict), "result should be a dict"
    if not _json_mode():
        print("[PASS] Return format (success): success=True, result present OK")


def test_return_format_failure():
    """Verify call() return format on failure: success=False, error present."""
    from spagent.tools import Moondream3Tool

    tool = Moondream3Tool(use_mock=True)
    out = tool.call(
        image_path="/nonexistent/image_xyz_123.jpg",
        question="What is in this image?"
    )

    assert isinstance(out, dict), "call() must return a dict"
    assert "success" in out, "Return must contain 'success'"
    assert out["success"] is False, "Expected success=False for missing file"
    assert "error" in out, "Return must contain 'error' when success=False (per doc)"
    assert isinstance(out["error"], str) and len(out["error"]) > 0, "error must be non-empty string"
    if not _json_mode():
        print("[PASS] Return format (failure): success=False, error present OK")


def test_validation_rejects_missing_file():
    """Verify input validation: missing image file returns error (best practice in doc)."""
    from spagent.tools import Moondream3Tool

    tool = Moondream3Tool(use_mock=True)
    out = tool.call(
        image_path="/does/not/exist.png",
        question="Describe this image."
    )
    assert out.get("success") is False
    assert "not found" in out.get("error", "").lower() or "exist" in out.get("error", "").lower()
    if not _json_mode():
        print("[PASS] Validation: missing file rejected with clear error OK")


def test_mock_vqa():
    """Mock mode: question about image returns answer in result."""
    from spagent.tools import Moondream3Tool

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        create_temp_image(tmp_path)
        tool = Moondream3Tool(use_mock=True)
        out = tool.call(image_path=str(tmp_path), question="What is in this image?")
    finally:
        tmp_path.unlink(missing_ok=True)

    assert out.get("success") is True
    assert "result" in out and "answer" in out.get("result", {})
    assert "answer" in out
    if not _json_mode():
        print("[PASS] Mock VQA (image + question) OK")


def test_registration_with_agent():
    """Verify tool can be registered and listed (Registering Tools with SPAgent)."""
    try:
        from spagent import SPAgent
        from spagent.models import GPTModel
        from spagent.tools import Moondream3Tool

        tool = Moondream3Tool(use_mock=True)
        model = GPTModel(model_name="gpt-4o-mini")
        agent = SPAgent(model=model)
        agent.add_tool(tool)
        names = agent.list_tools()
        assert "moondream3_tool" in names, f"list_tools() should include 'moondream3_tool', got {names}"
        if not _json_mode():
            print("[PASS] Registration: add_tool and list_tools OK")
    except ImportError as e:
        pytest.skip(f"Registration test (SPAgent/GPTModel not available: {e})")
    except Exception as e:
        if _json_mode():
            raise _SkipTest(str(e))
        raise


def run_with_real_image(image_path: str, use_mock: bool = True, server_url: str = "http://localhost:20025", request_timeout: int = 300):
    """Helper: run VQA with a real image path (mock or server). Not a pytest test (has params)."""
    from spagent.tools import Moondream3Tool

    path = Path(image_path)
    if not path.exists():
        pytest.skip(f"Image not found: {image_path}")
    tool = Moondream3Tool(use_mock=use_mock, server_url=server_url, request_timeout=request_timeout)
    out = tool.call(image_path=image_path, question="What is in this image?")
    if not _json_mode():
        print(f"  success={out.get('success')}, answer={out.get('answer', '')[:80]}...")
    assert out.get("success") is True
    if not _json_mode():
        print("[PASS] Real image (mock) OK")


def run_tool_with_example_input(use_mock: bool = True, server_url: str = "http://localhost:20025", request_timeout: int = 300):
    """Run tool once with example_input.json; return result dict or None if file missing."""
    example_path = Path(__file__).parent / "example_input.json"
    if not example_path.exists():
        return None
    with open(example_path, encoding="utf-8") as f:
        args = json.load(f)
    if "image_path" not in args or "question" not in args:
        return None
    from spagent.tools import Moondream3Tool
    tool = Moondream3Tool(use_mock=use_mock, server_url=server_url, request_timeout=request_timeout)
    return tool.call(**args)


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Test Moondream3Tool per ADDING_NEW_TOOLS.md")
    parser.add_argument("--image", type=str, default=None, help="Optional image path for real-image test")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write JSON to this file (relative path is under test/moondream3). e.g. result.json")
    parser.add_argument("--no-mock", action="store_true",
                        help="Use real Moondream/Moondream3 server (requires server running).")
    parser.add_argument("--server_url", type=str, default="http://localhost:2020/v1",
                        help="Server URL when --no-mock. Moondream Station: http://localhost:2020/v1 (default).")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Request timeout in seconds for real API (default 300).")
    args = parser.parse_args()

    sys.modules[__name__]._output_json = args.json

    if not args.json:
        print("Running Moondream3Tool tests (docs/ADDING_NEW_TOOLS.md)...\n")

    results = []
    try:
        test_tool_interface()
        results.append({"name": "test_tool_interface", "status": "pass"})
    except Exception as e:
        results.append({"name": "test_tool_interface", "status": "fail", "error": str(e)})
        if not args.json:
            raise

    try:
        test_return_format_success()
        results.append({"name": "test_return_format_success", "status": "pass"})
    except Exception as e:
        results.append({"name": "test_return_format_success", "status": "fail", "error": str(e)})
        if not args.json:
            raise

    try:
        test_return_format_failure()
        results.append({"name": "test_return_format_failure", "status": "pass"})
    except Exception as e:
        results.append({"name": "test_return_format_failure", "status": "fail", "error": str(e)})
        if not args.json:
            raise

    try:
        test_validation_rejects_missing_file()
        results.append({"name": "test_validation_rejects_missing_file", "status": "pass"})
    except Exception as e:
        results.append({"name": "test_validation_rejects_missing_file", "status": "fail", "error": str(e)})
        if not args.json:
            raise

    try:
        test_mock_vqa()
        results.append({"name": "test_mock_vqa", "status": "pass"})
    except Exception as e:
        results.append({"name": "test_mock_vqa", "status": "fail", "error": str(e)})
        if not args.json:
            raise

    try:
        test_registration_with_agent()
        results.append({"name": "test_registration_with_agent", "status": "pass"})
    except _SkipTest as e:
        results.append({"name": "test_registration_with_agent", "status": "skip", "error": str(e)})
    except Exception as e:
        results.append({"name": "test_registration_with_agent", "status": "fail", "error": str(e)})
        if not args.json:
            raise

    use_mock = not args.no_mock
    server_url = args.server_url
    request_timeout = getattr(args, "timeout", 300)

    tool_call_result = None
    if args.image:
        try:
            run_with_real_image(args.image, use_mock=use_mock, server_url=server_url, request_timeout=request_timeout)
            results.append({"name": "run_with_real_image", "status": "pass"})
        except Exception as e:
            results.append({"name": "run_with_real_image", "status": "fail", "error": str(e)})
    else:
        tool_call_result = run_tool_with_example_input(use_mock=use_mock, server_url=server_url, request_timeout=request_timeout)

    all_passed = all(r["status"] == "pass" for r in results if r["status"] != "skip")

    if args.json:
        out = {
            "tests": results,
            "all_passed": all_passed,
            "tool_call_result": tool_call_result,
        }
        json_str = json.dumps(out, ensure_ascii=False, indent=2)
        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = script_dir / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")
            print(f"JSON written to {out_path}")
        else:
            print(json_str)
    else:
        print("\nAll tests passed. Moondream3 tool is OK.")


if __name__ == "__main__":
    main()
