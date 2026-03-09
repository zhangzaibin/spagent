# Moondream3 test directory

This folder contains scripts to test `Moondream3Tool`: a full test suite and a real-API diagnostic script.

**Shared input:** Both scripts can use `example_input.json` in this directory. It must contain:

```json
{
    "image_path": "test/moondream3/dog.png",
    "question": "What is in this image?"
}
```

- `image_path`: path to an image file (absolute or relative to current working directory).
- `question`: question to ask the vision model.

---

## 1. test_moondream3_tool.py

Runs the full test suite for `Moondream3Tool` (interface, return format, validation, mock VQA, optional real server call).

### How to run

From project root:

```bash
python test/moondream3/test_moondream3_tool.py [options]
```

From this directory (`test/moondream3`):

```bash
python test_moondream3_tool.py [options]
```

### Input

- **Default:** reads `example_input.json` in this directory for the optional “run tool once” step (used when `--image` is not set).
- **Optional:** `--image PATH` — use this image for the real-image test instead of `example_input.json`. Path can be absolute or relative to current working directory.

### Arguments

| Argument | Description |
|----------|-------------|
| `--json` | Print results as JSON (tests + optional tool_call_result). |
| `--output FILE` | Write that JSON to `FILE`. If path is relative, it is under `test/moondream3`. Example: `--output result.json`. |
| `--no-mock` | Use real Moondream/Station server instead of mock. Requires a running server. |
| `--server_url URL` | Server base URL when using `--no-mock`. Default: `http://localhost:2020/v1`. |
| `--timeout N` | Request timeout in seconds for real API. Default: 300. |
| `--image PATH` | Image path for the single real-image test (see above). |

### Output

- **Without `--json`:** Human-readable lines like `[PASS] ...`, `[SKIP] ...`, then `All tests passed. Moondream3 tool is OK.` or an exception.
- **With `--json`:** A single JSON object printed to stdout (and optionally written to `--output`), e.g.:

```json
{
  "tests": [
    { "name": "test_tool_interface", "status": "pass" },
    { "name": "test_return_format_success", "status": "pass" },
    ...
  ],
  "all_passed": true,
  "tool_call_result": {
    "success": true,
    "answer": "...",
    "result": { ... }
  }
}
```

- `tests`: one object per test with `name`, `status` (`pass` / `fail` / `skip`), and optional `error`.
- `all_passed`: `true` if every non-skipped test passed.
- `tool_call_result`: result of one `tool.call()` when no `--image` is given (from `example_input.json`) or when running the real-image test; same shape as `Moondream3Tool.call()` (e.g. `success`, `answer`, `result`, or `error` on failure).

### Examples

```bash
# Mock only, human-readable
python test_moondream3_tool.py

# Mock, JSON to stdout
python test_moondream3_tool.py --json

# Mock, JSON to result.json, use dog.png for the image test
python test_moondream3_tool.py --json --output result.json --image test/moondream3/dog.png

# Real server (Station must be running)
python test_moondream3_tool.py --json --output result.json --no-mock
```

---

## 2. test_real_api.py

Checks that the environment is ready for a **real** API call (no mock), then performs one `tool.call()` and prints the raw return value. Use this to verify Station is reachable and to inspect the exact response.

### How to run

From this directory:

```bash
python test_real_api.py
```

Must be run from `test/moondream3` (or with Python path set so that `example_input.json` and the project root are correct).

### Input

- **Only:** `example_input.json` in this directory. It must contain `image_path` and `question`. The script does not accept command-line arguments for image or question.

### Output

All output is to **stdout**:

1. **Checks:** Lines like `[OK] Image exists: ...`, `[OK] requests is installed`, `[OK] Station reachable: ...`, or `[MISSING] ...` / `--- Currently missing ---` with a list of missing items.
2. If anything is missing (e.g. no `example_input.json`, image not found, no `requests`, Station not reachable), the script stops after printing the missing list and **does not** call the tool.
3. If all checks pass: `--- Sending real query request ---`, then the full JSON returned by `tool.call()` (e.g. `success`, `answer`, `result`, or `error`).

Example when everything is OK:

```
[OK] Image exists: test/moondream3/dog.png
[OK] requests is installed
[OK] Station reachable: http://localhost:2020/v1 (status 200)

--- Sending real query request ---
tool.call() returned:
{
  "success": true,
  "answer": "...",
  "result": { ... }
}
```

If `success` is true but `answer` is empty, a short note is printed suggesting to check Station’s response format (e.g. field name or empty body).

### Prerequisites

- `example_input.json` with valid `image_path` and `question`.
- Image file exists at `image_path`.
- `requests` installed: `pip install requests`.
- Moondream Station (or compatible server) running, e.g. `moondream-station`, with API at `http://localhost:2020/v1`.

---

## Summary

| Script | Purpose | Input | Output |
|--------|---------|--------|--------|
| `test_moondream3_tool.py` | Full test suite (mock or real) | `example_input.json` and/or `--image` | Pass/fail lines or JSON (optional file with `--output`) |
| `test_real_api.py` | Real-API readiness check + one call | `example_input.json` only | Stdout: checks + raw `tool.call()` JSON |
