"""
Diagnose Moondream3 real API calls: check environment, image, service and print missing items.

Usage (run from project root or test/moondream3):
  python test/moondream3/test_real_api.py

The script reads example_input.json in test/moondream3 to obtain image_path and question.
No CLI arguments for image/question; edit example_input.json or set env vars if needed.
"""
import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

def main():
    missing = []
    args = {"image_path": None, "question": None}

    # 1. Read example_input.json
    example_path = Path(__file__).parent / "example_input.json"
    if example_path.exists():
        with open(example_path, encoding="utf-8") as f:
            data = json.load(f)
        args["image_path"] = data.get("image_path")
        args["question"] = data.get("question")
    if not args["image_path"] or not args["question"]:
        missing.append("example_input.json must contain image_path and question")

    # 2. Check if image exists
    image_path = args["image_path"]
    if image_path:
        if not Path(image_path).exists():
            missing.append(f"Image does not exist: {image_path}")
        else:
            print(f"[OK] Image exists: {image_path}")

    # 3. Check requests
    try:
        import requests
        print("[OK] requests is installed")
    except ImportError:
        missing.append("Missing requests, run: pip install requests")

    # 4. Check if Station is reachable (default 2020/v1)
    server_url = "http://localhost:2020/v1"
    try:
        import requests as req
        r = req.get(server_url, timeout=2)
        print(f"[OK] Station reachable: {server_url} (status {r.status_code})")
    except Exception as e:
        missing.append(f"Cannot connect to Moondream Station ({server_url}). Run moondream-station in another terminal first.")
        print(f"[MISSING] Station connection: {e}")

    # 5. Make a real call and print raw response
    if missing:
        print("\n--- Currently missing ---")
        for m in missing:
            print("  -", m)
        return

    print("\n--- Sending real query request ---")
    from spagent.tools import Moondream3Tool
    tool = Moondream3Tool(use_mock=False, server_url=server_url)
    out = tool.call(image_path=image_path, question=args["question"])
    print("tool.call() returned:")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out.get("success") and not out.get("answer"):
        print("\n[Note] success is true but answer is empty; Station's JSON may use a different field name or return empty. Check Station docs or capture traffic to confirm response format.")

if __name__ == "__main__":
    main()
