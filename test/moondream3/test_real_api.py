"""
诊断 Moondream3 真实 API 调用：检查环境、图片、服务并打印缺失项。
运行：在 test/moondream3 目录下执行
  python test_real_api.py
  python test_real_api.py --image "你的图片路径" --question "问题"
"""
import json
import sys
from pathlib import Path

# 项目根加入 path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

def main():
    missing = []
    args = {"image_path": None, "question": None}

    # 1. 读取 example_input.json
    example_path = Path(__file__).parent / "example_input.json"
    if example_path.exists():
        with open(example_path, encoding="utf-8") as f:
            data = json.load(f)
        args["image_path"] = data.get("image_path")
        args["question"] = data.get("question")
    if not args["image_path"] or not args["question"]:
        missing.append("example_input.json 需包含 image_path 和 question")

    # 2. 检查图片是否存在
    image_path = args["image_path"]
    if image_path:
        if not Path(image_path).exists():
            missing.append(f"图片不存在: {image_path}")
        else:
            print(f"[OK] 图片存在: {image_path}")

    # 3. 检查 requests
    try:
        import requests
        print("[OK] requests 已安装")
    except ImportError:
        missing.append("缺少 requests，请执行: pip install requests")

    # 4. 检查 Station 是否可达（默认 2020/v1）
    server_url = "http://localhost:2020/v1"
    try:
        import requests as req
        r = req.get(server_url, timeout=2)
        print(f"[OK] Station 可访问: {server_url} (status {r.status_code})")
    except Exception as e:
        missing.append(f"无法连接 Moondream Station ({server_url})。请先另开终端运行: moondream-station")
        print(f"[缺] Station 连接: {e}")

    # 5. 真实调用一次并打印原始响应
    if missing:
        print("\n--- 当前缺少 ---")
        for m in missing:
            print("  -", m)
        return

    print("\n--- 发起真实 query 请求 ---")
    from spagent.tools import Moondream3Tool
    tool = Moondream3Tool(use_mock=False, server_url=server_url)
    out = tool.call(image_path=image_path, question=args["question"])
    print("tool.call() 返回:")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if out.get("success") and not out.get("answer"):
        print("\n[提示] success 为 true 但 answer 为空，可能是 Station 返回的 JSON 里 answer 字段名不同或为空。请查看 Station 文档或抓包确认响应格式。")

if __name__ == "__main__":
    main()
