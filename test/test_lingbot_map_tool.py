"""Tests for LingBotMapTool."""

import os
import sys
import base64
from pathlib import Path

import pytest
from PIL import Image

from spagent.tools import LingBotMapTool


def _make_frames(tmp_path: Path, count: int = 4) -> list[str]:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    paths = []
    for idx in range(count):
        path = frame_dir / f"{idx:06d}.png"
        image = Image.new("RGB", (96, 72), (40 + idx * 30, 90, 160))
        image.save(path)
        paths.append(str(path))
    return paths


def test_lingbot_map_import():
    assert LingBotMapTool is not None


def test_lingbot_map_schema_contains_required_inputs():
    schema = LingBotMapTool(use_mock=True).parameters
    assert "image_folder" in schema["properties"]
    assert "image_paths" in schema["properties"]
    assert "mask_sky" in schema["properties"]
    assert "oneOf" in schema


def test_mock_image_folder_mapping(tmp_path):
    frames = _make_frames(tmp_path)
    tool = LingBotMapTool(use_mock=True, output_dir=str(tmp_path / "out"))

    result = tool.call(image_folder=str(Path(frames[0]).parent), mask_sky=True)

    assert result["success"] is True
    assert result["num_frames"] == 4
    assert Path(result["preview_path"]).exists()
    assert Path(result["trajectory_path"]).exists()
    assert Path(result["point_cloud_path"]).exists()
    assert result["viewer_url"]


def test_mock_image_paths_mapping_with_frame_limits(tmp_path):
    frames = _make_frames(tmp_path, count=6)
    tool = LingBotMapTool(use_mock=True, output_dir=str(tmp_path / "out"))

    result = tool.call(image_paths=frames, keyframe_interval=2, max_frames=2)

    assert result["success"] is True
    assert result["num_frames"] == 2
    assert Path(result["preview_path"]).exists()


def test_rejects_missing_or_ambiguous_inputs(tmp_path):
    frames = _make_frames(tmp_path, count=1)
    tool = LingBotMapTool(use_mock=True)

    no_input = tool.call()
    assert no_input["success"] is False
    assert "exactly one" in no_input["error"]

    ambiguous = tool.call(image_folder=str(Path(frames[0]).parent), image_paths=frames)
    assert ambiguous["success"] is False
    assert "exactly one" in ambiguous["error"]


def test_rejects_invalid_paths_and_frame_options(tmp_path):
    tool = LingBotMapTool(use_mock=True)

    missing = tool.call(image_paths=[str(tmp_path / "missing.png")])
    assert missing["success"] is False
    assert "not found" in missing["error"]

    frames = _make_frames(tmp_path, count=1)
    bad_interval = tool.call(image_paths=frames, keyframe_interval=0)
    assert bad_interval["success"] is False
    assert "keyframe_interval" in bad_interval["error"]


def test_server_fake_official_cli_completion(tmp_path):
    from spagent.external_experts.LingBotMap import lingbot_map_server as server

    repo = tmp_path / "lingbot-map"
    repo.mkdir()
    demo = repo / "demo.py"
    demo.write_text(
        """
import json
import os
from pathlib import Path

out = Path(os.environ["LINGBOT_MAP_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "trajectory.json").write_text(json.dumps({"ok": True}))
(out / "point_cloud.ply").write_text("ply\\nformat ascii 1.0\\nelement vertex 0\\nend_header\\n")
""",
        encoding="utf-8",
    )
    model_path = tmp_path / "lingbot-map-long.pt"
    model_path.write_text("fake checkpoint", encoding="utf-8")
    frames = _make_frames(tmp_path, count=2)

    server.configure(repo_path=str(repo), model_path=str(model_path), python_bin=sys.executable)
    frame_dir = tmp_path / "server_frames"
    frame_dir.mkdir()
    for idx, frame in enumerate(frames):
        Image.open(frame).save(frame_dir / f"{idx:06d}.png")

    result = server._run_lingbot_map(
        frame_dir=frame_dir,
        output_dir=tmp_path / "server_out",
        mask_sky=True,
        wait_for_completion=True,
    )

    assert result["success"] is True
    assert result["trajectory_path"].endswith("trajectory.json")
    assert result["point_cloud_path"].endswith("point_cloud.ply")
    assert "--mask_sky" in result["command"]


def test_client_saves_server_outputs(tmp_path, monkeypatch):
    from spagent.external_experts.LingBotMap.lingbot_map_client import LingBotMapClient

    frames = _make_frames(tmp_path, count=1)
    preview = base64.b64encode(Path(frames[0]).read_bytes()).decode("utf-8")
    ply = base64.b64encode(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n").decode("utf-8")

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "success": True,
                "preview_image": preview,
                "point_cloud": ply,
                "viewer_url": "http://127.0.0.1:8080",
            }

    def fake_post(url, json, timeout):
        assert url.endswith("/infer")
        assert json["images"][0]["filename"].endswith(".png")
        return FakeResponse()

    monkeypatch.setattr("spagent.external_experts.LingBotMap.lingbot_map_client.requests.post", fake_post)
    client = LingBotMapClient(server_url="http://unused", output_dir=str(tmp_path / "client_out"))
    result = client.infer(image_paths=frames)

    assert result["success"] is True
    assert Path(result["preview_path"]).exists()
    assert Path(result["point_cloud_path"]).exists()


def test_server_http_route_with_fake_cli(tmp_path):
    pytest.importorskip("flask")
    from spagent.external_experts.LingBotMap import lingbot_map_server as server

    repo = tmp_path / "lingbot-map-http"
    repo.mkdir()
    (repo / "demo.py").write_text(
        """
import json
import os
from pathlib import Path

out = Path(os.environ["LINGBOT_MAP_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "trajectory.json").write_text(json.dumps({"route": True}))
(out / "point_cloud.ply").write_text("ply\\nformat ascii 1.0\\nelement vertex 0\\nend_header\\n")
""",
        encoding="utf-8",
    )
    model_path = tmp_path / "lingbot-map-long.pt"
    model_path.write_text("fake checkpoint", encoding="utf-8")
    frames = _make_frames(tmp_path, count=1)
    encoded = base64.b64encode(Path(frames[0]).read_bytes()).decode("utf-8")

    server.configure(repo_path=str(repo), model_path=str(model_path), python_bin=sys.executable)
    client = server.app.test_client()
    response = client.post(
        "/infer",
        json={
            "images": [{"filename": "frame.png", "data": encoded}],
            "wait_for_completion": True,
            "output_dir": str(tmp_path / "http_out"),
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["num_frames"] == 1
    assert "trajectory_json" in data
    assert "point_cloud" in data


@pytest.mark.skipif(
    os.environ.get("LINGBOT_MAP_REAL_TEST") != "1",
    reason="Set LINGBOT_MAP_REAL_TEST=1 to run against a live LingBot-Map server.",
)
def test_real_lingbot_map_server_smoke(tmp_path):
    frames = _make_frames(tmp_path, count=3)
    server_url = os.environ.get("LINGBOT_MAP_SERVER_URL", "http://127.0.0.1:20038")
    tool = LingBotMapTool(use_mock=False, server_url=server_url, output_dir=str(tmp_path / "out"))

    result = tool.call(image_paths=frames, mask_sky=False, wait_for_completion=False)

    assert result["success"] is True
    assert result["viewer_url"]
