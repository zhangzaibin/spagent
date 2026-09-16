"""Tests for LingBotMapTool."""

import os
import sys
import base64
import json
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "spagent"))

from spagent.tools import LingBotMapTool
from core.tool_result import ToolResult, validate_payload


def _make_frames(tmp_path: Path, count: int = 8) -> list[str]:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    paths = []
    for idx in range(count):
        path = frame_dir / f"{idx:06d}.png"
        image = Image.new("RGB", (96, 72), ((40 + idx * 30) % 256, 90, 160))
        image.save(path)
        paths.append(str(path))
    return paths


def _write_fake_runner(path: Path) -> None:
    path.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--repo_path')
parser.add_argument('--model_path')
parser.add_argument('--image_folder')
parser.add_argument('--output_dir')
parser.add_argument('--camera_num_iterations')
parser.add_argument('--use_sdpa', action='store_true')
parser.add_argument('--mask_sky', action='store_true')
args = parser.parse_args()
out = Path(args.output_dir)
out.mkdir(parents=True, exist_ok=True)
(out / 'trajectory.json').write_text(json.dumps({'frames': list(range(8))}))
(out / 'point_cloud.ply').write_text(
    'ply\\nformat ascii 1.0\\nelement vertex 4\\n'
    'property float x\\nproperty float y\\nproperty float z\\nend_header\\n'
    '0 0 0\\n1 0 0\\n0 1 0\\n0 0 1\\n'
)
(out / 'reconstruction_metadata.json').write_text(json.dumps({
    'points_count': 4,
    'checkpoint_path': str(Path(args.model_path).resolve()),
    'checkpoint_bytes': Path(args.model_path).stat().st_size,
}))
""",
        encoding="utf-8",
    )


def test_lingbot_map_import():
    assert LingBotMapTool is not None


def test_lingbot_map_schema_contains_required_inputs():
    schema = LingBotMapTool(use_mock=True).parameters
    assert "image_folder" in schema["properties"]
    assert "image_paths" in schema["properties"]
    assert "mask_sky" in schema["properties"]
    assert "oneOf" not in schema
    assert "wait_for_completion" not in schema["properties"]
    assert "exactly one" in schema["properties"]["image_folder"]["description"]
    assert "at least eight" in schema["properties"]["image_paths"]["description"]


def test_lingbot_map_is_in_tool_selection_guide():
    from core.prompts import build_tool_selection_guide

    guide = build_tool_selection_guide({"lingbot_map_tool"})
    assert "## 3D & Spatial" in guide
    assert "lingbot_map_tool" in guide
    assert "at least eight images" in guide


def test_image_folder_is_normalized_in_numeric_order_for_upload(tmp_path):
    frame_dir = tmp_path / "unordered_frames"
    frame_dir.mkdir()
    for index in [10, 2, 1, 7, 6, 5, 4, 3]:
        Image.new("RGB", (16, 12), color=(index, 20, 30)).save(frame_dir / f"{index}.png")

    valid, error, paths = LingBotMapTool(use_mock=True)._validate_inputs(
        image_folder=str(frame_dir),
        image_paths=None,
        keyframe_interval=1,
        max_frames=8,
    )

    assert valid is True
    assert error is None
    assert [Path(path).stem for path in paths] == ["1", "2", "3", "4", "5", "6", "7", "10"]


def test_mock_image_folder_mapping(tmp_path):
    frames = _make_frames(tmp_path)
    tool = LingBotMapTool(use_mock=True, output_dir=str(tmp_path / "out"))

    result = tool.call(image_folder=str(Path(frames[0]).parent), mask_sky=True)

    assert result["success"] is True
    assert isinstance(result, ToolResult)
    assert validate_payload(result, "3d_reconstruction")[0]
    assert result["num_frames"] == 8
    assert Path(result["preview_path"]).exists()
    assert Path(result["trajectory_path"]).exists()
    assert Path(result["point_cloud_path"]).exists()
    assert result["points_count"] == 4
    assert result["result"]["points_count"] == 4
    assert result["viewer_url"]


def test_mock_image_paths_mapping_with_frame_limits(tmp_path):
    frames = _make_frames(tmp_path, count=16)
    tool = LingBotMapTool(use_mock=True, output_dir=str(tmp_path / "out"))

    result = tool.call(image_paths=frames, keyframe_interval=2, max_frames=8)

    assert result["success"] is True
    assert result["num_frames"] == 8
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


def test_rejects_async_mode_without_reconstruction_artifacts(tmp_path):
    frames = _make_frames(tmp_path)
    result = LingBotMapTool(use_mock=True).call(
        image_paths=frames,
        wait_for_completion=False,
    )
    assert result["success"] is False
    assert "wait_for_completion=True" in result["error"]


def test_server_reencodes_jpeg_frames_as_png(tmp_path):
    from spagent.external_experts.LingBotMap import lingbot_map_server as server

    source = tmp_path / "jpeg_frames"
    source.mkdir()
    for idx in range(8):
        Image.new("RGB", (32, 24), (idx * 20, 50, 90)).save(source / f"{idx:06d}.jpeg")
    staged = server._prepare_frame_dir(
        image_folder=str(source),
        images=[],
        output_dir=tmp_path / "run",
        keyframe_interval=1,
        max_frames=8,
    )
    paths = sorted(staged.iterdir())
    assert len(paths) == 8
    assert all(path.suffix == ".png" for path in paths)
    assert all(Image.open(path).format == "PNG" for path in paths)


def test_server_fake_official_cli_completion(tmp_path):
    from spagent.external_experts.LingBotMap import lingbot_map_server as server

    repo = tmp_path / "lingbot-map"
    repo.mkdir()
    (repo / "demo.py").write_text("# fake official demo\n", encoding="utf-8")
    runner = tmp_path / "fake_runner.py"
    _write_fake_runner(runner)
    model_path = tmp_path / "lingbot-map-long.pt"
    model_path.write_text("fake checkpoint", encoding="utf-8")
    frames = _make_frames(tmp_path, count=8)

    server.configure(
        repo_path=str(repo),
        model_path=str(model_path),
        python_bin=sys.executable,
        runner_path=str(runner),
    )
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
    assert result["points_count"] == 4
    assert result["checkpoint_path"] == str(model_path.resolve())
    assert "--mask_sky" in result["command"]
    assert str(model_path) in result["command"]


def test_runner_exports_nonempty_ply_trajectory_and_metadata(tmp_path):
    from spagent.external_experts.LingBotMap.lingbot_map_runner import export_artifacts

    frame_count, height, width = 8, 2, 3
    points = np.arange(frame_count * height * width * 3, dtype=np.float32).reshape(frame_count, height, width, 3)
    predictions = {
        "world_points": points,
        "world_points_conf": np.full((frame_count, height, width), 2.0, dtype=np.float32),
        "images": np.full((frame_count, 3, height, width), 0.5, dtype=np.float32),
        "extrinsic": np.tile(np.eye(4, dtype=np.float32)[:3], (frame_count, 1, 1)),
        "intrinsic": np.tile(np.eye(3, dtype=np.float32), (frame_count, 1, 1)),
    }
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    image_paths = [f"frame_{idx:06d}.png" for idx in range(frame_count)]
    metadata = export_artifacts(
        predictions,
        image_paths,
        tmp_path / "export",
        checkpoint,
        point_stride=2,
        max_points=100,
    )

    assert metadata["points_count"] == 24
    assert (tmp_path / "export" / "point_cloud.ply").is_file()
    trajectory = json.loads((tmp_path / "export" / "trajectory.json").read_text())
    assert trajectory["convention"] == "camera_to_world"
    assert len(trajectory["frames"]) == frame_count
    assert (tmp_path / "export" / "preview.png").is_file()


def test_runner_converts_camera_to_world_before_depth_unprojection():
    from spagent.external_experts.LingBotMap.lingbot_map_runner import (
        _camera_to_world_to_world_to_camera,
    )

    camera_to_world = np.tile(np.eye(4, dtype=np.float32)[:3], (2, 1, 1))
    camera_to_world[0, 0, 3] = 2.0
    camera_to_world[1, 1, 3] = -3.0

    world_to_camera = _camera_to_world_to_world_to_camera(camera_to_world)

    assert world_to_camera.shape == (2, 3, 4)
    assert np.allclose(world_to_camera[0, :, 3], [-2.0, 0.0, 0.0])
    assert np.allclose(world_to_camera[1, :, 3], [0.0, 3.0, 0.0])
    assert np.allclose(world_to_camera[:, :3, :3], np.eye(3)[None])


def test_client_saves_server_outputs(tmp_path, monkeypatch):
    from spagent.external_experts.LingBotMap.lingbot_map_client import LingBotMapClient

    frames = _make_frames(tmp_path, count=8)
    preview = base64.b64encode(Path(frames[0]).read_bytes()).decode("utf-8")
    ply = base64.b64encode(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n").decode("utf-8")
    trajectory = base64.b64encode(b'{"frames": []}').decode("utf-8")
    metadata = base64.b64encode(b'{"points_count": 4}').decode("utf-8")
    log = base64.b64encode(b"completed").decode("utf-8")

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "success": True,
                "preview_image": preview,
                "trajectory_json": trajectory,
                "point_cloud": ply,
                "metadata_json": metadata,
                "log": log,
                "output_dir": "/server-only/path",
                "viewer_url": "http://127.0.0.1:8080",
            }

    def fake_post(url, json, timeout):
        assert url.endswith("/infer")
        assert json["images"][0]["filename"].endswith(".png")
        assert "output_dir" not in json
        return FakeResponse()

    monkeypatch.setattr("spagent.external_experts.LingBotMap.lingbot_map_client.requests.post", fake_post)
    client = LingBotMapClient(server_url="http://unused", output_dir=str(tmp_path / "client_out"))
    result = client.infer(image_paths=frames)

    assert result["success"] is True
    assert Path(result["preview_path"]).exists()
    assert Path(result["trajectory_path"]).exists()
    assert Path(result["point_cloud_path"]).exists()
    assert Path(result["metadata_path"]).read_text() == '{"points_count": 4}'
    assert Path(result["log_path"]).read_text() == "completed"
    assert result["output_dir"] == str(tmp_path / "client_out")


def test_client_uses_unique_artifact_paths(tmp_path):
    from spagent.external_experts.LingBotMap.lingbot_map_client import LingBotMapClient

    client = LingBotMapClient(output_dir=str(tmp_path / "client_out"))
    payload = {
        "success": True,
        "trajectory_json": base64.b64encode(b'{"frames": []}').decode("utf-8"),
        "point_cloud": base64.b64encode(b"ply\nformat ascii 1.0\nend_header\n").decode("utf-8"),
    }

    first = client._save_outputs(payload, None)
    second = client._save_outputs(payload, None)

    assert first["point_cloud_path"] != second["point_cloud_path"]
    assert first["trajectory_path"] != second["trajectory_path"]
    assert Path(first["point_cloud_path"]).is_file()
    assert Path(second["point_cloud_path"]).is_file()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("point_cloud", base64.b64encode(b"not a point cloud").decode("utf-8"), "PLY or PCD"),
        ("trajectory_json", base64.b64encode(b"not json").decode("utf-8"), "valid JSON"),
        ("preview_image", base64.b64encode(b"not an image").decode("utf-8"), "valid image"),
        ("video", base64.b64encode(b"not an mp4").decode("utf-8"), "valid MP4"),
        ("point_cloud", "not-base64!", "Invalid base64"),
    ],
)
def test_client_rejects_malformed_artifacts(tmp_path, key, value, message):
    from spagent.external_experts.LingBotMap.lingbot_map_client import LingBotMapClient

    client = LingBotMapClient(output_dir=str(tmp_path / "client_out"))
    with pytest.raises(ValueError, match=message):
        client._save_outputs({"success": True, key: value}, None)


@pytest.mark.parametrize(
    ("result_update", "message"),
    [
        ({"trajectory_path": None}, "trajectory"),
        ({"points_count": 0}, "points_count"),
        ({"point_cloud_path": None}, "point cloud"),
    ],
)
def test_tool_rejects_incomplete_success_response(tmp_path, result_update, message):
    frames = _make_frames(tmp_path)
    point_cloud = tmp_path / "point_cloud.ply"
    point_cloud.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text('{"frames": []}', encoding="utf-8")
    backend_result = {
        "success": True,
        "point_cloud_path": str(point_cloud),
        "trajectory_path": str(trajectory),
        "points_count": 1,
        "num_frames": 8,
    }
    backend_result.update(result_update)

    class IncompleteBackend:
        def infer(self, **kwargs):
            return backend_result

    tool = LingBotMapTool(use_mock=True)
    tool._client = IncompleteBackend()
    result = tool.call(image_paths=frames)

    assert result["success"] is False
    assert message in result["error"]


def test_server_http_route_with_fake_cli(tmp_path):
    pytest.importorskip("flask")
    from spagent.external_experts.LingBotMap import lingbot_map_server as server

    repo = tmp_path / "lingbot-map-http"
    repo.mkdir()
    (repo / "demo.py").write_text("# fake official demo\n", encoding="utf-8")
    runner = tmp_path / "fake_http_runner.py"
    _write_fake_runner(runner)
    model_path = tmp_path / "lingbot-map-long.pt"
    model_path.write_text("fake checkpoint", encoding="utf-8")
    frames = _make_frames(tmp_path, count=8)
    images = [
        {
            "filename": Path(frame).name,
            "data": base64.b64encode(Path(frame).read_bytes()).decode("utf-8"),
        }
        for frame in frames
    ]

    server.configure(
        repo_path=str(repo),
        model_path=str(model_path),
        python_bin=sys.executable,
        runner_path=str(runner),
        work_dir=str(tmp_path / "server_work"),
    )
    client = server.app.test_client()
    response = client.post(
        "/infer",
        json={
            "images": images,
            "wait_for_completion": True,
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["num_frames"] == 8
    assert "trajectory_json" in data
    assert "point_cloud" in data
    assert "metadata_json" in data
    assert "log" in data
    assert data["points_count"] == 4
    assert Path(data["output_dir"]).parent == tmp_path / "server_work"


def test_server_normalizes_jpeg_inputs_to_png(tmp_path):
    pytest.importorskip("flask")
    from spagent.external_experts.LingBotMap import lingbot_map_server as server

    source_dir = tmp_path / "jpeg_frames"
    source_dir.mkdir()
    for index in range(8):
        Image.new("RGB", (16, 12), color=(index, 20, 30)).save(
            source_dir / f"{index:06d}.jpeg", format="JPEG"
        )

    staged = server._prepare_frame_dir(
        image_folder=str(source_dir),
        images=[],
        output_dir=tmp_path / "server_work",
        keyframe_interval=1,
        max_frames=8,
    )
    staged_names = sorted(path.name for path in staged.iterdir())
    assert staged_names == [f"{index:06d}.png" for index in range(8)]


@pytest.mark.skipif(
    os.environ.get("LINGBOT_MAP_REAL_TEST") != "1",
    reason="Set LINGBOT_MAP_REAL_TEST=1 to run against a live LingBot-Map server.",
)
def test_real_lingbot_map_server_smoke(tmp_path):
    frames = _make_frames(tmp_path, count=8)
    server_url = os.environ.get("LINGBOT_MAP_SERVER_URL", "http://127.0.0.1:20040")
    tool = LingBotMapTool(use_mock=False, server_url=server_url, output_dir=str(tmp_path / "out"))

    result = tool.call(image_paths=frames, mask_sky=False, wait_for_completion=True)

    assert result["success"] is True
    assert result["points_count"] > 0
    assert Path(result["point_cloud_path"]).is_file()
    assert Path(result["trajectory_path"]).is_file()
