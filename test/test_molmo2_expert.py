"""
Unit tests for Molmo2 expert (mock client + HTTP client with mocks).

Run from repo root:
  python -m unittest test.test_molmo2_expert -v
"""

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from spagent.external_experts.Molmo2.mock_molmo2_service import MockMolmo2
from spagent.external_experts.Molmo2.molmo2_client import Molmo2Client
from spagent.tools.molmo2_tool import Molmo2Tool


class TestMockMolmo2(unittest.TestCase):
    def test_infer_path(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xd9")
            path = f.name
        try:
            m = MockMolmo2()
            r = m.infer_path(path, prompt="Hello")
            self.assertTrue(r.get("success"))
            self.assertIn("mock", r.get("text", ""))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_file(self):
        r = MockMolmo2().infer_path("/nonexistent/no.jpg")
        self.assertFalse(r.get("success"))


class TestMolmo2Client(unittest.TestCase):
    @patch("spagent.external_experts.Molmo2.molmo2_client.requests.post")
    def test_infer_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True, "text": "a cat"}
        mock_post.return_value = mock_resp

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xd9")
            path = f.name
        try:
            c = Molmo2Client(server_url="http://127.0.0.1:9")
            out = c.infer_path(path, prompt="What?")
            self.assertTrue(out.get("success"))
            self.assertEqual(out.get("text"), "a cat")
            mock_post.assert_called_once()
            body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
            self.assertIn("image", body)
            base64.b64decode(body["image"])
        finally:
            Path(path).unlink(missing_ok=True)


class TestMolmo2Tool(unittest.TestCase):
    def test_tool_mock_call(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xd9")
            path = f.name
        try:
            t = Molmo2Tool(use_mock=True)
            r = t.call(path, prompt="Describe")
            self.assertTrue(r.get("success"))
            self.assertIn("text", r)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
