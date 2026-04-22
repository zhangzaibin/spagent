import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class VaceRunner:
    """Run VACE first-frame pipeline command and collect outputs."""

    def __init__(
        self,
        vace_root: str,
        checkpoint_path: str,
        python_exec: str = "python",
    ):
        self.vace_root = Path(vace_root).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.python_exec = python_exec
        self._progress: Dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "denoise_step": 0,
            "denoise_total": 0,
            "message": "",
            "updated_at": 0.0,
        }
        self._progress_lock = threading.Lock()
        # Only one GPU inference at a time (parallel /infer requests otherwise OOM-kill the child, rc=-9).
        self._infer_lock = threading.Lock()

    def is_ready(self) -> Tuple[bool, str]:
        if not self.vace_root.exists():
            return False, f"VACE root does not exist: {self.vace_root}"

        pipeline_path = self.vace_root / "vace" / "vace_pipeline.py"
        if not pipeline_path.exists():
            return False, f"Pipeline file not found: {pipeline_path}"

        if not self.checkpoint_path.exists():
            return False, f"Checkpoint path not found: {self.checkpoint_path}"

        return True, "ok"

    def check_inference_runtime_deps(self) -> Tuple[bool, str]:
        """Packages needed for Wan2.1 (`import wan`) and VACE video IO (preprocessor)."""
        checks = [
            ("diffusers", "diffusers"),
            ("ftfy", "ftfy"),  # wan/modules/tokenizers.py
            ("accelerate", "accelerate"),
            ("einops", "einops"),  # wan/modules/vae.py
            ("decord", "decord"),  # vace/models/utils/preprocessor.py (MP4 frames)
            ("cv2", "opencv-python"),  # annotators/utils.py save_one_video OpenCV fallback
        ]
        for import_name, pip_name in checks:
            try:
                __import__(import_name)
            except ImportError as exc:
                # Almost always: `pip` / `pip list` targeted a different interpreter than this process.
                return (
                    False,
                    (
                        f"Cannot import `{import_name}` (PyPI: {pip_name}): {exc}. "
                        f"This server process uses Python: {sys.executable}. "
                        f"Install into the same interpreter, e.g.: "
                        f'"{sys.executable}" -m pip install -r requirements-vace.txt '
                        f"or \"{sys.executable}\" -m pip install {pip_name}"
                    ),
                )
        return True, "ok"

    def run_firstframe(
        self,
        image_path: str,
        prompt: str,
        base: str = "wan",
        task: str = "frameref",
        mode: str = "firstframe",
        timeout_seconds: int = 1800,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ready, message = self.is_ready()
        if not ready:
            return {"success": False, "error": message}

        absolute_image = Path(image_path)
        if not absolute_image.is_absolute():
            absolute_image = (self.vace_root / absolute_image).resolve()

        if not absolute_image.exists():
            return {"success": False, "error": f"Image file not found: {absolute_image}"}

        merged_extra_args = dict(extra_args or {})
        checkpoint_dir_for_run = self.checkpoint_path
        # Keep backward compatibility if caller still passes ckpt_dir via payload.
        if "ckpt_dir" in merged_extra_args:
            checkpoint_dir_for_run = Path(str(merged_extra_args.pop("ckpt_dir"))).expanduser().resolve()

        cmd = [
            self.python_exec,
            "vace/vace_pipeline.py",
            "--base",
            str(base),
            "--task",
            str(task),
            "--mode",
            str(mode),
            "--image",
            str(absolute_image),
            "--prompt",
            str(prompt),
            "--ckpt_dir",
            str(checkpoint_dir_for_run),
        ]

        if merged_extra_args:
            for key, value in merged_extra_args.items():
                if value is None:
                    continue
                flag = f"--{key}"
                cmd.extend([flag, str(value)])

        with self._infer_lock:
            return self._run_firstframe_locked(
                cmd=cmd,
                timeout_seconds=timeout_seconds,
            )

    def _run_firstframe_locked(
        self,
        cmd: List[str],
        timeout_seconds: int,
    ) -> Dict[str, Any]:
        results_dir = self.vace_root / "results"
        before_latest = self._latest_result_dir(results_dir)

        logger.info("Running VACE firstframe command...")
        logger.info("Command: %s", " ".join(cmd))
        run_env = os.environ.copy()
        # Wan2.1 upstream: `import wan` (see vace_wan_inference.py). Pip install may
        # pull flash_attn build; optional vendored clone: third_party/Wan2.1/wan/
        wan_root = self.vace_root / "third_party" / "Wan2.1"
        extra_paths = [str(self.vace_root), str(self.vace_root / "vace")]
        if wan_root.is_dir():
            extra_paths.insert(0, str(wan_root))
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        if existing_pythonpath:
            run_env["PYTHONPATH"] = os.pathsep.join(extra_paths + [existing_pythonpath])
        else:
            run_env["PYTHONPATH"] = os.pathsep.join(extra_paths)
        # Stream child logs in real time (capture_output buffers until exit — hides tqdm / denoise).
        run_env["PYTHONUNBUFFERED"] = "1"
        run_env["VACE_SERVER_PROGRESS"] = "1"

        with self._progress_lock:
            self._progress = {
                "active": True,
                "phase": "starting",
                "denoise_step": 0,
                "denoise_total": 0,
                "message": "starting pipeline",
                "updated_at": time.time(),
            }

        stdout_chunks: List[str] = []
        process: Optional[subprocess.Popen] = None

        def _parse_progress_line(line: str) -> None:
            text = line.strip()
            if not text.startswith("VACE_SERVER_PROGRESS"):
                return
            parts = text.split()
            if len(parts) >= 4 and parts[1] == "denoise":
                try:
                    cur, tot = int(parts[2]), int(parts[3])
                    with self._progress_lock:
                        self._progress["phase"] = "denoise"
                        self._progress["denoise_step"] = cur
                        self._progress["denoise_total"] = tot
                        self._progress["message"] = f"denoise {cur}/{tot}"
                        self._progress["updated_at"] = time.time()
                except ValueError:
                    pass

        def _drain_stdout() -> None:
            assert process is not None and process.stdout is not None
            try:
                for line in iter(process.stdout.readline, ""):
                    stdout_chunks.append(line)
                    stripped = line.rstrip()
                    if stripped:
                        _parse_progress_line(stripped)
                        if not stripped.startswith("VACE_SERVER_PROGRESS"):
                            logger.info("[vace] %s", stripped)
            finally:
                process.stdout.close()

        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.vace_root),
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            reader = threading.Thread(target=_drain_stdout, daemon=True)
            reader.start()
            try:
                rc = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=15)
                except Exception:
                    pass
                with self._progress_lock:
                    self._progress["active"] = False
                    self._progress["phase"] = "timeout"
                    self._progress["message"] = f"timed out after {timeout_seconds}s"
                    self._progress["updated_at"] = time.time()
                reader.join(timeout=5)
                full_out = "".join(stdout_chunks)
                return {
                    "success": False,
                    "error": f"VACE command timed out after {timeout_seconds}s",
                    "stdout": full_out,
                    "stderr": "",
                }
            reader.join(timeout=30)
            full_out = "".join(stdout_chunks)
        except Exception as exc:
            with self._progress_lock:
                self._progress["active"] = False
                self._progress["phase"] = "error"
                self._progress["message"] = str(exc)
                self._progress["updated_at"] = time.time()
            return {"success": False, "error": f"Failed to run VACE command: {exc}"}

        with self._progress_lock:
            self._progress["active"] = False
            if rc == 0:
                self._progress["phase"] = "done"
                self._progress["message"] = "completed"
            else:
                self._progress["phase"] = "error"
                self._progress["message"] = f"exit code {rc}"
            self._progress["updated_at"] = time.time()

        if rc != 0:
            err_msg = "VACE command failed"
            if rc == -9:
                err_msg = (
                    "VACE subprocess killed (SIGKILL, often OOM). "
                    "Avoid parallel VACE jobs; the server serializes /infer but the client must not flood requests."
                )
            return {
                "success": False,
                "error": err_msg,
                "return_code": rc,
                "stdout": full_out,
                "stderr": "",
            }

        latest_dir = self._latest_result_dir(results_dir)
        output_video = self._resolve_output_video(results_dir, before_latest, latest_dir)
        if output_video is None:
            return {
                "success": False,
                "error": "VACE succeeded but output video not found under results/",
                "stdout": full_out,
            }

        return {
            "success": True,
            "output_path": str(output_video),
            "result_dir": str(output_video.parent),
            "stdout": full_out,
            "stderr": "",
        }

    @staticmethod
    def _latest_result_dir(results_dir: Path) -> Optional[Path]:
        if not results_dir.exists():
            return None
        dirs = [p for p in results_dir.iterdir() if p.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _resolve_output_video(
        results_dir: Path, before_latest: Optional[Path], after_latest: Optional[Path]
    ) -> Optional[Path]:
        candidate_dirs = []
        if after_latest is not None:
            candidate_dirs.append(after_latest)
        if before_latest is not None and before_latest not in candidate_dirs:
            candidate_dirs.append(before_latest)

        for result_dir in candidate_dirs:
            out_path = result_dir / "out_video.mp4"
            if out_path.exists():
                return out_path.resolve()

        if results_dir.exists():
            matches = list(results_dir.glob("**/out_video.mp4"))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime).resolve()
        return None


runner: Optional[VaceRunner] = None


@app.route("/health", methods=["GET"])
def health_check():
    global runner
    if runner is None:
        return jsonify({"status": "unhealthy", "error": "Runner not initialized"}), 500
    ready, message = runner.is_ready()
    deps_ok, deps_message = runner.check_inference_runtime_deps()
    overall_ok = ready and deps_ok
    return jsonify(
        {
            "status": "healthy" if overall_ok else "unhealthy",
            "ready": ready,
            "message": message,
            "runtime_deps_ok": deps_ok,
            "runtime_deps_message": deps_message,
            "vace_root": str(runner.vace_root),
            "checkpoint_path": str(runner.checkpoint_path),
            "health_python_executable": sys.executable,
            "health_python_version": sys.version.split()[0],
        }
    )


@app.route("/progress", methods=["GET"])
def inference_progress():
    """Poll denoise step while a job is running (see run_firstframe streaming + wan_vace progress lines)."""
    global runner
    if runner is None:
        return jsonify({"error": "Runner not initialized"}), 500
    with runner._progress_lock:
        payload = dict(runner._progress)
    step = int(payload.get("denoise_step") or 0)
    total = int(payload.get("denoise_total") or 0)
    payload["denoise_percent"] = round(100.0 * step / total, 2) if total > 0 else 0.0
    return jsonify(payload)


@app.route("/test", methods=["POST"])
def test_infer():
    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image")
    prompt = payload.get("prompt", "A cinematic shot.")
    if not image_path:
        return jsonify({"success": False, "error": "Missing required field: image"}), 400
    result = infer_internal(image_path=image_path, prompt=prompt)
    return jsonify(result), (200 if result.get("success") else 500)


@app.route("/infer", methods=["POST"])
def infer():
    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image") or payload.get("image_path")
    prompt = payload.get("prompt")

    if not image_path:
        return jsonify({"success": False, "error": "Missing required field: image/image_path"}), 400
    if not prompt:
        return jsonify({"success": False, "error": "Missing required field: prompt"}), 400

    result = infer_internal(
        image_path=image_path,
        prompt=prompt,
        base=payload.get("base", "wan"),
        task=payload.get("task", "frameref"),
        mode=payload.get("mode", "firstframe"),
        timeout_seconds=int(payload.get("timeout_seconds", 1800)),
        extra_args=payload.get("extra_args"),
    )
    return jsonify(result), (200 if result.get("success") else 500)


def infer_internal(
    image_path: str,
    prompt: str,
    base: str = "wan",
    task: str = "frameref",
    mode: str = "firstframe",
    timeout_seconds: int = 1800,
    extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    global runner
    if runner is None:
        return {"success": False, "error": "Runner not initialized"}
    return runner.run_firstframe(
        image_path=image_path,
        prompt=prompt,
        base=base,
        task=task,
        mode=mode,
        timeout_seconds=timeout_seconds,
        extra_args=extra_args,
    )


def parse_args():
    default_vace_root = str(Path(__file__).resolve().parent)
    default_checkpoint_path = str(
        Path(__file__).resolve().parents[3] / "checkpoints" / "vace" / "Wan2.1-VACE-1.3B"
    )
    parser = argparse.ArgumentParser(description="VACE firstframe Flask server")
    parser.add_argument(
        "--vace_root",
        type=str,
        default=default_vace_root,
        help="Path to vendored VACE runtime root",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=default_checkpoint_path,
        help="Path to Wan VACE checkpoint directory",
    )
    parser.add_argument("--port", type=int, default=20034, help="Server port")
    parser.add_argument("--python_exec", type=str, default="python", help="Python executable")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = VaceRunner(
        vace_root=args.vace_root,
        checkpoint_path=args.checkpoint_path,
        python_exec=args.python_exec,
    )
    ready, message = runner.is_ready()
    logger.info(
        "Starting VACE server, root=%s, checkpoint_path=%s",
        args.vace_root,
        args.checkpoint_path,
    )
    if not ready:
        logger.warning("VACE environment check failed: %s", message)
    # threaded=True so /progress can be polled while /infer blocks on subprocess.
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
