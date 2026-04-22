# VACE Expert (Vendored Runtime)

This folder now contains a vendored VACE runtime so `spagent` can be used as a standalone project.

## What is included

- `vace/` (copied from the VACE project)
  - `vace_pipeline.py`
  - `vace_preproccess.py`
  - `vace_wan_inference.py`
  - required `annotators/`, `configs/`, `models/` runtime code
- `vace_server.py` (Flask service wrapper)
- `vace_client.py` (HTTP client wrapper)

## What is NOT included

- Model weights/checkpoints (very large files)

You need to prepare checkpoints manually under repository `checkpoints/`, e.g.:

- `checkpoints/vace/Wan2.1-VACE-1.3B/`

## External dependency: `import wan`

`vace_wan_inference.py` imports the **Wan2.1** Python package (`wan`).

**Recommended:** clone the upstream repo so `wan` is on `PYTHONPATH` without pip (avoids `flash_attn` build during `pip install wan`):

```bash
mkdir -p spagent/external_experts/vace/third_party
git clone https://github.com/Wan-Video/Wan2.1.git \
  spagent/external_experts/vace/third_party/Wan2.1
```

`vace_server.py` prepends `third_party/Wan2.1` to `PYTHONPATH` when that directory exists.

**Alternative:** `pip install "wan @ git+https://github.com/Wan-Video/Wan2.1"` if your environment can satisfy that package’s build (e.g. `flash_attn`).

Upstream also lists `flash_attn` in `third_party/Wan2.1/requirements.txt`; install a matching wheel for your CUDA/torch if inference fails without it.

## One-shot setup (copy-paste)

Run this from the `spagent` repository root:

```bash
set -e

# 1) Install VACE extra dependencies
python -m pip install -r requirements-vace.txt

# 2) Ensure HF CLI is available
python -m pip install -U "huggingface_hub[cli]"

# 3) Wan2.1 Python package (clone recommended; see section above)
mkdir -p spagent/external_experts/vace/third_party
git clone https://github.com/Wan-Video/Wan2.1.git \
  spagent/external_experts/vace/third_party/Wan2.1

# 4) Create checkpoints folder used by VACE runtime
mkdir -p checkpoints/vace

# 5) Download Wan2.1-VACE-1.3B weights (required for firstframe with --base wan)
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B \
  --local-dir checkpoints/vace/Wan2.1-VACE-1.3B \
  --local-dir-use-symlinks False

# 6) (Optional but recommended for other VACE tasks) Download annotator assets
huggingface-cli download ali-vilab/VACE-Annotators \
  --local-dir checkpoints/vace/VACE-Annotators \
  --local-dir-use-symlinks False
```

After this, the VACE server/tool can run without your separate local `VACE` project.

### If you see `ModuleNotFoundError` for `diffusers`, `ftfy`, `accelerate`, `einops`, or `decord`

Wan2.1’s `wan` package pulls several PyPI deps while importing (for example **diffusers** in `wan/modules/model.py`, **ftfy** in `wan/modules/tokenizers.py`, **einops** in `wan/modules/vae.py`). The vendored VACE preprocessor also uses **decord** to read intermediate MP4s (`vace/models/utils/preprocessor.py`). Install the full set from `requirements-vace.txt` rather than only one package.

```bash
pip install -r requirements-vace.txt
```

Or install the missing package only, e.g. `pip install ftfy`.

Run this in the **same conda/venv** you use to start `vace_server.py`, then restart the server. `GET /health` reports `runtime_deps_ok` and `runtime_deps_message` when something is still missing.

If logs mention **imageio** / **FFMPEG** / “no backend” for `.mp4`, install **`imageio-ffmpeg`** (or `pip install 'imageio[ffmpeg]'`) in the same environment as the server. The preprocessor’s `save_one_video` will **fall back to OpenCV** when imageio cannot write MP4, so a working install is still recommended for consistency. **decord** is only used to read those MP4s afterward.

## Start server

```bash
python spagent/external_experts/vace/vace_server.py \
  --checkpoint_path checkpoints/vace/Wan2.1-VACE-1.3B \
  --port 20034
```

By default, the server uses the current folder as `--vace_root`, and uses `checkpoints/vace/Wan2.1-VACE-1.3B` as `--checkpoint_path`.
