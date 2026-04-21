#!/bin/bash

# SGLang deployment script for Sana image generation
# Usage:
#   bash scripts/run_sana_30000.sh
#   bash scripts/run_sana_30000.sh --model-path Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers
#   bash scripts/run_sana_30000.sh --gpu-device 0 --port 30000 --vae-cpu-offload

MODEL_PATH="Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
HOST="0.0.0.0"
PORT=30000
GPU_DEVICE="0"
CONDA_ENV_NAME="Sana"
DISABLE_SOCKS_PROXY=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
case $1 in
  --model-path)
    MODEL_PATH="$2"
    shift 2
    ;;
  --host)
    HOST="$2"
    shift 2
    ;;
  --port)
    PORT="$2"
    shift 2
    ;;
  --gpu-device)
    GPU_DEVICE="$2"
    shift 2
    ;;
  --conda-env)
    CONDA_ENV_NAME="$2"
    shift 2
    ;;
  --keep-socks-proxy)
    DISABLE_SOCKS_PROXY=0
    shift 1
    ;;
  --dit-cpu-offload|--text-encoder-cpu-offload|--vae-cpu-offload|--pin-cpu-memory)
    EXTRA_ARGS+=("$1")
    shift 1
    ;;
  --help)
    echo "Usage: $0 [--model-path MODEL] [--host HOST] [--port PORT] [--gpu-device IDS]"
    echo ""
    echo "Options:"
    echo "  --model-path MODEL              Hugging Face model id or local model path"
    echo "  --host HOST                     Host to bind to (default: 0.0.0.0)"
    echo "  --port PORT                     Port to bind to (default: 30000)"
    echo "  --gpu-device IDS                CUDA_VISIBLE_DEVICES value (default: 0)"
    echo "  --conda-env NAME                Conda env name to use when sglang is not on PATH (default: Sana)"
    echo "  --keep-socks-proxy              Keep ALL_PROXY/all_proxy instead of unsetting them"
    echo "  --dit-cpu-offload               Offload DiT to CPU"
    echo "  --text-encoder-cpu-offload      Offload text encoder to CPU"
    echo "  --vae-cpu-offload               Offload VAE to CPU"
    echo "  --pin-cpu-memory                Pin CPU memory for faster transfer"
    echo ""
    echo "Examples:"
    echo "  bash $0"
    echo "  bash $0 --gpu-device 0 --vae-cpu-offload --text-encoder-cpu-offload"
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    echo "Use --help for usage information"
    exit 1
    ;;
esac
done

SG_LANG_CMD=()
if command -v sglang >/dev/null 2>&1; then
  SG_LANG_CMD=("sglang")
elif [ -x "/home/ubuntu/anaconda3/bin/conda" ]; then
  SG_LANG_CMD=("/home/ubuntu/anaconda3/bin/conda" "run" "-n" "$CONDA_ENV_NAME" "sglang")
else
  echo "Error: 'sglang' command not found, and conda was not available for fallback."
  echo "Either activate the correct environment first or install SGLang."
  echo "Example install: uv pip install 'sglang[diffusion]' --prerelease=allow"
  exit 1
fi

echo "========================================="
echo "Sana SGLang Deployment Configuration"
echo "========================================="
echo "Model Path:       $MODEL_PATH"
echo "Host:             $HOST"
echo "Port:             $PORT"
echo "GPU Device:       $GPU_DEVICE"
echo "Conda Env:        $CONDA_ENV_NAME"
if [ "$DISABLE_SOCKS_PROXY" -eq 1 ]; then
  echo "SOCKS Proxy:      disabled (ALL_PROXY/all_proxy will be unset)"
else
  echo "SOCKS Proxy:      kept from current environment"
fi
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  echo "Extra Args:       ${EXTRA_ARGS[*]}"
else
  echo "Extra Args:       (none)"
fi
echo "========================================="
echo ""

export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
if [ "$DISABLE_SOCKS_PROXY" -eq 1 ]; then
  unset ALL_PROXY
  unset all_proxy
fi

"${SG_LANG_CMD[@]}" serve \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${EXTRA_ARGS[@]}"
