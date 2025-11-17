#!/bin/bash

# vLLM deployment script for Qwen3-VL-4B-Instruct
# Usage: bash run_qwen.sh [--host HOST] [--port PORT] [--tensor-parallel-size N]

# Default parameters
MODEL_PATH="/data/wyh/pretrained_weights/Qwen3-VL-4B-Instruct"
HOST="0.0.0.0"
PORT=8000
TENSOR_PARALLEL_SIZE=4
DEVICE="cuda"
GPU_DEVICE=3,4,5,6

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --gpu-device)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--host HOST] [--port PORT] [--tensor-parallel-size N] [--gpu-device N]"
      echo ""
      echo "Options:"
      echo "  --host HOST               Host to bind to (default: 0.0.0.0)"
      echo "  --port PORT               Port to bind to (default: 8000)"
      echo "  --tensor-parallel-size N  Number of GPUs to use (default: 1)"
      echo "  --gpu-device N            GPU device ID to use (default: 0)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path does not exist: $MODEL_PATH"
  exit 1
fi

# Print configuration
echo "========================================="
echo "vLLM Qwen Deployment Configuration"
echo "========================================="
echo "Model Path:       $MODEL_PATH"
echo "Host:             $HOST"
echo "Port:             $PORT"
echo "Tensor Parallel:  $TENSOR_PARALLEL_SIZE"
echo "Device:           $DEVICE"
echo "GPU Device:       $GPU_DEVICE"
echo "========================================="
echo ""

# Set CUDA_VISIBLE_DEVICES to use only the specified GPU
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE

# Launch vLLM API server
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --served-model-name "Qwen3-VL-4B-Instruct" \
  --trust-remote-code \
  --gpu-memory-utilization 0.9

