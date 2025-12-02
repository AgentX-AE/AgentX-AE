#!/bin/bash
# run_real_latency.sh
# Usage:
#   bash run_real_latency.sh <model_name> <model_path> <model_size> [gpu_id]
#
# Example:
#   bash run_real_latency.sh qwen3-32b /data1/Qwen3-32B 32B 0

chmod +x run_real_latency.sh

MODEL_NAME="$1"
MODEL_PATH="$2"
MODEL_SIZE="$3"
GPU_ID="${4:-0}"   # default to GPU 0 if not provided

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ] || [ -z "$MODEL_SIZE" ]; then
  echo "Usage: $0 <model_name> <model_path> <model_size> [gpu_id]"
  exit 1
fi

# 1) Set vLLM / FlashAttention related environment variables
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_FLASH_ATTN_VERSION=3      # use FlashAttention v3
unset VLLM_USE_TRITON_FLASH_ATTN      # make sure Triton FA is not forced

# Optional: set log level
export VLLM_LOG_LEVEL=INFO

# 2) Run the Python latency test script
LOG_FILE="latency_${MODEL_NAME}_gpu${GPU_ID}.log"

echo "Running REAL latency test for model: ${MODEL_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Model size: ${MODEL_SIZE}"
echo "GPU: ${GPU_ID}"
echo "Log file: ${LOG_FILE}"
echo "------------------------------------------------------"

python real_vllmtest.py \
  --name "${MODEL_NAME}" \
  --path "${MODEL_PATH}" \
  --model_size "${MODEL_SIZE}" \
  --device "${GPU_ID}" 2>&1 | tee "${LOG_FILE}"
