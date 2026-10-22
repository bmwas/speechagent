#!/usr/bin/env bash
# MiniCPM-o 4.5 (duplex) or Qwen2.5-Omni-3B (turn). GPU-box only.
set -euo pipefail

MODEL="${MODEL_NAME:-openbmb/MiniCPM-o-4_5}"
PORT="${PORT:-8199}"
HOST="${HOST:-0.0.0.0}"
HF_HOME="${HF_HOME:-/models/hf}"
mkdir -p "${HF_HOME}"
TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
if [ -n "${TOKEN}" ]; then
  export HUGGINGFACE_TOKEN="${TOKEN}"
  export HF_TOKEN="${TOKEN}"
  export HUGGING_FACE_HUB_TOKEN="${TOKEN}"
fi
export HF_HOME
unset TOKEN

# nvidia-container-runtime already selected the host GPU. Inside the
# container that card is cuda:0. Leaving NVIDIA_VISIBLE_DEVICES=1 (host
# index) makes StageEngineCoreProc children look for a second device,
# fall back to CPU, and die with "Available KV cache memory: 0.0 GiB".
unset NVIDIA_VISIBLE_DEVICES || true
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "model-server MODEL=${MODEL} port=${PORT} HF_HOME=${HF_HOME}"
echo "model-server CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "model-server FATAL: nvidia-smi missing — image has no GPU runtime" >&2
  exit 1
fi
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv || true

python - <<'PY'
import os, sys
try:
    import torch
except Exception as exc:
    print("model-server FATAL: torch import failed:", exc, file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print(
        "model-server FATAL: torch sees no CUDA device "
        f"(count={torch.cuda.device_count()}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}). "
        "StageEngineCoreProc would start on CPU and die with 0.0 GiB KV cache.",
        file=sys.stderr,
    )
    sys.exit(1)
free, total = torch.cuda.mem_get_info()
name = torch.cuda.get_device_name(0)
print(
    f"model-server torch cuda:0 {name} "
    f"free={free/1024**3:.2f}GiB total={total/1024**3:.2f}GiB"
)
need = 12 * 1024**3 if "minicpm" in os.environ.get("MODEL_NAME", "").lower() else 6 * 1024**3
if free < need:
    print(
        f"model-server FATAL: only {free/1024**3:.2f} GiB free on {name}. "
        "Stop msestream interviewer-engine on this Ada GPU or switch MODEL_NAME "
        "to Qwen/Qwen2.5-Omni-3B.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

find_yaml() {
  local name="$1"
  local cand
  for cand in \
    "/deploy/${name}" \
    /usr/local/lib/python3.12/dist-packages/vllm_omni/deploy/"${name}" \
    /usr/local/lib/python3.11/dist-packages/vllm_omni/deploy/"${name}"
  do
    if [ -f "${cand}" ]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

DEPLOY=""
case "${MODEL}" in
  *MiniCPM*|*minicpm*)
    DEPLOY="$(find_yaml minicpmo_4_5.yaml || true)"
    ;;
  *Qwen2.5-Omni*|*qwen2.5-omni*|*Qwen2_5*)
    DEPLOY="$(find_yaml qwen2_5_omni.yaml || true)"
    ;;
esac

EXTRA=(
  --omni
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
)
if [ -n "${DEPLOY}" ]; then
  echo "model-server using deploy-config ${DEPLOY}"
  EXTRA+=(--deploy-config "${DEPLOY}")
else
  echo "model-server no deploy-config; --omni auto with gpu_memory_utilization"
  EXTRA+=(
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --max-model-len "${MAX_MODEL_LEN:-2048}"
    --enforce-eager
  )
fi

exec vllm serve "${MODEL}" "${EXTRA[@]}"
