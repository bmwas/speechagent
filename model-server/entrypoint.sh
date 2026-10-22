#!/usr/bin/env bash
# Select MiniCPM-o 4.5 (duplex) or Qwen2.5-Omni-3B (turn) from MODEL_NAME.
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

echo "model-server MODEL=${MODEL} port=${PORT} HF_HOME=${HF_HOME}"

# Prefer an in-image deploy yaml when it exists; otherwise --omni auto-detects.
DEPLOY=""
case "${MODEL}" in
  *MiniCPM*|*minicpm*)
    for cand in \
      /usr/local/lib/python3.*/site-packages/vllm_omni/deploy/minicpmo_4_5.yaml \
      /opt/vllm/vllm_omni/deploy/minicpmo_4_5.yaml
    do
      if [ -f $cand ]; then
        DEPLOY="$(ls $cand | head -n1)"
        break
      fi
    done
    ;;
  *Qwen2.5-Omni*|*qwen2.5-omni*|*Qwen2_5*)
    for cand in \
      /usr/local/lib/python3.*/site-packages/vllm_omni/deploy/qwen2_5_omni.yaml \
      /opt/vllm/vllm_omni/deploy/qwen2_5_omni.yaml
    do
      if [ -f $cand ]; then
        DEPLOY="$(ls $cand | head -n1)"
        break
      fi
    done
    ;;
esac

if [ -n "${DEPLOY}" ]; then
  echo "model-server using deploy-config ${DEPLOY}"
  exec vllm serve "${MODEL}" --omni --host "${HOST}" --port "${PORT}" \
    --trust-remote-code --deploy-config "${DEPLOY}"
fi

echo "model-server no deploy-config found; --omni auto"
exec vllm serve "${MODEL}" --omni --host "${HOST}" --port "${PORT}" --trust-remote-code
