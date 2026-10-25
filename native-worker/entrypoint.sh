#!/usr/bin/env bash
# Resolve the HuggingFace snapshot directory, then exec OpenBMB's worker-backend
# entrypoint. Snapshot files are blobs reached via relative symlinks, so the
# whole models--openbmb--MiniCPM-o-4_5 tree must be mounted at WEIGHTS_ROOT.
set -euo pipefail

WEIGHTS_ROOT="${WEIGHTS_ROOT:-/weights}"
FALLBACK_PATH="${MODEL_PATH:-/models/MiniCPM-o-4_5}"

if [ -d "${WEIGHTS_ROOT}/snapshots" ]; then
  snap="$(ls -1 "${WEIGHTS_ROOT}/snapshots" | head -1 || true)"
  if [ -n "${snap}" ] && [ -d "${WEIGHTS_ROOT}/snapshots/${snap}" ]; then
    export MODEL_PATH="${WEIGHTS_ROOT}/snapshots/${snap}"
  fi
fi

if [ ! -d "${MODEL_PATH:-}" ]; then
  if [ -d "${FALLBACK_PATH}" ]; then
    export MODEL_PATH="${FALLBACK_PATH}"
  fi
fi

if [ -z "${MODEL_PATH:-}" ] || [ ! -d "${MODEL_PATH}" ]; then
  echo "[native-worker] MODEL_PATH is missing. Mount the HF cache at ${WEIGHTS_ROOT}" >&2
  echo "[native-worker] expected ${WEIGHTS_ROOT}/snapshots/<hash>/" >&2
  exit 1
fi

echo "[native-worker] MODEL_PATH=${MODEL_PATH}"
cd /app
exec /bin/bash docker/entrypoint-worker-backend.sh
