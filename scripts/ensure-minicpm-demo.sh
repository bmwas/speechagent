#!/usr/bin/env bash
# Make sure third_party/MiniCPM-o-Demo has OpenBMB's worker Dockerfile.
# Prefer the git submodule; if that is empty or unavailable, clone the pinned SHA.
# Idempotent. Safe to run from a laptop, the GPU box, or CI.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="${ROOT}/native-worker/source.pin"
# shellcheck disable=SC1090
set -a
# shellcheck source=/dev/null
. "${PIN}"
set +a

DEST="${MINICPM_DEMO_DIR:-${ROOT}/third_party/MiniCPM-o-Demo}"
MARKER="${DEST}/docker/Dockerfile.worker-backend"

if [ -z "${MINICPM_REPO:-}" ] || [ -z "${MINICPM_SHA:-}" ]; then
  echo "[ensure-minicpm-demo] ${PIN} must set MINICPM_REPO and MINICPM_SHA" >&2
  exit 1
fi

if [ -f "${MARKER}" ]; then
  echo "[ensure-minicpm-demo] ready ${MARKER}"
  exit 0
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[ensure-minicpm-demo] git is required to fetch MiniCPM-o-Demo" >&2
  exit 1
fi

mkdir -p "$(dirname "${DEST}")"

if [ -d "${ROOT}/.git" ] && [ -f "${ROOT}/.gitmodules" ]; then
  echo "[ensure-minicpm-demo] git submodule update --init -- ${DEST#${ROOT}/}"
  git -C "${ROOT}" submodule update --init --recursive -- third_party/MiniCPM-o-Demo || true
  if [ -f "${MARKER}" ]; then
    echo "[ensure-minicpm-demo] submodule ready ${MARKER}"
    exit 0
  fi
fi

echo "[ensure-minicpm-demo] cloning ${MINICPM_REPO} @ ${MINICPM_SHA}"
tmp="$(mktemp -d)"
cleanup() { rm -rf "${tmp}"; }
trap cleanup EXIT
git clone "${MINICPM_REPO}" "${tmp}/src"
git -C "${tmp}/src" checkout --detach "${MINICPM_SHA}"
mkdir -p "${DEST}"
# Keep dest/.git if this is a submodule gitlink; only fill the working tree.
tar -C "${tmp}/src" --exclude=.git -cf - . | tar -C "${DEST}" -xf -

if [ ! -f "${MARKER}" ]; then
  echo "[ensure-minicpm-demo] FATAL: ${MARKER} still missing after clone" >&2
  exit 1
fi
echo "[ensure-minicpm-demo] cloned ${MINICPM_SHA} -> ${DEST}"
