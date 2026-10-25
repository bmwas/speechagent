# syntax=docker/dockerfile:1
#
# Build context is ./native-worker (always present in this repo).
# OpenBMB MiniCPM-o-Demo is cloned at the SHA in source.pin so
# `docker compose build` does not need the git submodule checked out.
# Runtime still uses native-worker/entrypoint.sh (HF snapshot mount).
#
# Keep the pip/torch layers in sync with
# third_party/MiniCPM-o-Demo/docker/Dockerfile.worker-backend.

FROM python:3.12-slim-bookworm AS src
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY source.pin /tmp/source.pin
WORKDIR /src
RUN set -a && . /tmp/source.pin && set +a \
    && git clone "${MINICPM_REPO}" . \
    && git checkout --detach "${MINICPM_SHA}"

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    TORCHINDUCTOR_CACHE_DIR=/app/torch_compile_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=src /src/requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 torchaudio==2.8.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "minicpmo-utils[all]>=1.0.5" \
    && grep -vE '^[[:space:]]*minicpmo-utils(\[all\])?>=' requirements.txt > /tmp/requirements-worker.txt \
    && pip install -r /tmp/requirements-worker.txt \
    && rm /tmp/requirements-worker.txt

COPY --from=src /src /app

RUN chmod +x docker/entrypoint-worker-backend.sh \
    && mkdir -p /app/data /app/tmp /app/torch_compile_cache

EXPOSE 22500
EXPOSE 22400

ENTRYPOINT ["docker/entrypoint-worker-backend.sh"]
