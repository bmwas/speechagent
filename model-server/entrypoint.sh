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
_model_lc = os.environ.get("MODEL_NAME", "").lower()
if "minicpm" in _model_lc:
    need = 12 * 1024**3
elif "qwen3-omni" in _model_lc or "qwen3_omni" in _model_lc:
    # 4-bit thinker (~18 GiB) + BF16 talker/code2wav + KV share this card.
    need = 20 * 1024**3
else:
    need = 6 * 1024**3
if free < need:
    print(
        f"model-server FATAL: only {free/1024**3:.2f} GiB free on {name}. "
        "Stop msestream interviewer-engine on this Ada GPU or switch MODEL_NAME "
        "to Qwen/Qwen2.5-Omni-3B.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

ensure_mod() {
  local mod="$1"
  python -c "import ${mod}" >/dev/null 2>&1
}

# MiniCPM-o 4.5 Code2Wav (stage 2) imports in-tree Step-Audio2 Token2wav,
# which needs s3tokenizer + onnxruntime + hyperpyyaml. The public
# vllm/vllm-omni:v0.28.0 image does not ship them, so stage 2 crashes,
# the orchestrator never serves :8199, and the agent drops every mic
# chunk as stream.drop_not_ready.
# `step_audio2` (PyPI, wheels cosyvoice2 + flashcosyvoice) provides the
# flow class `cosyvoice2.flow.flow.CausalMaskedDiffWithXvec` and
# `flashcosyvoice.modules.hifigan.HiFTGenerator` that vllm-omni's in-tree
# StepAudio2 Token2wav imports. Without it, hyperpyyaml cannot resolve the
# !new:cosyvoice2... tag in assets/token2wav/flow.yaml and stage 2 dies with
#   ImportError: There is no such class as cosyvoice2.flow.flow.CausalMaskedDiffWithXvec
install_minicpm_vocoder() {
  local missing=()
  ensure_mod s3tokenizer || missing+=(s3tokenizer)
  ensure_mod onnxruntime || missing+=(onnxruntime)
  ensure_mod hyperpyyaml || missing+=(hyperpyyaml)
  ensure_mod flashcosyvoice || missing+=(step_audio2)
  if [ "${#missing[@]}" -eq 0 ]; then
    echo "model-server MiniCPM vocoder imports ok (s3tokenizer, onnxruntime, hyperpyyaml, step_audio2/flashcosyvoice)"
    return 0
  fi
  echo "model-server installing MiniCPM-o Code2Wav deps: ${missing[*]}"
  mkdir -p /models/pip
  python -m pip install \
    --cache-dir /models/pip \
    --disable-pip-version-check \
    --no-warn-script-location \
    "s3tokenizer>=0.3.0" \
    "onnxruntime>=1.17.0" \
    "hyperpyyaml" \
    "step_audio2"
  local fail=0
  ensure_mod s3tokenizer || { echo "model-server FATAL: s3tokenizer still missing after pip install." >&2; fail=1; }
  ensure_mod flashcosyvoice || { echo "model-server FATAL: flashcosyvoice still missing after pip install step_audio2." >&2; fail=1; }
  if ! python -c "from cosyvoice2.flow.flow import CausalMaskedDiffWithXvec" >/dev/null 2>&1; then
    echo "model-server FATAL: cosyvoice2.flow.flow.CausalMaskedDiffWithXvec not importable (needed by assets/token2wav/flow.yaml)." >&2
    fail=1
  fi
  if [ "$fail" -ne 0 ]; then
    echo "model-server FATAL: Stage-2 Code2Wav cannot start; deps above unresolved." >&2
    exit 1
  fi
  echo "model-server MiniCPM vocoder imports ok after install"
}

# v0.30.0 MiniCPM-o live-duplex tunings that are not YAML knobs:
# 1) HiFT max_lazy_graphs is hardcoded at 8. After 8 unseen chunk
#    shapes, Code2Wav falls back to eager HiFT (5-15 s holes).
# 2) Predefined HiFT buckets are only (48,0)/(56,3840). Live first/last
#    chunks use even sizes 10-54; each new shape recaptures mid-utterance.
# 3) A Talker ledger miss raises and kills the duplex session; skip
#    that one handoff instead so the socket stays up.
tune_minicpm_omni() {
  python - <<'PY'
from pathlib import Path

roots = [
    Path("/usr/local/lib/python3.12/dist-packages/vllm_omni"),
    Path("/usr/local/lib/python3.11/dist-packages/vllm_omni"),
]


def patch_one(path: Path, needle: str, repl: str, label: str) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if repl in text and needle not in text:
        print(f"model-server {label} already applied ({path})")
        return
    if needle not in text:
        print(f"model-server skip {label}: pattern not in {path}")
        return
    path.write_text(text.replace(needle, repl, 1), encoding="utf-8")
    print(f"model-server applied {label} ({path})")


def wrap_talker_handoff(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "skipping talker handoff" in text:
        print(f"model-server talker-handoff skip already applied ({path})")
        return
    marker = "tts_hidden_slice = _native_duplex_forwarded_hidden_rows("
    idx = text.find(marker)
    if idx < 0:
        print(f"model-server skip talker-handoff wrap: call not in {path}")
        return
    line_start = text.rfind("\n", 0, idx) + 1
    indent = text[line_start:idx]
    paren = text.find("(", idx)
    depth = 0
    end = None
    for j, ch in enumerate(text[paren:], start=paren):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end is None:
        print(f"model-server skip talker-handoff wrap: unterminated call in {path}")
        return
    call = text[idx:end]
    wrapped = (
        f"{indent}try:\n"
        f"{indent}    {call}\n"
        f"{indent}except ValueError as exc:\n"
        f"{indent}    logger.warning(\n"
        f'{indent}        "MiniCPM-o native duplex: skipping talker handoff (%s)",\n'
        f"{indent}        exc,\n"
        f"{indent}    )\n"
        f"{indent}    tts_token_ids_slice = None\n"
        f"{indent}    tts_hidden_slice = None"
    )
    path.write_text(text[:line_start] + wrapped + text[end:], encoding="utf-8")
    print(f"model-server applied talker-handoff skip ({path})")


for root in roots:
    hift = root / "model_executor/models/minicpmo_4_5/cuda_graph_wrapper.py"
    patch_one(
        hift,
        "self.max_lazy_graphs = 8",
        "self.max_lazy_graphs = 32",
        "hift-lazy-graphs-32",
    )
    patch_one(
        hift,
        """    def capture(self):
        for batch_size in self.capture_batch_sizes:
            for mel_frames, source_cache_len in zip(
                self.capture_bucket_size,
                self.capture_source_cache_len,
                strict=True,
            ):
                self._capture(batch_size, mel_frames, source_cache_len)
""",
        """    def capture(self):
        # Live first/last codec chunks miss the 48/56 body buckets and
        # were recaptured mid-utterance. Warm even 10-56 at boot instead.
        frames = sorted(set(int(x) for x in self.capture_bucket_size) | set(range(10, 57, 2)))
        caches = sorted(set(int(x) for x in self.capture_source_cache_len))
        for batch_size in self.capture_batch_sizes:
            for mel_frames in frames:
                for source_cache_len in caches:
                    try:
                        self._capture(batch_size, mel_frames, source_cache_len)
                    except Exception as exc:
                        logger.warning(
                            "HiFT warmup skip shape (%s, %s, %s): %s",
                            batch_size,
                            mel_frames,
                            source_cache_len,
                            exc,
                        )
""",
        "hift-warmup-even-frames",
    )
    wrap_talker_handoff(
        root / "model_executor/stage_input_processors/minicpmo_4_5_omni.py"
    )
PY
}

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

# v0.30.0 talker crashes on the first /v1/video/chat/stream query:
# sampling_metadata.prompt_token_ids is None, the pad-id remap TypeErrors,
# and the orchestrator evicts stage-1 forever ("Stage-1 has no live replica").
# Fail startup if the line cannot be patched — a silent miss boots a mute bot.
patch_qwen25_talker_pad_remap() {
  python /deploy/patch_qwen_talker.py
}

DEPLOY=""
case "${MODEL}" in
  *MiniCPM*|*minicpm*)
    DEPLOY="$(find_yaml minicpmo_4_5.yaml || true)"
    install_minicpm_vocoder
    tune_minicpm_omni
    ;;
  *Qwen3-Omni*|*qwen3-omni*|*Qwen3_Omni*|*qwen3_omni*)
    # Qwen3-Omni MoE. Realtime /v1/video/chat/stream is the Qwen3-Omni path
    # (Qwen2.5-Omni has no streaming audio). No talker pad-remap patch — that
    # TypeError is Qwen2.5-only. Thinker-only 4-bit checkpoint; stages 1/2
    # keep checkpoint precision (see deploy/qwen3_omni.yaml hf_overrides).
    DEPLOY="$(find_yaml qwen3_omni.yaml || true)"
    ;;
  *Qwen2.5-Omni*|*qwen2.5-omni*|*Qwen2_5*)
    DEPLOY="$(find_yaml qwen2_5_omni.yaml || true)"
    patch_qwen25_talker_pad_remap
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
  python - "${DEPLOY}" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    import yaml
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    eager = [(s.get("stage_id"), s.get("enforce_eager"), s.get("async_scheduling")) for s in d.get("stages") or []]
    print("model-server deploy stages (id, enforce_eager, async_scheduling):", eager)
except Exception as exc:
    print("model-server deploy parse failed:", exc)
PY
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
