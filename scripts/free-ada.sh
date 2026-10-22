#!/bin/sh
# Stop anything using the Ada GPU (host nvidia-smi index ADA_GPU_INDEX, default 1)
# so MiniCPM-o 4.5 can load. Does not touch Blackwell (index 0 / sglang).
# Run on the GPU box via: docker compose run --rm gpu-free
# or automatically before model-server on compose up.
set -eu

IDX="${ADA_GPU_INDEX:-1}"
KEEP_RE="${KEEP_CONTAINER_REGEX:-speechagent-model-server|speechagent-gpu-free|speechagent-gpu-stats}"
MAX_WAIT="${ADA_FREE_WAIT_SECS:-45}"
MAX_USED_MB="${ADA_FREE_MAX_USED_MB:-1024}"

nvsmi() {
  # Host nvidia-smi is glibc; this image is Alpine. Enter the host PID/mount
  # namespace so we see real GPU indices and host PIDs we can signal.
  if command -v nsenter >/dev/null 2>&1; then
    nsenter --target 1 --mount --uts --ipc --net --pid nvidia-smi "$@"
    return
  fi
  if command -v apk >/dev/null 2>&1; then
    apk add --no-cache util-linux >/dev/null
    nsenter --target 1 --mount --uts --ipc --net --pid nvidia-smi "$@"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi "$@"
    return
  fi
  echo "gpu-free FATAL: cannot run nvidia-smi on the host" >&2
  exit 1
}

csv1() {
  # One csv,noheader field; strip spaces. nvidia-smi -i already selected the GPU.
  nvsmi "$@" | head -n1 | tr -d '[:space:]'
}

if [ "${FREE_ADA_GPU:-1}" = "0" ]; then
  echo "gpu-free skipped (FREE_ADA_GPU=0)"
  exit 0
fi

echo "gpu-free target nvidia-smi index ${IDX}"
nvsmi -L || true
nvsmi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv || true

UUID=$(csv1 -i "$IDX" --query-gpu=uuid --format=csv,noheader)
if [ -z "${UUID}" ]; then
  echo "gpu-free FATAL: no GPU at index ${IDX}" >&2
  nvsmi -L >&2 || true
  exit 1
fi
echo "gpu-free Ada uuid=${UUID}"

uses_ada() {
  cid="$1"
  req=$(docker inspect -f '{{json .HostConfig.DeviceRequests}}' "$cid" 2>/dev/null || echo null)
  envs=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$cid" 2>/dev/null || true)
  nv=$(printf '%s\n' "$envs" | sed -n 's/^NVIDIA_VISIBLE_DEVICES=//p' | head -n1)
  cvd=$(printf '%s\n' "$envs" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n1)

  echo "$req" | grep -q "\"${IDX}\"" && return 0
  echo "$req" | grep -Fq "${UUID}" && return 0
  case "$nv" in
    "${IDX}"|"${UUID}") return 0 ;;
  esac
  if echo "$req" | grep -q '"Count":-1'; then
    case "$cvd" in
      "${IDX}") return 0 ;;
    esac
  fi
  case "$nv" in
    all|"")
      case "$cvd" in
        "${IDX}") return 0 ;;
      esac
      ;;
  esac
  return 1
}

stop_list=""
for cid in $(docker ps -q); do
  name=$(docker inspect -f '{{.Name}}' "$cid" 2>/dev/null || echo "")
  name=${name#/}
  case "$name" in
    "") continue ;;
  esac
  echo "$name" | grep -Eq "$KEEP_RE" && continue
  if uses_ada "$cid"; then
    echo "gpu-free stopping container ${name} (${cid}) — holds Ada"
    stop_list="${stop_list} ${cid}"
  fi
done

if [ -n "${stop_list}" ]; then
  # docker stop is a manual stop: restart=unless-stopped will stay down.
  # shellcheck disable=SC2086
  docker stop -t 20 ${stop_list} || true
else
  echo "gpu-free no other Docker containers reserved Ada"
fi

keep_proc() {
  case "$1" in
    nvidia-persistenced|nv-hostengine|nvidia-cuda-mps-control|nvidia-cuda-mps-server|Xorg|X|gnome-shell|gdm|gdm-x-session|dockerd|containerd|containerd-shim-runc-v2)
      return 0 ;;
  esac
  return 1
}

i=0
while [ "$i" -lt 3 ]; do
  nvsmi -i "$IDX" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F',' '{
        pid=$1; name=$2
        gsub(/ /,"",pid)
        gsub(/^ +| +$/,"",name)
        if (pid ~ /^[0-9]+$/) print pid, name
      }' | while read -r pid pname; do
          base=$(basename "$pname")
          if keep_proc "$base"; then
            echo "gpu-free keep pid=${pid} ${base}"
            continue
          fi
          echo "gpu-free SIGTERM pid=${pid} ${base} (Ada compute)"
          kill -TERM "$pid" 2>/dev/null || true
        done
  i=$((i + 1))
  sleep 2
done

nvsmi -i "$IDX" --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
  | awk -F',' '{
      pid=$1; name=$2
      gsub(/ /,"",pid)
      gsub(/^ +| +$/,"",name)
      if (pid ~ /^[0-9]+$/) print pid, name
    }' | while read -r pid pname; do
      base=$(basename "$pname")
      keep_proc "$base" && continue
      echo "gpu-free SIGKILL pid=${pid} ${base}"
      kill -KILL "$pid" 2>/dev/null || true
    done

waited=0
used=99999
while [ "$waited" -lt "$MAX_WAIT" ]; do
  used=$(csv1 -i "$IDX" --query-gpu=memory.used --format=csv,noheader,nounits)
  used=$(echo "$used" | tr -cd '0-9')
  used=${used:-99999}
  echo "gpu-free Ada memory.used=${used} MiB (want <= ${MAX_USED_MB})"
  if [ "$used" -le "$MAX_USED_MB" ]; then
    nvsmi -i "$IDX" --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv
    echo "gpu-free Ada is clear"
    exit 0
  fi
  sleep 2
  waited=$((waited + 2))
done

echo "gpu-free FATAL: Ada still has ${used} MiB in use after ${MAX_WAIT}s" >&2
nvsmi -i "$IDX" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true
exit 1
