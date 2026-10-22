#!/usr/bin/env bash
# Run on the Precision (GPU box), from the speechagent repo, then paste stdout.
#   cd ~/Downloads/speechagent && bash scripts/diag-remote.sh
set -u
cd "$(dirname "$0")/.." || exit 1
echo "===== host ====="
hostname; date -u; pwd
echo "===== compose still running? ====="
pgrep -af 'docker compose|docker-compose' || echo '(no compose process)'
echo "===== images (vllm / speechagent / python / node) ====="
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
  | grep -Ei 'vllm|speechagent|python|node:|nginx' || echo '(none matching)'
echo "===== compose ps ====="
docker compose ps -a || true
echo "===== model-server inspect ====="
docker compose ps -a --format json model-server 2>/dev/null | head -c 4000; echo
echo "===== last 80 lines model-server ====="
docker compose logs --tail=80 --no-color model-server 2>&1 || true
echo "===== last 40 lines agent / livekit / frontend ====="
docker compose logs --tail=40 --no-color agent livekit frontend 2>&1 || true
echo "===== docker events last 10m (pull/create) ====="
docker events --since 10m --until 0s --filter type=image --filter type=container 2>/dev/null | tail -n 40 || true
echo "===== disk ====="
df -h / /var/lib/docker 2>/dev/null | head
echo "===== nvidia ====="
nvidia-smi -L 2>&1 | head
echo "===== HUGGINGFACE_TOKEN in this shell? (value hidden) ====="
if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then echo "HUGGINGFACE_TOKEN=set len=${#HUGGINGFACE_TOKEN}"; else echo "HUGGINGFACE_TOKEN=UNSET"; fi
echo "===== grafana :3300 (not msestream :3000) ====="
curl -sS -m 5 http://127.0.0.1:3300/api/health || true
echo
curl -sS -m 5 http://127.0.0.1:3300/api/search || true
echo
echo "===== ports ====="
ss -lntup 2>/dev/null | grep -E ':6173|:8860|:8880|:8199|:3300|:3310' || netstat -lntup 2>/dev/null | grep -E ':6173|:8860|:8880|:8199|:3300|:3310' || true
echo "===== done ====="
