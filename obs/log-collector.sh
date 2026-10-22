#!/bin/sh
# Copy Compose container logs onto the host bind ./logs.
set -eu
mkdir -p /logs /logs/.dumped /logs/loki
PROJECT="${PROJECT:-speechagent}"
INTERVAL="${INTERVAL:-30}"
MAX_BYTES="${MAX_BYTES:-52428800}"

_trim() {
  path="$1"
  if [ -f "$path" ]; then
    sz=$(wc -c < "$path" 2>/dev/null || echo 0)
    if [ "$sz" -gt "$MAX_BYTES" ]; then
      tail -c "$MAX_BYTES" "$path" > "${path}.tmp" && mv "${path}.tmp" "$path"
    fi
  fi
}

_canon_name() {
  echo "$1" | sed 's#^/##; s/^[0-9a-f]\{12\}_//'
}

echo "log-collector project=${PROJECT} interval=${INTERVAL}s"

while true; do
  DATE=$(date -u +%Y%m%d)

  docker ps --filter "label=com.docker.compose.project=${PROJECT}" \
            --format '{{.ID}}|{{.Names}}' 2>/dev/null \
  | while IFS='|' read -r id raw; do
    [ -z "$id" ] || [ -z "$raw" ] && continue
    name=$(_canon_name "$raw")
    case "$name" in
      *log-collector*|*log-janitor*) continue ;;
    esac
    out="/logs/${name}-${DATE}.log"
    docker logs --since "${INTERVAL}s" --timestamps "$id" >> "$out" 2>&1 || true
    _trim "$out"
  done

  docker ps -a --filter "label=com.docker.compose.project=${PROJECT}" \
               --filter "status=exited" \
               --format '{{.ID}}|{{.Names}}' 2>/dev/null \
  | while IFS='|' read -r id raw; do
    [ -z "$id" ] || [ -z "$raw" ] && continue
    name=$(_canon_name "$raw")
    case "$name" in
      *log-collector*|*log-janitor*) continue ;;
    esac
    marker="/logs/.dumped/${id}"
    [ -f "$marker" ] && continue
    out="/logs/${name}-${DATE}.log"
    docker logs --timestamps "$id" >> "$out" 2>&1 || true
    : > "$marker"
    _trim "$out"
  done

  sleep "$INTERVAL"
done
