#!/bin/sh
# Wipe host log copies and Loki chunks when docker compose down sends SIGTERM.
# Model weights in ./models are NOT touched.
set -eu

wipe() {
  echo "log-janitor: wiping /logs on shutdown"
  find /logs -mindepth 1 -exec rm -rf {} + 2>/dev/null || true
  mkdir -p /logs
  echo "log-janitor: /logs is empty"
}

trap 'wipe; exit 0' TERM INT

echo "log-janitor: idle until compose down"
while true; do
  sleep 3600 &
  wait $! || true
done
