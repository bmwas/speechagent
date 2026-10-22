#!/bin/sh
# Render keys + node_ip/turn.domain from the same env the agent uses.
set -eu
KEY="${LIVEKIT_API_KEY:?LIVEKIT_API_KEY is required}"
SECRET="${LIVEKIT_API_SECRET:?LIVEKIT_API_SECRET is required}"
NODE_IP="${LIVEKIT_NODE_IP:-${MSE_LAN_IP:-}}"

if [ -z "$NODE_IP" ]; then
  NODE_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}') || true
fi
if [ -z "$NODE_IP" ]; then
  NODE_IP=$(hostname -i 2>/dev/null | awk '{print $1}') || true
fi
case "$NODE_IP" in
  127.*|::1|"") NODE_IP="" ;;
esac

if [ -n "$NODE_IP" ]; then
  USE_EXT="false"
  NODE_LINE="  node_ip: \"${NODE_IP}\""
  TURN_DOMAIN="  domain: \"${NODE_IP}\""
else
  USE_EXT="true"
  NODE_LINE=""
  TURN_DOMAIN=""
fi

cat > /tmp/livekit.yaml <<EOF
port: 7880
bind_addresses:
  - "0.0.0.0"
log_level: info
rtc:
  tcp_port: 7881
  port_range_start: 50000
  port_range_end: 50100
  use_external_ip: ${USE_EXT}
${NODE_LINE}
room:
  empty_timeout: 300
  max_participants: 10
turn:
  enabled: true
${TURN_DOMAIN}
  udp_port: 7882
  tls_port: 0
  external_tls: false
keys:
  ${KEY}: ${SECRET}
EOF

echo "livekit config node_ip=${NODE_IP:-'(probe)'} use_external_ip=${USE_EXT}"

BIN=livekit-server
if ! command -v livekit-server >/dev/null 2>&1; then
  BIN=/livekit-server
fi
exec "$BIN" --config /tmp/livekit.yaml --keys "${KEY}: ${SECRET}"
