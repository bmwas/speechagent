#!/bin/sh
# Render keys + node_ip/turn.domain from the same env the agent uses.
# Ports are offset from msestream (7880/7881/7882 / UDP 50000-50100).
set -eu
KEY="${LIVEKIT_API_KEY:?LIVEKIT_API_KEY is required}"
SECRET="${LIVEKIT_API_SECRET:?LIVEKIT_API_SECRET is required}"
NODE_IP="${LIVEKIT_NODE_IP:-${MSE_LAN_IP:-}}"
LK_PORT="${LIVEKIT_PORT:-8880}"
LK_TCP="${LIVEKIT_TCP_PORT:-8881}"
LK_TURN="${LIVEKIT_TURN_PORT:-8882}"
LK_RTC0="${LIVEKIT_RTC_PORT_START:-51000}"
LK_RTC1="${LIVEKIT_RTC_PORT_END:-51100}"

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
port: ${LK_PORT}
bind_addresses:
  - "0.0.0.0"
log_level: info
rtc:
  tcp_port: ${LK_TCP}
  port_range_start: ${LK_RTC0}
  port_range_end: ${LK_RTC1}
  use_external_ip: ${USE_EXT}
${NODE_LINE}
room:
  empty_timeout: 300
  max_participants: 10
turn:
  enabled: true
${TURN_DOMAIN}
  udp_port: ${LK_TURN}
  tls_port: 0
  external_tls: false
keys:
  ${KEY}: ${SECRET}
EOF

echo "livekit config port=${LK_PORT} tcp=${LK_TCP} turn=${LK_TURN} rtc=${LK_RTC0}-${LK_RTC1} node_ip=${NODE_IP:-'(probe)'} use_external_ip=${USE_EXT}"

BIN=livekit-server
if ! command -v livekit-server >/dev/null 2>&1; then
  BIN=/livekit-server
fi
exec "$BIN" --config /tmp/livekit.yaml --keys "${KEY}: ${SECRET}"
