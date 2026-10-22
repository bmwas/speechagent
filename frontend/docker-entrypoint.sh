#!/bin/sh
# Generate a self-signed cert so a remote Chrome origin is a secure context.
# getUserMedia is undefined on http://<LAN-IP> (Chrome); https:// even with a
# bench cert is enough after the user clicks through the warning.
set -e
CERT_DIR="${TLS_CERT_DIR:-/app/certs}"
mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_DIR/key.pem" ] || [ ! -f "$CERT_DIR/cert.pem" ]; then
  SAN="DNS:localhost,DNS:capture-ui,IP:127.0.0.1"
  IPS="${CAPTURE_TLS_IP:-${MSE_LAN_IP:-}}"
  for ip in $IPS; do
    case "$ip" in
      *:*) ;; # skip IPv6
      *[0-9]*) SAN="${SAN},IP:${ip}" ;;
    esac
  done
  echo "capture-ui TLS SAN=${SAN}"
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 825 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -subj "/CN=speechagent-capture" \
    -addext "subjectAltName=${SAN}"
fi

export TLS_CERT_DIR="$CERT_DIR"
exec "$@"
