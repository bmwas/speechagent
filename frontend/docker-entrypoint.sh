#!/bin/sh
# Generate a self-signed cert so a remote Chrome origin is a secure context.
# getUserMedia is undefined on http://<LAN-IP> (Chrome); https:// even with a
# bench cert is enough after the user clicks through the warning.
set -e
CERT_DIR="${TLS_CERT_DIR:-/app/certs}"
mkdir -p "$CERT_DIR"

detected=""
if command -v ip >/dev/null 2>&1; then
  detected=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | tr '\n' ' ') || true
fi

IPS="${CAPTURE_TLS_IP:-} ${MSE_LAN_IP:-} ${detected}"
SAN="DNS:localhost,DNS:capture-ui,IP:127.0.0.1"
seen=" 127.0.0.1 "
for ip in $IPS; do
  case "$ip" in
    *:*) continue ;;
    127.*) continue ;;
    172.*) continue ;; # docker bridges; including them rotated the cert on every rebuild
    169.254.*) continue ;;
    *[0-9]*)
      case "$seen" in
        *" $ip "*) continue ;;
      esac
      SAN="${SAN},IP:${ip}"
      seen="${seen}${ip} "
      ;;
  esac
done

need_new=0
if [ ! -f "$CERT_DIR/key.pem" ] || [ ! -f "$CERT_DIR/cert.pem" ]; then
  need_new=1
else
  text=$(openssl x509 -in "$CERT_DIR/cert.pem" -noout -text 2>/dev/null || true)
  for ip in $IPS; do
    case "$ip" in
      *:*) continue ;;
      127.*) continue ;;
      172.*) continue ;;
      169.254.*) continue ;;
      *[0-9]*)
        echo "$text" | grep -q "IP Address:${ip}" || need_new=1
        ;;
    esac
  done
fi

if [ "$need_new" = "1" ]; then
  echo "capture-ui TLS SAN=${SAN}"
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 825 \
    -keyout "$CERT_DIR/key.pem" \
    -out "$CERT_DIR/cert.pem" \
    -subj "/CN=speechagent-capture" \
    -addext "subjectAltName=${SAN}"
else
  echo "capture-ui TLS cert already covers ${SAN}"
fi

export TLS_CERT_DIR="$CERT_DIR"
exec "$@"
