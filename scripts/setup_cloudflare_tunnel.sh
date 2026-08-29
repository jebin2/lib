#!/usr/bin/env bash
set -e

# Cloudflare Tunnel setup — single or multi-service
#
# Usage:
#   bash setup_cloudflare_tunnel.sh [--name TUNNEL_NAME] hostname1:port1 [hostname2:port2 ...]
#
# One-liner:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jebin2/lib/main/scripts/setup_cloudflare_tunnel.sh)" bash ttt.voidall.com:7860
#
# Multi-service:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/jebin2/lib/main/scripts/setup_cloudflare_tunnel.sh)" bash ttt.voidall.com:7860 nvr.voidall.com:2126
#
# Optional --name overrides the auto-derived tunnel name (defaults to first subdomain label).

# ---- Parse args ----
TUNNEL_NAME=""
SERVICES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            TUNNEL_NAME="$2"
            shift 2
            ;;
        *)
            SERVICES+=("$1")
            shift
            ;;
    esac
done

if [[ ${#SERVICES[@]} -eq 0 ]]; then
    echo "❌ No services specified."
    echo ""
    echo "Usage:"
    echo "  bash $0 [--name TUNNEL_NAME] hostname1:port1 [hostname2:port2 ...]"
    echo ""
    echo "Examples:"
    echo "  bash $0 ttt.voidall.com:7860"
    echo "  bash $0 ttt.voidall.com:7860 nvr.voidall.com:2126"
    echo "  bash $0 --name myserver ttt.voidall.com:7860"
    exit 1
fi

# Validate each service is in hostname:port format
for svc in "${SERVICES[@]}"; do
    if [[ "$svc" != *:* ]]; then
        echo "❌ Invalid format '$svc' — expected hostname:port (e.g. ttt.voidall.com:7860)"
        exit 1
    fi
done

# Auto-derive tunnel name from first hostname if not provided
if [[ -z "$TUNNEL_NAME" ]]; then
    FIRST_HOST="${SERVICES[0]%%:*}"
    TUNNEL_NAME="${FIRST_HOST%%.*}"
fi

CLOUDFLARED_DIR="$HOME/.cloudflared"
CONFIG_FILE="$CLOUDFLARED_DIR/config.yml"

echo "📋 Configuration:"
echo "   Tunnel Name : $TUNNEL_NAME"
for svc in "${SERVICES[@]}"; do
    HOST="${svc%%:*}"
    PORT="${svc##*:}"
    echo "   $HOST → localhost:$PORT"
done
echo ""

# ---- Step 1: Login ----
echo "🔐 Step 1: Cloudflare login"
if [ ! -f "$CLOUDFLARED_DIR/cert.pem" ]; then
    cloudflared tunnel login
else
    echo "   ↳ Already logged in (cert.pem exists)"
fi

# ---- Step 2: Create tunnel ----
echo "🚇 Step 2: Creating tunnel: $TUNNEL_NAME"
if cloudflared tunnel list | grep -q "$TUNNEL_NAME"; then
    echo "   ↳ Tunnel '$TUNNEL_NAME' already exists, skipping creation"
else
    cloudflared tunnel create "$TUNNEL_NAME"
fi

TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
if [ -z "$TUNNEL_ID" ]; then
    echo "❌ Failed to detect tunnel UUID"
    exit 1
fi

# Recreate if credentials file is missing
CRED_FILE="$CLOUDFLARED_DIR/$TUNNEL_ID.json"
if [ ! -f "$CRED_FILE" ]; then
    echo "⚠️  Credentials file missing for '$TUNNEL_NAME', recreating..."
    cloudflared tunnel delete -f "$TUNNEL_ID" || true
    cloudflared tunnel create "$TUNNEL_NAME"
    TUNNEL_ID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
    if [ -z "$TUNNEL_ID" ]; then
        echo "❌ Failed to detect new tunnel UUID after recreation"
        exit 1
    fi
fi

echo "🆔 Tunnel UUID: $TUNNEL_ID"

# ---- Step 3: DNS routes ----
echo "🌐 Step 3: Creating DNS routes"
for svc in "${SERVICES[@]}"; do
    HOST="${svc%%:*}"
    echo "   ↳ $HOST"
    cloudflared tunnel route dns --overwrite-dns "$TUNNEL_NAME" "$HOST"
done

# ---- Step 4: Write config ----
echo "📝 Step 4: Writing config.yml"
mkdir -p "$CLOUDFLARED_DIR"

{
    echo "tunnel: $TUNNEL_ID"
    echo "credentials-file: /etc/cloudflared/$TUNNEL_ID.json"
    echo ""
    echo "ingress:"
    for svc in "${SERVICES[@]}"; do
        HOST="${svc%%:*}"
        PORT="${svc##*:}"
        echo "  - hostname: $HOST"
        echo "    service: http://localhost:$PORT"
        echo "    originRequest:"
        echo "      noTLSVerify: true"
        echo ""
    done
    echo "  - service: http_status:404"
} > "$CONFIG_FILE"

echo "✅ Config written to $CONFIG_FILE"

# ---- Step 5: Install service ----
echo "⚙️  Step 5: Installing cloudflared as system service"
sudo mkdir -p /etc/cloudflared
sudo cp "$CONFIG_FILE" /etc/cloudflared/config.yml
sudo cp "$CLOUDFLARED_DIR/$TUNNEL_ID.json" /etc/cloudflared/
echo "   ↳ Copied credentials to /etc/cloudflared/"
sudo cloudflared service install || echo "   ↳ Service may already be installed"

# ---- Step 6: Start ----
echo "▶️  Step 6: Starting tunnel service"
sudo systemctl enable cloudflared
sudo systemctl restart cloudflared

echo ""
echo "🎉 Done!"
for svc in "${SERVICES[@]}"; do
    HOST="${svc%%:*}"
    PORT="${svc##*:}"
    echo "   👉 https://$HOST → localhost:$PORT"
done
echo ""
echo "📡 Status:"
echo "   sudo systemctl status cloudflared"
echo "   journalctl -u cloudflared -f"
