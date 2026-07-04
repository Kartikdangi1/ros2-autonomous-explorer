#!/usr/bin/env bash
# Idempotent .env writer for ros2-explorer workspace.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
RECONFIGURE=false
[[ "${1:-}" == "--reconfigure" ]] && RECONFIGURE=true
HOST_UID=$(id -u); HOST_GID=$(id -g)
TIMEZONE=$(cat /etc/timezone 2>/dev/null || echo "UTC")
NETWORK_INTERFACE=$(ip route | grep default | grep -oP 'dev \K\S+' | head -1 || echo "eth0")

if [[ ! -f "$ENV_FILE" || "$RECONFIGURE" == "true" ]]; then
    echo "=== ros2-explorer environment setup ==="
    read -rp "Image name [ros2_explorer]: " IMAGE_NAME; IMAGE_NAME="${IMAGE_NAME:-ros2_explorer}"
    while true; do
        read -rp "ROS_DOMAIN_ID [1-231, default 42]: " ROS_DOMAIN_ID
        ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
        [[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] && (( ROS_DOMAIN_ID >= 1 && ROS_DOMAIN_ID <= 231 )) && break
        echo "  Invalid: must be 1–231"
    done
    read -rp "Include TORCH? [y/N]: " TORCH_INPUT
    TORCH="false"; [[ "${TORCH_INPUT,,}" == "y" ]] && TORCH="true"
    cat > "$ENV_FILE" <<EOF
image_name=${IMAGE_NAME}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
TORCH=${TORCH}
UID=${HOST_UID}
GID=${HOST_GID}
NETWORK_INTERFACE=${NETWORK_INTERFACE}
TIMEZONE=${TIMEZONE}
EOF
    echo "Written: $ENV_FILE"
else
    sed -i "s/^UID=.*/UID=${HOST_UID}/" "$ENV_FILE"
    sed -i "s/^GID=.*/GID=${HOST_GID}/" "$ENV_FILE"
    sed -i "s/^TIMEZONE=.*/TIMEZONE=${TIMEZONE}/" "$ENV_FILE"
    sed -i "s/^NETWORK_INTERFACE=.*/NETWORK_INTERFACE=${NETWORK_INTERFACE}/" "$ENV_FILE"
fi
