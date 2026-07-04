#!/usr/bin/env bash
# ros2-explorer container lifecycle
# Usage: ./container.bash [-b] [-b -c] [-t] [-d] [-d -c] [-s] [-g] [-r] [--training] [--monitor]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/.docker"
COMPOSE_FILE="$DOCKER_DIR/docker-compose.yaml"

BUILD=false; CLEAN=false; DELETE=false; STOP=false
TERMINAL=false; GPU=false; RECONFIGURE=false; TRAINING=false; MONITOR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build)      BUILD=true ;;
        -c|--clean)      CLEAN=true ;;
        -d|--delete)     DELETE=true ;;
        -s|--stop)       STOP=true ;;
        -t|--terminal)   TERMINAL=true ;;
        -g|--gpu)        GPU=true ;;
        -r|--reconfigure) RECONFIGURE=true ;;
        --training)      TRAINING=true ;;
        --monitor)       MONITOR=true ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
    shift
done

$RECONFIGURE && bash "$DOCKER_DIR/env_setup.sh" --reconfigure
bash "$DOCKER_DIR/env_setup.sh"
source "$DOCKER_DIR/.env"
export USER image_name ROS_DOMAIN_ID UID GID TIMEZONE TORCH

has_nvidia_gpu() { command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; }
docker_has_nvidia_runtime() { docker info 2>/dev/null | grep -q "nvidia"; }

if ! $GPU && has_nvidia_gpu && docker_has_nvidia_runtime; then
    echo "[container.bash] NVIDIA GPU detected — using dev_gpu service"
    GPU=true
fi

SERVICE="dev"; $GPU && SERVICE="dev_gpu"
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
$GPU && COMPOSE_ARGS+=(--profile dev_gpu)
$TRAINING && COMPOSE_ARGS+=(--profile training)
$MONITOR && COMPOSE_ARGS+=(--profile monitoring)

$STOP && { docker compose "${COMPOSE_ARGS[@]}" stop; exit 0; }
$DELETE && {
    docker compose "${COMPOSE_ARGS[@]}" down
    $CLEAN && docker compose "${COMPOSE_ARGS[@]}" down --volumes
    exit 0
}
$BUILD && {
    if $CLEAN; then
        docker compose "${COMPOSE_ARGS[@]}" down --volumes 2>/dev/null || true
        docker compose "${COMPOSE_ARGS[@]}" build --no-cache --pull dev
    else
        docker compose "${COMPOSE_ARGS[@]}" build dev
    fi
}

if $TRAINING; then
    docker compose "${COMPOSE_ARGS[@]}" up -d gazebo trainer
    echo "Training started. Monitor: ./container.bash --monitor"
    exit 0
fi

docker compose "${COMPOSE_ARGS[@]}" up -d "$SERVICE"
CONTAINER_ID=$(docker compose "${COMPOSE_ARGS[@]}" ps -q "$SERVICE")

if $TERMINAL; then
    docker exec -it "$CONTAINER_ID" bash
else
    REMOTE_AUTHORITY="attached-container+$(printf "%s" "$CONTAINER_ID" | od -An -tx1 | tr -d ' \n')"
    code --new-window --folder-uri "vscode-remote://${REMOTE_AUTHORITY}/ros2_ws"
    echo "VS Code attached to $CONTAINER_ID"
fi
