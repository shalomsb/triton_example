#!/usr/bin/env bash
#
# launch.sh -- one-script orchestration for the yolo11x Triton ensemble.
#
#   ./launch.sh -b   build the docker image, then build model.plan inside it
#                    and drop it at triton_models/yolo11x/1/model.plan
#   ./launch.sh -r   run the Triton server (foreground)
#   ./launch.sh -h   help
#

set -euo pipefail

IMAGE_NAME="yolo11x-triton:local"
CONTAINER_NAME="yolo11x_triton"
WEIGHTS="${WEIGHTS:-yolo11x.pt}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-640}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="$SCRIPT_DIR/triton_models"
ENGINE_PATH="$MODEL_REPO/yolo11x/1/model.plan"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-b | -r | -h]

  -b    Build docker image (if missing) and build TensorRT engine inside
        the container. Output: triton_models/yolo11x/1/model.plan
  -r    Run Triton server in the foreground.
        Ports: 8000 (HTTP) 8001 (gRPC) 8002 (metrics)
  -h    Show this help.

Env overrides:
  WEIGHTS   .pt file relative to this script         [default: yolo11x.pt]
  BATCH     max batch for the engine                 [default: 8]
  IMGSZ     input size                               [default: 640]
EOF
}

ensure_image() {
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "[launch] building image $IMAGE_NAME"
        docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"
    else
        echo "[launch] image $IMAGE_NAME exists (delete to force rebuild: docker rmi $IMAGE_NAME)"
    fi
}

build_engine() {
    ensure_image

    if [[ ! -f "$SCRIPT_DIR/$WEIGHTS" ]]; then
        echo "[launch] weights not found at $SCRIPT_DIR/$WEIGHTS – downloading..."
        local url="https://github.com/ultralytics/assets/releases/download/v8.3.0/${WEIGHTS}"
        if command -v wget >/dev/null 2>&1; then
            wget -q --show-progress -O "$SCRIPT_DIR/$WEIGHTS" "$url"
        elif command -v curl >/dev/null 2>&1; then
            curl -L --progress-bar -o "$SCRIPT_DIR/$WEIGHTS" "$url"
        else
            echo "[launch] ERROR: neither wget nor curl found – install one or download manually:"
            echo "         $url"
            exit 1
        fi
        echo "[launch] downloaded $WEIGHTS"
    fi

    echo "[launch] building TensorRT engine inside container -> $ENGINE_PATH"
    docker run --rm --gpus all \
        -v "$SCRIPT_DIR":/workspace \
        -w /workspace \
        --entrypoint python3 \
        "$IMAGE_NAME" \
        build_engine.py \
            --weights "$WEIGHTS" \
            --output "triton_models/yolo11x/1/model.plan" \
            --batch "$BATCH" \
            --imgsz "$IMGSZ" \
            --fp16

    echo "[launch] engine ready: $ENGINE_PATH"
}

run_server() {
    ensure_image

    if [[ ! -f "$ENGINE_PATH" ]]; then
        echo "[launch] ERROR: engine not found at $ENGINE_PATH. Run with -b first."
        exit 1
    fi

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker rm -f "$CONTAINER_NAME" >/dev/null
    fi

    echo "[launch] starting Triton (ctrl-c to stop)"
    docker run --rm --gpus all \
        --name "$CONTAINER_NAME" \
        --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        -v "$MODEL_REPO":/models:ro \
        "$IMAGE_NAME" \
        tritonserver \
            --model-repository=/models \
            --strict-model-config=false \
            --log-verbose=0
}

if [[ $# -eq 0 ]]; then usage; exit 1; fi

while getopts ":brh" opt; do
    case $opt in
        b) build_engine ;;
        r) run_server ;;
        h) usage; exit 0 ;;
        \?) echo "Unknown option: -$OPTARG"; usage; exit 1 ;;
    esac
done
