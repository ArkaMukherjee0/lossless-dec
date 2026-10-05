#!/usr/bin/env bash
# Usage: GPUS=0,1 docker/run.sh [cmd...]   (GPUS = GCD indices, default all)
# Mounts the repo at /workspace and the local-NVMe model cache at /models.
set -euo pipefail
IMAGE=${IMAGE:-lossless-dec:vllm0.30}
MODELS=${MODELS:-/var/tmp/arkamukh}
REPO=$(cd "$(dirname "$0")/.." && pwd)

TTY=-i; [ -t 0 ] && TTY=-it

exec docker run --rm $TTY ${NAME:+--name "$NAME"} \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --shm-size=64g --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}" \
  --user "$(id -u):$(id -g)" -e HOME=/models/home -e VLLM_CACHE_ROOT="/models/home/.cache/vllm-gcd${GPUS//,/_}" -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HF_TOKEN="${HF_TOKEN:-}" -e TARGET -e ATTN -e EAGER -e PREFIX_CACHE \
  -v "$REPO":/workspace -v "$MODELS":/models \
  "$IMAGE" "${@:-bash}"
