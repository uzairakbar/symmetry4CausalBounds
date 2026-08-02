#!/usr/bin/env bash
set -euo pipefail

module load uv

export UV_CACHE_DIR="$HOME/scratch/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$HOME/scratch/uv_envs/symmetry4CausalBounds"


cc=$(nvidia-smi --query-gpu=compute_cap \
    --format=csv,noheader 2>/dev/null \
    | head -1 \
    | tr -d ' .')


if [ -z "$cc" ]; then
    BACKEND=cpu
elif [ "$cc" -lt 75 ]; then
    # Pascal/Volta (P100/V100)
    BACKEND=cu126
else
    # Turing/Ampere/Hopper/Blackwell
    BACKEND=auto
fi


echo "GPU compute capability '$cc' -> reinstalling torch ($BACKEND)"


uv pip install . \
    --reinstall-package torch \
    --reinstall-package torchvision \
    --torch-backend="$BACKEND"