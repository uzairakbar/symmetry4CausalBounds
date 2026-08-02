#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# PACE configuration
# ------------------------------------------------------------

if [[ "$(hostname)" == *"pace.gatech.edu"* ]]; then
    module load uv

    export UV_CACHE_DIR="$HOME/scratch/.cache/uv"
    export UV_PROJECT_ENVIRONMENT="$HOME/scratch/uv_envs/symmetry4CausalBounds"

    mkdir -p "$HOME/scratch/uv_envs"

    # local shortcut
    if [ ! -e .venv ]; then
        ln -s "$UV_PROJECT_ENVIRONMENT" .venv
    fi
fi


# ------------------------------------------------------------
# Create environment
# ------------------------------------------------------------

uv venv --python 3.10


# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

uv pip install .