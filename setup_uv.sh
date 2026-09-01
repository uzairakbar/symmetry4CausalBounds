#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# PACE configuration
# ------------------------------------------------------------

if [[ "$(hostname)" == *"pace.gatech.edu"* ]]; then
    module load uv

    # keep caches, managed pythons and the env itself on scratch (bigger quota)
    export UV_CACHE_DIR="$HOME/scratch/.cache/uv"
    export UV_PYTHON_INSTALL_DIR="$HOME/scratch/.local/uv/python"
    export UV_PROJECT_ENVIRONMENT="$HOME/scratch/uv_envs/symmetry4CausalBounds-py313"

    mkdir -p "$HOME/scratch/uv_envs"

    # local shortcut
    if [ ! -e .venv ]; then
        ln -s "$UV_PROJECT_ENVIRONMENT" .venv
    fi
fi


# ------------------------------------------------------------
# Create environment and install from uv.lock
# ------------------------------------------------------------

uv venv --python 3.13

uv sync
