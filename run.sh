#!/usr/bin/env bash
# Launcher for Linux and macOS. Runs the same steps as _RUN_DepthCrafter_GUI_Seg.bat.
set -eu
cd "$(dirname "$0")"
ROOT="$(pwd)"

# Portable setup: venv, uv cache, Hugging Face models and ffmpeg stay inside this folder.
export UV_CACHE_DIR="$ROOT/uv-cache"
export HF_HOME="$ROOT/models_cache"
export PATH="$ROOT/tools/ffmpeg/bin:$HOME/.local/bin:$PATH"

# 1. uv (the uv installer puts it in ~/.local/bin)
if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] uv was not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 2. Python packages. uv skips anything already installed and reuses the cache.
#    --inexact keeps packages installed outside pyproject.toml (such as triton-windows on Windows).
if ! uv sync --inexact; then
    echo "[ERROR] uv sync failed. Run this script again: the download cache is reused."
    exit 1
fi

# 3. ffmpeg and Hugging Face models: downloads what is missing, skips what is present.
if ! ./.venv/bin/python tools/preflight_check.py; then
    echo
    echo "[WARN] Some items are still missing. See MISSING_MANUAL.txt in this folder."
    exit 1
fi

# 4. GPU selection (NVIDIA only; skipped on macOS and on machines without nvidia-smi)
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | grep -c GPU || true)
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo "================================================="
        echo "Multiple GPUs detected:"
        nvidia-smi --query-gpu=index,name --format=csv,noheader
        echo "================================================="
        read -r -p "Enter the ID of the GPU you want to use (e.g. 0 or 1): " GPU_ID
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "Script will now run using ONLY GPU $GPU_ID."
    else
        echo "Only $GPU_COUNT GPU detected. Running on default."
    fi
fi

# 5. Run the app with the project's own interpreter
exec ./.venv/bin/python depthcrafter_gui_seg.py
