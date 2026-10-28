#!/usr/bin/env bash
# Start the pi0.5-DROID joint-position policy server (openpi fork xuningy/openpi@aa64205) on one GPU.
# Usage: GPU=0 PORT=8765 bash robolab_lang/serve_pi05.sh   (runs in the foreground; wrap in nohup)
# CONFIG / CKPT override the policy (e.g. pi0_fast_droid_jointpos).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
GPU="${GPU:-0}"
PORT="${PORT:-8765}"
CONFIG="${CONFIG:-pi05_droid_jointpos}"
CKPT="${CKPT:-gs://openpi-assets-simeval/pi05_droid_jointpos}"
OPENPI_DIR="${OPENPI_DIR:-$HOME/robolab_ws/openpi_robolab}"
# Shared host: do not let JAX take half of a 98 GB card (RoboLab's example uses 0.5).
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.2}"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$OPENPI_DIR"
exec uv run scripts/serve_policy.py --port "$PORT" policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$CKPT"
