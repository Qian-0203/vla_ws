#!/usr/bin/env bash
# One-time setup of the RoboLab eval stack used for the language pilot, in $WS (default ~/robolab_ws):
#   $WS/RoboLab          NVlabs/RoboLab at b6cfcf6, with git-lfs assets (~6 GB)
#   $WS/openpi_robolab   xuningy/openpi at aa64205 (policy server; has the *_droid_jointpos configs)
#   $WS/robolab_lang     this folder (task files, launchers)
#   images robolab:b6cfcf6 (RoboLab's docker/Dockerfile on nvcr.io/nvidia/isaac-lab:2.2.0) and
#          robolab:b6cfcf6-ffmpeg
# Needs: docker + nvidia-container-toolkit, git, git-lfs, uv, ~60 GB free disk. Takes 1-2 h, mostly downloads.
# Usage: bash robolab_lang/setup_robolab.sh
set -euo pipefail
WS="${WS:-$HOME/robolab_ws}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$WS"

for cmd in docker git git-lfs uv; do
    command -v "$cmd" >/dev/null || { echo "missing: $cmd"; exit 1; }
done

# 1. this folder
if [[ "$HERE" != "$WS/robolab_lang" ]]; then
    rm -rf "$WS/robolab_lang"
    cp -r "$HERE" "$WS/robolab_lang"
fi

# 2. RoboLab with assets
if [[ ! -d "$WS/RoboLab/.git" ]]; then
    git lfs install
    git clone https://github.com/NVlabs/RoboLab.git "$WS/RoboLab"
fi
git -C "$WS/RoboLab" checkout b6cfcf6ccd2113c4cd1216d35b33024a7d5691da
git -C "$WS/RoboLab" lfs pull

# 3. images
docker pull nvcr.io/nvidia/isaac-lab:2.2.0
docker build -f "$WS/RoboLab/docker/Dockerfile" -t robolab:b6cfcf6 "$WS/RoboLab"
docker build -f "$WS/robolab_lang/docker/Dockerfile.ffmpeg" -t robolab:b6cfcf6-ffmpeg "$WS/robolab_lang/docker"

# 4. policy server code (checkpoint downloads on first serve to ~/.cache/openpi, ~12 GB)
if [[ ! -d "$WS/openpi_robolab/.git" ]]; then
    git clone https://github.com/xuningy/openpi.git "$WS/openpi_robolab"
fi
git -C "$WS/openpi_robolab" checkout aa6420561529593114160d05e5ad155792b272f3
git -C "$WS/openpi_robolab" submodule update --init --recursive
(cd "$WS/openpi_robolab" && GIT_LFS_SKIP_SMUDGE=1 uv sync)

echo "setup done in $WS"
