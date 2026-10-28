#!/usr/bin/env bash
# Run one RoboLab client job in the RoboLab image on one GPU, talking to a policy server on this host.
# Usage: GPU=0 PORT=8765 bash robolab_lang/run_robolab.sh <output_folder_name> [run_pi0.py args...]
# e.g.   bash robolab_lang/run_robolab.sh gateA_pi05 --task BananaInBowlTask --num-envs 10 --num-runs 5
#        bash robolab_lang/run_robolab.sh lang_pi05_neg --lang-task /lang/tasks/LangMustardInLeftBinTask.py \
#             --instruction-type negation --num-envs 10 --num-runs 2
# Outputs land in $WS/output/<output_folder_name>/ (RoboLab layout). Re-running the same name resumes.
# The image runs as root (Isaac Lab base); every mounted directory is chowned back to the caller on exit.
set -euo pipefail
WS="${WS:-$HOME/robolab_ws}"
GPU="${GPU:-0}"
PORT="${PORT:-8765}"
IMAGE="${IMAGE:-robolab:b6cfcf6}"
NAME="$1"; shift
mkdir -p "$WS/output" "$WS/cache/ov" "$WS/cache/kit" "$WS/cache/nv"
U="$(id -u):$(id -g)"
exec docker run --rm --gpus "\"device=$GPU\"" --network host \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e PRIVACY_CONSENT=Y -e PYTHONNOUSERSITE=1 \
    -v "$WS/robolab_lang:/lang:ro" \
    -v "$WS/output:/workspace/robolab/output" \
    -v "$WS/cache/ov:/root/.cache/ov" \
    -v "$WS/cache/kit:/isaac-sim/kit/cache" \
    -v "$WS/cache/nv:/root/.nv/ComputeCache" \
    --entrypoint bash "$IMAGE" -c "
        trap 'chown -R $U /workspace/robolab/output /root/.cache/ov /isaac-sim/kit/cache /root/.nv/ComputeCache' EXIT
        cd /workspace/robolab
        /workspace/isaaclab/_isaac_sim/python.sh /lang/run_pi0.py --policy pi05 --headless \
            --remote-port $PORT --output-folder-name $NAME $*"
