#!/usr/bin/env bash
# RoboLab language pilot, Option A: 13 tasks in 4 shared scenes x instruction keys x 20 episodes, pi0.5-DROID.
# One run per instruction key (results-storage convention: run.json first, then RoboLab output, then records).
#
# Prereqs (docs/robolab/run_option_a.md): image robolab:b6cfcf6-ffmpeg built, policy server up on $PORT
# (robolab_lang/serve_pi05.sh), this folder at $WS/robolab_lang.
#
# Usage:  GPU=0 PORT=8765 HOST_TAG=vmrl bash robolab_lang/run_option_a.sh
#   KEYS="default blank no_target"   instruction keys, one run each (default: all three = 780 episodes)
#   NUM_ENVS=10 NUM_RUNS=2           parallel envs x runs = episodes per task (default 20)
#   TASKS="LangMustardInLeftBinTask ..."   subset of task files (default: all 13)
# Re-running with the same RUN_PREFIX resumes RoboLab's output folders.
set -euo pipefail
WS="${WS:-$HOME/robolab_ws}"
GPU="${GPU:-0}"
PORT="${PORT:-8765}"
IMAGE="${IMAGE:-robolab:b6cfcf6-ffmpeg}"
HOST_TAG="${HOST_TAG:-$(hostname -s)}"
KEYS="${KEYS:-default blank no_target}"
NUM_ENVS="${NUM_ENVS:-10}"
NUM_RUNS="${NUM_RUNS:-2}"
STAGE="${STAGE:-screen}"
TASKS="${TASKS:-LangMustardInLeftBinTask LangMustardInRightBinTask LangBowlStackingLeftOnRightTask LangBowlStackingRightOnLeftTask LangLargerObjectRaisinBoxInBinTask LangSmallerObjectButterInBinTask LangRubiksCubeLeftOfBowlTask LangRubiksCubeBehindBowlTask LangRubiksCubeInFrontOfBowlTask LangRubiksCubeAndBananaTask LangRubiksCubeOrBananaTask LangRubiksCubeThenBananaTask LangBananaThenRubiksCubeTask}"
RUN_PREFIX="${RUN_PREFIX:-$(date -u +%Y%m%d-%H%M)}"
HERE="$(cd "$(dirname "$0")" && pwd)"

docker image inspect "$IMAGE" >/dev/null || { echo "image $IMAGE missing (see docs/robolab/run_option_a.md)"; exit 1; }
(exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null || { echo "no policy server on port $PORT (start robolab_lang/serve_pi05.sh)"; exit 1; }

task_files=()
for t in $TASKS; do
    [[ -f "$WS/robolab_lang/tasks/$t.py" ]] || { echo "missing $WS/robolab_lang/tasks/$t.py"; exit 1; }
    task_files+=("/lang/tasks/$t.py")
done
n_tasks=$(wc -w <<<"$TASKS")
mkdir -p "$WS/runs" "$WS/logs"

for key in $KEYS; do
    run_id="${RUN_PREFIX}_robolab_pi05_${STAGE}_${HOST_TAG}_${key}"
    run_dir="$WS/runs/$run_id"
    mkdir -p "$run_dir"
    if [[ ! -f "$run_dir/run.json" ]]; then
        RUN_ID="$run_id" KEY="$key" N_TASKS="$n_tasks" GPU="$GPU" IMAGE="$IMAGE" STAGE="$STAGE" \
            NUM_ENVS="$NUM_ENVS" NUM_RUNS="$NUM_RUNS" TASKS="$TASKS" PURPOSE="${PURPOSE:-}" python3 - "$run_dir/run.json" <<'EOF'
import json, os, subprocess, sys
e = os.environ
gpu = subprocess.run(["nvidia-smi", "-i", e["GPU"], "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
img = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", e["IMAGE"]],
                     capture_output=True, text=True).stdout.strip()
n_envs, n_runs = int(e["NUM_ENVS"]), int(e["NUM_RUNS"])
json.dump({
    "run_id": e["RUN_ID"], "stage": e["STAGE"], "host": os.uname().nodename, "gpu": f'{e["GPU"]} ({gpu})',
    "started_utc": subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip(),
    "operator": e.get("USER", ""),
    "isaac_sim": "5.0.0-rc.45", "isaac_lab": "2.2.0", "robolab_commit": "b6cfcf6ccd2113c4cd1216d35b33024a7d5691da",
    "image": e["IMAGE"], "image_id": img, "policy": "pi05", "policy_config": "pi05_droid_jointpos",
    "checkpoint": "gs://openpi-assets-simeval/pi05_droid_jointpos", "openpi_repo": "xuningy/openpi",
    "openpi_commit": "aa6420561529593114160d05e5ad155792b272f3",
    "client": "RoboLab Pi0DroidJointposClient via robolab_lang/run_pi0.py",
    "purpose": (e.get("PURPOSE") or "RoboLab language pilot Option A (original shared scenes, paired instructions)"),
    "tasks": e["TASKS"].split(), "instruction_key": e["KEY"], "num_envs": n_envs, "num_runs": n_runs,
    "episodes_planned": int(e["N_TASKS"]) * n_envs * n_runs,
    "raw_output_dir": f'runs/{e["RUN_ID"]}/robolab_output',
}, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
EOF
    fi
    echo "[$(date -u +%FT%TZ)] $run_id: $n_tasks tasks x $NUM_ENVS envs x $NUM_RUNS runs, key=$key"
    GPU="$GPU" PORT="$PORT" IMAGE="$IMAGE" WS="$WS" bash "$HERE/run_robolab.sh" "$run_id" \
        --lang-task "${task_files[@]}" --instruction-type "$key" \
        --num-envs "$NUM_ENVS" --num-runs "$NUM_RUNS" 2>&1 | tee -a "$WS/logs/$run_id.log"

    # RoboLab writes to $WS/output/<run_id>; move it under the run directory, then write the records.
    if [[ -d "$WS/output/$run_id" ]]; then
        rm -rf "$run_dir/robolab_output"
        mv "$WS/output/$run_id" "$run_dir/robolab_output"
    fi
    docker run --rm --entrypoint bash -v "$WS/robolab_lang:/lang:ro" -v "$WS/runs:/runs" \
        -e PYTHONNOUSERSITE=1 "$IMAGE" -c \
        "/workspace/isaaclab/_isaac_sim/python.sh /lang/make_run_record.py /runs/$run_id; chown -R $(id -u):$(id -g) /runs/$run_id"
    (cd "$run_dir" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    echo "[$(date -u +%FT%TZ)] $run_id done: $run_dir/summary.csv"
    cat "$run_dir/summary.csv" || true
done
