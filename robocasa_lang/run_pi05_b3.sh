#!/bin/bash
# π0.5 on RoboCasa365 on Berkeley-3 (no docker): one openpi server per GPU, a pool of client workers, one cell per job.
#   bash run_pi05_b3.sh serve "0 1 2"                       start servers on ports 8800+gpu (refuses a taken port)
#   bash run_pi05_b3.sh run <run_id> <cells.jsonl> <workers> "0 1 2"   run all cells, then write the run record
#   bash run_pi05_b3.sh stop                                stop the servers this script started
set -u
Q=/mnt/dt-vol-00494956a3ff3a063-20000G/qian; W=$Q/robocasa_ws; RL=$W/robocasa_lang
CKPT=$W/checkpoints/pi05_pretrain_human300/multitask_learning/75000
LOGS=$W/logs; PIDS=$LOGS/pi05_servers.pid; mkdir -p $LOGS
export HF_HOME=$Q/cache/hf

port_of() { echo $((8800 + $1)); }
port_open() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(s.connect_ex(('127.0.0.1',$1)))"; }

case "$1" in
serve)
  for g in $2; do
    p=$(port_of $g)
    if port_open $p; then echo "port $p already taken; not starting GPU $g"; exit 1; fi
    ( cd $W/openpi_robocasa && CUDA_VISIBLE_DEVICES=$g XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
        exec nohup .venv/bin/python $RL/servers/pi05_server.py --ckpt $CKPT --port $p ) > $LOGS/pi05_server_gpu$g.log 2>&1 < /dev/null &
    echo $! >> $PIDS; echo "GPU $g -> port $p pid $!"
  done
  for g in $2; do
    p=$(port_of $g); for i in $(seq 120); do port_open $p && break; sleep 5; done
    port_open $p && echo "port $p up" || { echo "port $p did not come up"; tail -5 $LOGS/pi05_server_gpu$g.log; exit 1; }
  done ;;
stop)
  [ -f $PIDS ] && kill $(cat $PIDS) 2>/dev/null; rm -f $PIDS; echo stopped ;;
run)
  RID=$2; CELLS=$3; NW=$4; GPUS=($5); RD=$W/runs/$RID
  mkdir -p $RD/parts; cp $CELLS $RD/cells.jsonl
  for g in "${GPUS[@]}"; do port_open $(port_of $g) || { echo "no server on GPU $g"; exit 1; }; done
  START=$(date -u +%FT%TZ); echo "[$START] START $RID"
  python3 -c "import json,sys; [print(json.loads(l)['cell']) for l in open(sys.argv[1]) if l.strip()]" $CELLS \
    | awk -v n=${#GPUS[@]} -v g="${GPUS[*]}" 'BEGIN{split(g,G," ")} {print $1, G[(NR-1)%n+1]}' \
    | xargs -P $NW -L 1 bash -c '
        c=$0; g=$1; p=$((8800 + g))
        cd /tmp && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$g OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
          '$W'/.venv-sim/bin/python '$RL'/run_lang_eval.py --policy pi05 --port $p --run-dir '$RD' --run-id '$RID' \
          --cells '$RD'/cells.jsonl --cell-names $c --episodes-file parts/$c.jsonl > '$RD'/parts/$c.log 2>&1
        echo "[$(date -u +%FT%TZ)] cell $c gpu $g exit=$?"'
  END=$(date -u +%FT%TZ)
  cat $RD/parts/*.jsonl > $RD/episodes.jsonl
  cd /tmp && $W/.venv-sim/bin/python $RL/runrecord.py summarize --episodes $RD/episodes.jsonl --out $RD/summary.csv
  $W/.venv-sim/bin/python - "$RD" "$RID" "$START" "$END" "${GPUS[*]}" <<'P'
import json, subprocess, sys
rd, rid, start, end, gpus = sys.argv[1:]
W = "/mnt/dt-vol-00494956a3ff3a063-20000G/qian/robocasa_ws"
git = lambda d: subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
rows = [json.loads(l) for l in open(f"{rd}/episodes.jsonl") if '"success"' in l]
json.dump(dict(run_id=rid, platform="robocasa", model="pi05", checkpoint="robocasa/robocasa365_checkpoints:"
               "pi05_pretrain_human300/multitask_learning/75000 (params + assets)", config="pi05_pretrain_human300",
               server="robocasa_lang/servers/pi05_server.py (openpi fork robocasa-benchmark/openpi @ %s)" % git(f"{W}/openpi_robocasa"),
               client="robocasa_lang/run_lang_eval.py --policy pi05 (replan 5, 224 px, 3 cameras)",
               robocasa=git(f"{W}/robocasa"), robosuite=git(f"{W}/robosuite"), mujoco="3.3.1", split="pretrain",
               host="berkeley-3 (ip-172-31-64-36, 8x H200)", gpus=gpus, started=start, ended=end,
               episodes=len(rows), skipped=sum(1 for l in open(f"{rd}/episodes.jsonl") if '"skipped"' in l)),
          open(f"{rd}/run.json", "w"), indent=1)
P
  ( cd $RD && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
  echo "[$END] END $RID episodes=$(grep -c '"success"' $RD/episodes.jsonl)" ;;
esac
