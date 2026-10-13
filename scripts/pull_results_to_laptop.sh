#!/usr/bin/env bash
# Pull eval results from the GPU servers into a folder on this machine, as a second copy next to the HF mirror
# (scripts/sync_results_hf.sh). Servers are replaced without notice, so run this at every milestone and on a timer.
#
#   bash scripts/pull_results_to_laptop.sh                 # all servers and workspaces
#   HOSTS=berkeley-2 WORKSPACES="vla_ws_oft2" bash scripts/pull_results_to_laptop.sh
#
# Copies only openvla/experiments/logs/results/{*.jsonl,*.meta.json} (the raw rollouts and their run metadata) and
# the queue markers (*.done, *.fail, jobs.tsv) of scripts/gpu_queue.py. Rollout videos and text logs are not
# copied; they are large and the HF mirror has the logs. Nothing is deleted here, and a file under DEST that is
# newer than the server's copy is left alone (rsync --update).
set -uo pipefail

DEST="${DEST:-$HOME/vla_results_backup/libero_ckpt_campaign_2026-10-12}"
HOSTS="${HOSTS:-berkeley berkeley-2}"
WORKSPACES="${WORKSPACES:-vla_ws vla_ws_oft2 vla_ws_va vla_ws_va2 vla_ws_xvla vla_ws_gr00t vla_ws_lang vla_ws_lang2 vla_ws_xr0 vla_ws_oftm}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
STAMP="$(date -u +%FT%TZ)"
mkdir -p "${DEST}"

for h in ${HOSTS}; do
  for w in ${WORKSPACES}; do
    out="${DEST}/${h}/${w}/results"
    mkdir -p "${out}"
    rsync -a --update -e "${SSH[*]}" \
      --include='*.jsonl' --include='*.meta.json' --exclude='*' \
      "${h}:${w}/openvla/experiments/logs/results/" "${out}/" 2>/dev/null \
      || echo "[${STAMP}] ${h}:${w}: no results dir (or host unreachable)"
  done
  q="${DEST}/${h}/queue_state"
  mkdir -p "${q}"
  rsync -a --update -e "${SSH[*]}" \
    --include='queue_*/' --include='*.done' --include='*.fail' --include='jobs.tsv' --include='mine.tsv' --exclude='*' \
    "${h}:logs/" "${q}/" 2>/dev/null || true
done

n=$(cat "${DEST}"/*/*/results/*.jsonl 2>/dev/null | wc -l)
echo "[${STAMP}] pulled to ${DEST}: ${n} rollouts in $(ls "${DEST}"/*/*/results/*.jsonl 2>/dev/null | wc -l) files; $(du -sh "${DEST}" | cut -f1) on disk"
