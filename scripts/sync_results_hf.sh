#!/usr/bin/env bash
# Mirror this server's eval results and text logs to the private HF dataset. GCP flexstart instances
# are replaced without notice: the official-checkpoint run of 2026-10-05..07 was recovered only
# because its disk reappeared on another instance on 2026-10-10. Install from cron on every server,
# before launching anything:
#
#   (crontab -l 2>/dev/null; echo "*/30 * * * * bash $HOME/vla_ws/scripts/sync_results_hf.sh >> $HOME/logs/sync_results_hf.log 2>&1") | crontab -
#
# Uploads openvla/experiments/logs/results/{*.jsonl,*.meta.json} to results/ and the text logs
# (*.txt, *.out) to logs/<hostname>/. Rollout videos are not mirrored. Needs a Hugging Face token on
# the machine (uvx --from huggingface_hub hf auth login) and either uv on PATH or the eval image.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${HF_RESULTS_REPO:-Qian0203/vla_ws-eval-results}"
LOGS="${RESULTS_LOG_DIR:-${ROOT}/openvla/experiments/logs}"
HOST="$(hostname)"
STAMP="$(date -u +%FT%TZ)"
export PATH="${HOME}/.local/bin:${PATH}"

exec 9>"/tmp/sync_results_hf.$(id -un).lock"  # cron sets no $USER
flock -n 9 || { echo "[${STAMP}] previous sync still running; skipped"; exit 0; }
[[ -d "${LOGS}/results" ]] || { echo "[${STAMP}] no ${LOGS}/results yet; nothing to sync"; exit 0; }

if command -v uvx >/dev/null; then
  HF=(uvx --from "huggingface_hub[hf_xet]==2.0.0" hf)
  "${HF[@]}" upload "${REPO}" "${LOGS}/results" results --repo-type=dataset \
    --include "*.jsonl" --include "*.meta.json" --commit-message "sync results ${STAMP} (${HOST})" >/dev/null
  rc1=$?
  "${HF[@]}" upload "${REPO}" "${LOGS}" "logs/${HOST}" --repo-type=dataset \
    --include "*.txt" --include "*.out" --exclude "results/*" --commit-message "sync logs ${STAMP} (${HOST})" >/dev/null
  rc2=$?
else
  # No uv: use huggingface_hub from the eval image (same flags as the 2026-10-05 checkpoint download).
  docker run --rm --entrypoint python3 --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONNOUSERSITE=1 \
    -e LIBERO_CONFIG_PATH=/tmp/.libero -e HF_HOME=/hf -e HF_HUB_CACHE=/tmp/hub \
    -v "${HOME}/.cache/huggingface:/hf:ro" -v "${LOGS}:/logs:ro" -w /tmp "${IMAGE_NAME:-openvla-libero:blackwell}" -c "
from huggingface_hub import HfApi
a = HfApi()
a.upload_folder(repo_id='${REPO}', repo_type='dataset', folder_path='/logs/results', path_in_repo='results',
                allow_patterns=['*.jsonl', '*.meta.json'], commit_message='sync results ${STAMP} (${HOST})')
a.upload_folder(repo_id='${REPO}', repo_type='dataset', folder_path='/logs', path_in_repo='logs/${HOST}',
                allow_patterns=['*.txt', '*.out'], ignore_patterns=['results/*'], commit_message='sync logs ${STAMP} (${HOST})')
" >/dev/null
  rc1=$?; rc2=$rc1
fi
n=$(cat "${LOGS}"/results/*.jsonl 2>/dev/null | wc -l)
echo "[${STAMP}] ${HOST}: results exit ${rc1}, logs exit ${rc2}, ${n} rollouts in results"
