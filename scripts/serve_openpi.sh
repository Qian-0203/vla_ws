#!/usr/bin/env bash
# Start one openpi policy server per GPU for `MODEL_FAMILY=openpi bash docker/openvla_libero/run_eval.sh`.
# Server i runs on the i-th GPU in GPUS and listens on POLICY_PORT_BASE + i, which is the port
# run_eval.sh's shard i connects to. Servers run on the host in openpi's own uv environment
# (JAX), not in the eval image.
#
#   GPUS=0,1,2,3 bash scripts/serve_openpi.sh          # start (default policy: pi05_libero)
#   bash scripts/serve_openpi.sh stop                  # stop all servers started from here
#
# Env: OPENPI_DIR (clone location, default ~/openpi), OPENPI_COMMIT (pinned, matches the
# openpi-client in the Dockerfiles), OPENPI_ENV (serve_policy.py --env, default LIBERO, which
# serves gs://openpi-assets/checkpoints/pi05_libero), POLICY_PORT_BASE (default 8000).
# The checkpoint downloads to ~/.cache/openpi on first start.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$HOME/openpi}"
OPENPI_COMMIT="${OPENPI_COMMIT:-215abfb217dbac7d5f1273282331b9b1866c0479}"
OPENPI_ENV="${OPENPI_ENV:-LIBERO}"
GPUS="${GPUS:-0}"
POLICY_PORT_BASE="${POLICY_PORT_BASE:-8000}"
LOG_DIR="${ROOT}/openvla/experiments/logs/openpi_servers"
PID_FILE="${LOG_DIR}/servers.pid"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" == stop ]]; then
  [[ -f "${PID_FILE}" ]] || { echo "No servers recorded in ${PID_FILE}."; exit 0; }
  while read -r pid; do kill "${pid}" 2>/dev/null && echo "stopped ${pid}" || true; done <"${PID_FILE}"
  rm -f "${PID_FILE}"
  exit 0
fi

command -v uv >/dev/null || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

if [[ ! -d "${OPENPI_DIR}/.git" ]]; then
  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
fi
git -C "${OPENPI_DIR}" fetch -q origin "${OPENPI_COMMIT}" 2>/dev/null || true
git -C "${OPENPI_DIR}" checkout -q "${OPENPI_COMMIT}"
git -C "${OPENPI_DIR}" submodule update -q --init --recursive
# openpi's README: skip LFS smudge for the lerobot dependency.
(cd "${OPENPI_DIR}" && GIT_LFS_SKIP_SMUDGE=1 uv sync -q)

IFS=',' read -r -a GPU_ARR <<<"${GPUS}"
: >"${PID_FILE}"
for idx in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$idx]}"
  port=$((POLICY_PORT_BASE + idx))
  log="${LOG_DIR}/server-gpu${gpu}-port${port}.log"
  # JAX grabs 75% of GPU memory by default; these GPUs may be shared, so allocate on demand.
  (cd "${OPENPI_DIR}" && CUDA_VISIBLE_DEVICES="${gpu}" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    nohup uv run scripts/serve_policy.py --env "${OPENPI_ENV}" --port "${port}" >"${log}" 2>&1 </dev/null &
    echo $! >>"${PID_FILE}")
  echo "server ${idx}: GPU ${gpu}, port ${port}, log ${log}"
done

echo "Waiting for servers to accept connections (first start also downloads the checkpoint)..."
for idx in "${!GPU_ARR[@]}"; do
  port=$((POLICY_PORT_BASE + idx))
  until (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; do
    sleep 5
    if ! kill -0 "$(sed -n "$((idx + 1))p" "${PID_FILE}")" 2>/dev/null; then
      echo "Server on port ${port} exited; see ${LOG_DIR}/server-gpu${GPU_ARR[$idx]}-port${port}.log" >&2
      exit 1
    fi
  done
  echo "port ${port} ready"
done
