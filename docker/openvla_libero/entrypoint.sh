#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${LIBERO_CONFIG_PATH}"

if [[ -d /workspace/LIBERO/libero/libero ]]; then
  cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<'YAML'
assets: /workspace/LIBERO/libero/libero/assets
bddl_files: /workspace/LIBERO/libero/libero/bddl_files
benchmark_root: /workspace/LIBERO/libero/libero
datasets: /workspace/openvla
init_states: /workspace/LIBERO/libero/libero/init_files
YAML
fi

exec "$@"
