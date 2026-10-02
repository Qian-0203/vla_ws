#!/usr/bin/env bash
# Move an eval checkpoint through a private Hugging Face model repo instead of rsync-ing ~15GB
# from the laptop to every new server. Upload once from wherever the checkpoint lives; each
# server then pulls it at datacenter speed.
#
#   bash scripts/hf_checkpoint.sh upload   <local_checkpoint_dir> <hf_user>/<repo>
#   bash scripts/hf_checkpoint.sh download <hf_user>/<repo> [local_dir]
#
# <local_checkpoint_dir> is the merged HF checkpoint itself (the directory holding config.json,
# model-*.safetensors, dataset_statistics.json). download's local_dir defaults to
# openvla/checkpoint/<repo>, i.e. inside the workspace where server.env's CHECKPOINT expects it.
#
# Needs `uv` on PATH (runs the `hf` CLI via uvx, no install) and a token with write access
# (upload) or read access (download): run `uvx --from huggingface_hub hf auth login` once per
# machine, or export HF_TOKEN.
set -euo pipefail

# Pinned: the hf CLI renames/removes subcommands between releases (2.0 dropped upload-large-folder).
HF=(uvx --from "huggingface_hub[hf_xet]==2.0.0" hf)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

command -v uvx >/dev/null || {
  echo "uvx not found. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

case "${1:-}" in
  upload)
    [[ $# -eq 3 ]] || usage
    src="$2"; repo="$3"
    for f in config.json dataset_statistics.json model.safetensors.index.json; do
      [[ -f "$src/$f" ]] || { echo "$src/$f missing: pass the merged checkpoint directory itself." >&2; exit 1; }
    done
    # Always private: this is an unreleased research checkpoint. Create the repo explicitly so
    # it is private even if the upload step below would otherwise pick a default. Rerunning after
    # a dropped connection is safe: Xet storage skips chunks the Hub already has.
    "${HF[@]}" repos create "$repo" --repo-type=model --private --exist-ok
    "${HF[@]}" upload "$repo" "$src" . --repo-type=model --private \
      --commit-message "Upload checkpoint $(basename "$src")"
    ;;
  download)
    [[ $# -ge 2 && $# -le 3 ]] || usage
    repo="$2"; dst="${3:-$ROOT/openvla/checkpoint/${repo##*/}}"
    "${HF[@]}" download "$repo" --repo-type=model --local-dir "$dst"
    echo
    echo "Checkpoint at: $dst"
    echo "Next: CHECKPOINT=$dst python3 scripts/preflight.py"
    ;;
  *) usage ;;
esac
