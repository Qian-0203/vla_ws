# CLAUDE.md

Operating guide for working in this repo. For the research spec (splits, hypotheses, metrics),
see `benchmark_split.md` — this file is operational only; don't duplicate that content here.

## What this is

Systematic evaluation of an `openvla-7b` LoRA checkpoint (fine-tuned on LIBERO's `libero_spatial`)
against prompt-phrasing, distractor-placement, scene-clutter, and grounding-type variations, to
separate real spatial/language grounding from overfitting to the training distribution.

## Repository map — three git repos, one workspace

```
vla_ws/                    <- THIS repo. Docs, config, docker orchestration. No model/eval code.
  benchmark_split.md          canonical research spec
  eval_results.md              results (append-only, never overwrite)
  config/                      machine profiles (laptop.env, server.env.example)
  docker/openvla_libero/       Dockerfiles + run_eval.sh (the one eval launcher)
  scripts/                     aggregate_results.py, preflight.py

  openvla/                  <- SEPARATE git repo (fork github.com/Qian-0203/openvla), gitignored
                                here. Owns all eval CODE.
    experiments/robot/libero/
      run_libero_eval.py         THE canonical eval entry point
      eval_registry.py           THE canonical benchmark registry (splits -> suite/condition)
      instructions.py            prompt condition text, keyed by task name
      libero_utils.py            env/image/video helpers
    experiments/robot/openvla_utils.py, robot_utils.py   model loading, action decoding
    checkpoint/                  LoRA checkpoint (~15GB, gitignored, not in git history)
    experiments/logs/            eval outputs (gitignored)

  LIBERO/                   <- SEPARATE git repo (fork github.com/Qian-0203/LIBERO), gitignored
                                here. Owns all SCENE/TASK definitions.
    libero/libero/benchmark/{__init__.py,libero_suite_task_map.py}   suite registry
    libero/libero/bddl_files/<suite>/                                scene definitions
    libero/libero/init_files/<suite>/                                pre-sampled init states
```

**Why three repos:** `openvla/` and `LIBERO/` are your own forks with their own commit history and
remotes — changes to eval code or scenes belong in those repos' git history, not this one. This
repo only orchestrates: it mounts both into a Docker container and runs one script. When editing
eval logic or scenes, `cd` into the relevant fork and commit there separately.

## Canonical sources of truth

- **Which splits exist / how a split maps to (suite, unnorm_key, condition):**
  `openvla/experiments/robot/libero/eval_registry.py::SPLITS`
- **Prompt text per condition:** `openvla/experiments/robot/libero/instructions.py`
- **Task suites / scenes:** `LIBERO/libero/libero/benchmark/__init__.py` (registration) +
  `libero_suite_task_map.py` (task lists) + `bddl_files/<suite>/*.bddl` (scene geometry)
- **Research semantics (hypotheses, metrics, what's implemented vs. open):** `benchmark_split.md`
- **Historical results:** `eval_results.md` (append new experiments, never edit old ones)

## Environment setup

Everything runs inside Docker — no host Python env needed for eval itself.

1. `docker --version` and `nvidia-container-toolkit` must be installed (already true if
   `docker run --gpus all ...` works).
2. Build the image for your GPU (see "Known pitfalls" below for which one):
   ```
   docker build -f docker/openvla_libero/Dockerfile -t openvla-libero:cuda12.1 .
   docker build -f docker/openvla_libero/Dockerfile.blackwell -t openvla-libero:blackwell .
   ```
3. `python scripts/preflight.py --checkpoint <path> --image <image>` — checks GPU/VRAM, docker,
   checkpoint completeness, and whether it'll fit at the precision you plan to use. Host-side,
   no torch/CUDA import needed.

## Running eval

One entry point (`docker/openvla_libero/run_eval.sh`), one registry (`--split`):

```bash
# Laptop, 1 GPU:
MACHINE_CONFIG=config/laptop.env bash docker/openvla_libero/run_eval.sh --split spatial/default

# Server, multiple GPUs (sharded round-robin across the suite's tasks):
MACHINE_CONFIG=config/server.env GPUS=0,1,2,3,4 bash docker/openvla_libero/run_eval.sh --split spatial_3bowl/irrelevant

# Quick smoke test (1 task, 1 trial):
MACHINE_CONFIG=config/laptop.env bash docker/openvla_libero/run_eval.sh --split spatial/default --task_ids 0 --num_trials_per_task 1

# Resume an interrupted run / refuse-by-default on re-run into existing results:
... run_eval.sh --split spatial/default --resume True     # continue
... run_eval.sh --split spatial/default --overwrite True  # discard and restart
```

Copy `config/server.env.example` to `config/server.env` (gitignored — has no secrets, but is
machine-specific) and fill in `SERVER_ROOT`/`CHECKPOINT` before first use on a new server.
Any flag not consumed by `run_eval.sh` (`--split`, `--resume`, ...) forwards straight to
`run_libero_eval.py` — see that file's `GenerateConfig` for the full flag list.

**Config precedence:** CLI args to `run_eval.sh` > variables already exported in your shell >
`MACHINE_CONFIG` file > `run_eval.sh`'s own fallback defaults.

## Results & logs

- Text log: `openvla/experiments/logs/EVAL-{suite}-openvla-{timestamp}[--note][--shardXofY].txt`
- Structured results (used for resume + aggregation): `openvla/experiments/logs/results/{suite}--{condition}[--note][--shardXofY].jsonl` — one JSON line per rollout: `{task_id, task_name, episode_idx, success, num_steps}`
- Run metadata (config, checkpoint, git commit, python/torch/GPU, timestamp): sibling `*.meta.json`
- Rollout videos: `openvla/rollouts/{date}/`
- Aggregate: `python scripts/aggregate_results.py [--filter <suite substring>]` — per-task +
  suite-wide success rate (mean of per-task rates, matching `eval_results.md`'s convention).
- After a real run, append a summary to `eval_results.md` by hand (it's a narrative document,
  not auto-generated) — don't overwrite existing experiment sections.

## Hardware constraints

- Laptop: 1x GPU, commonly ~8GB VRAM. A bf16 7B checkpoint is ~15GB — **does not fit**. Use
  `--load_in_4bit True` (already the `config/laptop.env` default, ~4GB).
- Check `nvidia-smi --query-gpu=compute_cap --format=csv`: compute capability 12.0 = Blackwell
  (e.g. RTX 50-series laptop GPUs). The `Dockerfile` (PyTorch 2.2.0/CUDA 12.1) predates that
  architecture and the model will fail to load — use `Dockerfile.blackwell` instead (no flash-attn
  wheel for this arch either, hence `OPENVLA_ATTN_IMPLEMENTATION=sdpa` in `laptop.env`).
- Server: plenty of VRAM, `Dockerfile`/`cuda12.1` image, full precision, flash-attn, multiple GPUs
  (`GPUS=0,1,2,3,4`, sharded round-robin across the suite's tasks — see `run_eval.sh`).

## Known pitfalls

- **A built image can silently drift from its Dockerfile.** The `openvla-libero:cuda12.1` image
  on this machine had `mujoco==3.10.0` installed despite the Dockerfile pinning `2.3.2` — breaks
  ALL env stepping (`mj_fullM(): incompatible function arguments`) with an error that looks
  GPU-related but isn't. If you hit this, rebuild the image from the current Dockerfile; don't
  assume an existing tagged image matches its source.
- **LIBERO env stepping/rendering does NOT need CUDA compute** — only `torch.cuda` model
  inference does. You can generate/verify init states and render contact sheets even on a GPU
  whose architecture the built image's PyTorch doesn't support, as long as MuJoCo/EGL rendering
  works (which only needs the host driver, not a CUDA-arch match).
- `.libero/` and `.cache/` at the repo root can end up root-owned if a container was ever run
  without `--user "$(id -u):$(id -g)"` (run_eval.sh always sets this). If you hit permission
  errors writing there, that's why — fix ownership rather than deleting, since `.cache/huggingface`
  may hold downloaded weights.
- Both `openvla/` and `LIBERO/` are gitignored from this repo *by design* — don't try to `git add`
  them here; `cd` in and commit there.

## Coding & import conventions

- Eval code imports as `experiments.robot.libero.*` / `experiments.robot.*` (relative to
  `openvla/`, set via `PYTHONPATH=/workspace/openvla:/workspace/LIBERO` in the Docker image).
- LIBERO scene files use a **shared region catalog** — every task's BDDL defines the same full
  set of named regions (`plate_region`, `next_to_ramekin_region`, `table_center`, cabinet/stove
  sub-regions, etc.), even ones that task doesn't use. New scene variants should place new
  objects on these existing regions where possible instead of inventing new numeric ranges —
  far lower risk of overlap, and no new geometry to verify.

## How to add a benchmark split

1. If it needs a new scene: add BDDL file(s) under `LIBERO/libero/libero/bddl_files/<suite>/`,
   register the suite in `LIBERO/libero/libero/benchmark/libero_suite_task_map.py` (task list)
   and `benchmark/__init__.py` (`@register_benchmark` class + `libero_suites` list), then
   generate + verify init states:
   ```
   docker run ... openvla-libero:cuda12.1 bash -c \
     "cd LIBERO && python3 scripts/gen_suite_init_states.py <suite> && python3 scripts/verify_suite_init_states.py <suite>"
   ```
   (pin `mujoco==2.3.2` first if you hit the drift pitfall above). Inspect the generated contact
   sheet (`LIBERO/scratch_render/<suite>/<suite>_init_grid.png`) before trusting it.
2. If it needs new prompt text: add a dict to `openvla/experiments/robot/libero/instructions.py`,
   keyed by `task.name`.
3. Add one entry to `eval_registry.SPLITS` (and `CONDITIONS` if it's a new condition, not just a
   new suite).
4. Add a row to `benchmark_split.md` (§3) describing the hypothesis/metrics — code and doc must
   agree.
5. Commit scene changes in `LIBERO/`, code changes in `openvla/`, and doc changes here —
   separately, in their own repos.

## Protected files

- `eval_results.md` — append-only, historical record. Never edit or delete existing sections.
- `openvla/checkpoint/`, `openvla/libero_spatial/` (training data, not read at eval time) — large,
  gitignored, never delete without explicit instruction.
- Canonical `libero_spatial` BDDL files — never modify (the 84% baseline depends on exact scene
  geometry); create a new suite instead, even for a one-line change.

## Pre-change / completion checklist

Before: read `benchmark_split.md` for the split you're touching; check `git status` in whichever
repo(s) you're about to edit — don't clobber uncommitted work in `openvla/`/`LIBERO/`.

After: `python3 -m py_compile` on any touched `.py`; if you touched scene BDDLs, re-run
`verify_suite_init_states.py` and eyeball the contact sheet; if you touched `eval_registry.py`,
confirm `benchmark_split.md` still matches; run `python scripts/preflight.py` before claiming a
new machine is eval-ready.
