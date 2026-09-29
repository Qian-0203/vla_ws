# LIBERO-Spatial Grounding Benchmark

A set of controlled variations of LIBERO's `libero_spatial` suite. Each variation changes **one
thing at a time**: the prompt wording, where an extra distractor bowl sits, scene clutter, or how the
target location is described. The goal is to tell real spatial/language grounding apart from
overfitting to the training prompts and scenes. It runs any OpenVLA-format checkpoint fine-tuned on
`libero_spatial`.

## Quickstart

Requirements: Linux, an NVIDIA GPU, Docker, and `nvidia-container-toolkit`. A bf16 run needs about
17 GB of VRAM.

```bash
git clone --recursive https://github.com/Qian-0203/vla_ws.git && cd vla_ws

# 1. Build the image. Check `nvidia-smi --query-gpu=compute_cap --format=csv`:
docker build -f docker/openvla_libero/Dockerfile -t openvla-libero:cuda12.1 .            # compute_cap < 12.0
# docker build -f docker/openvla_libero/Dockerfile.blackwell -t openvla-libero:blackwell . # compute_cap 12.0
#   (Blackwell also needs: export IMAGE_NAME=openvla-libero:blackwell OPENVLA_ATTN_IMPLEMENTATION=sdpa)

export CHECKPOINT=/path/to/your/openvla/checkpoint   # can live anywhere on the host
export GPUS=0                                        # e.g. 0,1,2,3 shards tasks across GPUs

# 2. Host-side checks: GPU, image, checkpoint format, and norm-stats key
python3 scripts/preflight.py

# 3. Sanity gate: the unmodified baseline (500 rollouts)
SPLITS=spatial/default bash scripts/run_benchmark.sh

# 4. Everything else (about 4,400 more rollouts). Already-finished splits are skipped with --resume.
bash scripts/run_benchmark.sh --resume True
```

At the end the script prints the per-task and per-split success rates. Raw results are written to
`openvla/experiments/logs/results/*--<run note>*.jsonl`, with one line per rollout plus a
`.meta.json` holding the config. Rollout videos go to `openvla/rollouts/`. The run note defaults to
the checkpoint's directory name; override it with `RUN_NOTE=...`.

## Before you trust the numbers: checklist, most important first

The first items can drop success rate by tens of points without raising any error. Work through them
in order.

1. **Checkpoint format.** The checkpoint must be a *merged* Hugging Face OpenVLA checkpoint:
   `config.json`, `model-*.safetensors`, and `dataset_statistics.json`. A bare LoRA adapter has to be
   merged into `openvla-7b` first. `preflight.py` checks this.
2. **Action un-normalization key.** Every split un-normalizes actions with the `libero_spatial` (or
   `libero_spatial_no_noops`) statistics from `dataset_statistics.json`. If your model was trained on
   a mixture, that key has to be present. `preflight.py` checks this.
3. **`CENTER_CROP` must match training.** Leave it at `True` (the default) if you fine-tuned with
   OpenVLA's `--image_aug`. Otherwise set `CENTER_CROP=False`.
4. **Precision.** Use bf16 (the default). With `LOAD_IN_4BIT=True`, the numbers can't be compared to
   the reference column below.
5. **Environment.** Use the image that matches your GPU architecture. The reference baseline gives
   the same result either way: 84.0% on H200 with flash-attn and 84.4% on Blackwell with sdpa. If
   env stepping fails with `mj_fullM(): incompatible function arguments`, the image has drifted to
   the wrong MuJoCo version; rebuild it. If you see
   `MUJOCO_EGL_DEVICE_ID ... between 0 and -1`, the host is missing
   `libnvidia-gl-<driver>-server`.
6. **Protocol.** Keep 50 trials/task and seed 7, which are the defaults. `num_steps_wait=10` and
   `max_steps=220` are fixed in code.
7. **Sanity gate.** Your `spatial/default` result should be close to your model's own
   `libero_spatial` success rate before you read anything into the other splits. Noise: ±3.3 pts
   pooled over 500 rollouts, ±5–7 pts on a single task.

## Splits

Reference column: `openvla-7b` LoRA r32 fine-tuned on `libero_spatial_no_noops`, bf16, 50 trials/task,
seed 7. Rows are ordered as `run_benchmark.sh` runs them: baseline first, then roughly by how much
each split moved the reference model. Task ids are the same across all scenes.

| Split id | Probe | What changes vs. baseline | Tasks | Reference SR |
|---|---|---|---|--:|
| `spatial/default` | 1 · Prompt | nothing (baseline: 2 bowls, native prompt) | 0–9 | 84.0% |
| `spatial/positive_contrast` | 1 · Prompt | prompt also states where the distractor bowl is | 0–9 | 32.4% |
| `spatial/negative_contrast` | 1 · Prompt | prompt names the distractor and negates it ("not the one …") | 0–9 | 36.8% |
| `grounding/target_cue_region` | 4b · Cue type | target described as a table zone ("back-left of the table") | 0,1,3,5–9 | 17.0% |
| `grounding/target_cue_landmark` | 4b · Cue type | "on X" rephrased as "next to X" | 3,5,7,9 | 30.5% (default on these tasks: 80.5%) |
| `grounding/target_cue_proximity_novel` | 4c · Cue type | "on X" rephrased as "close to X" (never seen in training) | 3,5,7,9 | 53.0% |
| `spatial_3bowl/landmark_with_hardneg_prompt` | 1×2 | `landmark` scene + prompt that disambiguates the target | 0–9 | 41.2% |
| `spatial_3bowl/drawer_open` | 3 · Clutter | 3 bowls + cabinet top drawer open | 0–9 | 60.0% (73.1% excl. tasks 3,6,7, which the drawer physically blocks) |
| `spatial_3bowl/landmark` | 2 · Distractor | 3rd bowl near the target's own landmark (hard negative) | 0–9 | 80.6% |
| `spatial_3bowl/irrelevant` | 2 · Distractor | 3rd bowl far from everything | 0–9 | 85.2% |
| `spatial_3bowl/semantic` | 2 · Distractor | 3rd bowl at a different named landmark | 0–9 | 85.2% |
| `grounding/surface_landmark` | 4a · Scene | surface-cue target + landmark-cue distractor | 0 (1-task suite) | 88.0% |
| `grounding/region_surface` | 4a · Scene | region-cue target + surface-cue distractor | 0 (1-task suite) | 92.0% |

`run_benchmark.sh` skips the `*_legacy` splits in the registry. They are retired scene definitions,
kept only so that old numbers stay traceable. The one exception you might want is the closed-drawer
counterpart of `drawer_open` (80.2% reference), which Split 3's Δ is measured against. Run it with
`SPLITS=spatial_3bowl/center_fixed_legacy bash scripts/run_benchmark.sh`.

## Repository layout

```
vla_ws/                      entry point: Docker, launchers, docs
  scripts/run_benchmark.sh     run all current splits for one checkpoint, then aggregate
  scripts/preflight.py         host-side checks before a run
  scripts/aggregate_results.py per-task / per-split success rates from the results JSONL
  docker/openvla_libero/       Dockerfiles + run_eval.sh (runs a single split)
  config/                      machine profiles (laptop.env, server.env.example); optional
  docs/                        research record (see below)
  openvla/   (submodule)       eval code: experiments/robot/libero/{run_libero_eval,eval_registry,instructions}.py
  LIBERO/    (submodule)       scenes: libero/libero/bddl_files/<suite>/, init_files/<suite>/
```

- **Split → (scene suite, prompt condition):** `openvla/experiments/robot/libero/eval_registry.py`
- **Prompt text for each condition:** `openvla/experiments/robot/libero/instructions.py`
- **Scene renders:** `LIBERO/scratch_render/<suite>/<suite>_init_grid.png` (every current suite
  except `libero_spatial_3bowl_open`)

To run a single split by hand:
`bash docker/openvla_libero/run_eval.sh --split <id> [--task_ids 3,5,7,9] [--num_trials_per_task 1]`.

## Further reading

- `docs/benchmark_split_plan.md`: the hypothesis behind each split, scene design, and distractor
  coordinates.
- `docs/benchmark_split_result.md`: exact prompt text, renders, full per-task results, and analysis
  for every condition.
- `docs/eval_log.md`: chronological launch log for the reference numbers above, plus the
  hardware and server-setup reference.
- `docs/mechanistic_localization.md`: an optional diagnostic that probes *where* in the network the
  prompt-induced failures originate (logit lens, attention mass, decision confidence). It is not
  needed to run the benchmark.
