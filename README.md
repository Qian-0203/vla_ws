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

# 4. Everything else (about 8,300 more rollouts). Already-finished splits are skipped with --resume.
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
seed 7. These are the originally recorded values. For 11 conditions the original raw files were lost and the
conditions were re-run on 2026-10-11 (`docs/result_tables/` uses the re-run): six match per task exactly, the
others differ by 0.4–3.2 pts (`default` 84.4, `negative_contrast` 40.0, `drawer_open` 60.8, `irrelevant` 86.0,
closed-drawer control 82.2). Rows are ordered as `run_benchmark.sh` runs them: baseline first, then grouped by probe,
roughly by how much each group moved the reference model. Task ids are the same across all scenes.

| Split id | Probe | What changes vs. baseline | Tasks | Reference SR |
|---|---|---|---|--:|
| `spatial/default` | 1 · Prompt | nothing (baseline: 2 bowls, native prompt) | 0–9 | 84.0% |
| `spatial/positive_contrast` | 1 · Prompt | prompt also states where the distractor bowl is | 0–9 | 32.4% |
| `spatial/negative_contrast` | 1 · Prompt | prompt names the distractor and negates it ("not the one …") | 0–9 | 36.8% |
| `spatial/length_control_suffix` | 1 · Length control | native prompt + a content-free clause at the end (+13 tokens, matches `positive_contrast`'s length) | 0–9 | 53.6% |
| `spatial/length_control_infix` | 1 · Length control | native prompt + a content-free clause mid-sentence (+11 tokens, matches `negative_contrast`'s length) | 0–9 | 62.8% |
| `spatial/negation_only` | 1 · Prompt | target named only by negating the distractor ("the black bowl that is not on …") | 0–9 | 2.6% |
| `grounding/target_cue_region` | 4b · Cue type | **withdrawn (wording used swapped axes, see below)** — superseded by `grounding/target_cue_region_img` | 0,1,3,5–9 | ~~17.0%~~ |
| `grounding/target_cue_region_v2` | 4b · Cue type | **withdrawn** — superseded by `grounding/target_cue_region_v2_img` | 0,1,3,5–9 | ~~20.0%~~ |
| `grounding/target_cue_region_v3` | 4b · Cue type | **withdrawn** — superseded by `grounding/target_cue_region_v3_img` | 0,1,3,5–9 | ~~18.8%~~ |
| `grounding/target_cue_region_img` | 4b · Cue type | target described as a table zone, direction words in the policy's image frame ("at the back-right of the table") | 0,1,3,5–9 | not run on the reference (see below) |
| `grounding/target_cue_region_v2_img` | 4b · Cue type | same zones, 2nd wording ("in the back-right area of the table") | 0,1,3,5–9 | not run on the reference (see below) |
| `grounding/target_cue_region_v3_img` | 4b · Cue type | same zones, 3rd wording (side-first) | 0,1,3,5–9 | not run on the reference (see below) |
| `grounding/target_cue_landmark` | 4b · Cue type | "on X" rephrased as "next to X" | 3,5,7,9 | 30.5% (default on these tasks: 80.5%) |
| `grounding/target_cue_proximity_novel` | 4c · Cue type | "on X" rephrased as "close to X" (never seen in training) | 3,5,7,9 | 53.0% |
| `grounding/target_cue_proximity_beside` | 4c · Cue type | "on X" rephrased as "beside X" | 3,5,7,9 | 38.0% |
| `grounding/target_cue_proximity_near` | 4c · Cue type | "on X" rephrased as "near X" | 3,5,7,9 | 49.0% |
| `grounding/target_cue_proximity_adjacent` | 4c · Cue type | "on X" rephrased as "adjacent to X" | 3,5,7,9 | 37.5% |
| `spatial_3bowl/landmark_with_hardneg_prompt` | 1×2 | `landmark` scene + prompt that disambiguates the target | 0–9 | 41.2% |
| `spatial_3bowl/drawer_open` | 3 · Clutter | 3 bowls + cabinet top drawer open | 0–9 | 60.0% (73.1% excl. tasks 3,6,7, which the drawer physically blocks) |
| `spatial_3bowl/landmark` | 2 · Distractor | 3rd bowl near the target's own landmark (hard negative) | 0–9 | 80.6% |
| `spatial_3bowl/irrelevant` | 2 · Distractor | 3rd bowl far from everything | 0–9 | 85.2% |
| `spatial_3bowl/semantic` | 2 · Distractor | 3rd bowl at a different named landmark | 0–9 | 85.2% |
| `grounding/paraphrase_lexical` | 4 · Same-cue paraphrase | one synonym swap in the native cue ("on" → "on top of", "next to" → "beside") | 0–9 | 72.4% |
| `grounding/paraphrase_syntactic` | 4 · Same-cue paraphrase | native words in a relative clause ("the black bowl that is on X") | 0–9 | 82.2% |
| `grounding/surface_landmark` | 4a · Scene | surface-cue target + landmark-cue distractor | 0 (1-task suite) | 88.0% |
| `grounding/region_surface` | 4a · Scene | region-cue target + surface-cue distractor | 0 (1-task suite) | 92.0% |

**Region-cue erratum (2026-10-11).** The first three `target_cue_region*` rows used left/right and
front/back words derived from a swapped coordinate assumption (+x is toward the camera and +y is
image-right, not the reverse), so 7 of 8 prompts named the wrong place. Their numbers are struck
through and must not be used as reference values; `run_benchmark.sh` now runs the corrected `_img`
conditions instead. Those have no number for the reference checkpoint; they were run on the eight public
checkpoints below (52.5–84.0% for OFT, OFT 4-suite, VLA-Adapter Pro, X-VLA, GR00T and Xiaomi-R0; 29.0–36.0% for pi0 and VLA-Adapter Spatial). Details: `docs/benchmark_split_result.md`,
Erratum E1.

`run_benchmark.sh` skips the `*_legacy` splits in the registry. They are retired scene definitions,
kept only so that old numbers stay traceable. The one exception you might want is the closed-drawer
counterpart of `drawer_open` (80.2% reference), which Split 3's Δ is measured against. Run it with
`SPLITS=spatial_3bowl/center_fixed_legacy bash scripts/run_benchmark.sh`.

### Other checkpoints

The same splits were run on eleven other checkpoints: the official `openvla-7b-finetuned-libero-spatial`,
QwenVLA (run by Ken Zheng), `pi05_libero`, and, on 2026-10-12 → 2026-10-13, `pi0_libero`, OpenVLA-OFT,
VLA-Adapter Spatial and Spatial-Pro, X-VLA, GR00T N1.5, OpenVLA-OFT 4-suite
(`openvla-7b-oft-finetuned-libero-spatial-object-goal-10`) and Xiaomi-Robotics-0. Except the official OpenVLA checkpoint, these are not
OpenVLA-format checkpoints: they run through their own clients or policy servers and queue scripts
(`scripts/queue_*.sh`, `scripts/run_eval_qwenvla.sh`; setup in `docs/eval_log.md`), not through the quickstart above.
One run per condition, seed 7, 50 trials/task, success counted within 230 environment steps. The policies use
different action protocols (chunk sizes, cameras, absolute vs. delta actions; `docs/benchmark_split_result.md` §13),
so read each column against its own `spatial/default`, not against another column.

| Split | official OpenVLA | QwenVLA | pi05 | pi0 | OFT | VLA-Ad. Spatial | VLA-Ad. Pro | X-VLA | GR00T N1.5 | OFT 4-suite | Xiaomi-R0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `spatial/default` | 84.0% | 79.8% | 98.4% | 96.4% | 98.2% | 95.8% | 97.8% | 97.4% | 89.0% | 97.4% | 98.8% |
| `spatial/positive_contrast` | 32.8% | 54.8% | 92.2% | 62.6% | 77.8% | 53.6% | 71.4% | 82.0% | 73.2% | 86.0% | 78.4% |
| `spatial/negative_contrast` | 42.4% | 37.2% | 87.0% | 55.0% | 76.4% | 55.6% | 70.4% | 89.6% | 73.2% | 85.2% | 55.0% |
| `spatial/length_control_suffix` | 51.2% | 76.2% | 99.0% | 69.2% | 97.8% | 92.6% | 98.4% | 97.0% | 87.4% | 96.6% | 97.8% |
| `spatial/length_control_infix` | 63.6% | 74.8% | 99.2% | 81.4% | 97.2% | 91.8% | 98.4% | 97.0% | 87.4% | 97.6% | 98.4% |
| `spatial/negation_only` | 5.0% | 1.8% | 44.8% | 32.8% | 43.6% | 22.6% | 34.2% | 65.8% | 35.8% | 72.6% | 35.8% |
| `grounding/target_cue_region_img` | — | — | — | 32.5% | 71.8% | 31.5% | 52.5% | 64.5% | 59.5% | 83.5% | 66.8% |
| `grounding/target_cue_region_v2_img` | — | — | — | 30.5% | 73.8% | 35.5% | 54.8% | 69.0% | 55.5% | 82.8% | 67.0% |
| `grounding/target_cue_region_v3_img` | — | — | — | 36.0% | 72.2% | 29.0% | 54.2% | 65.2% | 57.5% | 84.0% | 67.5% |
| `grounding/target_cue_landmark` | 25.0% | 39.5% | 87.0% | 46.0% | 85.5% | 63.5% | 83.5% | 91.5% | 78.0% | 96.0% | 94.5% |
| `grounding/target_cue_proximity_novel` | 37.0% | 46.0% | 96.0% | 74.0% | 96.5% | 42.0% | 85.5% | 91.0% | 82.5% | 95.5% | 99.5% |
| `grounding/target_cue_proximity_beside` | 31.5% | 49.0% | 97.0% | 93.0% | 97.0% | 66.5% | 89.5% | 95.0% | 76.0% | 95.5% | 96.0% |
| `grounding/target_cue_proximity_near` | 45.0% | 50.5% | 96.0% | 96.0% | 93.5% | 72.5% | 88.5% | 96.5% | 80.5% | 96.0% | 97.5% |
| `grounding/target_cue_proximity_adjacent` | 31.0% | 44.5% | 97.0% | 82.0% | 96.5% | 61.5% | 88.0% | 92.5% | 78.0% | 97.0% | 98.0% |
| `spatial_3bowl/landmark_with_hardneg_prompt` | 42.2% | 47.4% | 66.2% | 46.6% | 59.6% | 46.6% | 70.0% | 66.0% | 65.8% | 59.6% | 83.4% |
| `spatial_3bowl/drawer_open` | 64.6% | 68.2% | 43.4% | 40.2% | 27.2% | 33.4% | 49.4% | 39.4% | 48.4% | 13.6% | 64.2% |
| `spatial_3bowl/landmark` | 77.0% | 68.6% | 72.0% | 69.0% | 65.6% | 67.4% | 75.8% | 75.4% | 74.4% | 63.6% | 80.0% |
| `spatial_3bowl/irrelevant` | 82.8% | 79.0% | 80.6% | 86.0% | 73.6% | 80.6% | 81.2% | 82.8% | 88.2% | 67.8% | 75.2% |
| `spatial_3bowl/semantic` | 83.2% | 74.0% | 91.0% | 92.2% | 81.0% | 83.0% | 87.0% | 78.8% | 81.8% | 50.8% | 85.8% |
| `grounding/paraphrase_lexical` | 79.0% | 80.0% | 97.2% | 90.8% | 97.2% | 71.8% | 98.6% | 92.6% | 86.8% | 96.2% | 98.6% |
| `grounding/paraphrase_syntactic` | 84.8% | 81.0% | 97.8% | 97.0% | 97.2% | 92.8% | 99.8% | 90.2% | 87.4% | 97.4% | 99.2% |
| `grounding/surface_landmark` | 88.0% | 90.0% | 100.0% | 94.0% | 90.0% | 98.0% | 100.0% | 90.0% | 62.0% | 98.0% | 96.0% |
| `grounding/region_surface` | 82.0% | 98.0% | 100.0% | 96.0% | 100.0% | 88.0% | 92.0% | 90.0% | 96.0% | 100.0% | 100.0% |
| `no_location` (instruction file) | 17.6% | 27.6% | 81.4% | 51.2% | 66.0% | 30.2% | 63.0% | 59.6% | 65.8% | 84.4% | 76.8% |
| `target_swap` (instruction file) | 57.2% | 50.2% | 56.8% | 68.2% | 37.4% | 36.4% | 51.2% | 32.2% | 40.8% | 21.8% | 38.6% |

`no_location` drops the location phrase and scores the original target (with two identical bowls it has no unique
correct answer); `target_swap` names the other bowl and scores that bowl (instruction files under
`screening/language_stress/candidates/`). `surface_landmark` and `region_surface` are 50 rollouts each (one task);
the four proximity cues and `target_cue_landmark` 200 (tasks 3, 5, 7, 9); the `_img` region cues 400. QwenVLA's
`negation_only` / `no_location` / `target_swap` come from a separate 2026-10-11 package and are not yet in
`docs/result_tables/`. Per-task numbers, which-bowl breakdowns and caveats: `docs/benchmark_split_result.md`
(summary at the top, §11, §13, §15) and `docs/result_tables/`. Data coverage and what is left out:
`docs/data_status_2026-10-13.md`.

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
- `docs/result_tables/`: condition x checkpoint success rates and which-bowl counts as CSV, for ten checkpoints.
- `docs/data_status_2026-10-13.md`: which checkpoint has which condition, data kept out of the tables
  (exploratory, withdrawn, excluded), repeated runs, and where the raw files are.
- `docs/mechanistic_localization.md`: an optional diagnostic that probes *where* in the network the
  prompt-induced failures originate (logit lens, attention mass, decision confidence). It is not
  needed to run the benchmark.
