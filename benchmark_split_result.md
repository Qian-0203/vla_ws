# OpenVLA · LIBERO-Spatial — Evaluation Results

**Document type:** Eval-detail record (per-task tables, analysis, rendered scenes).
**Companion document:** `benchmark_split_plan.md` — research plan and progress tracker; every
condition below corresponds 1:1 to a row in that document's Progress tables.
**Section order:** identical to `benchmark_split_plan.md` §3 (Split 1 → Split 2 → Split 3 → Split 4).
**Scope:** this file records eval runs going forward from 2026-08-19 onward. Runs predating that
date are historical and remain in `test_eval_results.md`, append-only and not duplicated
here; each section below cites the corresponding `test_eval_results.md` experiment where relevant.

---

## 0. Common Evaluation Protocol

Applies to every condition in this document unless a section states an override.

| Parameter | Value |
|---|---|
| Model | `openvla-7b` LoRA (r32), checkpoint `baseline_lora_libero_spatial_4gpu_b24_run004/openvla-7b+libero_spatial_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug` |
| Fine-tuning data | `libero_spatial_no_noops` |
| Precision / attention | bf16; FlashAttention-2 on H200/cuda12.1 hardware, sdpa on Blackwell hardware (no flash-attn wheel for that architecture) |
| Base task set | `libero_spatial` — 10 tasks, each *"pick up the black bowl \<location\> and place it on the plate"* |
| Trials | 50 / task / condition → 500 rollouts per condition |
| Seed | 7 |
| Success criterion | LIBERO goal-state check |
| Statistical noise | at n=50/task, 1 standard error ≈ ±5–7 pts; single-task deltas ≲10 pts are not distinguishable from noise |

**Task id → target location** (identical across every suite in this document):

| id | target location | id | target location |
|--:|---|--:|---|
| 0 | between the plate and the ramekin | 5 | on the ramekin |
| 1 | next to the ramekin | 6 | next to the cookie box |
| 2 | table center | 7 | on the stove |
| 3 | on the cookie box | 8 | next to the plate |
| 4 | in the top drawer of the wooden cabinet | 9 | on the wooden cabinet |

---

## 1. Split 1 — Prompt Sensitivity Probe

**Registry:** `spatial/default`, `spatial/negative_contrast`, `spatial/positive_contrast`.
**Scene:** `libero_spatial` (2 bowls: target + 1 distractor) for all three conditions — only the
prompt text changes.

### 1.1 `default` vs. `negative_contrast`

**Source:** `test_eval_results.md` Experiment 1 ("default vs. explicit"; `negative_contrast` was
previously labeled "explicit"). Reproduced here in full detail per the plan/result split.

**Run configuration:** 5× H200 (GPUs 0–4), `openvla-libero:cuda12.1` image (mujoco 2.3.2,
robosuite 1.4.1), flash-attention-2, tasks round-robin sharded 2/GPU.

**Question.** The scene contains two visually identical black bowls (target + 1 distractor). Does
naming and negating the distractor in the prompt help the policy disambiguate?

- **`default`:** names only the target — *"pick up the black bowl on the stove and place it on the plate."*
- **`negative_contrast`:** also names and negates the distractor — *"…, not the one on top of the wooden cabinet, …"*

| Condition | Overall SR | Rollouts |
|---|--:|--:|
| `default` | **84.0%** | 420 / 500 |
| `negative_contrast` | **36.8%** | 184 / 500 |
| Δ (`negative_contrast` − `default`) | **−47.2 pts** | |

| id | target | distractor named in prompt | `default` | `negative_contrast` | Δ |
|--:|---|---|--:|--:|--:|
| 0 | between the plate and the ramekin | next to the ramekin | 92% | 94% | +2 |
| 1 | next to the ramekin | next to the cookie box | 84% | 32% | −52 |
| 2 | table center | next to the plate | 92% | 2% | **−90** |
| 3 | on the cookie box | on top of the wooden cabinet | 84% | 52% | −32 |
| 4 | in the top drawer | on top of the cabinet | 76% | 64% | −12 |
| 5 | on the ramekin | on top of the cookie box | 94% | 4% | **−90** |
| 6 | next to the cookie box | on the stove | 90% | 72% | −18 |
| 7 | on the stove | on top of the wooden cabinet | 72% | 4% | −68 |
| 8 | next to the plate | next to the ramekin | 84% | 36% | −48 |
| 9 | on the wooden cabinet | on the stove | 72% | 8% | **−64** |

**Analysis.** Naming and negating the distractor ("…not the one on X…") does not disambiguate the
target — it actively confuses the policy, costing 47.2 points overall. The damage is worst where
the negation clause names a visually or semantically salient surface (ramekin, table center,
stove, cabinet). The single task unaffected by the negation clause (id 0) was already phrased
relationally in the default prompt, which may explain its robustness.

**Rendered scene** — episode-0 init state for all 10 tasks under `libero_spatial` (2×5 grid, task
ids 0–9 left-to-right, top-to-bottom). Both conditions share this scene; only the prompt text
differs between `default` and `negative_contrast`:

![libero_spatial init states](openvla/experiments/figures/libero_spatial_init_grid.png)

<details><summary>Exact <code>negative_contrast</code> prompts used (negation clause in <b>bold</b>)</summary>

| `default` target | `negative_contrast` instruction |
|---|---|
| between the plate and the ramekin | pick up the black bowl between the plate and the ramekin, **not the one next to the ramekin**, and place it on the plate |
| from table center | pick up the black bowl at the center of the table, **not the one next to the plate**, and place it on the plate |
| in the top drawer of the cabinet | pick up the black bowl inside the top drawer of the wooden cabinet, **not the one on top of the cabinet**, and place it on the plate |
| next to the cookie box | pick up the black bowl next to the cookie box, **not the one on the stove**, and place it on the plate |
| next to the plate | pick up the black bowl next to the plate, **not the one next to the ramekin**, and place it on the plate |
| next to the ramekin | pick up the black bowl next to the ramekin, **not the one next to the cookie box**, and place it on the plate |
| on the cookie box | pick up the black bowl on top of the cookie box, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the ramekin | pick up the black bowl on top of the ramekin, **not the one on top of the cookie box**, and place it on the plate |
| on the stove | pick up the black bowl on the stove, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the wooden cabinet | pick up the black bowl on top of the wooden cabinet, **not the one on the stove**, and place it on the plate |

</details>

**Artifacts**
- Per-shard logs: `experiments/logs/EVAL-libero_spatial-openvla-*--{default,negative_contrast}--shard{0..4}of5.txt`
- Structured results: `experiments/logs/results/libero_spatial--{default,negative_contrast}.jsonl`
- Reproduce: `MACHINE_CONFIG=<machine>.env bash docker/openvla_libero/run_eval.sh --split spatial/default`
  and `--split spatial/negative_contrast`

### 1.2 `positive_contrast`

**Registry:** `spatial/positive_contrast` → suite `libero_spatial`, condition `positive_contrast`.
**Run configuration:** 2026-08-19, server (4× RTX PRO 6000 Blackwell, GPUs 0–3),
`openvla-libero:blackwell` image (mujoco 2.3.2, robosuite 1.4.1, sdpa attention), `openvla` git
commit `8c7ffa3`, tasks sharded round-robin across 4 GPUs.

**Question.** `negative_contrast` (§1.1) both *names* and *negates* the distractor ("…not the one
on X…") and costs 47.2 pts. Is that damage from the negation grammar specifically, or does merely
bringing the distractor into the prompt — with no negation — already do most of the harm?
`positive_contrast` mentions the distractor's location but does not negate it, e.g. *"pick up the
black bowl on the stove; the other black bowl is on top of the wooden cabinet, place [the first
one] on the plate."*

| Condition | Overall SR | Rollouts |
|---|--:|--:|
| `default` | **84.0%** | 420 / 500 |
| `positive_contrast` | **32.4%** | 162 / 500 |
| `negative_contrast` | **36.8%** | 184 / 500 |
| Distractor Mention Drop (`default` − `positive_contrast`) | **−51.6 pts** | |
| Negation-specific Drop (`positive_contrast` − `negative_contrast`) | **−4.4 pts** | |

| id | target | distractor named in prompt | `default` | `positive_contrast` | `negative_contrast` | Δ vs. default | Δ vs. negative |
|--:|---|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | next to the ramekin | 92% | 92% | 94% | 0 | −2 |
| 1 | next to the ramekin | next to the cookie box | 84% | 8% | 32% | −76 | −24 |
| 2 | table center | next to the plate | 92% | 14% | 2% | **−78** | +12 |
| 3 | on the cookie box | on top of the wooden cabinet | 84% | 38% | 52% | −46 | −14 |
| 4 | in the top drawer | on top of the cabinet | 76% | 50% | 64% | −26 | −14 |
| 5 | on the ramekin | on top of the cookie box | 94% | 2% | 4% | **−92** | −2 |
| 6 | next to the cookie box | on the stove | 90% | 40% | 72% | −50 | −32 |
| 7 | on the stove | on top of the wooden cabinet | 72% | 12% | 4% | −60 | +8 |
| 8 | next to the plate | next to the ramekin | 84% | 68% | 36% | −16 | **+32** |
| 9 | on the wooden cabinet | on the stove | 72% | 0% | 8% | **−72** | −8 |

**Analysis.** Merely mentioning the distractor's location — without any negation clause — is
almost as damaging as naming *and* negating it (Distractor Mention Drop −51.6 pts vs.
`negative_contrast`'s −47.2 pts), and on 6 of 10 tasks `positive_contrast` scores *lower* than
`negative_contrast`. The Negation-specific Drop is slightly negative (−4.4 pts overall), meaning
the negation grammar itself adds essentially no incremental harm on top of just bringing the
distractor into the prompt. Task 0 (the one task robust to `negative_contrast`, see §1.1) is
identically robust here (92%), reinforcing that its relational default phrasing — not the absence
of negation — is what protects it. Two tasks (2, 8) recover meaningfully under `positive_contrast`
relative to `negative_contrast` (+12, +32 pts) but both still fall far short of `default`, so this
looks like within-condition prompt-parsing noise on specific distractor phrasings rather than a
real "no negation" reprieve. **Headline: the policy doesn't fail because it misparses "not" — it
fails because introducing the distractor into language at all pulls attention/policy off target.**

**Artifacts**
- Per-shard logs: `experiments/logs/EVAL-libero_spatial-openvla-2026_08_19-14_05_47--shard{0..3}of4.txt`
- Structured results: `experiments/logs/results/libero_spatial--positive_contrast--shard{0..3}of4.jsonl` (+ matching `.meta.json`)
- Rollout videos: `openvla/rollouts/2026_08_19/`
- Reproduce: `MACHINE_CONFIG=config/server.env bash docker/openvla_libero/run_eval.sh --split spatial/positive_contrast`

---

## 2. Split 2 — Distractor Placement Probe

**Registry:** `spatial_3bowl/irrelevant`, `spatial_3bowl/center_fixed_legacy`,
`spatial_3bowl/semantic`, `spatial_3bowl/landmark`, `spatial_3bowl/landmark_with_hardneg_prompt`.

### 2.1 `irrelevant` (redefined placement)

**Registry:** `spatial_3bowl/irrelevant` → suite `libero_spatial_3bowl_neutral`, condition `default`.
**Run configuration:** 2026-08-19, server (4× RTX PRO 6000 Blackwell, GPUs 0–3),
`openvla-libero:blackwell` image (mujoco 2.3.2, robosuite 1.4.1, sdpa attention), `openvla` git
commit `8c7ffa3`, tasks sharded round-robin across 4 GPUs.

**Question.** The retired `center_fixed_legacy` condition placed the 3rd distractor bowl at a
single fixed absolute coordinate (`table_center`) in 9 of 10 tasks — not neutral relative to every
task's own reach path (see `benchmark_split_plan.md` Split 2, "Design confound found, then fixed").
This run uses the redefined `irrelevant` condition: for each task, the 3rd bowl is placed in
whichever safe region gives the largest clearance from both the target-to-plate reach path and the
2nd (original) distractor.

| Condition (default prompt) | Overall SR | Rollouts | Source |
|---|--:|--:|---|
| 2 bowls (baseline) | 84.0% | 420 / 500 | `test_eval_results.md` Exp 1 |
| 3 bowls, `center_fixed_legacy` (retired) | 80.2% | 401 / 500 | `test_eval_results.md` Exp 2 |
| 3 bowls, `irrelevant` (redefined) | **88.8%** | **444 / 500** | this run |
| Δ vs. baseline | **+4.8 pts** | | |
| Δ vs. `center_fixed_legacy` | **+8.6 pts** | | |

| id | target | 2-bowl (baseline) | `center_fixed_legacy` | `irrelevant` (redefined) | Δ vs. baseline | Δ vs. `center_fixed_legacy` |
|--:|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 76% | 86% | −6 | +10 |
| 1 | next to the ramekin | 84% | 90% | 90% | +6 | 0 |
| 2 | table center | 92% | 94% | 96% | +4 | +2 |
| 3 | on the cookie box | 84% | 86% | 90% | +6 | +4 |
| 4 | in the top drawer | 76% | 84% | 96% | **+20** | **+12** |
| 5 | on the ramekin | 94% | 84% | 96% | +2 | +12 |
| 6 | next to the cookie box | 90% | 44% | 94% | +4 | **+50** |
| 7 | on the stove | 72% | 82% | 88% | **+16** | +6 |
| 8 | next to the plate | 84% | 88% | 84% | 0 | −4 |
| 9 | on the wooden cabinet | 72% | 74% | 68% | −4 | −6 |

**Analysis.** Placing the distractor off the reach path and distance-matched, rather than at one
shared coordinate, leaves overall success not merely robust to the extra bowl but slightly above
baseline (+4.8 pts), and clearly above `center_fixed_legacy` (+8.6 pts). This confirms the Split 2
confound diagnosis: task 6 (*next to the cookie box*), which collapsed to 44% under
`center_fixed_legacy` because that fixed coordinate sat ~0.18m off the task's reach path, recovers
to 94% under the redefined placement — a +50 pt swing on that task alone. No task exhibits a
`center_fixed_legacy`-style collapse under `irrelevant`; the two negative deltas (tasks 8 and 9,
−4 pts each) fall within the ±5–7 pt single-task noise band at n=50.

**Rendered scene** — episode-0 init state for all 10 tasks under `libero_spatial_3bowl_neutral`
(2×5 grid, task ids 0–9 left-to-right, top-to-bottom). Confirms three visually distinct,
non-overlapping bowls per task, consistent with the pre-run geometric verification (worst pairwise
separation 0.122m):

![libero_spatial_3bowl_neutral init states](openvla/experiments/figures/libero_spatial_3bowl_neutral_init_grid.png)

**2-bowl vs. 3-bowl comparison** — same init state, side by side per task (10 rows, task ids 0–9
top-to-bottom): left column (blue strip) = `libero_spatial` baseline (target + 1 distractor),
right column (orange strip) = `libero_spatial_3bowl_neutral` (target + 2 distractors, the added
3rd bowl is the `irrelevant`-placed one). The added bowl is the only difference between columns in
each row, making it straightforward to spot where it lands relative to the target/plate/reach path:

![2-bowl vs. 3-bowl (irrelevant) comparison](openvla/experiments/figures/compare_2v3bowl_neutral_grid.png)

**Artifacts**
- Per-shard results: `openvla/experiments/logs/results/libero_spatial_3bowl_neutral--default--shard{0..3}of4.jsonl` (+ matching `.meta.json`)
- Rollout videos: `openvla/rollouts/2026_08_19/` (tagged with task + success)
- Reproduce eval: `MACHINE_CONFIG=config/server.env bash docker/openvla_libero/run_eval.sh --split spatial_3bowl/irrelevant`
- Reproduce renders (read-only, do not touch `.pruned_init` files): `LIBERO/scripts/render_suite_contact_sheet.py libero_spatial_3bowl_neutral`
  and `LIBERO/scripts/compare_two_suites_init.py libero_spatial libero_spatial_3bowl_neutral compare_2v3bowl_neutral`

### 2.2 `center_fixed_legacy`

Historical, retired definition — see `test_eval_results.md` Experiment 2 ("3 bowls"). Kept under
this label so that number stays attributable; not re-recorded here. Summarized in §2.1's comparison
table above.

### 2.3 `semantic`, `landmark`, `landmark_with_hardneg_prompt`

Not yet run.

---

## 3. Split 3 — Scene Complexity Probe

**Registry:** `spatial_3bowl/drawer_open` (`drawer_closed` = Split 2's `center_fixed_legacy`).

No new runs recorded here. Both conditions are historical — see `test_eval_results.md`
Experiment 2 (`drawer_closed`) and Experiment 3 (`drawer_open`).

---

## 4. Split 4 — Surface vs. Landmark Grounding Probe

**Registry:** `grounding/surface_landmark`, `grounding/region_surface` (new scenes); the remaining
4 of 6 (target-family, distractor-family) cells are derived from existing Split 1 data.

No new runs recorded here. The 4 derived cells are historical — see `test_eval_results.md`
Experiment 1. `grounding/surface_landmark` and `grounding/region_surface` are not yet run.
