# OpenVLA · LIBERO-Spatial Benchmark — Detailed Results

**What this file is.** The detailed record for every split/condition: the exact eval setting
(instruction/prompt text, scene render, distractor placement), render comparisons against baseline,
full per-task results, and the analysis/takeaway for each. It also carries the status dashboard
(run vs. not, computed drop metrics) and the cross-experiment findings as they accumulate. This file
is the authoritative source for *what a condition's setting was and what its numbers were*.

**What it is not.** Not the launch record — *when* something was run, in what batch/order, on what
hardware, and the exact relaunch command live in `eval_log.md`. Not the split design/rationale either
— hypotheses, why a split exists, and open design questions live in `benchmark_split_plan.md`.

**Update rule: every time a real GPU eval finishes, update this file** (add/extend the condition's
detail section, the status table, and the cross-experiment findings if the picture changed) **and
append an entry to `eval_log.md`.** `benchmark_split_plan.md` only needs an update when a split's
*definition* changes (new condition, redefined placement, newly authored scene).

## 0. Common eval setting

Applies to every condition below unless a section says otherwise.

- **Model:** `openvla-7b` LoRA (r32) fine-tuned on `libero_spatial_no_noops`, bf16 (hardware/attention
  kernel per batch: `eval_log.md`).
- **Base suite:** `libero_spatial` — 10 tasks; each = *"pick up the black bowl \<location\> and place
  it on the plate."*
- **Protocol:** 50 trials/task = 500 rollouts per condition, seed 7. Success = LIBERO's own goal
  predicate for that task.
- **Task ids** are identical across every suite, so columns/rows line up 1:1 everywhere below:

  | id | target location | id | target location |
  |--:|---|--:|---|
  | 0 | between the plate and the ramekin | 5 | on the ramekin |
  | 1 | next to the ramekin | 6 | next to the cookie box |
  | 2 | table center | 7 | on the stove |
  | 3 | on the cookie box | 8 | next to the plate |
  | 4 | in the top drawer of the wooden cabinet | 9 | on the wooden cabinet |

- **Noise:** at 50 trials/task, 1 standard error ≈ ±5–7 pts on a single task, ≈ ±3.3 pts pooled over
  500 — treat single-task swings ≲10 pts, or overall swings ≲3–4 pts, as noise.
- **Baseline reference:** `spatial/default` — 2 bowls, default prompt, `libero_spatial` scene, 84.0%
  (420/500) — everything below compares against this unless noted.

## 1. Status overview

| Split | Registry status | Data status |
|---|---|---|
| 1. Prompt Sensitivity | 3/3 conditions implemented + 2 length controls (`length_control_infix`/`_suffix`, authored 2026-10-01) | 3/3 run (`default`, `negative_contrast`, `positive_contrast`); both length controls run 2026-10-01 — `length_control_infix` 62.8%, `length_control_suffix` 53.6% (500/500 each): a content-free clause alone costs 21–30 pts (§2) |
| 2. Distractor Placement | 3/4 conditions implemented (`path` not authored) | All implemented conditions run. `irrelevant` and `semantic` each redefined a second time (see §3) — current suites (`libero_spatial_3bowl_front` 85.2%, `libero_spatial_3bowl_semantic2` 85.2%, both 500/500) now run; prior data survives relabeled `irrelevant_v1_legacy` (88.8%) / `semantic_v1_legacy` (84.8%); original fixed-coordinate data survives as `center_fixed_legacy`. Only unauthored `path` remains beyond that |
| 3. Scene Complexity | Implemented | Run (both conditions) |
| 4. Surface vs. Landmark Grounding | 4a: all 6 cells implemented; 4b: implemented as a target cue-type probe; 4c: implemented as a familiar-vs-novel proximity-cue probe | 4a/4b/4c fully run — 4c's `target_cue_proximity_novel` (53.0% pooled, tasks 3/5/7/9) came in *above* `target_cue_landmark` (30.5%), the opposite of the plan's predicted direction; full three-way synthesis in §5.4. 4b/4c paraphrase controls (`paraphrase_lexical`/`_syntactic`, `target_cue_region_v2`/`_v3`, `target_cue_proximity_beside`/`_near`/`_adjacent`) all run 2026-10-01 (§5.5): same-cue paraphrases 72.4% / 82.2%; region cue 17.0 / 20.0 / 18.8% across three wordings; novel proximity synonyms 37.5–53.0% |
| VLM Bowl-Pointing Probe (§8, not a `SPLITS` entry) | Script implemented (`probe_bowl_pointing.py`) | OpenVLA itself: dead end confirmed on 3 angles — no language-responsive text channel. Qwen2-VL-7B alternative (§8.1): a marker-placement bug (§8.2) was found and fixed; re-run scores 70% on both 2-bowl distractor-mention conditions and `hardneg` (was 40-60%). `default`/`hardneg_default` no-mention baselines added (§8.3): 50% (2-bowl, exactly chance) and 60% (3-bowl, above chance) — distractor-mention phrasing is a mild *disambiguating* cue for Qwen in both scenes, not a difficulty source |
| Bowl-Attraction Probe (§8.5, not a `SPLITS` entry) | Script implemented (`probe_bowl_attraction.py`) | Run 2026-09-02/03/04, all 10 `libero_spatial` tasks for `default`+`negative_contrast` (280 rollouts); `target_cue_landmark` + `target_cue_proximity_novel` on the 4-task surface cohort (80 more). Reads OpenVLA's own failure mode via instrumented action rollouts (not VQA): pooled at full scale, "arm never approaches either bowl" is the *majority* failure mode (58.1% of `negative_contrast` failures) — success rates closely match the real 500/200-trial evals. `target_cue_proximity_novel` (no distractor, no misleading template) fails the same way as `target_cue_landmark` despite a much higher success rate — a second line of evidence against distractor-pull. Task 3 shows a persistent second mode (correct approach, still fails). See §8.6 for the synthesis and open gaps |
| Cross-model reference: `pi05_libero` (§9) | Runs through the same registry via `--model_family openpi` | All 22 current conditions run 2026-09-30 → 2026-10-01 (8,300 rollouts). `spatial/default` 98.4%; prompt-only conditions 87–99% except the region cue (75–78%); 3-bowl scene conditions 43–91% |
| Mechanistic Localization Probe (not a `SPLITS` entry) | Script implemented (`probe_mechanistic_localization.py`) | Run 2026-09-04 → 2026-09-10 on task 5. Real, n=50-replicated early-window vision-attention/resolution-layer effects for `target_cue_landmark`; no layer/module localized; confidence diagnostic null. Full record: `mechanistic_localization.md` |

---

## 2. Split 1 — Prompt Sensitivity Probe

**Question.** The 2-bowl scene has a target + 1 identical distractor bowl. Does *telling the model
which bowl to avoid* help, and does *how* it's told (mention vs. negate) matter? Only the **prompt
string** changes across conditions; scene and init states are identical.

| Condition | Status | Overall SR | Rollouts |
|---|---|--:|--:|
| `default` — names only the target | ✅ run | **84.0%** | 420/500 |
| `positive_contrast` — states distractor's location as fact, no negation | ✅ run | **32.4%** | 162/500 |
| `negative_contrast` — names + negates distractor ("…not the one…") | ✅ run | **36.8%** | 184/500 |
| `length_control_infix` — native wording + content-free clause mid-sentence (+11 tokens) | ✅ run | **62.8%** | 314/500 |
| `length_control_suffix` — native wording + content-free clause at the end (+13 tokens) | ✅ run | **53.6%** | 268/500 |

Computed: `Negative Contrast Drop = 84.0 − 36.8 = 47.2 pts`. `Distractor Mention Drop = 84.0 − 32.4 =
51.6 pts`. `Negation-specific Drop = 32.4 − 36.8 = −4.4 pts` — negative, meaning the negation clause
is not the source of the damage; bare mention of a second location does effectively all of it alone.

`Length-only Drop = 84.0 − 62.8 = 21.2 pts` (infix) and `84.0 − 53.6 = 30.4 pts` (suffix);
`Content Drop = 62.8 − 36.8 = 26.0 pts` (negative) and `53.6 − 32.4 = 21.2 pts` (positive). So about
half of each contrast drop is reproduced by content-free text of the same length, and "bare mention
does effectively all of it" above holds only for the half that remains (see the length-control
setting below).

**Split fully run.**

### Setting: `default` vs. `negative_contrast`

- **Default:** *"pick up the black bowl on the stove and place it on the plate."*
- **Negative contrast:** also names the distractor — *"…, not the one on top of the wooden cabinet, …"*
- Prompts: `openvla/experiments/robot/libero/instructions.py::LIBERO_SPATIAL_NEGATIVE_CONTRAST_INSTRUCTIONS`
  (historically named "explicit").
- Scene: stock `libero_spatial` (2 bowls), unmodified — no render check needed.

| id | target | distractor named in prompt | Default | Neg. contrast | Δ |
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
| 9 | on the wooden cabinet | on the stove | 72% | 8% | −64 |

<details><summary>Exact negative-contrast prompts used (distractor clause in <b>bold</b>)</summary>

| Default target | Negative-contrast instruction |
|---|---|
| between the plate and the ramekin | pick up the black bowl between the plate and the ramekin, **not the one next to the ramekin**, and place it on the plate |
| table center | pick up the black bowl at the center of the table, **not the one next to the plate**, and place it on the plate |
| in the top drawer of the cabinet | pick up the black bowl inside the top drawer of the wooden cabinet, **not the one on top of the cabinet**, and place it on the plate |
| next to the cookie box | pick up the black bowl next to the cookie box, **not the one on the stove**, and place it on the plate |
| next to the plate | pick up the black bowl next to the plate, **not the one next to the ramekin**, and place it on the plate |
| next to the ramekin | pick up the black bowl next to the ramekin, **not the one next to the cookie box**, and place it on the plate |
| on the cookie box | pick up the black bowl on top of the cookie box, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the ramekin | pick up the black bowl on top of the ramekin, **not the one on top of the cookie box**, and place it on the plate |
| on the stove | pick up the black bowl on the stove, **not the one on top of the wooden cabinet**, and place it on the plate |
| on the wooden cabinet | pick up the black bowl on top of the wooden cabinet, **not the one on the stove**, and place it on the plate |

</details>

**Analysis.** The "…not the one on X…" clause *confuses* the policy instead of disambiguating it
(−47.2 pts overall). Damage is worst where the clause names a salient surface (ramekin, table center,
stove, cabinet). The only unhurt task (0) was already phrased relationally.

**Replication (2026-09-10).** An independent full re-run of `default` (500 rollouts, different
hardware; `eval_log.md` 2026-09-10) reproduced the baseline within noise: **84.4% (422/500) vs. the
original 84.0%**.
Per-task numbers: 90/80/92/88/80/92/96/78/82/66% (ids 0-9) vs. the original 92/84/92/84/76/94/90/72/
84/72% — individual tasks move by up to ±6-10pts either direction (consistent with the run-to-run GPU
non-determinism documented in `mechanistic_localization.md` §8.11), but the pooled mean and every task's qualitative level match.

### Setting: `positive_contrast`

- **Positive contrast:** states the distractor's location as a plain fact, no negation — *"…the other
  black bowl is on X."*
- Prompts: `openvla/experiments/robot/libero/instructions.py::LIBERO_SPATIAL_POSITIVE_CONTRAST_INSTRUCTIONS`.
- Scene: stock `libero_spatial` (2 bowls), unmodified — no render check needed.

| id | target | Default | Positive contrast | Negative contrast | Δ (pos − default) | Δ (neg − pos) |
|--:|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 92% | 94% | 0 | +2 |
| 1 | next to the ramekin | 84% | 8% | 32% | **−76** | +24 |
| 2 | table center | 92% | 14% | 2% | **−78** | −12 |
| 3 | on the cookie box | 84% | 38% | 52% | −46 | +14 |
| 4 | in the top drawer | 76% | 50% | 64% | −26 | +14 |
| 5 | on the ramekin | 94% | 2% | 4% | **−92** | +2 |
| 6 | next to the cookie box | 90% | 40% | 72% | −50 | +32 |
| 7 | on the stove | 72% | 12% | 4% | −60 | −8 |
| 8 | next to the plate | 84% | 68% | 36% | −16 | −32 |
| 9 | on the wooden cabinet | 72% | 0% | 8% | **−72** | +8 |

**Analysis.** The negation-specific drop is *negative* — adding "not the one…" on top of a bare
mention slightly **helps** rather than hurts (+4.4 pts), and that's within noise at this trial count
anyway. Nearly all of `negative_contrast`'s −47 pt damage survives with the negation removed entirely
(−51.6 pts here). So the earlier hypothesis ("negation confuses the policy") was misattributed: the
confusion comes from **naming a second spatial location in the prompt at all** — the checkpoint was
fine-tuned on target-only prompts and has no practice grounding a second referent, negated or not.
Task-level pattern matches `negative_contrast` closely (tasks 1, 2, 5, 9 hit hardest in both).

Results: `results/libero_spatial--positive_contrast--shard{0..3}of4.jsonl`.

### Setting: `length_control_infix` / `length_control_suffix` (run 2026-10-01)

- **What changes:** every task keeps its native wording verbatim and gains one content-free
  politeness clause (no object, no location, no second referent), sized to the contrast clauses' mean
  length in the checkpoint's own Llama-2 tokens (`benchmark_split_plan.md` Split 1, "Length control").
  - `length_control_infix` (+11 tokens, stands in for `negative_contrast`): *"pick up the black bowl
    on the ramekin, if it is not too much trouble for you, and place it on the plate"*.
  - `length_control_suffix` (+13 tokens, stands in for `positive_contrast`): *"pick up the black bowl
    on the ramekin and place it on the plate; thank you so very much in advance for your help with
    this"*.
- Prompts: `instructions.py::LIBERO_SPATIAL_LENGTH_CONTROL_INFIX_INSTRUCTIONS` /
  `LIBERO_SPATIAL_LENGTH_CONTROL_SUFFIX_INSTRUCTIONS`.
- Scene: stock `libero_spatial` (2 bowls), unmodified — no render check needed.

| id | target | Default | Infix | Δ (infix − default) | Negative contrast | Suffix | Δ (suffix − default) | Positive contrast |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 90% | −2 | 94% | 86% | −6 | 92% |
| 1 | next to the ramekin | 84% | 70% | −14 | 32% | 60% | −24 | 8% |
| 2 | table center | 92% | 62% | −30 | 2% | 66% | −26 | 14% |
| 3 | on the cookie box | 84% | 96% | +12 | 52% | 78% | −6 | 38% |
| 4 | in the top drawer | 76% | 52% | −24 | 64% | 56% | −20 | 50% |
| 5 | on the ramekin | 94% | 88% | −6 | 4% | 84% | −10 | 2% |
| 6 | next to the cookie box | 90% | 72% | −18 | 72% | 48% | −42 | 40% |
| 7 | on the stove | 72% | 22% | **−50** | 4% | 22% | **−50** | 12% |
| 8 | next to the plate | 84% | 66% | −18 | 36% | 34% | **−50** | 68% |
| 9 | on the wooden cabinet | 72% | 10% | **−62** | 8% | 2% | **−70** | 0% |
| **Overall** | | **84.0%** | **62.8%** | **−21.2** | 36.8% | **53.6%** | **−30.4** | 32.4% |

```
Length-only Drop (infix)  = SR(default: 84.0%) − SR(length_control_infix: 62.8%)  = 21.2 pts
Length-only Drop (suffix) = SR(default: 84.0%) − SR(length_control_suffix: 53.6%) = 30.4 pts
Content Drop (negative)   = SR(length_control_infix: 62.8%)  − SR(negative_contrast: 36.8%) = 26.0 pts
Content Drop (positive)   = SR(length_control_suffix: 53.6%) − SR(positive_contrast: 32.4%) = 21.2 pts
```

**Analysis.**

1. **The length-only drop is far outside the ±3.3 pt noise band.** A clause that names no object and
   no location costs 21.2 pts (infix) and 30.4 pts (suffix). By the plan's reading, prompt length or
   generic off-template text is by itself a sufficient cause of a large drop.
2. **It does not account for the whole contrast drop.** A further 26.0 pts (negative) and 21.2 pts
   (positive) remain after subtracting the matched length control. So roughly 45% of
   `negative_contrast`'s 47.2 pt drop and 59% of `positive_contrast`'s 51.6 pt drop is reproduced by
   content-free text of the same length, and the rest is tied to what the contrast clause says.
3. **The per-task profile differs from the contrast conditions.** The length controls hit tasks 7 and
   9 hardest (22% / 2–10%) and leave task 5 nearly intact (84–88%), whereas both contrast conditions
   take task 5 to 2–4%. Task 5's collapse under the contrast prompts is therefore content-specific,
   while tasks 7 and 9 are fragile to any added text.
4. **Caveat.** One filler wording per position. The two controls also differ from each other by
   9.2 pts, so position, the 2 extra tokens, and the specific wording are not separated here.

Results: `results/libero_spatial--length_control_{infix,suffix}--openvla-7b-libero-spatial-lora-r32--shard{0..3}of4.jsonl`.

---

## 3. Split 2 — Distractor Placement Probe

**Question.** Does an extra distractor's *position* — not just its presence — drive failures? Each
condition keeps the ordinary (target-only) prompt and adds a **third** `akita_black_bowl` at a
different kind of location relative to the target. Coordinate catalog and per-task placement table
for every condition live in `benchmark_split_plan.md` §Split 2 — this section covers rendered
outcome, results, and analysis per condition. For the same numbers reorganized **by task** instead —
placement + render + instruction + SR side by side for `default`/`irrelevant`/`semantic`/`landmark`,
one table per task — see `split2_distractor_comparison.md`.

| Condition | Status | Overall SR | Rollouts |
|---|---|--:|--:|
| `irrelevant` (current, `libero_spatial_3bowl_front`) | ✅ run | 85.2% | 500/500 |
| `semantic` (current, `libero_spatial_3bowl_semantic2`) | ✅ run | 85.2% | 500/500 |
| `landmark` | ✅ run | 80.6% | 403/500 |
| `landmark_with_hardneg_prompt` (Split 1×2 combo) | ✅ run | 41.2% | 412/500 |
| `path` | ⬜ not authored | — | — |

Retired condition definitions (`irrelevant_v1_legacy`, `semantic_v1_legacy`, `center_fixed_legacy`)
were run and superseded by the current `irrelevant`/`semantic` above; their numbers are preserved in
git history and `eval_log.md`, not repeated here.

Computed `Distractor-type Drop = SR(spatial/default) − SR(condition)`:

| Condition | Drop | Interpretation |
|---|--:|---|
| `irrelevant` (current) | −1.2 pts | Negative — costs nothing; scores *above* baseline. Task 6 the one soft spot (−14 pts on that task alone), no other task collapses |
| `semantic` (current) | −1.2 pts | Negative — no measurable cost; no task collapses (weakest: task 1 at 72%) |
| `landmark` | +3.4 pts | The one real-cost result — concentrated almost entirely in two tasks |
| `landmark_with_hardneg_prompt` | +42.8 pts (vs. baseline); +39.4 pts vs. `landmark` on the identical scene | Adding a disambiguating prompt to the `landmark` scene does not rescue the affected tasks and wrecks the rest of the suite |

### Per-task render table: `default` vs. `irrelevant` vs. `semantic` vs. `hardneg`

One row per task id, one column per Split 2 condition (episode-0 init state, post physics-settle —
see §7 — so no floating bowls). `default` = `libero_spatial` (2 bowls); `irrelevant`/`semantic` are
the **current** suites (`libero_spatial_3bowl_front`/`libero_spatial_3bowl_semantic2`, 140x140
resized — not center-cropped — from the raw 256x256 render, since a center crop cuts off
`irrelevant`'s 3rd bowl at the table's front edge); `hardneg` (`landmark`'s scene) is
`libero_spatial_3bowl_hardneg`, unchanged since first authored. All copied to
`openvla/experiments/figures/per_task_render/` since the source `LIBERO/scratch_render/<suite>/`
dirs are scratch and can be overwritten by a future pass.

| id | target | `default` | `irrelevant` | `semantic` | `hardneg` |
|--:|---|---|---|---|---|
| 0 | between the plate and the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t0.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t0.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t0.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t0.png) |
| 1 | next to the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t1.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t1.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t1.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t1.png) |
| 2 | table center | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t2.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t2.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t2.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t2.png) |
| 3 | on the cookie box | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t3.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t3.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t3.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t3.png) |
| 4 | in the top drawer of the wooden cabinet | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t4.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t4.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t4.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t4.png) |
| 5 | on the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t5.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t5.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t5.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t5.png) |
| 6 | next to the cookie box | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t6.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t6.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t6.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t6.png) |
| 7 | on the stove | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t7.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t7.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t7.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t7.png) |
| 8 | next to the plate | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t8.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t8.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t8.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t8.png) |
| 9 | on the wooden cabinet | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/default_t9.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/irrelevant_v2_thumb_t9.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/semantic_v2_thumb_t9.png) | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/per_task_render/hardneg_t9.png) |

**Split's implemented conditions are fully run; only the unauthored `path` condition remains.**

### Setting: `irrelevant` (current)

**Question.** Split 2's control: a 3rd bowl placed somewhere not tied to any relational language,
to isolate whether an extra bowl's mere *presence* costs anything, independent of where it sits.
Redefined twice (see `benchmark_split_plan.md` §Split 2 for the full history) — this current
definition places bowl_3 at the literal front edge of the table (`table_front`) in every task,
instead of a per-task "least-crowded named region" pick that the first redefinition used, which
reused `next_to_ramekin_region` (another task's real target landmark) for 5/10 tasks and undercut
the goal of a placement carrying no relational meaning at all. Four tasks (1, 3, 5, 6) fall back to
`table_center` where `table_front` overlaps an existing bowl.

**2026-08-27 offset fine-tune.** Both anchor coordinates were nudged another 5cm apart, within the
`libero_spatial_3bowl_front` suite only (not the shared catalog default): `table_front` +0.05m further
toward the front edge (x: 0.19–0.21 → 0.24–0.26), used in the 6 non-fallback task files; `table_center`
0.05m further back (x: −0.10––0.05 → −0.15––0.10), used only in the 4 fallback task files (1, 3, 5,
6) — task 2's own `table_center` bowl_1 target (its "from table center" semantic anchor) was left
untouched since it's not part of this bowl_3 placement. The `table_center` fallback stays ≈0.29m from
`stove_region` (−0.42,−0.15)–(−0.40,−0.13) either way — moving it back doesn't approach the stove.

- Suite: `libero_spatial_3bowl_front`. Full per-task region table and the redefinition rationale are
  in `benchmark_split_plan.md` §Split 2 ("Second redefinition").
- Render check: `verify_suite_init_states.py` → `PASS`, worst separation 0.122 m (task 4, unrelated
  to bowl_3 — same persistent bowl_1/bowl_2 constraint present in every 3-bowl suite); every other
  task's separation improved with the wider offsets (0.148–0.301 m, vs. 0.122–0.276 m before the
  fine-tune). Contact sheet re-rendered and eyeballed — 3 distinct bowls per task, no
  overlaps/clipping, all resting flat; bowl_3 visibly farther toward the front edge (front tasks) or
  farther back (fallback tasks) than the pre-fine-tune render.

| id | target | Default (2-bowl) | Irrelevant (3-bowl) | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 88% | −4 |
| 1 | next to the ramekin | 84% | 84% | 0 |
| 2 | table center | 92% | 94% | +2 |
| 3 | on the cookie box | 84% | 96% | **+12** |
| 4 | in the top drawer | 76% | 88% | **+12** |
| 5 | on the ramekin | 94% | 86% | −8 |
| 6 | next to the cookie box | 90% | 76% | **−14** |
| 7 | on the stove | 72% | 92% | **+20** |
| 8 | next to the plate | 84% | 80% | −4 |
| 9 | on the wooden cabinet | 72% | 68% | −4 |

**Analysis.** 500/500 rollouts, overall 85.2% vs. the 84.0% baseline — Drop = −1.2 pts, essentially
free, continuing the pattern of every Split 2 condition except `landmark`. Task 6 (next to the cookie
box) is the one real soft spot (−14 pts) — notably the same task that collapsed hardest under
`center_fixed_legacy`'s confounded single-coordinate placement (90%→44%, see the design-confound note
in `benchmark_split_plan.md`); it no longer collapses, but it's still this condition's weakest point,
worth watching if a third redefinition ever touches this task. Task 7 (+20) and tasks 3/4 (+12 each)
gain the most, matching `irrelevant_v1_legacy`'s pattern of tasks 4 and 7 being the biggest gainers —
consistent gains on the same tasks across two different neutral-placement definitions suggests this
is a real property of those tasks' scenes, not placement-specific noise.

Results: `results/libero_spatial_3bowl_front--default--shard{0..3}of4.jsonl`.

### Setting: `semantic` (current)

**Question.** Places the 3rd bowl at a **named landmark that is not the target's own** — testing
whether sitting at *any* nameable relational spot pulls the policy, even when that landmark doesn't
match the prompt. Redefined once (see `benchmark_split_plan.md` §Split 2 for the full history): task
4's (in the top drawer) bowl_3 moved from `next_to_plate_region` (~0.58m from the target — outside
the 0.33-0.50m band the other 9 tasks land in, so it behaved more like a neutral placement than a
genuine semantic distractor) to `between_plate_ramekin_region` (~0.48m, task 0's real target
landmark). All 9 other tasks unchanged.

- Suite: `libero_spatial_3bowl_semantic2`. Redefinition rationale in `benchmark_split_plan.md`
  §Split 2 ("Second redefinition").
- Render check: `verify_suite_init_states.py` → `PASS`, worst separation 0.122 m (task 4;
  `next_to_box_region` was tried first and failed at 0.063 m — the open drawer's footprint collides
  with it); contact sheet eyeballed — 3 distinct bowls per task, no overlaps/clipping.

| id | target | 3rd bowl's landmark | Success | n | Δ vs `semantic_v1_legacy` |
|--:|---|---|--:|--:|--:|
| 0 | between the plate and the ramekin | next to the cookie box | 86% | 50 | 0 |
| 1 | next to the ramekin | next to the plate | 72% | 50 | 0 |
| 2 | table center | next to the ramekin | 96% | 50 | 0 |
| 3 | on the cookie box | next to the ramekin | 90% | 50 | 0 |
| 4 | in the top drawer | between the plate and the ramekin | 92% | 50 | **+4** |
| 5 | on the ramekin | next to the plate | 86% | 50 | 0 |
| 6 | next to the cookie box | next to the ramekin | 94% | 50 | 0 |
| 7 | on the stove | next to the box | 88% | 50 | 0 |
| 8 | next to the plate | next to the box | 80% | 50 | 0 |
| 9 | on the wooden cabinet | next to the ramekin | 68% | 50 | 0 |

**Analysis.** 500/500 rollouts, overall 85.2% vs. the 84.0% baseline — Drop = −1.2 pts, in line with
every Split 2 condition except `landmark`. The 9 unchanged tasks land bit-for-bit identical to
`semantic_v1_legacy`'s numbers (same seed, same scene) — a clean internal-consistency check that the
redefinition genuinely isolated task 4. Task 4 itself improved +4 pts moving its bowl_3 from the
out-of-band `next_to_plate_region` to the in-band `between_plate_ramekin_region`, a small move in the
direction of "closer semantic distractor costs slightly more," though well within noise for a single
task at n=50. Task 1 (next to the ramekin, 72%) remains this condition's weakest task, unchanged from
`semantic_v1_legacy` since task 1's scene didn't change.

Results: `results/libero_spatial_3bowl_semantic2--default--shard{0..3}of4.jsonl`.

### Setting: `landmark`

**Question.** Split 2's hardest confusability test: the 3rd bowl sits near the target's **own**
landmark — the same relational word the prompt uses — farther from the exact target point than the
real bowl. A genuine look-alike for "the bowl near X," unlike `semantic` (different landmark) or
`irrelevant` (no landmark).

- Suite: `libero_spatial_3bowl_hardneg`. Per-task `hardneg_region` coordinates in
  `benchmark_split_plan.md` §Split 2.
- Render check: BDDL/init states pre-existed but were unchecked with the current pipeline;
  regenerated (byte-identical, confirming determinism), verified (min separation 0.121 m vs. the
  0.12 m threshold — passes narrowly, task 3 tightest), contact sheet eyeballed — task 3's close pair
  confirmed as two separate bowls, not merged.

| id | target | Default (2-bowl) | Landmark (3-bowl) | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 48% | **−44** |
| 1 | next to the ramekin | 84% | 92% | +8 |
| 2 | table center | 92% | 98% | +6 |
| 3 | on the cookie box | 84% | 88% | +4 |
| 4 | in the top drawer | 76% | 84% | +8 |
| 5 | on the ramekin | 94% | 92% | −2 |
| 6 | next to the cookie box | 90% | 80% | −10 |
| 7 | on the stove | 72% | 86% | +14 |
| 8 | next to the plate | 84% | 84% | 0 |
| 9 | on the wooden cabinet | 72% | 54% | **−18** |

**Analysis.** The overall drop (−3.4 pts) looks mild — comparable to `center_fixed_legacy`'s plain
extra distractor (−3.8 pts) — but that headline number hides a **concentrated, task-specific effect**:
task 0 collapses 92%→48% and task 9 drops 72%→54%, both far beyond the ~±7 pt single-task noise band
at n=50; every other task is flat or actually improves. This is the first Split 2 condition where the
distractor's *placement relative to the target's own landmark* — not just presence or placement at
some other landmark — produces a real, attributable failure. Together with `semantic`, this narrows
*where* scene-driven failures come from: proximity to the target's own landmark matters; unrelated
landmarks and neutral placement don't.

Results: `results/libero_spatial_3bowl_hardneg--default--shard{0..3}of4.jsonl`.

### Setting: `landmark_with_hardneg_prompt` (Split 1×2 combo)

**Question.** Can a disambiguating prompt — *"pick up the black bowl closest to X, not the one farther
from it, …"* — rescue `landmark`'s task 0/9 collapse? Same scene as `landmark`
(`libero_spatial_3bowl_hardneg`), condition = `hardneg` prompt
(`instructions.py::LIBERO_SPATIAL_HARDNEG_INSTRUCTIONS`).

- Scene: identical to `landmark`, no changes — no new render check needed.
- Prompt: adds an explicit closer/farther disambiguation clause on top of the `landmark` scene.

| id | target | Landmark scene, default prompt | + disambiguating prompt | Δ |
|--:|---|--:|--:|--:|
| 0 | between the plate and the ramekin | 48% | 38% | −10 |
| 1 | next to the ramekin | 92% | 34% | **−58** |
| 2 | table center | 98% | 50% | **−48** |
| 3 | on the cookie box | 88% | 62% | −26 |
| 4 | in the top drawer | 84% | 80% | −4 |
| 5 | on the ramekin | 92% | 38% | **−54** |
| 6 | next to the cookie box | 80% | 52% | −28 |
| 7 | on the stove | 86% | 4% | **−82** |
| 8 | next to the plate | 84% | 48% | −36 |
| 9 | on the wooden cabinet | 54% | 6% | **−48** |

**Analysis.** The disambiguating prompt does not rescue the two tasks that actually needed help —
task 0 barely changes (48%→38%) and task 9 gets *worse*, not better (54%→6%) — while every task that
was fine without it collapses instead, several catastrophically (task 7: 86%→4%, task 1: 92%→34%,
task 5: 92%→38%). Net effect is strongly negative (−39.4 pts on the same scene) and lands the overall
number (41.2%) close to Split 1's `negative_contrast` on the plain 2-bowl scene (36.8%) — consistent
with Split 1's finding that *any* prompt referencing a second bowl/location hurts, regardless of
whether the scene actually contains a confusable distractor. Combining a hard scene with a hard prompt
doesn't compound narrowly on the hard cases; the prompt damage dominates and spreads to tasks the
scene alone never touched. Strongest evidence in the project that language, not scene design, is the
primary lever on this checkpoint's failures.

Results: `results/libero_spatial_3bowl_hardneg--hardneg--shard{0..3}of4.jsonl`.

---

## 4. Split 3 — Scene Complexity Probe

**Question.** Keep the `center_fixed_legacy` 3-bowl scene and add clutter/occlusion by **opening the
wooden cabinet's top ("first") drawer** in every task. Does a protruding open drawer degrade the
policy further, or does it just block the arm?

### Blocked tasks — tasks 3, 6, 7 excluded from the adjusted overall

Rollout review showed that for **task 3** (*on the cookie box*), **task 6** (*next to the cookie
box*), and **task 7** (*on the stove*), the open top drawer **physically blocks the trajectory**: the
protruding drawer sits directly in the pick-and-place corridor these tasks must traverse, so the arm
cannot complete the motion regardless of what the policy predicts. Failures on these three tasks
therefore measure the *scene's physical feasibility*, not the *policy's robustness*, and are excluded
from the adjusted overall below. (Task 6's 0% is the clearest case — a hard geometric block, not a
perception error.) **Data-quality caveat:** this exclusion was a manual rollout-video review, not an
automated feasibility check — treat the adjusted number as reviewed-and-reasoned, not
machine-verified ground truth.

**Raw** (all 10 tasks) and **adjusted** (7 feasible tasks, 350 rollouts) overalls:

| Condition (default prompt) | Raw overall (10 tasks) | Adjusted overall (7 tasks†) |
|---|--:|--:|
| 3 bowls, drawer closed (`center_fixed_legacy`) | 80.2% (401/500) | 84.3% (295/350) |
| 3 bowls, **drawer open** | 60.0% (300/500) | **73.1%** (256/350) |
| **Δ (open − closed)** | −20.2 pts | **−11.1 pts** |

† Same-subset comparison: tasks {0, 1, 2, 4, 5, 8, 9} in both conditions. Against the original 2-bowl
baseline on the same 7 tasks (84.9%, 297/350), the adjusted drop is **−11.7 pts** — vs. the raw
headline of −24.0, which conflates policy degradation with physically infeasible tasks.

| id | target | 3-bowl (drawer closed) | +drawer open | Δ | |
|--:|---|--:|--:|--:|---|
| 0 | between the plate and the ramekin | 76% | 66% | −10 | |
| 1 | next to the ramekin | 90% | 86% | −4 | |
| 2 | table center | 94% | 80% | −14 | |
| 3 | on the cookie box | 86% | 42% | −44 | ‡ blocked |
| 4 | in the top drawer | 84% | 84% | 0 | |
| 5 | on the ramekin | 84% | 84% | 0 | |
| 6 | next to the cookie box | 44% | 0% | −44 | ‡ blocked |
| 7 | on the stove | 82% | 46% | −36 | ‡ blocked |
| 8 | next to the plate | 88% | 68% | −20 | |
| 9 | on the wooden cabinet | 74% | 44% | −30 | |

‡ Trajectory physically obstructed by the open drawer — excluded from the adjusted overall.

**Setting.** Suite `libero_spatial_3bowl_open`, same `center_fixed_legacy` bowl placement as Split 2
+ cabinet top-drawer joint forced open. Verified across all 500 open-drawer init states: bowls never
overlap (worst separation 0.122 m) and the drawer joint is open (qpos −0.141 m) in every trial. Adding
the drawer state does not change the state-vector size (105 dims) — the drawer joint already exists in
the cabinet model; only its position changes.

**Render compare** (left = drawer closed `libero_spatial_3bowl`, right = drawer open
`libero_spatial_3bowl_open`; rows = task ids 0–9):

![3-bowl: drawer closed vs open](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/compare_3bowl_closed_vs_open_grid.png)

**Analysis.** The raw −20.2 pt drop overstates the policy effect: nearly half of it comes from the
three blocked tasks (3, 6, 7), where no policy could succeed because the open drawer occupies the
motion corridor. On the seven tasks that remain physically feasible, the open drawer still costs the
policy **−11.1 pts** vs. the closed-drawer scene — a genuine (occlusion/robustness) effect,
concentrated around the cabinet: task 9 (*on the wooden cabinet*, −30) and task 8 (*next to the
plate*, −20) drop well beyond noise, with smaller dips on 2 and 0. The two unaffected tasks (4 *in the
top drawer*, 5 *on the ramekin*) are telling: task 4's target is inside the drawer, so an open drawer
is expected there. The protruding drawer occludes the cabinet side of the table, which is exactly
where the still-degraded targets sit or are approached — but the effect is a robustness gap (~−11
pts), not the collapse the raw number suggests.

**Split fully run.**

---

## 5. Split 4 — Surface vs. Landmark Grounding Probe

**Split fully run** — 4a's two missing cells and 4b (redesigned as a target cue-type probe)
completed 2026-08-20; 4c (familiar-vs-novel proximity-cue probe) completed 2026-09-02. See
`eval_log.md` for both launch batches, and §5.4 for the combined three-way synthesis.

### 5.0 — What this split is testing, in plain language

Every task in this suite is "pick up the black bowl \<somewhere\> and place it on the plate." The
"\<somewhere\>" is always describable in one of three ways, depending on what's physically next to
the bowl in that task's scene:

| Relation family | What it means | Example (task id) |
|---|---|---|
| **landmark** | bowl sits *beside* another named object | "next to the ramekin" (task 1), "next to the cookie box" (task 6), "next to the plate" (task 8), "between the plate and the ramekin" (task 0) |
| **surface** | bowl sits *on top of* another named object | "on the cookie box" (task 3), "on the ramekin" (task 5), "on the stove" (task 7), "on the wooden cabinet" (task 9) |
| **region** | bowl sits in an open area with nothing nearby to name | "table center" (task 2) |

Split 4 asks the same underlying question two different ways:

- **4a asks it from the *scene* side:** does it matter, physically, whether the target bowl and the
  (unmentioned) distractor bowl each happen to be a landmark-type, surface-type, or region-type
  placement? The prompt text stays the standard "pick up the black bowl \<location\>" the whole
  time — only where the bowls actually sit in the scene changes.
- **4b asks it from the *prompt* side:** for a target bowl that is physically sitting in one place
  (say, resting on the cookie box), does it matter whether the instruction *describes* that same
  spot with surface-style words ("on the cookie box"), landmark-style words ("next to the cookie
  box"), or region-style words ("near the center of the table")? Nothing about the scene moves —
  only the sentence changes, and it's checked to still be a true description of where the bowl is.

A third question follows directly from 4b: when a rephrasing *does* happen to reuse a sentence
pattern seen natively elsewhere in training (e.g. "next to X"), is it protected because it's
*familiar*, or is that just a coincidence of which relation type it happens to express?

- **4c asks it from the *wording-familiarity* side:** holding relation type fixed at "proximity"
  (so 4a/4b's relation-family confound can't explain the answer either way), does describing the
  same true fact with a phrase the checkpoint saw at fine-tuning time ("next to X") do any better
  than an equally true, equally approximate phrase it never saw ("close to X")?

**Bottom line (read the full analysis in §5.1/§5.2/§5.3, synthesized in §5.4, for the numbers):**
the scene-side manipulation (4a) barely matters — success rates stay within about 8 points of each
other no matter which relation family the target or distractor happens to be. The prompt-side
manipulation (4b) is the single biggest effect measured anywhere in this project, 50–67 points of
success lost by rewording a completely true, completely unambiguous sentence about a bowl whose
position never changed and whose distractor is never mentioned. 4b's own results suggested a tidy
explanation for part of that — "next to X" survives better than region-style phrasing because it's
at least a *familiar* template — but 4c tested that explanation directly and it doesn't hold: the
*novel* phrasing beat the *familiar* one by 22.5 points. Put together, this says the checkpoint
isn't reasoning about *where things are*, and it isn't simply rewarding *any* sentence pattern seen
during fine-tuning either — it's bound to specific phrase-to-scene associations from fine-tuning,
and reusing a familiar phrase on the *wrong* scene can actively mislead it more than a phrase it
has no prior for at all. See §5.4 for the full three-way synthesis.

### 5.1 — 4a: Grounding-by-scene probe (complete, all 6 cells)

4 of 6 cells reuse rollouts already collected under `spatial/default` (Split 1) — no new GPU run
needed for those, just re-aggregating existing per-task numbers by relation-family cell. The other 2
cells (`grounding/surface_landmark`, `grounding/region_surface`) are new 50-trial/task runs on the
gap-fill scenes described in §7's render-check log. Per-task `default` success rates are copied from
Split 1's `default` table (task 4, containment, is excluded from this matrix — see
`benchmark_split_plan.md` §3).

| (target, distractor) family | Tasks (id: SR) | Cell SR | Status |
|---|---|--:|---|
| (landmark, landmark) | 0: 92%, 1: 84%, 8: 84% | **86.7%** | ✅ derived from existing data |
| (region, landmark) | 2: 92% | **92.0%** | ✅ derived from existing data |
| (surface, surface) | 3: 84%, 5: 94%, 7: 72%, 9: 72% | **80.5%** | ✅ derived from existing data |
| (landmark, surface) | 6: 90% | **90.0%** | ✅ derived from existing data |
| (surface, landmark) | new suite, task 0: 88% (44/50) | **88.0%** | ✅ run |
| (region, surface) | new suite, task 0: 92% (46/50) | **92.0%** | ✅ run |

**Conclusion (now conclusive, all 6 cells run — supersedes the earlier "preliminary" read).**
Pooling cells by target family (simple mean of that family's cell SRs, same convention as the
original 4-cell read):

| Target family | Cells (SR) | Family mean |
|---|---|--:|
| landmark | (landmark,landmark)=86.7%, (landmark,surface)=90.0% | **88.35%** |
| surface | (surface,surface)=80.5%, (surface,landmark)=88.0% | **84.25%** |
| region | (region,landmark)=92.0%, (region,surface)=92.0% | **92.0%** |

The hypothesis ("landmark-target scenes are more attraction-prone than surface/region-target
scenes") is **not supported** — landmark-target (88.35%) scores *above* surface-target (84.25%),
and region-target scores highest of all three (92.0%). The 2 new cells confirm rather than
overturn the earlier 4-cell reading: whatever drives this checkpoint's failures, it isn't the
target's own relation-family under a fixed, correctly-phrased prompt. Given 4b's results below,
the far larger effect by orders of magnitude is *prompt phrasing*, not scene relation type.

### 5.2 — 4b: Target Cue-Type Probe (complete)

**Question.** Holding the scene and distractor completely fixed (distractor never mentioned), does
rephrasing *only* how the target is described — landmark ("next to X") vs. surface ("on X") vs.
region (table-zone) — change success, independent of what relation family the scene actually is?
Full design (truthfulness tiers, which tasks get which rephrasing and why) is in
`benchmark_split_plan.md` Split 4's 4b section.

**Result: this is the single largest effect measured anywhere in this project so far, and it comes
from a manipulation that never mentions a second bowl at all.**

**Concretely, what changed.** Task 5's bowl physically rests on top of the ramekin, and nothing
about the scene or robot's view differs between these three runs — only the sentence:

| Condition | Instruction given to the policy |
|---|---|
| `spatial/default` (native, "surface" phrasing) | "pick up the black bowl on the ramekin and place it on the plate" |
| `target_cue_landmark` | "pick up the black bowl next to the ramekin and place it on the plate" |
| `target_cue_region` | "pick up the black bowl at the back-left of the table and place it on the plate" |

All three sentences are true descriptions of the exact same bowl in the exact same spot. Success
on this one task went 94% → 10% → 28% across those three sentences respectively — see the
per-task table below for every task's exact wording pair (full instruction dicts:
`LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS` / `LIBERO_SPATIAL_TARGET_CUE_LANDMARK_INSTRUCTIONS`
in `openvla/experiments/robot/libero/instructions.py`).

| Condition | Tasks | Overall SR | Rollouts |
|---|---|--:|--:|
| `target_cue_region` | 0,1,3,5,6,7,8,9 (8 tasks) | **17.0%** | 400/400 |
| `target_cue_landmark` | 3,5,7,9 (4 tasks) | **30.5%** | 200/200 |

Per-task, against the `spatial/default` baseline for the same task:

| id | target (native phrasing) | Default SR | `target_cue_region` | Δ | `target_cue_landmark` | Δ |
|--:|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin (landmark) | 92% | 58% | −34 | — | — |
| 1 | next to the ramekin (landmark) | 84% | 10% | −74 | — | — |
| 3 | on the cookie box (surface) | 84% | 28% | −56 | 52% | −32 |
| 5 | on the ramekin (surface) | 94% | 28% | −66 | 10% | **−84** |
| 6 | next to the cookie box (landmark) | 90% | 2% | **−88** | — | — |
| 7 | on the stove (surface) | 72% | 0% | −72 | 44% | −28 |
| 8 | next to the plate (landmark) | 84% | 10% | −74 | — | — |
| 9 | on the wooden cabinet (surface) | 72% | 0% | −72 | 16% | −56 |

**Metrics** (SR(default) restricted to the same task subset each condition covers, matching
`benchmark_split_plan.md`'s formulas):

```
Region-cue Drop   = SR(default, 8 tasks: 84.0%) − SR(target_cue_region: 17.0%)   = 67.0 pts
Landmark-cue Drop = SR(default, 4 tasks: 80.5%) − SR(target_cue_landmark: 30.5%) = 50.0 pts

Region-cue Drop, landmark-family pool {0,1,6,8}: 87.5% → 20.0%  = 67.5 pts
Region-cue Drop, surface-family pool  {3,5,7,9}: 80.5% → 14.0%  = 66.5 pts
```

**Analysis.** Three findings, in order of how surprising they are:

1. **The effect size is enormous** — 50 to 67 points overall, with several tasks collapsing to
   0-10%. This is *larger* than Split 1's `negative_contrast`/`positive_contrast` drops (47.2 /
   51.6 pts), the previous largest effect in the project — and 4b's prompts never reference a
   second bowl or location at all. Purely restating the *same true fact about the same bowl* in a
   different relation-family's words is enough to devastate the policy.
2. **No landmark-vs-surface asymmetry in the region-cue condition** — landmark-family (67.5 pt
   drop) and surface-family (66.5 pt drop) are damaged almost identically when forced into
   region-style phrasing. This rules out the original hypothesis's language-side analogue
   (landmark language being harder to compute than surface language) — the damage tracks
   *phrasing-template mismatch*, not relation-type difficulty.
3. **Landmark-cue (the "approximate/disclosed" relaxation) hurts less than region-cue for the same
   4 surface-family tasks** (50.0 vs. 66.5 pts) — describing a bowl resting on X as "next to X"
   survives better than describing it by table position. Plausibly because "next to X" is at least
   *structurally* the phrasing template the checkpoint was fine-tuned on for other tasks (landmark
   phrasing appears natively in 4/10 training tasks), whereas region-style phrasing
   ("at the back-left of the table") never appears in any of the 10 original `libero_spatial`
   prompts in any form.
   > **Superseded by 4c (§5.3).** This "familiarity is protective" reading was the natural
   > hypothesis at the time, but it was never tested against a matched novel-wording control —
   > point 3 here only compares "next to X" (familiar) against region-style phrasing (also novel,
   > *and* a different relation type). 4c isolates wording familiarity alone and finds the
   > opposite: a novel proximity phrase ("close to X") beats the familiar one by 22.5 pts. Point 3's
   > numbers stand, but its explanation doesn't — see §5.4.

This finding's relationship to 4a and 4c is synthesized in full in §5.4, after 4c's results below.

Results: `results/libero_spatial--target_cue_region--shard{0,1}of2.jsonl`,
`results/libero_spatial--target_cue_landmark--shard{0,1}of2.jsonl`,
`results/libero_spatial_grounding_surface_landmark--default--shard0of2.jsonl`,
`results/libero_spatial_grounding_region_surface--default--shard0of2.jsonl`.

### 5.3 — 4c: Familiar vs. Novel Proximity-Cue Probe (complete)

**Open question left by 4b.** `target_cue_landmark`'s ~50pt drop rephrases a surface-family target
("on the ramekin") using "next to X" — which happens to be the *exact* template tasks 0/1/6/8 already
use natively. That leaves the cause ambiguous: is the damage from the relation-type change itself
(surface → proximity), or from the sentence simply not matching any template seen at fine-tuning
time, with "next to X" only looking special because it happens to coincide with one? 4c holds
relation type fixed at "proximity" and swaps only the wording — "next to X" (familiar, reused
verbatim from tasks 0/1/6/8) vs. "close to X" (a proximity synonym that appears in none of the 10
native `libero_spatial` prompts) — on the same 4 surface-family tasks (3, 5, 7, 9), same scene, same
distractor-never-mentioned protocol as 4b.

| Label | Condition id | Example (task 5) | Status |
|---|---|---|---|
| surface_native | `default` (reused) | "on the ramekin" | ✅ already run — 80.5% pooled over tasks 3/5/7/9 (see §5.2 table above) |
| proximity_familiar | `target_cue_landmark` (reused) | "next to the ramekin" | ✅ already run — 30.5% pooled |
| proximity_novel | `target_cue_proximity_novel` (**new**) | "close to the ramekin" | ✅ run 2026-09-02 — 53.0% pooled |

Run 2026-09-02 on `--task_ids 3 5 7 9`, 200/200 rollouts (launch details: `eval_log.md`).

| id | native ("on X") | proximity_familiar ("next to X") | proximity_novel ("close to X") |
|--:|---|--:|--:|
| 3 | cookie box | 84% → 52% | 84% → **56%** |
| 5 | ramekin | 94% → 10% | 94% → **66%** |
| 7 | stove | 72% → 44% | 72% → **64%** |
| 9 | wooden cabinet | 72% → 16% | 72% → **26%** |
| **Pooled** | | **80.5% → 30.5%** | **80.5% → 53.0%** |

```
Novel-cue Drop     = SR(default, 4 tasks: 80.5%) − SR(target_cue_proximity_novel: 53.0%) = 27.5 pts
Familiar-cue Drop  = SR(default, 4 tasks: 80.5%) − SR(target_cue_landmark: 30.5%)         = 50.0 pts
Familiarity Gap    = SR(target_cue_landmark: 30.5%) − SR(target_cue_proximity_novel: 53.0%) = −22.5 pts
```

**Analysis — the result inverts the plan's predicted direction.** The plan's dichotomy (§4c) was:
Familiarity Gap ≈ 0 → relation-type-driven damage; Familiarity Gap clearly positive (novel drops
*more*) → template-familiarity-driven damage, "the more likely outcome." What actually happened is
a third case neither branch anticipated: the **novel** phrasing ("close to X") is clearly *less*
damaging than the **familiar** one ("next to X") — a −22.5pt gap, well outside the ~±7pt noise
band, in the opposite sign from the predicted branch. Task 5 shows the effect starkest (10% vs.
66%, a 56pt swing on a single task from swapping one word). This rules out the naive
template-familiarity story as stated (reusing a seen-at-finetuning-time phrase does *not* help
here — it hurts, a lot) while also not supporting pure relation-type invariance (a 22.5pt gap is
not noise). The more consistent reading, tying back to finding 17's action-collapse evidence: "next
to X" is not just *a* familiar template, it is the template natively bound to tasks 0/1/6/8's
specific scenes/targets — reusing it verbatim on a surface-family task's bowl may actively trigger
those tasks' learned reach patterns (a wrong, specific prior) rather than merely failing to match
any template (a generic, template-mismatch collapse). A phrase the model has *no* strong prior for
at all ("close to X") apparently degrades more gracefully than one it has a strong, wrong prior
for. This reframes template "familiarity" from a uniformly protective property to one that can be
actively harmful when the familiar template is bound to the wrong scene.

Results: `results/libero_spatial--target_cue_proximity_novel--shard{0,1,2,3}of4.jsonl` (shards 0/2
empty by design).

### 5.4 — Synthesis: 4a + 4b + 4c together (complete)

All three legs of Split 4 are now run. Laid side by side, on the same 4 surface-family tasks
(3, 5, 7, 9) where every leg overlaps:

| Leg | What moved | Condition | Pooled SR | Δ from default (80.5%) |
|---|---|---|--:|--:|
| baseline | — | `default` ("on X") | 80.5% | — |
| 4a | scene only (target/distractor relation family), prompt fixed | (surface,surface) cell | 80.5% | 0 pts (same data) |
| 4b | prompt only, region wording | `target_cue_region` | 14.0% | −66.5 pts |
| 4b | prompt only, landmark wording ("next to X", familiar) | `target_cue_landmark` | 30.5% | −50.0 pts |
| 4c | prompt only, proximity wording ("close to X", novel) | `target_cue_proximity_novel` | 53.0% | −27.5 pts |

Three results, read together, rule out every single-factor explanation tried so far:

1. **It is not the scene's relation type.** 4a moved where the bowls physically sit (landmark vs.
   surface vs. region placement) while keeping the prompt's phrasing standard, and success stayed
   within ~8 points across all 6 (target, distractor) cells (§5.1). Whatever breaks the policy in
   4b/4c is not triggered by scene geometry.
2. **It is not "does the sentence match *some* fine-tuning-time template."** If it were, `next to X`
   (verbatim from tasks 0/1/6/8) should score at or near `default`, and `close to X` (matching no
   training sentence at all) should score at or below `target_cue_region` (also matching nothing).
   Instead the ranking is default (80.5%) > novel (53.0%) > familiar (30.5%) > region (14.0%) —
   novel phrasing *beats* familiar phrasing by 22.5 points, the opposite of what generic
   template-matching predicts.
3. **It is not pure relation-type sensitivity either.** 4c held relation type fixed at "proximity"
   across both its legs, and still found a 22.5-point gap between them — a same-relation-type,
   wording-only change with a large, non-noise effect.

**What's left standing: template *binding*, not template *matching*.** The consistent story across
4a + 4b + 4c + the bowl-attraction probe (§8.5, finding 17) is that the checkpoint doesn't parse
"next to X" as a general proximity relation it can apply anywhere (that would predict 4a-style
scene-invariance carrying over to 4b/4c, which it doesn't) and doesn't just check the sentence
against a bag of seen templates (that would predict `target_cue_landmark` ≥ `target_cue_proximity_novel`,
which is backwards). Instead, "next to X" appears to be a phrase *specifically bound* to the
scenes/targets of tasks 0/1/6/8 during fine-tuning. Reusing it verbatim on a different task's bowl
doesn't fail to match anything — it matches the *wrong* thing, pulling the action decoder toward
those other tasks' learned reach behavior (finding 17's "confident wrong-target" mode). A phrase
with no training-time association at all ("close to X") has nothing wrong to pull toward, and
degrades more gracefully, closer to (though still well short of) `default`. Region phrasing
("at the back-left of the table") is worst of all because it's both unfamiliar *and* structurally
farthest from anything in the 10 native prompts (no object-relative phrasing at all).

Net effect: the split's original question — "does this checkpoint's apparent grounding reflect real
spatial/language understanding?" — gets a firmly negative answer, but a more specific one than 4b
alone suggested. It's not merely brittle to unfamiliar phrasing; specific familiar phrases carry
scene-specific baggage that can actively misdirect the policy on a task they weren't fine-tuned for,
which is a more concerning failure mode for practical prompt engineering than simple
out-of-distribution brittleness would be (a novel synonym is not a safe fallback in general — it
happened to help here only because the *alternative* familiar phrase was specifically mis-bound, not
because novelty itself is safe).

---

### 5.5 — 4b/4c paraphrase controls (complete)

**Question.** Every 4b/4c number above rests on one hand-written phrasing per task, and 4b only
measured *cross*-cue rewordings. These controls describe the target in its **own** native cue type,
truthfully, in different words, and add extra wordings for the region cue and the novel proximity
cue. Design: `benchmark_split_plan.md` Split 4, "4b/4c paraphrase controls". Scene and init states
are identical to `spatial/default`; the distractor is never mentioned.

| Condition | Example (task 5) | Tasks | Overall SR | Rollouts |
|---|---|---|--:|--:|
| `paraphrase_lexical` | "on top of the ramekin" | 0–9 | **72.4%** | 362/500 |
| `paraphrase_syntactic` | "that is on the ramekin" | 0–9 | **82.2%** | 411/500 |
| `target_cue_region_v2` | "in the back-left area of the table" | 0,1,3,5–9 | **20.0%** | 80/400 |
| `target_cue_region_v3` | "on the left side of the table, toward the back" | 0,1,3,5–9 | **18.8%** | 75/400 |
| `target_cue_proximity_beside` | "beside the ramekin" | 3,5,7,9 | **38.0%** | 76/200 |
| `target_cue_proximity_near` | "near the ramekin" | 3,5,7,9 | **49.0%** | 98/200 |
| `target_cue_proximity_adjacent` | "adjacent to the ramekin" | 3,5,7,9 | **37.5%** | 75/200 |

All seven run 2026-10-01 (launch details: `eval_log.md`). `pi05_libero` on the same conditions: §9.

#### Same-cue paraphrases: `paraphrase_lexical`, `paraphrase_syntactic`

`paraphrase_lexical` makes one synonym swap or argument reorder per task. `paraphrase_syntactic`
keeps the native words and wraps the location in a relative clause ("the black bowl **that is** on
the ramekin"; task 2: "from table center" → "that is at table center"). Prompts:
`instructions.py::LIBERO_SPATIAL_PARAPHRASE_LEXICAL_INSTRUCTIONS` / `..._SYNTACTIC_INSTRUCTIONS`.

| id | native → lexical paraphrase | Default | Lexical | Δ | Syntactic | Δ |
|--:|---|--:|--:|--:|--:|--:|
| 0 | "between the plate and the ramekin" → "between the ramekin and the plate" | 92% | 98% | +6 | 94% | +2 |
| 1 | "next to the ramekin" → "beside the ramekin" | 84% | 78% | −6 | 86% | +2 |
| 2 | "from table center" → "from the middle of the table" | 92% | 44% | **−48** | 82% | −10 |
| 3 | "on the cookie box" → "on top of the cookie box" | 84% | 76% | −8 | 86% | +2 |
| 4 | "in the top drawer" → "inside the top drawer" | 76% | 58% | −18 | 76% | 0 |
| 5 | "on the ramekin" → "on top of the ramekin" | 94% | 76% | −18 | 84% | −10 |
| 6 | "next to the cookie box" → "beside the cookie box" | 90% | 92% | +2 | 90% | 0 |
| 7 | "on the stove" → "on top of the stove" | 72% | 66% | −6 | 80% | +8 |
| 8 | "next to the plate" → "beside the plate" | 84% | 84% | 0 | 80% | −4 |
| 9 | "on the wooden cabinet" → "on top of the wooden cabinet" | 72% | 52% | −20 | 64% | −8 |
| **Overall** | | **84.0%** | **72.4%** | **−11.6** | **82.2%** | **−1.8** |

#### Region cue, three wordings: `target_cue_region` / `_v2` / `_v3`

Same table zone per task, three phrasings (task 5: "at the back-left of the table" / "in the
back-left area of the table" / "on the left side of the table, toward the back"). Prompts:
`LIBERO_SPATIAL_TARGET_CUE_REGION_INSTRUCTIONS`, `..._REGION_V2_...`, `..._REGION_V3_...`.

| id | target (native phrasing) | Default | `target_cue_region` | `_v2` | `_v3` |
|--:|---|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 92% | 58% | 58% | 64% |
| 1 | next to the ramekin | 84% | 10% | 12% | 0% |
| 3 | on the cookie box | 84% | 28% | 36% | 38% |
| 5 | on the ramekin | 94% | 28% | 38% | 40% |
| 6 | next to the cookie box | 90% | 2% | 2% | 4% |
| 7 | on the stove | 72% | 0% | 0% | 2% |
| 8 | next to the plate | 84% | 10% | 14% | 2% |
| 9 | on the wooden cabinet | 72% | 0% | 0% | 0% |
| **Overall (8 tasks)** | | **84.0%** | **17.0%** | **20.0%** | **18.8%** |

#### Novel proximity cue, four synonyms: "close to" / "beside" / "near" / "adjacent to"

| id | native ("on X") | Default | `target_cue_landmark` ("next to") | "close to" | "beside" | "near" | "adjacent to" |
|--:|---|--:|--:|--:|--:|--:|--:|
| 3 | cookie box | 84% | 52% | 56% | 60% | 54% | 54% |
| 5 | ramekin | 94% | 10% | 66% | 28% | 56% | 42% |
| 7 | stove | 72% | 44% | 64% | 48% | 56% | 46% |
| 9 | wooden cabinet | 72% | 16% | 26% | 16% | 30% | 8% |
| **Pooled** | | **80.5%** | **30.5%** | **53.0%** | **38.0%** | **49.0%** | **37.5%** |

"close to" is §5.3's `target_cue_proximity_novel`, run 2026-09-02; the other three are new.

#### Metrics

```
Same-cue Paraphrase Drop = SR(default: 84.0%) − SR(paraphrase_lexical: 72.4%)   = 11.6 pts
                           SR(default: 84.0%) − SR(paraphrase_syntactic: 82.2%) =  1.8 pts
Region-cue Drop, 8 tasks = 84.0% − 17.0 / 20.0 / 18.8 = 67.0 / 64.0 / 65.3 pts   (region / v2 / v3)
Cue-type Drop (excess)   = SR(paraphrase_lexical, 8 tasks: 77.8%) − SR(region*)  = 60.8 / 57.8 / 59.0 pts
Novel-cue Drop, 4 tasks  = 80.5% − 53.0 / 38.0 / 49.0 / 37.5 = 27.5 / 42.5 / 31.5 / 43.0 pts
Familiarity Gap (robust) = mean of 4 novel synonyms (44.4%) − SR(target_cue_landmark: 30.5%) = +13.9 pts
```

**Analysis.**

1. **A same-cue rewording costs little; 4b's collapse is about the cue type.** The relative-clause
   paraphrase is inside the noise band (−1.8 pts). The synonym swap costs 11.6 pts. Against that, a
   region-cue rewording of the same 8 tasks costs 64–67 pts. So the checkpoint does not fail on
   every surface-form change, which rules out the strongest template-matching reading.
2. **The lexical drop is concentrated.** Task 2 alone ("the middle of the table") loses 48 pts.
   Without it the other nine tasks go from 83.1% to 75.6% (−7.6). The three "beside" tasks (1, 6, 8)
   are essentially flat (−6, +2, 0), while "on" → "on top of" costs 6–20 pts on tasks 3/5/7/9.
3. **The region-cue result does not depend on the wording.** Three phrasings give 17.0 / 20.0 /
   18.8%, a 3-pt spread, with the same per-task profile: task 0 survives (58–64%), tasks 3 and 5
   reach 28–40%, and tasks 1, 6, 7, 8, 9 are at 0–14%.
4. **The novel proximity cue varies a lot with the synonym, and 4c's "novel beats familiar" result
   is weaker than the single wording suggested.** The four synonyms span 37.5–53.0%. All four are
   above "next to" (30.5%), but "beside" and "adjacent to" only by 7.0–7.5 pts, which is about 1.5
   standard errors at 200 rollouts each. The gap averaged over wordings is 13.9 pts, against the
   22.5 pts measured with "close to" alone. Task 5 carries most of it (10% for "next to", 28–66%
   for the synonyms).
5. **"beside" is fine as a same-cue swap and harmful as a cross-cue one.** On tasks 1/6/8, where it
   replaces the native "next to", it costs nothing. On tasks 3/5/7/9, where it replaces "on", it
   costs 42.5 pts. The word is the same; what differs is whether the relation still matches the
   one the task was trained with.
6. **Bound on `negative_contrast`'s target rewording.** `negative_contrast` rewords tasks 3/5/9's
   target the same way as `paraphrase_lexical` ("on" → "on top of"). That rewording alone gives
   76% / 76% / 52%, against 52% / 4% / 8% under `negative_contrast`, so it explains little of that
   drop on these tasks.

Results: `results/libero_spatial--{paraphrase_lexical,paraphrase_syntactic,target_cue_region_v2,target_cue_region_v3,target_cue_proximity_beside,target_cue_proximity_near,target_cue_proximity_adjacent}--openvla-7b-libero-spatial-lora-r32--shard{0..3}of4.jsonl`.

---

## 6. Cross-experiment findings

1. **Negative-contrast prompts badly hurt** the policy (-47.2 pts overall; up to -90 on some
   tasks) — the model does not know how to use a "not the one X" clause, it seems to actively
   confuse it.
2. **One extra distractor barely dents overall success** (-3.8 pts, `center_fixed_legacy`) but
   concentrates almost entirely on one task (next to cookie box: 90% -> 44%) where the extra bowl
   lands on the direct path. This was the empirical motivation for Split 2's `irrelevant`
   redefinition — the fixed `table_center` spot used for every task wasn't a uniformly "neutral"
   placement, since it happens to sit near task 6's reach path specifically (see
   `benchmark_split_plan.md` Split 2's confound note).
3. **Open-drawer clutter's raw drop (-20.2 pts) overstates the policy effect** — about half is 3
   tasks where the drawer physically blocks the arm's path, not a perception failure. Adjusted
   (7 feasible tasks): -11.1 to -11.7 pts, a real but smaller robustness gap.
4. **Split 4a is complete (all 6 cells) and confirms the landmark-attraction hypothesis is wrong,
   not just under-tested.** Target-family-pooled SR: landmark 88.35%, surface 84.25%, region 92.0%
   — landmark-target is *not* lower than surface-target, and region-target scores highest of all
   three. The 2 new-scene cells ((surface,landmark)=88.0%, (region,surface)=92.0%) landed close to
   their family's existing cells rather than overturning the pattern — see §5.1.
5. **The negation clause wasn't the source of the damage.** Bare mention of the distractor's
   location, with no "not the one…" clause, is just as bad (-51.6 pts) as mentioning + negating it
   (-47.2 pts) — `positive_contrast` (32.4%) even scores slightly *below* `negative_contrast`
   (36.8%). The policy has no practice grounding a second referent at all; negation specifically
   isn't the issue.
6. **Split 2's first real result cut against its own hypothesis, and the redefined `semantic` confirms
   it.** A distractor placed at an unrelated named landmark costs ~0 pts either way: `semantic_v1_legacy`
   84.8% (+0.8 vs. baseline), current `semantic` 85.2% (+1.2 vs. baseline) — both well inside noise.
   `semantic_v1_legacy`'s task 4 had sat ~0.58m from the target, outside the 0.33-0.50m band the other
   9 tasks land in (likely behaving like a neutral placement rather than a genuine semantic distractor
   for that one task); the redefined `semantic` (`libero_spatial_3bowl_semantic2`, run 2026-08-27,
   500/500) moved it to ~0.48m, in-band, and the 9 unchanged tasks landed bit-for-bit identical while
   task 4 ticked up +4 pts — a properly in-band semantic distractor costs nothing either. Combined with
   finding 5, the pattern holds: failures track *what the prompt says*, not *what's on the table*.
7. **The harder version of the test — `landmark` — does show a real effect, but a concentrated
   one.** A distractor near the target's *own* landmark costs -3.4 pts overall (80.6% vs. 84.0%),
   similar in size to finding 2's plain extra bowl. But like finding 2, that headline number hides
   a task-specific collapse: task 0 (92%→48%) and task 9 (72%→54%) account for nearly all of it,
   every other task flat or improved. Revised picture: scene changes barely matter *in general*,
   but a distractor placed to be a genuine near-miss for the target's own landmark can badly hurt
   specific tasks — proximity-to-own-landmark is the one scene manipulation so far with a real,
   attributable (if concentrated) cost.
8. **Split 2's three-way distractor-position comparison is complete, and the pattern holds under both
   the original and the redefined `irrelevant`/`semantic`.** Current `irrelevant` (+1.2 pts) and
   `semantic` (+1.2 pts) — as well as their retired `irrelevant_v1_legacy` (+4.8 pts) and
   `semantic_v1_legacy` (+0.8 pts) predecessors — all cost nothing; if anything they trend positive
   with no overall-task collapse (`irrelevant`'s task 6 is the one notable single-task soft spot,
   -14 pts, but the suite as a whole doesn't collapse). Only `landmark` (-3.4 pts, concentrated in 2
   tasks) shows a real effect. Combined with findings 5-6, the picture across this entire project so
   far: failures
   come from **language that references a second location** (catastrophic, -47 to -52 pts) or from
   **a distractor that's a genuine look-alike for the target's own described location**
   (task-concentrated, tens of pts on the affected tasks) — not from scene clutter, extra-object
   presence, or a distractor merely sitting at *some* named place in general.
9. **A disambiguating prompt does not rescue the `landmark` scene's task-0/9 collapse — it makes
   things much worse overall.** `landmark_with_hardneg_prompt` (same hard-negative scene as finding
   7, plus "closest to X, not the one farther") scores 41.2%, a further -39.4 pts on top of
   `landmark`'s already-reduced 80.6%. Tasks 0 and 9 barely improve or get worse (48%→38%, 54%→6%),
   while every task that was fine without the prompt collapses instead (task 7: 86%→4%). Language
   that references a second location dominates regardless of whether the scene actually contains a
   confusable distractor — it doesn't compound narrowly on the hard cases, it spreads damage to the
   whole suite. Strongest evidence in the project that this checkpoint's failures are fundamentally
   about prompt grounding, not scene design.
10. **The single largest effect in the whole project: rephrasing the target's cue type — with the
    distractor never mentioned — costs 50 to 67 pts.** Split 4b held the scene, distractor, and the
    referenced fact's truth completely fixed, and varied only whether the target was described via
    its native relation-family wording or an alternate one (region-zone or a disclosed-approximate
    landmark phrasing). `target_cue_region` (8 tasks): 84.0%→17.0% (−67.0 pts). `target_cue_landmark`
    (4 surface-family tasks): 80.5%→30.5% (−50.0 pts). This *exceeds* finding 1's negative-contrast
    damage (−47.2 pts) despite never introducing a second referent — see §5.2.
11. **Findings 1-9 already showed language dominates scene; finding 10 shows it's not really about
    "a second referent" at all — it's template matching on exact phrasing.** Region-cue damage is
    statistically indistinguishable between landmark-family targets (−67.5 pts) and surface-family
    targets (−66.5 pts) — ruling out relation-type difficulty as the mechanism. And landmark-cue
    phrasing (present in 4/10 training tasks' native language) hurts less than region-cue phrasing
    (present in none) on the identical 4 tasks (−50.0 vs. −66.5 pts). Combined with finding 9 (a
    disambiguating prompt spreads damage to unaffected tasks rather than fixing the hard ones), the
    project's overall picture is now: this checkpoint's "grounding" is narrow surface-pattern
    matching against the exact phrasing templates seen at fine-tuning time — not scene-relation
    reasoning (4a, ≤8 pt spread) and not really referent-counting either (a single bowl, truthfully
    described in unfamiliar words, is nearly as damaging as describing two).
12. **Findings 1-11 diagnose failure patterns from success-rate deltas alone, and that turns out to
    be the only tool available.** A direct attempt to separate grounding from action decoding — ask
    the checkpoint, via VQA, which bowl it thinks the target is instead of asking it to act — hit a
    structural dead end confirmed 3 independent ways (see §8): both the fine-tuned checkpoint and the
    unmodified base `openvla-7b` unconditionally emit action-bin tokens after the eval prompt
    template regardless of what's asked (even a content-free control question), and a
    generation-free restricted-logit comparison showed no ranking change across real vs. unrelated
    vs. mismatched instructions for the same image. This is a consequence of OpenVLA's action-only
    continued-pretraining recipe, not something specific to distractor-mention phrasing — but it
    means every interpretation above (11 in particular: "template matching, not real grounding")
    rests on outcome-level evidence and cannot currently be corroborated by directly interrogating
    the model's internal grounding.
13. **The §8 dead end's deferred alternative was pursued (§8.1) and didn't rescue the interpretation
    either way.** A genuinely separate general-purpose VLM (`Qwen2-VL-7B-Instruct`, never trained on
    OpenVLA's action-only template) shown the identical numbered-bowl images/instructions scored
    40-60% against 33-50% chance baselines across all 3 conditions — not convincingly above chance —
    and its raw answers show a numeric/positional bias (zero "1" answers across all 10 `hardneg`
    queries; a 7/10 skew toward "2" on `positive_contrast`) rather than instruction-tracking, even
    though the same model *does* change its answer on 5/10 tasks when only the prompt (not the image)
    differs between `negative_contrast` and `positive_contrast`. So this project's probe rendering
    itself may be too weak a stimulus to cleanly separate "is this language ambiguous" from "is this
    model's grounding broken" — finding 12's caveat (findings 1-11 rest on outcome-level evidence
    alone) still stands, now for a different reason: the direct-interrogation approach hit a second,
    independent dead end even switching model families.
14. **Finding 13 was itself a rendering artifact — the probe's marker placement had a real bug, and
    fixing it flipped the §8.1 conclusion.** User-driven inspection of the probe images found the
    markers were both too large (occluding the target bowl) and, for at least one bowl per scene,
    projected up to ~46px off the bowl's true rendered position (confirmed against MuJoCo's own
    segmentation render as ground truth — see §8.2). After fixing marker placement (segmentation-based
    centroid instead of a hand-projected 3D→2D transform) and shrinking the markers so they no longer
    cover the bowl, re-running the identical Qwen2-VL battery scored 70% on all 3 conditions — clearly
    above chance, versus 40-60% (not convincingly above chance) before the fix. The `negative_contrast`
    /`positive_contrast` phrasing-sensitivity result in finding 13 also reversed: the two conditions
    now agree on 10/10 tasks (previously 5/10 disagreed), meaning that "phrasing sensitivity" was very
    likely the model reacting to noisy markers, not real wording sensitivity. Net effect: §8.1's
    "language may be ambiguous even to a capable VLM" alternative is weaker than it looked — a capable
    VLM resolves the referring expression fine once the stimulus itself isn't broken — but this still
    doesn't corroborate findings 1-11 about OpenVLA specifically (finding 12's dead end for that model
    family is untouched by this fix).
15. **Adding a `default` (no-distractor-mention) baseline to the corrected bowl-pointing probe (§8.3)
    shows distractor-mention phrasing is not inherently harder to ground — for Qwen it's easier.**
    `default` (LIBERO's own target-only task language, distractor never mentioned) scored 5/10 (50%,
    exactly chance) on `libero_spatial`'s 2-bowl scenes, versus 7/10 (70%) for both
    `negative_contrast` and `positive_contrast` on the identical images — the two distractor-mention
    phrasings each correctly resolve 2 tasks (both target=bowl "1") that the target-only phrasing gets
    wrong. So the distractor mention itself is, if anything, a disambiguating cue for a capable VLM,
    not a source of difficulty. This narrows the "maybe the phrasing is just inherently harder to
    ground" reading of findings 1/5/9/10 further than finding 14 already did — it's specifically
    OpenVLA's behavior under that phrasing that's unexplained by the language itself being hard,
    since a different, capable model finds the *same* phrasing easier than no phrasing at all.
16. **The same no-mention-vs-mention comparison holds at 3 bowls, but the gap shrinks and the scene
    itself clears chance easily either way.** `hardneg_default` (3-bowl scene, distractor family never
    mentioned) scored 6/10 (60%, comfortably above the 33% chance floor) versus `hardneg`'s 7/10
    (70%) — the two conditions disagree on exactly 1 of 10 tasks (task id 5, where the mention flips a
    wrong guess to right), a much smaller mention-effect than the 2-bowl comparison's 2/10 (finding
    15). Task id 0 is the single hardest case across the entire 5-condition, 50-query battery — wrong
    regardless of scene or phrasing — while task id 2 is wrong only in the two 3-bowl conditions,
    isolating the extra distractor bowl (not wording) as that task's specific difficulty. Combined with
    finding 15: a capable VLM's accuracy on this referring expression is barely dented by adding a
    third bowl (60% vs. 50%, no-mention baselines) and is never *hurt* by naming the distractor family,
    in either scene — a further data point against reading OpenVLA's findings 1-11 collapse as evidence
    that the language or clutter itself is intrinsically hard to ground.
17. **A first direct behavioral read on OpenVLA's own failure mode (not just outcome deltas) points to
    template mismatch, not distractor-pull, as the dominant mechanism.** Findings 1-16 diagnose OpenVLA
    purely from success/failure counts, since §8 showed its text channel can't be interrogated directly.
    A new instrumented-rollout probe (§8.5) reads behavior instead of language: per-step end-effector-
    to-bowl distance logged across real action rollouts on task 5 ("on the ramekin", the largest
    single-task collapse: 94%→4% under `negative_contrast`, 94%→10% under `target_cue_landmark`). Under
    `default` the arm reaches for the target first in 10/10 episodes (mean closest approach 5cm). Under
    both `negative_contrast` and `target_cue_landmark`, the dominant failure mode is *neither* bowl
    being coherently approached (60% and 80% of episodes respectively) — not the arm confidently
    grasping the wrong bowl (30% under `negative_contrast`, 0% under `target_cue_landmark`, where
    there's no linguistic reason to be pulled toward the untouched second bowl at all). The two
    conditions' "neither" rates are close (60% vs 80%) despite one mentioning a second bowl and the
    other never doing so, while a genuine minority distractor-pull effect (30%) shows up only in
    `negative_contrast`. This is the first evidence in the project that speaks directly to mechanism
    rather than just outcome: the checkpoint's action decoder mostly fails to lock onto a target at all
    once the prompt deviates from its fine-tuning template, and only a minority of `negative_contrast`'s
    damage looks like actual language-driven misdirection toward the named distractor.
18. **"Familiar" phrasing can hurt more than novel phrasing when the familiar template is bound to
    the wrong scene — familiarity alone doesn't explain 4b's landmark-cue drop.** Split 4c (§5.3) held
    relation type fixed at "proximity" and varied only wording familiarity: `target_cue_landmark`
    reuses "next to X" verbatim from tasks 0/1/6/8's native phrasing (30.5% pooled, tasks 3/5/7/9);
    `target_cue_proximity_novel` uses "close to X," which appears in none of the 10 native prompts
    (53.0% pooled, same tasks) — a 22.5pt gap in the *opposite* direction from the plan's predicted
    "template familiarity is protective" branch (task 5 alone: 10% vs. 66%). Reusing a seen-at-
    finetuning phrase does not generically help; it appears to actively bind the policy to the wrong
    scene-specific behavior when that exact phrase is natively associated with different tasks/targets,
    which fits finding 17's action-collapse mechanism better than a simple template-match/no-match
    story: a phrase with no strong prior at all degrades more gracefully than one with a strong, wrong
    prior. Full three-way synthesis with 4a and 4b in `benchmark_split_result.md` §5.4.
19. **Mechanistic localization finds real internal differences, but not a localized mechanism.**
    With the eval reproduced faithfully (center-crop, teacher-forced diagnostics so that requesting
    attentions doesn't change the executed policy) and the episode as the statistical unit,
    `target_cue_landmark` shows a replicated early-window shift in vision-attention mass and
    resolution layer vs. `default` (n=50, d=0.62 / 0.45). `target_cue_proximity_novel` diverges in the
    opposite direction, and final-layer decision confidence doesn't differ by condition at all.
    Net: the prompt changes *where the network looks and how fast it resolves*, not *how decisively
    wrong* it is. Gap 3 (`benchmark_split_plan.md` §11.3) is narrowed, not closed. The full trail,
    including two instrumentation bugs that invalidated earlier runs, is in
    `mechanistic_localization.md`.

20. **Prompt length / off-template filler is itself a large cause of OpenVLA's contrast drop, but
    not all of it** (2026-10-01). A content-free politeness clause of the contrast clause's length
    costs 21.2 pts mid-sentence and 30.4 pts at the end (§2). That is 45% of `negative_contrast`'s
    drop and 59% of `positive_contrast`'s; the remaining 26.0 / 21.2 pts depend on the clause's
    content. This revises finding 1 and the "bare mention does all of it" reading of Split 1: the
    damage is part generic off-template text, part distractor mention. The per-task profile also
    differs — filler hits tasks 7 and 9, the contrast clauses additionally collapse task 5.
21. **Same-cue paraphrases cost OpenVLA little; cross-cue rewordings cost a lot, whatever the
    wording** (2026-10-01, §5.5). A relative-clause paraphrase costs 1.8 pts and a synonym swap
    11.6 pts, while the region cue costs 64–67 pts in each of three phrasings (17.0 / 20.0 / 18.8%).
    4b's collapse is therefore about cue type, and it is not an artifact of one hand-written
    phrasing. The checkpoint does not fail on every surface-form change.
22. **`pi05_libero` is far more robust to prompts and less robust to an extra bowl** (2026-10-01,
    §9). Length controls and same-cue paraphrases cost it about 0–1 pt and the contrast prompts
    6–11 pts, where OpenVLA loses 2–52. The region cue is its one clear prompt weakness (75–78%
    across three wordings). On the 3-bowl scenes it scores below OpenVLA on `irrelevant`,
    `landmark`, and `drawer_open`, with single tasks dropping to 0–22%. The two models differ in
    training data (40 LIBERO tasks vs. `libero_spatial` only), camera inputs, and control loop, so
    this is a reference point, not a controlled comparison.

23. **4c's "novel phrasing beats the familiar one" holds on average but is wording-dependent**
    (2026-10-01, §5.5). Four proximity synonyms never seen in training give 37.5–53.0% on tasks
    3/5/7/9, all above "next to" (30.5%), but "beside" and "adjacent to" only by about 7 pts. The
    gap averaged over wordings is 13.9 pts, not the 22.5 pts in finding 18, which used "close to"
    alone. Treat finding 18's size as an upper end.

## 7. Render / contact-sheet check log

Every new or previously-unchecked scene gets its init-state contact sheet rendered
(`LIBERO/scripts/render_suite_contact_sheet.py <suite>`) and eyeballed before spending GPU time on
it. Where a run adds a distractor to the 2-bowl baseline, a side-by-side comparison is also
rendered (`LIBERO/scripts/compare_two_suites_init.py libero_spatial <suite> <outname>` — left/blue
= 2-bowl baseline, right/orange = the 3-bowl variant, one row per task id 0–9) and saved under
`openvla/experiments/figures/` for permanent reference (the raw per-suite contact sheets under
`LIBERO/scratch_render/` are scratch and get overwritten each pass). The figures themselves are
embedded inline in each condition's section above (§3-4).

**2026-08-26 render-settling fix.** `render_suite_contact_sheet.py` rendered immediately after
`set_init_state()` with no settle steps, so some per-task renders (including cells in §3's per-task
render table) showed bowls still mid-fall/floating from their sampled init height instead of resting
on the surface — `compare_two_suites_init.py` never had this problem since it already stepped a
dummy action first. Both scripts now settle for 10 steps with the same no-op action
`run_libero_eval.py` uses via `cfg.num_steps_wait`, so every render reflects what the real eval
actually sees. Regenerated: the 4 suites feeding §3's per-task render table (`libero_spatial`,
`libero_spatial_3bowl_neutral`, `libero_spatial_3bowl_semantic`, `libero_spatial_3bowl_hardneg`) and
their thumbnails; not regenerated (unaffected — already had a settle step, unchanged appearance at
10 vs. the prior 12 steps): the five `compare_*_grid.png` figures embedded in §3-4.

**2026-08-27 follow-up.** `libero_spatial_3bowl_front`/`libero_spatial_3bowl_semantic2` missed the
2026-08-26 pass above (authored the same day, but their contact sheets had been captured via
`gen_suite_init_states.py`'s own inline preview, which has no settle step at all — same root cause,
different script). Caught when the `libero_spatial_3bowl_front` grid was eyeballed again and a bowl
was visibly still falling in one panel. Re-rendered both with `render_suite_contact_sheet.py`
(read-only over the already-generated/verified `.pruned_init` files, so `verify_suite_init_states.py`'s
PASS results below are unaffected — this only redid the preview image, not the init states); both
now show every bowl resting flat with contact shadows.

| Suite | Numeric verify (`verify_suite_init_states.py`) | Visual eyeball | Result |
|---|---|---|---|
| `libero_spatial_3bowl_neutral` (`irrelevant_v1_legacy`) | PASS, worst sep 0.122m | ✅ | 3 distinct bowls per task, no overlaps |
| `libero_spatial_3bowl_semantic` (`semantic_v1_legacy`) | verified | ✅ | 3 distinct bowls per task, no overlaps/clipping, drawer open only on task 4 (expected — task 4's target lives in the drawer) |
| `libero_spatial_3bowl_front` (`irrelevant`, current, 2026-08-26) | First pass FAIL (`table_front` alone) — worst sep 0.107m, tasks 3 & 5 overlap. Fixed by falling back tasks 1, 3, 5, 6 to `table_center`; re-verify PASS, worst sep 0.122m | ✅ (re-eyeballed 2026-08-27 after the settle-timing fix above) | 3 distinct bowls per task, no overlaps/clipping, every bowl resting flat with a contact shadow (first-pass grid had one still mid-fall); 3rd bowl visibly isolated near the table's front edge in every panel |
| `libero_spatial_3bowl_front` offset fine-tune (2026-08-27) | `table_front` +5cm front / `table_center` fallback +5cm back (still ≈0.29m from `stove_region`); re-verify PASS, worst sep 0.122m (task 4, unchanged, unrelated to bowl_3), all other tasks 0.148–0.301m (up from 0.122–0.276m) | ✅ | Bowl_3 visibly farther toward the front edge in the 6 front tasks, farther back (away from the cluster) in the 4 fallback tasks; still resting flat, no overlaps/clipping |
| `libero_spatial_3bowl_semantic2` (`semantic`, current, 2026-08-26) | First pass FAIL (task 4 at `next_to_box_region`) — worst sep 0.063m, open drawer's 3D footprint collides. Fixed by moving task 4 to `between_plate_ramekin_region`; re-verify PASS, worst sep 0.122m | ✅ (re-eyeballed 2026-08-27 after the settle-timing fix above) | 3 distinct bowls per task, no overlaps/clipping, every bowl resting flat with a contact shadow; task 4's 3rd bowl visibly near the plate/ramekin cluster instead of the far corner |
| `libero_spatial_3bowl_hardneg` (`landmark` / `landmark_with_hardneg_prompt`, same scene) | PASS but narrow — min sep 0.121m vs. 0.12m threshold, task 3 tightest | ✅ | 3 distinct bowls per task; task 3's close pair confirmed as two separate bowls, not merged |
| `libero_spatial_grounding_surface_landmark` (`grounding/surface_landmark`) | PASS, sep 0.393m | ✅ | Single task (`on_the_ramekin`). Cross-checked exact xyz against the BDDL, not just pixels: target `akita_black_bowl_1` = (-0.210, 0.192, z=1.080) — inside `ramekin_region` (-0.21,0.19)-(-0.19,0.21), elevated (on top of the ramekin, as intended). Distractor `akita_black_bowl_2` = (0.116, -0.067, z=0.970) — inside `next_to_box_region` (0.12,-0.08)-(0.14,-0.06) (0.004m outside on x, negligible), flat on the table (not elevated) — confirms it moved off `cookies_1` (was elevated/surface in the original task 5) to a landmark placement, as designed. No overlap/clipping. |
| `libero_spatial_grounding_region_surface` (`grounding/region_surface`) | PASS, sep 0.210m | ✅ | Single task (`from_table_center`). Target `akita_black_bowl_1` = (-0.075, 0.003, z=0.970) — inside `table_center` (-0.10,-0.01)-(-0.05,0.01), flat on table (region cue, as intended). Distractor `akita_black_bowl_2` = (-0.263, -0.137, z=1.010) — y matches `stove_region`'s -0.14 almost exactly, x offset from the stove's base anchor (-0.41) is consistent with `flat_stove_1_cook_region` being the stove's own top surface (same region used by tasks 6/9's distractors elsewhere in the suite), elevated (on top of the stove) — confirms surface placement, moved off `next_to_plate_region` (landmark in the original task 2). No overlap/clipping. |

`libero_spatial` (baseline) and `libero_spatial_3bowl`/`libero_spatial_3bowl_open`
(`center_fixed_legacy`/`drawer_open`) predate this render-before-run practice being tracked here;
no issues have surfaced in their data, but no dedicated check is logged. The two grounding suites
above were checked here before running — both have since been run, see §5.1.

Note on method: for these two single-task suites, plain pixel-diffing the new render against the
original `libero_spatial` task's render (t5 for surface_landmark, t2 for region_surface) was tried
first and was inconclusive — moving one bowl shifts shadows/specular highlights across a large
fraction of a 256x256 frame, so a large diff bounding box doesn't distinguish "one bowl moved" from
"something is wrong." Reading the actual simulator joint `qpos` for each bowl (as tabulated above)
and comparing against the BDDL region catalog is unambiguous and is the more reliable check for a
single-object scene change — recommended over pixel-diffing for any future single-task gap-fill
suite in this project.

## 8. VLM Bowl-Pointing Probe — grounding-vs-action-decoding diagnostic (2026-08-25)

**Question.** Every distractor-mention condition (`negative_contrast`, `positive_contrast`,
`landmark_with_hardneg_prompt`) collapses this checkpoint's task success, but end-to-end success can't
say *why*: is vision-language grounding itself broken once a second referent is mentioned, or is
grounding fine and only action-decoding falls apart on out-of-distribution phrasing? This probe tries
to isolate the two by showing the model the same scene with each black bowl overlaid with a random
number (Set-of-Mark style) and asking, in free text, which numbered bowl matches the (unmodified)
failing instruction — swapping "output an action" for "output a number" while holding scene and
language fixed. Ground truth: every task's BDDL goal names `akita_black_bowl_1` as the target.

**Method.** `openvla/experiments/robot/libero/probe_bowl_pointing.py`: render the exact episode-0 init
state the real eval would see, project each bowl's 3D position into the frame, overlay a shuffled
marker number per bowl, then call the checkpoint's `.generate()` directly (bypassing action-only
decoding) with the instruction reformatted as "which numbered bowl...".

**Result — dead end on OpenVLA, confirmed three ways.**

1. Free-text generation on the fine-tuned checkpoint always returns action-bin tokens (ids in
   `[31744, 31999]`, the tail-of-vocabulary range reserved for the 256 action bins) regardless of the
   question — a real bowl-pointing query and a content-free control ("What is 2+2?") both decode to
   garbage.
2. The base, pre-LIBERO-finetune `openvla/openvla-7b` does the same — this is structural to the
   OpenVLA checkpoint family's action-only continued-pretraining recipe, not a symptom of this
   project's fine-tune or its distractor-mention conditions.
3. A restricted-logit comparison (raw next-token logits over just the candidate digit tokens, no
   generation) found the ranking among candidates tracks only *which image* was shown, identically
   across three different prompt variants (real instruction / unrelated question / mismatched
   instruction) — whatever tiny logit gap exists isn't responsive to language at all.

**Conclusion.** OpenVLA has no text-output channel causally responsive to language at the point this
diagnostic needs to query it. Grounding cannot be separated from action decoding via a VQA-style probe
on this model family; end-to-end task success remains the only measurable signal for it. Confirmed on
3 independent angles, so the full battery wasn't run on OpenVLA — it would only reproduce the same
negative result. The script is left in place and works end-to-end; reusable for a future checkpoint
without this collapse, or for probing a genuinely separate VLM (pursued below).

### 8.1 Qwen2-VL-7B-Instruct: is the referring expression resolvable in principle? (2026-08-25)

**Superseded by §8.2's marker-placement fix — numbers below are historical, see §8.2 for corrected ones.**

`probe_bowl_pointing_qwen.py` reuses the same render/annotate/score pipeline but queries
`Qwen/Qwen2-VL-7B-Instruct` via ordinary free-text `.generate()`. Full battery (3 conditions × 10
tasks, no parse failures):

| Condition | Scene | Accuracy | Chance |
|---|---|---|---|
| `negative_contrast` | `libero_spatial` (2 bowls) | 6/10 (60%) | 50% |
| `positive_contrast` | `libero_spatial` (2 bowls) | 4/10 (40%) | 50% |
| `hardneg` | `libero_spatial_3bowl_hardneg` (3 bowls) | 6/10 (60%) | 33% |

None cleared chance convincingly, and answers showed a numeric/positional bias rather than
target-tracking (e.g. Qwen answered "1" zero times across all 10 `hardneg` queries, regardless of
where the target actually was). Read at the time as "not resolvable in principle even by a capable
VLM" — revised in §8.2.

### 8.2 Marker-placement bug found and fixed — Qwen re-run (2026-08-26)

**Bug.** User inspection flagged large, occluding markers and some that looked off-bowl. Sweeping
`num_steps_wait` ruled out physics-settle timing as the cause, but cross-checking the hand-projected
marker position against MuJoCo's own segmentation render exposed a real bug:
`project_points_from_world_to_camera()`'s output was off by ~46px (20% of the frame) for one bowl on
one task — enough to land the marker on bare table. Root mechanism in the projection call not
identified, but segmentation-based ground truth is trustworthy by construction (same draw call as the
RGB frame).

**Fix.** `bowl_pointing_common.py` now reads each bowl's segmentation mask directly and uses its pixel
centroid as the marker position, sidestepping the projection call entirely; markers also changed from
occluding filled circles to small outline dots with an offset label.

**Re-run, same model/prompts/scoring, only the images changed:**

| Condition | Before fix (§8.1) | After fix | Chance |
|---|---|---|---|
| `negative_contrast` | 60% | **70%** | 50% |
| `positive_contrast` | 40% | **70%** | 50% |
| `hardneg` | 60% | **70%** | 33% |

All three now clear chance clearly. A second thing changed with the fix: `negative_contrast` and
`positive_contrast` (identical images, only wording differs) disagreed on 5/10 tasks before the fix,
now agree on 10/10 — the earlier "phrasing sensitivity" finding was very likely noisy-marker artifact,
not a real wording effect.

**Revised conclusion.** With accurate markers, the referring expression IS resolvable well above
chance by a capable VLM — §8.1's "not resolvable in principle" was itself a rendering artifact, not
evidence about the language.

### 8.3 `default` (no-distractor-mention) baselines added, 2-bowl and 3-bowl (2026-08-26)

**Motivation.** §8.1/§8.2 only tested distractor-mention phrasing. Missing: how Qwen does on the same
scenes under LIBERO's own native, target-only task language (Split 1's `default` condition) — the
natural comparison point.

**Result — both `default` baselines score at or below their distractor-mention counterpart, not above:**

| Condition | Scene | Accuracy | Chance |
|---|---|---|---|
| `default` (no mention) | `libero_spatial` (2 bowls) | **50%** | 50% |
| `negative_contrast` | `libero_spatial` (2 bowls) | 70% | 50% |
| `positive_contrast` | `libero_spatial` (2 bowls) | 70% | 50% |
| `hardneg_default` (no mention) | `libero_spatial_3bowl_hardneg` (3 bowls) | **60%** | 33% |
| `hardneg` | `libero_spatial_3bowl_hardneg` (3 bowls) | 70% | 33% |

`default` disagrees with the distractor-mention conditions on 2/10 tasks (both target=bowl "1"): given
only the target's own location, Qwen picks the wrong bowl; stating where the *other* bowl is fixes
both. `hardneg_default` vs. `hardneg` differ on just 1/10. Both 3-bowl conditions clear the 33% chance
baseline comfortably even with zero distractor mention.

**Reading.** For this scene/model, naming the distractor family — negated, positive, or hard-negative —
is a mild *disambiguating* cue for Qwen, not a difficulty source, in both scenes. This sharpens the
finding that OpenVLA's distractor-mention collapse (findings 1-11) can't be explained by "the mention
itself makes the scene harder to ground" — a capable VLM grounds it at least as well as no mention at
all.

### 8.4 Per-task render + marker table (2026-08-26)

Every rendered scene actually fed to Qwen, markers baked in, with each condition's `target→answer`
verdict (✓ = correct). Marker color: **1**=red, **2**=green, **3**=blue (shuffled per task).

**2-bowl scenes (`libero_spatial`) — `default`/`negative_contrast`/`positive_contrast` show the identical image per task:**

| id | task | scene | `default` | `negative_contrast` | `positive_contrast` |
|--:|---|---|---|---|---|
| 0 | between the plate and the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t0.png) | 1→2 ✗ | 1→2 ✗ | 1→2 ✗ |
| 1 | next to the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t1.png) | 2→2 ✓ | 2→2 ✓ | 2→2 ✓ |
| 2 | from table center | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t2.png) | 1→1 ✓ | 1→1 ✓ | 1→1 ✓ |
| 3 | on the cookie box | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t3.png) | 2→2 ✓ | 2→2 ✓ | 2→2 ✓ |
| 4 | in the top drawer of the wooden cabinet | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t4.png) | 1→2 ✗ | 1→1 ✓ | 1→1 ✓ |
| 5 | on the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t5.png) | 1→2 ✗ | 1→1 ✓ | 1→1 ✓ |
| 6 | next to the cookie box | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t6.png) | 1→1 ✓ | 1→1 ✓ | 1→1 ✓ |
| 7 | on the stove | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t7.png) | 1→1 ✓ | 1→1 ✓ | 1→1 ✓ |
| 8 | next to the plate | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t8.png) | 2→1 ✗ | 2→1 ✗ | 2→1 ✗ |
| 9 | on the wooden cabinet | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial--negative_contrast--t9.png) | 1→2 ✗ | 1→2 ✗ | 1→2 ✗ |

**3-bowl scenes (`libero_spatial_3bowl_hardneg`) — `hardneg_default`/`hardneg` show the identical image per task:**

| id | task | scene | `hardneg_default` | `hardneg` |
|--:|---|---|---|---|
| 0 | between the plate and the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t0.png) | 1→2 ✗ | 1→2 ✗ |
| 1 | next to the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t1.png) | 3→2 ✗ | 3→1 ✗ |
| 2 | from table center | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t2.png) | 2→3 ✗ | 2→3 ✗ |
| 3 | on the cookie box | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t3.png) | 2→2 ✓ | 2→2 ✓ |
| 4 | in the top drawer of the wooden cabinet | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t4.png) | 2→2 ✓ | 2→2 ✓ |
| 5 | on the ramekin | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t5.png) | 3→2 ✗ | 3→3 ✓ |
| 6 | next to the cookie box | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t6.png) | 2→2 ✓ | 2→2 ✓ |
| 7 | on the stove | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t7.png) | 2→2 ✓ | 2→2 ✓ |
| 8 | next to the plate | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t8.png) | 2→2 ✓ | 2→2 ✓ |
| 9 | on the wooden cabinet | ![](https://raw.githubusercontent.com/Qian-0203/openvla/main/experiments/figures/probe_bowl_pointing/libero_spatial_3bowl_hardneg--hardneg--t9.png) | 3→3 ✓ | 3→3 ✓ |

Source images: `openvla/experiments/figures/probe_bowl_pointing/` (`openvla` commit `1b27db3`).

### 8.5 Bowl-attraction probe — instrumented action rollouts, not VQA (2026-09-02 through 2026-09-04)

**Question.** Since VQA can't probe OpenVLA (§8's dead end), read its only language-responsive channel
— the action output — directly: instrument real rollouts with per-step end-effector-to-bowl distance,
and classify each failed episode by which bowl (if any) the gripper reached for first. Distinguishes
(a) language pulls the arm toward the wrong bowl (grounding-adjacent misdirection) vs. (b) the arm
fails to lock onto *any* target once the prompt deviates from its fine-tuning template.

**Method.** `probe_bowl_attraction.py` runs the real `run_libero_eval.py` action loop (same model,
preprocessing, seed/init-state protocol) and additionally logs each bowl's distance to the
end-effector every step. Per episode: running min distance to each bowl, and the first step (if any)
the gripper came within 8cm of each — `first_bowl_approached` is whichever bowl that was first, `None`
if neither. `libero_spatial` (2-bowl scene), 10 episodes/condition/task, same seed-7 init states as the
real eval. Ran across four rounds as scope grew (task 5 only → tasks 3/7/9 → all 10 tasks →
`target_cue_proximity_novel` added); full launch-by-launch history in `eval_log.md`'s 2026-09-02/03/04
entries. This section reports the final, consolidated numbers only.

**Result — pooled success rates closely reproduce the real eval, validating the probe:**

| Task | `default` | `negative_contrast` | `target_cue_landmark` | `target_cue_proximity_novel` |
|--:|--:|--:|--:|--:|
| 0 | 80% | 90% | — | — |
| 1 | 70% | 40% | — | — |
| 2 | 90% | 10% | — | — |
| 3 | 90% | 40% | 50% | 70% |
| 4 | 90% | 60% | — | — |
| 5 | 100% | 0% | 0% | 50% |
| 6 | 100% | 70% | — | — |
| 7 | 80% | 0% | 20% | 50% |
| 8 | 60% | **70%** | — | — |
| 9 | 50% | 0% | 20% | 30% |
| **Pooled** | **81.0%** | **38.0%** | **22.5%** | **50.0%** |
| Real 500/200-trial SR | 84.0% | 36.8% | 30.5% | 53.0% |

Task 8 is a flagged outlier (`negative_contrast` scored above `default`, opposite the real eval's
direction; instructions verified correct, read as n=10 sampling noise). Every other task tracks the
real numbers' direction and rough magnitude.

**Result — failure-mode breakdown among failed episodes, matched 4-task cohort (3, 5, 7, 9):**

| Condition (n=40) | Success | Target-first, still failed | Distractor-first | Neither |
|---|--:|--:|--:|--:|
| `default` | 80.0% | 62.5% | 0% | 37.5% |
| `negative_contrast` | 10.0% | 27.8% | 16.7% | **55.6%** |
| `target_cue_landmark` | 22.5% | 38.7% | 0% | **61.3%** |
| `target_cue_proximity_novel` | 50.0% | 40.0% | 0% | **60.0%** |

**`target_cue_proximity_novel` is the key data point.** It mentions no distractor, like
`target_cue_landmark` — but where `target_cue_landmark` reuses "next to X" (a phrase natively bound to
*other* tasks, finding 18), `target_cue_proximity_novel`'s "close to X" has no such training-time
association. Despite that, its failure shape is nearly identical (60.0% vs. 61.3% neither, 0%
distractor-first in both). The two conditions differ enormously in *how often* they fail (50.0% vs.
22.5% success) but, conditional on failing, fail the *same way* — independent evidence that
distractor-pull isn't the mechanism: a condition that structurally cannot exhibit it still shows
target-selection collapse as its majority failure mode.

**Result — full 10-task cohort, `default`/`negative_contrast`, among failed episodes:**

| Condition (n=100) | Success | Target-first, still failed | Distractor-first | Neither |
|---|--:|--:|--:|--:|
| `default` | 81.0% | 63.2% | 0% | 36.8% |
| `negative_contrast` | 38.0% | 27.4% | 14.5% | **58.1%** |

At full scale, "arm never approaches either bowl" is the outright *majority* failure mode for
`negative_contrast` (58.1%). Distractor-directed misdirection is real but secondary (14.5%,
tasks 1/2/5/9). Target-approached-but-still-failed (27.4%) is concentrated in task 3 across every
condition it was tested under — this probe doesn't instrument grasp/lift/place, so it can't say why,
but it's consistent with off-template phrasing disrupting the policy at more than one stage for that
specific task.

Artifacts: `openvla/experiments/logs/probe_bowl_attraction/` (per-task JSONL + summaries, gitignored);
rollout videos under `openvla/rollouts/2026_09_0{2,3,4}/`. Code:
`experiments/robot/libero/probe_bowl_attraction.py`. Full launch history (smoke tests, operational
incidents, hardware): `eval_log.md`'s 2026-09-02/03/04 entries.

### 8.6 Synthesis — why the distractor-mention collapse shows up in the VLA but not the VLM (2026-09-02, updated through 2026-09-10)

**What's ruled out.** Not that the language is ambiguous and only OpenVLA fails to parse it (§8.1-8.3
— Qwen2-VL resolves distractor-mention phrasing at 70%, scoring it *easier* to ground than LIBERO's
own target-only language). Not "the distractor pulls the arm toward the wrong object" as the dominant
mechanism either (§8.5 — misdirection was only 14.5% of `negative_contrast` failures; the majority was
the arm never approaching *either* bowl). `target_cue_proximity_novel` strengthens this further: no
distractor mention and no misleading template association, yet the same majority-"neither" failure
shape.

**Best-supported hypothesis.** §8.5's instrumented rollouts, pooled across all 10 tasks (280 rollouts
`default`+`negative_contrast`, plus `target_cue_landmark`/`target_cue_proximity_novel` on the 4-task
surface cohort), point to *template-mismatch action collapse*: once the prompt deviates from the
fine-tuning template, the action decoder most often stops committing to any target at all (58.1% of
`negative_contrast` failures — a majority). A model with no action-decoding head (Qwen) never exhibits
this because it's never asked to act; a model whose only output is action tokens, fine-tuned on one
narrow template per task (finding 18, "template *binding*, not template *matching*"), collapses on
phrasing a VLM finds easy.

**Three gaps, current status:**

1. **Sample size — closed.** Extended from 1 task to all 10 (280 rollouts). The "arm never commits"
   pattern is the outright majority of failures at full scale (58.1%), not a one-task artifact. Task
   3's "approached correctly, still failed" persists as a genuine second mechanism (27.4% of failures,
   task-concentrated). Pooled success rates closely reproduce the real eval (81%/38% vs. 84.0%/36.8%).
2. **Length/complexity control — partially addressed.** `target_cue_landmark`/`target_cue_proximity_novel`
   are word-for-word the same length as `default` (single relation-word swap, no added clause) yet
   still collapse to the same majority-"neither" shape — rules out "it's simply a longer sentence" as
   the explanation for *this* failure shape. Still open: whether `negative_contrast`'s own extra clause
   (vs. its distractor content specifically) matters, since no condition adds comparable filler length
   without semantic content.
3. **Mechanistic localization — open, substantially narrowed.** Real, replicated early-window
   vision-attention/resolution-layer effects for `target_cue_landmark`, no module localized, and the
   confidence diagnostic is null. See `mechanistic_localization.md` §2.

**Net.** The VLA/VLM split is well explained at the outcome level (findings 12-17). "Arm fails to
commit to a target" is the best-supported majority mechanism, directly evidenced (not inferred) at full
10-task scale, with a second, task-concentrated "approached but still failed" mode. Mechanistic
localization has moved from "two coarse diagnostics, both null" to "vision-attention is a real,
replicated, condition- and window-dependent effect that isn't yet a clean localization" — see
`mechanistic_localization.md`.

### 8.7 Qwen bowl-pointing probe, re-run with sampled (not greedy-only) decoding (2026-09-04)

**Motivation.** Every Qwen run so far drew one greedy decode per query — no way to tell a stable
tendency from a lucky/unlucky draw.

**Method.** `probe_bowl_pointing_qwen.py` changed to draw 10 samples/query at temperature 0.7 (batched
`num_return_sequences=10`), reporting per-query sample accuracy and majority vote. Same 5 conditions ×
10 tasks as §8.3 (500 generations).

**Result — zero within-query disagreement: the sampled "trend" is that there isn't one.** All 50
queries landed all 10 samples on the same bowl number (`sample_accuracy` exactly 0.0 or 1.0, never
between).

| Condition | Sample-mean accuracy | §8.3 greedy accuracy |
|---|--:|--:|
| `default` | 50.0% | 50% |
| `negative_contrast` | 60.0% | 70% |
| `positive_contrast` | 80.0% | 70% |
| `hardneg_default` | 60.0% | 60% |
| `hardneg` | 70.0% | 70% |

`default`/`hardneg`/`hardneg_default` reproduce §8.3 exactly, task-for-task. `negative_contrast` and
`positive_contrast` each move by one task (unanimous across all 10 samples both times, not a close
call) — an unexplained but small delta between the old single-sequence and new batched `generate()`
call path (bf16 batching-precision is the natural suspect, not investigated). Effect size doesn't
change either condition's qualitative reading. The main result — no query shows genuine draw-to-draw
disagreement — confirms §8.1-8.6's conclusions rather than undercutting them.

### 8.8 Qwen3-VL-8B-Instruct re-run — does a newer/stronger VLM raise the ceiling? (2026-09-04)

**Motivation.** §8.7's numbers (48-90%, chance 33-50%) still looked low for a "resolvable in
principle" upper bound. Same pipeline, same conditions/tasks, only the model swapped.

**Result — large 2-bowl gains, a 3-bowl regression, not a uniform upgrade:**

| Condition | Scene | Qwen2-VL (§8.7) | Qwen3-VL-8B | Δ |
|---|---|--:|--:|--:|
| `default` | 2 bowls | 50.0% | **81.0%** | +31.0 |
| `negative_contrast` | 2 bowls | 60.0% | **90.0%** | +30.0 |
| `positive_contrast` | 2 bowls | 80.0% | **84.0%** | +4.0 |
| `hardneg_default` | 3 bowls | 60.0% | **42.0%** | −18.0 |
| `hardneg` | 3 bowls | 70.0% | **48.0%** | −22.0 |

Every condition still clears its chance baseline, so §8.6's core synthesis holds for Qwen3-VL too — but
it isn't a strictly-better replacement: +30pt on the 2-bowl scene, −18/−22pt on the 3-bowl `hardneg`
scene. Task 3 ("on the cookie box") flips correct→incorrect in **all 5 conditions**, unanimously — the
clearest single-task regression, concentrated rather than diffuse. Genuine sample-to-sample
disagreement appeared on 5/50 queries (vs. 0/50 for Qwen2-VL at the same temperature) — a real
trend to characterize for the first time, task 1 (2-bowl scene) being the most consistently uncertain.

Artifacts (§8.1-8.8): `openvla/experiments/logs/probe_bowl_pointing_qwen{,3}/*.jsonl` (gitignored,
local only); images `openvla/experiments/figures/probe_bowl_pointing/`. Code:
`bowl_pointing_common.py`, `probe_bowl_pointing{,_qwen,_qwen3}.py` (`openvla` fork; `bowl_pointing_common.py`
+ `probe_bowl_pointing.py`/`probe_bowl_pointing_qwen.py`/`probe_bowl_pointing_qwen3.py` at commit
`1b27db3`, later changes uncommitted as of this write-up).

### 8.9–8.15 Mechanistic localization probe — moved to `mechanistic_localization.md`

Where in the network the "never commits to a target" failure (§8.5) originates is now recorded in its
own file, `mechanistic_localization.md`, with its original section numbers (8.9–8.15) kept intact.
Current conclusion, in one line: early-window vision-attention and resolution-layer effects for
`target_cue_landmark` are real and replicated (n=50), but no layer or module has been localized, and
final-layer decision confidence does not differ by condition.

---

## 9. Cross-model reference: `pi05_libero` on the whole benchmark (2026-10-01)

**What this is.** The first non-OpenVLA policy run through the benchmark: openpi's `pi05_libero`
checkpoint on all 13 current splits plus the 9 conditions authored 2026-10-01 — 22 conditions,
8,300 rollouts. Same registry, scenes, init states, seed (7), `env.seed(0)`, and 50 trials/task as
the OpenVLA reference. Launch details: `eval_log.md`, 2026-09-30 → 2026-10-01.

**Read these caveats before comparing the two columns.**

- **Training data differs.** `pi05_libero` was trained on all four LIBERO suites (40 tasks). The
  OpenVLA reference was fine-tuned on `libero_spatial` only.
- **Observation and control differ.** pi05 sees agentview + wrist images (resize-with-pad) and
  proprio state, and replans every 5 steps. OpenVLA sees one center-cropped agentview image and
  acts every step.
- **One run per condition.** The noise bands from §0 apply: ±3.3 pts pooled over 500, ±5–7 pts on a
  single task.

### 9.1 — Overall, against the OpenVLA reference

| Split | Probe | Tasks | OpenVLA | pi05_libero | Rollouts (pi05) |
|---|---|---|--:|--:|--:|
| `spatial/default` | 1 · Prompt | 0–9 | 84.0% | **98.4%** | 492/500 |
| `spatial/positive_contrast` | 1 · Prompt | 0–9 | 32.4% | **92.2%** | 461/500 |
| `spatial/negative_contrast` | 1 · Prompt | 0–9 | 36.8% | **87.0%** | 435/500 |
| `spatial/length_control_infix` | 1 · Length control | 0–9 | 62.8% | **99.2%** | 496/500 |
| `spatial/length_control_suffix` | 1 · Length control | 0–9 | 53.6% | **99.0%** | 495/500 |
| `grounding/paraphrase_lexical` | 4 · Same-cue paraphrase | 0–9 | 72.4% | **97.2%** | 486/500 |
| `grounding/paraphrase_syntactic` | 4 · Same-cue paraphrase | 0–9 | 82.2% | **97.8%** | 489/500 |
| `grounding/target_cue_region` | 4b · Cue type | 0,1,3,5–9 | 17.0% | **75.5%** | 302/400 |
| `grounding/target_cue_region_v2` | 4b · Cue type | 0,1,3,5–9 | 20.0% | **77.5%** | 310/400 |
| `grounding/target_cue_region_v3` | 4b · Cue type | 0,1,3,5–9 | 18.8% | **74.8%** | 299/400 |
| `grounding/target_cue_landmark` | 4b · Cue type | 3,5,7,9 | 30.5% | **87.0%** | 174/200 |
| `grounding/target_cue_proximity_novel` | 4c · "close to" | 3,5,7,9 | 53.0% | **96.0%** | 192/200 |
| `grounding/target_cue_proximity_beside` | 4c · "beside" | 3,5,7,9 | 38.0% | **97.0%** | 194/200 |
| `grounding/target_cue_proximity_near` | 4c · "near" | 3,5,7,9 | 49.0% | **96.0%** | 192/200 |
| `grounding/target_cue_proximity_adjacent` | 4c · "adjacent to" | 3,5,7,9 | 37.5% | **97.0%** | 194/200 |
| `spatial_3bowl/irrelevant` | 2 · Distractor | 0–9 | 85.2% | **80.6%** | 403/500 |
| `spatial_3bowl/semantic` | 2 · Distractor | 0–9 | 85.2% | **91.0%** | 455/500 |
| `spatial_3bowl/landmark` | 2 · Distractor | 0–9 | 80.6% | **72.0%** | 360/500 |
| `spatial_3bowl/landmark_with_hardneg_prompt` | 1×2 | 0–9 | 41.2% | **66.2%** | 331/500 |
| `spatial_3bowl/drawer_open` | 3 · Clutter | 0–9 | 60.0% | **43.4%** | 217/500 |
| `grounding/surface_landmark` | 4a · Scene | 0 | 88.0% | **100.0%** | 50/50 |
| `grounding/region_surface` | 4a · Scene | 0 | 92.0% | **100.0%** | 50/50 |

### 9.2 — Per-task: prompt-only conditions (stock 2-bowl scene)

| id | target | Default | Pos. contrast | Neg. contrast | Len. infix | Len. suffix | Para. lexical | Para. syntactic |
|--:|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 1 | next to the ramekin | 100% | 96% | 82% | 98% | 100% | 90% | 100% |
| 2 | table center | 96% | 100% | 100% | 100% | 100% | 100% | 100% |
| 3 | on the cookie box | 98% | 98% | 100% | 98% | 100% | 98% | 98% |
| 4 | in the top drawer | 98% | 94% | 94% | 100% | 100% | 96% | 94% |
| 5 | on the ramekin | 98% | 64% | 58% | 98% | 98% | 98% | 94% |
| 6 | next to the cookie box | 98% | 98% | 96% | 100% | 100% | 100% | 100% |
| 7 | on the stove | 100% | 92% | 84% | 100% | 98% | 94% | 94% |
| 8 | next to the plate | 100% | 94% | 90% | 100% | 98% | 98% | 100% |
| 9 | on the wooden cabinet | 96% | 86% | 66% | 98% | 96% | 98% | 98% |
| **Overall** | | **98.4%** | **92.2%** | **87.0%** | **99.2%** | **99.0%** | **97.2%** | **97.8%** |

| id | target | Region | Region v2 | Region v3 | Landmark ("next to") | "close to" | "beside" | "near" | "adjacent to" |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 100% | 100% | 100% | — | — | — | — | — |
| 1 | next to the ramekin | 76% | 66% | 70% | — | — | — | — | — |
| 3 | on the cookie box | 96% | 100% | 98% | 98% | 100% | 98% | 98% | 100% |
| 5 | on the ramekin | 84% | 86% | 80% | 96% | 96% | 98% | 96% | 100% |
| 6 | next to the cookie box | 98% | 100% | 100% | — | — | — | — | — |
| 7 | on the stove | 26% | 28% | 16% | 68% | 88% | 92% | 94% | 92% |
| 8 | next to the plate | 100% | 100% | 100% | — | — | — | — | — |
| 9 | on the wooden cabinet | 24% | 40% | 34% | 86% | 100% | 100% | 96% | 96% |
| **Overall** | | **75.5%** | **77.5%** | **74.8%** | **87.0%** | **96.0%** | **97.0%** | **96.0%** | **97.0%** |

### 9.3 — Per-task: scene conditions (default prompt unless noted)

| id | target | `irrelevant` | `semantic` | `landmark` | `landmark` + hardneg prompt | `drawer_open` |
|--:|---|--:|--:|--:|--:|--:|
| 0 | between the plate and the ramekin | 98% | 100% | 22% | 18% | 38% |
| 1 | next to the ramekin | 16% | 32% | 42% | 2% | 50% |
| 2 | table center | 100% | 100% | 100% | 100% | 100% |
| 3 | on the cookie box | 84% | 98% | 96% | 100% | 44% |
| 4 | in the top drawer | 100% | 98% | 84% | 86% | 60% |
| 5 | on the ramekin | 16% | 98% | 74% | 80% | 36% |
| 6 | next to the cookie box | 96% | 96% | 84% | 80% | 0% |
| 7 | on the stove | 100% | 100% | 96% | 96% | 70% |
| 8 | next to the plate | 96% | 96% | 88% | 94% | 0% |
| 9 | on the wooden cabinet | 100% | 92% | 34% | 6% | 36% |
| **Overall** | | **80.6%** | **91.0%** | **72.0%** | **66.2%** | **43.4%** |

Suites: `irrelevant` = `libero_spatial_3bowl_front`, `semantic` = `libero_spatial_3bowl_semantic2`,
`landmark` = `libero_spatial_3bowl_hardneg`, `drawer_open` = `libero_spatial_3bowl_open`. The two 4a
gap-fill cells (`grounding/surface_landmark`, `grounding/region_surface`; task 0 only) are 50/50 each.

### 9.4 — Metrics (same formulas as the plan)

```
Split 1
  Negative Contrast Drop   = 98.4 − 87.0 = 11.4 pts      (OpenVLA: 47.2)
  Distractor Mention Drop  = 98.4 − 92.2 =  6.2 pts      (OpenVLA: 51.6)
  Negation-specific Drop   = 92.2 − 87.0 =  5.2 pts      (OpenVLA: −4.4)
  Length-only Drop         = 98.4 − 99.2 = −0.8 (infix), 98.4 − 99.0 = −0.6 (suffix)   (OpenVLA: 21.2 / 30.4)
  Content Drop (residual)  = 99.2 − 87.0 = 12.2 (negative), 99.0 − 92.2 = 6.8 (positive) (OpenVLA: 26.0 / 21.2)

Split 4
  Same-cue Paraphrase Drop = 98.4 − 97.2 = 1.2 (lexical), 98.4 − 97.8 = 0.6 (syntactic)  (OpenVLA: 11.6 / 1.8)
  Region-cue Drop, 8 tasks = 98.75 − 75.5 / 77.5 / 74.75 = 23.3 / 21.3 / 24.0 pts for region / v2 / v3   (OpenVLA: 67.0 / 64.0 / 65.3)
    landmark-family pool {0,1,6,8}: 99.5% → 93.5 / 91.5 / 92.5
    surface-family pool  {3,5,7,9}: 98.0% → 57.5 / 63.5 / 57.0
  Cue-type Drop (excess)   = SR(paraphrase_lexical, 8 tasks: 97.0%) − SR(region*) = 21.5 / 19.5 / 22.3 pts
  Landmark-cue Drop        = SR(default, 4 tasks: 98.0%) − 87.0 = 11.0 pts                (OpenVLA: 50.0)
  Novel-cue Drop           = 98.0 − 96.0 / 97.0 / 96.0 / 97.0 = 2.0 / 1.0 / 2.0 / 1.0 pts  (OpenVLA: 27.5 / 42.5 / 31.5 / 43.0)
  Familiarity Gap (robust) = mean of 4 novel synonyms (96.5%) − SR(target_cue_landmark: 87.0%) = +9.5 pts   (OpenVLA: +13.9)

Splits 2 / 3 (against pi05's own 2-bowl default, 98.4%; pi05 has no closed-drawer 3-bowl run)
  irrelevant −17.8, semantic −7.4, landmark −26.4, landmark + hardneg prompt −32.2, drawer_open −55.0
```

### 9.5 — What the pi05 numbers show

1. **Prompt robustness is far higher.** Content-free added text costs pi05 nothing (99.2% / 99.0%
   against 98.4%), and same-cue paraphrases cost about 1 pt. The same conditions cost OpenVLA 21–30
   pts (length) and 1.8–11.6 pts (paraphrase).
2. **Mentioning the distractor still costs something, and negation adds to it.** 6.2 pts for a bare
   mention and 11.4 pts with negation, concentrated on task 5 (64% / 58%) and task 9 (86% / 66%).
   Because the length controls are flat, this residual is content-specific for pi05.
3. **The region cue is the one prompt manipulation that clearly hurts pi05, and it is stable across
   three wordings** (75.5 / 77.5 / 74.8%). It is carried by tasks 7 and 9 (16–40%) and task 1
   (66–76%); the other five tasks stay at 80–100%.
4. **The familiar "next to X" cue is again worse than every novel proximity synonym** (87.0% against
   96–97% for all four), driven by task 7 (68%). This is the same direction as OpenVLA's 4c result,
   at a much smaller size.
5. **pi05 is not uniformly more robust: a third bowl hurts it more than it hurts OpenVLA on three of
   five scene conditions.** `irrelevant` 80.6% (OpenVLA 85.2%), `landmark` 72.0% (80.6%),
   `drawer_open` 43.4% (60.0%). The losses are sharply task-specific: tasks 1 and 5 drop to 16% in
   `irrelevant`, tasks 0 and 9 to 22% / 34% in `landmark`, and tasks 6 and 8 to 0% in `drawer_open`.
   No rollout videos have been reviewed for these, so the failure mode behind them is not established.
6. **The disambiguating prompt does not rescue the hard-negative scene for pi05 either**: 66.2% with
   it against 72.0% without, with task 1 going from 42% to 2% and task 9 from 34% to 6%.

Results: `results/*--pi05_libero--shard{0..3}of4.jsonl`.
