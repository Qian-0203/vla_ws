# Language-stress screening

A cheap pre-screen for new language conditions, run on every policy before any of them becomes a
benchmark split. Motivation: π0.5-LIBERO scores 87–99% on every current prompt condition except the
table-region cue (75–78%), so the benchmark has no language condition yet on which all three model
families fail. The candidates here look for one.

Candidates live outside the canonical registry (`openvla/experiments/robot/libero/eval_registry.py`).
A candidate that moves every policy by a large margin is promoted the normal way (instructions.py dict
+ `SPLITS` entry + docs, see `CLAUDE.md` "How to add a benchmark split") and then run at the full
50 trials/task.

## How it works

- `scripts/language_stress/make_candidates.py` writes one JSON per condition to `candidates/`:
  `condition`, `family`, `hypothesis`, `task_ids`, `swap_target`, and `instructions` keyed by LIBERO
  task name. Every candidate keeps the stock 2-bowl `libero_spatial` scene, init states and seed.
  Distractor locations come from the `negative_contrast` prompts in instructions.py.
- `run_libero_eval.py --instruction_file <json>` (openvla branch `language-stress-screen`) uses the
  JSON instead of a registry condition. `swap_target: true` retargets LIBERO's goal predicate from
  `akita_black_bowl_1` to `akita_black_bowl_2` at env construction, so success means the bowl the
  instruction names reaches the plate. Run metadata records the file and its sha256.
- `scripts/language_stress/screen.sh` runs all candidates on one policy at 5 trials/task (50 rollouts
  per condition, about ±7 pts). `summarize.py` prints a candidate × policy table with the change from
  each policy's own `screen_default`.

```bash
python3 scripts/language_stress/make_candidates.py
CHECKPOINT=/path/to/ckpt RUN_NOTE=openvla-ours GPUS=0,1,2,3 bash scripts/language_stress/screen.sh
python3 scripts/language_stress/summarize.py openvla-ours--screen pi05_libero--screen qwenvla--screen
```

## Candidates (v1)

| Condition | Family | What changes (task 7 example) | Why it might break all three |
|---|---|---|---|
| `screen_default` | baseline | "pick up the black bowl on the stove and place it on the plate" | Same-budget reference |
| `target_swap` | language following | "pick up the black bowl on top of the wooden cabinet …"; success = that bowl on the plate | Each task has its own layout, so a policy can succeed without reading the location. If it executes the layout's trained motion, it scores near 0 here. |
| `no_location` | language ablation | "pick up the black bowl and place it on the plate" | Diagnostic, not a failure test: a high score shows the layout alone identifies the target |
| `negation_only` | negation | "pick up the black bowl that is not on top of the wooden cabinet …" | The only location named is the distractor's; negation is weak in VLMs |
| `distractor_first` | reference order | "leave the black bowl on top of the wooden cabinet where it is; pick up the black bowl on the stove …" | Mentioning the distractor already costs π0.5 6–11 pts; naming it first may cost more |
| `coreference` | discourse | "there is a black bowl on the stove. pick it up and place it on the plate." | Two sentences and a pronoun; never seen in fine-tuning |
| `noun_synonym` | object names | "… on the cooktop …" (ramekin → small baking dish, cookie box → biscuit box, wooden cabinet → wooden cupboard) | Same-cue paraphrases changed relation words only; object names are what vision grounds |
| `translate_zh` | multilingual | "拿起炉子上的黑色碗，把它放到盘子上" | Backbones are multilingual, fine-tuning was English only |
| `translate_es` | multilingual | "toma el tazón negro que está sobre la estufa y colócalo en el plato" | Same, in Spanish |

`target_swap` and `no_location` read together: if `no_location` stays high and `target_swap` is near 0
for every policy, the policies largely ignore the location phrase whenever the layout determines the
task, and π0.5's prompt robustness is that, not better grounding.

## Candidates (v2, 2026-10-10, written after `negation_only`)

| Condition | Family | What changes (task 7 example) | Why |
|---|---|---|---|
| `negation_swap` | negation | "pick up the black bowl that is not on the stove …"; success = the *other* bowl | Negates the target's own location. A policy that drops "not" grabs the named bowl and scores near 0 |
| `negation_except` | negation | "pick up any black bowl except the one on top of the wooden cabinet …" | Exclusion without the word "not" |
| `negation_other` | negation | "pick up the other black bowl, not the one on top of the wooden cabinet, …" | `negative_contrast` minus the target's location |
| `self_correction` | discourse (repair) | "… on top of the wooden cabinet, sorry, I mean the one on the stove, …" | The distractor is named first, then retracted |
| `functional_landmark` | lexical (no landmark noun) | "… on the appliance you cook on …" (8 tasks) | No landmark noun left to match |
| `code_switch` | multilingual (mixed) | "pick up the 黑色碗 on the 炉子 and place it on the 盘子" | Between `screen_default` and `translate_zh` |

## Results so far (2026-10-10, 5 trials/task, ±7 pts)

| Candidate | OpenVLA, official checkpoint | π0.5-LIBERO |
|---|--:|--:|
| `screen_default` | 82% | 100% |
| `negation_only` | 2% (−80) | 44% (−56) |
| `translate_zh` | 0% (−82) | 78% (−22) |
| `noun_synonym` | 62% (−20) | 88% (−12) |
| `target_swap` | — | 60% (−40) |
| `no_location` | — | 82% (−18) |
| `negation_swap` | 0% (−82) | 0% (−100) |
| `negation_except` | 6% (−76) | 44% (−56) |
| `negation_other` | 8% (−74) | 46% (−54) |
| `self_correction` | 32% (−50) | 84% (−16) |
| `functional_landmark` | 30% (−52) | 87.5% (−12.5) |
| `code_switch` | 4% (−78) | 80% (−20) |

Results: `results/libero_spatial--<candidate>--{openvla-official-libero-spatial,pi05_libero}--screen.jsonl`
(HF mirror `Qian0203/vla_ws-eval-results`). Launch details: `docs/eval_log.md`, 2026-10-10.

**Promoted:** `negation_only` → `spatial/negation_only` (openvla `62487f7`). `make_candidates.py`
no longer writes it, because the eval refuses a screening condition that shares a registry name.

## Not in v1

- Comparative / ordinal references ("the bowl closer to the plate", "the second bowl from the left")
  need per-task geometry from the init states to stay truthful.
- Translations need a native-speaker check before any promotion.
- 3-bowl or other scenes: v1 is 2-bowl only because `retarget_goal_to_distractor` assumes exactly two
  black bowls.
