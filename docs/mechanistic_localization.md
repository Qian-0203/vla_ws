# OpenVLA · LIBERO-Spatial — Mechanistic Localization Probe

**What this file is.** The full record of the mechanistic-localization probe: the question, the
design, every run with its results, and the current conclusion. It is a diagnostic, not a benchmark
split. It has no `eval_registry.SPLITS` entry, and its script
(`openvla/experiments/robot/libero/probe_mechanistic_localization.py`) is run directly, not via
`run_eval.sh --split`.

**What it is not.** Launch details (hardware, exact commands, artifact paths per run) live in
`eval_log.md`, in the "Mechanistic localization probe" entries from 2026-09-04 to 2026-09-10. The
behavioral results that motivated this probe stay in `benchmark_split_result.md`: §8.5 (bowl-attraction
probe) and §8.6 (synthesis). Any §-reference below 8.9 (§8.5, §8.6, §2, ...) points into that file.
Sections 8.9–8.15 keep the numbers they had when they lived in `benchmark_split_result.md` §8, so
existing references to them elsewhere (in particular in the append-only `eval_log.md`) still resolve.

**Update rule.** After every new probe run, add a §8.x section at the end, update §2 below if the
conclusion changed, and append an `eval_log.md` entry.

## 1. Question

The benchmark splits show *that* prompt deviations from the fine-tuning template collapse success, and
the bowl-attraction probe (`benchmark_split_result.md` §8.5) shows *what the arm does* when it fails:
most often it never commits to any bowl. Neither says *where in the network* that failure originates.
The vision encoder, the language projector, or the action-token head could each independently produce
"never commits to a target". This is gap 3 in `benchmark_split_plan.md` §11.3.

## 2. Current conclusion (as of 2026-09-10, after §8.15)

Open, but substantially narrowed.

- **Real, replicated effect: where the network looks and how fast it resolves.** Measured in a fixed
  early window (steps 0–9, n=50 episodes, task 5), `target_cue_landmark` differs from `default` on
  vision-attention mass (p=0.0009, d=0.62) and on resolution layer (p=0.019, d=0.45) (§8.14). The
  window is used because whole-episode averages turned out to track episode length/outcome rather
  than condition content (§8.13). That early-window design removes the confound by construction.
- **But not a clean story.** `target_cue_proximity_novel` also diverges from `default`, in the opposite
  direction (vision-attention d=−0.42). Resolution layer's direction also flips between whole-episode
  and early-window measurement (§8.14).
- **Null: how decisively wrong it is.** Final-layer logit margin/entropy, alone or correlated with
  per-step progress toward the target, shows no condition-level difference (§8.15). The "confident
  misexecution" reading suggested by §8.14 is not supported.
- **Not yet achieved:** attributing the failure to a specific layer or module.

## 3. Design

| Field | Description |
|---|---|
| Goal | Localize the "never commits to a target" failure (`benchmark_split_result.md` §8.5) inside the network |
| Method | Instrumented real rollouts, two passes per step (§8.10). Pass 1 is plain `generate()`, identical to `predict_action()`, and chooses the executed action. Pass 2 is a teacher-forced forward pass on those exact tokens with `output_attentions=True`, used only to read the diagnostics |
| Diagnostics | **Resolution layer:** the earliest LLM layer (0–32) whose logit-lens argmax matches the final token (§8.9). **Vision-attention mass:** the share of attention on the 256 vision patches (§8.9). **Final-layer margin/entropy** (§8.15). **Per-step distance** from the end-effector to each bowl (`dist_by_bowl`, §8.11) |
| Conditions | `default` vs. `negative_contrast` (§8.9–§8.12), then `default` vs. `target_cue_landmark` vs. `target_cue_proximity_novel` (§8.13–§8.15). All reuse existing benchmark prompt conditions |
| Task | Task 5 (*on the ramekin*): 92–94% success under `default` vs. 2–4% under `negative_contrast` in the real eval |
| Episodes | 15/condition over whole episodes (§8.9–§8.13); 50/condition truncated to 10 instrumented steps (`--max_env_steps_to_instrument 10`, §8.14–§8.15) |
| Statistical unit | The episode, not the token (§8.12). Tests: Mann–Whitney U with tie correction and Cohen's d |
| Precision | bf16 only. 4-bit perturbs exactly the logits being measured (§8.15) |
| Code | `probe_mechanistic_localization.py` in the `openvla` fork, including the §8.15 confidence diagnostic (PR [#1](https://github.com/Qian-0203/openvla/pull/1), merged to `main` 2026-09-29) |

## 4. Methodological lessons (apply to any future probe on this checkpoint)

1. **A bare `output_attentions=True` is not a passive side-channel.** It silently swaps `sdpa` for
   `eager` attention, and on this bf16 checkpoint that flips the argmax on many continuous action dims.
   You end up diagnosing a different policy from the one being evaluated. Use the two-pass,
   teacher-forced structure (§8.10).
2. **Reproduce the eval's preprocessing exactly.** Missing center-crop dropped both conditions to
   ~0% success and erased the contrast the diagnostics need (§8.9 → §8.10).
3. **The episode is the independent unit.** Steps within a rollout are autocorrelated, and token-level
   pooling hid a large effect (§8.12).
4. **Whole-episode averages are confounded by outcome.** Failing episodes run to the step cap, so use
   a fixed early window (§8.13 → §8.14).
5. **Instrument in the same rollout you analyze.** Separate process launches don't reproduce
   trajectories, even with a matching seed and greedy decoding (§8.11).

## 5. Run log

### 8.9 Mechanistic localization — logit lens + attention mass, first pass (run 2026-09-04, written up 2026-09-06)

**Question (§8.6 gap 3).** §8.5 shows *what the arm does*, not *where in the network* "never commits"
originates — vision encoder, language projector, or action-token head could each independently produce
it.

**Method.** `probe_mechanistic_localization.py` instruments real rollouts via `vla.generate(...,
output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)`, computing two
diagnostics per action-token prediction: (1) **resolution layer** — the earliest LLM layer (0-32) whose
logit-lens argmax already matches the final generated token; (2) **vision-attention mass** — fraction
of attention (mean over heads) landing on the 256-patch vision span. Task 5, `default` vs.
`negative_contrast`, 15 full-length episodes each.

**Disclosed confound.** `center_crop` was accepted as a flag but never applied — 0/15 succeeded in
*both* conditions (vs. the real eval's 92-94%/2-4%), so this run likely never contained the
commits-vs-never-commits contrast the diagnostics need to see.

**Result — no condition-level difference on either diagnostic** (mean resolution-layer frac 0.843 vs.
0.856; vision-attn last-layer 0.116 vs. 0.101 — well under half a token-level stdev apart). Gripper
resolves markedly earlier than the 6 continuous dims in both conditions alike; that pattern is
consistent across conditions, the *level* isn't.

**Reading.** A genuine null on these two diagnostics, but not conclusive: possibly washed out by
per-token/per-condition averaging, possibly hidden by the missing-center-crop confound collapsing both
conditions into a similar bad regime, and n=1 task. See §8.10 for the fix.

Artifacts: `probe_mechanistic_localization/libero_spatial--t5--fullcompare1--2026_09_04-15_09_35.jsonl`.
Code: `experiments/robot/libero/probe_mechanistic_localization.py`.

### 8.10 Fixed: center-crop + teacher-forced diagnostics (2026-09-06)

**Fix #1 (center-crop) — necessary but not sufficient.** Added, `default` success rose only to
1/15 (6.7%) — a second, independent problem remained.

**Fix #2 — root cause: `output_attentions=True` changes which action gets executed, not just what's
observed.** `sdpa` (the model's normal attention kernel) doesn't support returning attention weights,
so HF silently falls back to numerically-different `eager` whenever diagnostics are requested. A
controlled single-frame test found this flips the argmax on **4 of 7 action dims on the very first
prediction of a fresh episode** — this bf16 checkpoint's continuous-dim decisions often have a small
top-1/runner-up logit margin, and `sdpa` vs. `eager`'s different floating-point summation order is
enough to flip it. Every §8.9 run had therefore been diagnosing a policy that wasn't the one actually
evaluated, on top of the center-crop bug.

**Redesign — two-pass, teacher-forced.** Pass 1: plain `generate()` (matches `predict_action()`
exactly) decides the real executed action. Pass 2: a separate, non-incremental forward pass,
teacher-forced on those exact tokens, `output_attentions=True` purely to read diagnostics — `eager`'s
own argmax is recorded but never used to act. Mathematically equivalent hidden-states/attentions to the
old incremental loop, just computed in one shot.

**Validation — success now matches the real eval:** `default` 14/15 (93.3%) vs. real 92-94%;
`negative_contrast` 0/15 (0%) vs. real 2-4%.

**Result on now-valid data — still no condition-level difference:** resolution-layer frac 0.8467 vs.
0.8477; vision-attn last-layer 0.1163 vs. 0.0964. Same qualitative null as §8.9, but now unconfounded —
93 points of real success separate the conditions and these two aggregate diagnostics still don't move.

**A genuinely new finding: the sdpa/eager mismatch rate is large and tracks resolution-layer margin.**
The 6 continuous action dims (late-resolving, ~0.83-0.90 frac) disagree between kernels on 12-48% of
predictions; the gripper dim (early-resolving, ~0.66 frac) agrees ~88-90% of the time. Late resolution
*is* a small decision margin, and a small margin is what makes a dimension sensitive to this kind of
numerical perturbation — holds equally in both conditions, so it's a property of the checkpoint's
general calibration, not something the failing condition induces.

**Reading.** Gap 3 now has a real, unconfounded null on both diagnostics — most likely explanation:
they're coarse, per-token, within-condition averages, and "never commits" may be a per-episode
phenomenon that averaging washes out (tested next, §8.11). Any future probe on this checkpoint needing
both real behavior *and* introspection needs this two-pass, teacher-forced structure — a bare
`output_attentions=True` call is not a passive side-channel on this model at bf16.

Artifacts: `probe_mechanistic_localization/libero_spatial--t5--tf_full--2026_09_06-10_14_08.jsonl`
(validated run). Code: same file, both fixes documented inline.

### 8.11 Regrouping by §8.5's approach behavior instead of prompt condition (2026-09-06)

**Motivation.** §8.9/§8.10's shared caveat: pooling by *condition* could wash out a per-episode effect.
§8.5's per-episode label (`first_bowl_approached`: target/distractor/neither) tests that directly.

**A separately-launched re-run of `probe_bowl_attraction.py` was not joinable.** Two process launches
of nominally identical rollouts (same seed, same greedy decoding) diverged after a handful of steps —
ordinary GPU run-to-run non-determinism (unseeded cuDNN/cuBLAS kernel selection), the same family of
issue as the sdpa/eager finding, but between two runs of the *identical* code path. **Consequence for
the whole project:** per-episode data from two separately-launched rollouts can't be assumed to match
beyond the first few steps, even with matching seeds — any future cross-script join needs single-run
instrumentation.

**Fix.** Added the same bowl-distance bookkeeping directly into `probe_mechanistic_localization.py`'s
own rollout loop, so behavior label and diagnostics come from one trajectory by construction. Re-run
reproduced §8.10's outcomes exactly, confirming this script's own rollouts *are* reproducible run-to-run.

**First pooled regroup looked promising, then turned out to be a confound:** pooling across *both*
conditions by behavior, `target_first` (0.1137 vision-attn) looked higher than `distractor_first`/
`neither` (0.0956/0.0980) — but `target_first` is 15/17 `default` episodes and only 2/17
`negative_contrast`, so this mostly just restates the known *condition*-level gap. Confirmed directly:
`default`'s `target_first` episodes average 0.1162; `negative_contrast`'s 2 `target_first` episodes
average only 0.0946 — same label, wildly different level, tracking condition not behavior.

**The valid test — hold condition fixed, vary only behavior, within `negative_contrast`:**

| Behavior | Episodes | Vision-attn, last layer | Resolution layer (frac) |
|---|--:|--:|--:|
| `target_first` | 2 | 0.0946 | 0.859 |
| `distractor_first` | 7 | 0.0956 | 0.836 |
| `neither` | 6 | 0.0980 | 0.858 |

Vision-attention indistinguishable across groups. Resolution-layer shows a small, suggestive gap
(`distractor_first` earlier-resolving) but `target_first` has only 2 episodes — not enough to treat as
a finding.

**Reading.** The null survives the exact test §8.9/§8.10 recommended, once done properly (a confound
caught and corrected along the way). Gap 3 has now been tested at both the condition level and the
per-episode-behavior level — neither shows a reliable difference on these two diagnostics, on this one
task. Side note: 2/15 `negative_contrast` episodes reached the target first and still failed — the same
"approached but still failed" pattern as task 3 (§8.5/§8.6), here at a lower rate (13%).

Artifacts: `probe_mechanistic_localization/libero_spatial--t5--dist_full--2026_09_06-11_40_47.jsonl`.
Code: same file (now committed in `openvla`, `0bd16d9`/`360f56e`).

### 8.12 Episode-level reanalysis: vision-attention diagnostic isn't actually null (2026-09-08)

**Motivation.** §8.10/§8.11 compared each condition's mean against the pooled *per-token* population
stdev — but steps within a 15-episode, up-to-220-step rollout are highly autocorrelated; the true
independent unit is the episode (n=15/condition). Re-analyzed the existing `dist_full` JSONL at the
correct level, purely by reanalysis (no new rollouts), plus two more angles.

**Result 1 — vision-attention is a large, highly significant condition-level effect at the episode level; resolution-layer is not:**

| Diagnostic (episode-level mean, cont. dims) | `default` (n=15) | `negative_contrast` (n=15) | p | Cohen's d |
|---|--:|--:|--:|--:|
| resolution_layer_frac | 0.8752 | 0.8796 | 0.12 | -0.38 (null) |
| vision_attn_last_layer | 0.1140 | 0.0951 | **<0.0001** | **1.88** |

Episode-level variance is far tighter than token-level variance (each mean already averages 88-220
autocorrelated steps), so the same raw gap that looked "under half a token-level stdev" is a huge,
highly significant effect by the correct test — present from steps 0-9 already, not just once the arm
gives up.

**This is very likely a prompt-length artifact, not a new mechanistic signal.** Attention is
softmax-normalized over the whole sequence; `negative_contrast`'s instruction has a whole extra clause
(~24 vs. ~15 words), so more text tokens compete for the same softmax mass regardless of any
target-selection mechanism. `target_cue_landmark`/`target_cue_proximity_novel` are word-for-word the
same length as `default` and had never been run through this probe — the direct test, launched next
(§8.13).

**Result 2 — phase breakdown, redone on valid data:** resolution-layer's early-phase-only (steps 0-9)
comparison shows a small gap (p=0.034) that disappears whole-episode (p=0.12) — one borderline test,
not leaned on without replication.

**Result 3 — distance/attention time-course join: inconclusive.** Per-episode correlation between
distance-to-target-bowl and vision-attention is noisy and inconsistent within every behavioral group,
no clean sign in any — confirms rather than resolves the existing "underpowered" caveat.

Artifacts: pure re-analysis of §8.11's `dist_full` JSONL; ad hoc stdlib-only analysis script (not
checked into either repo).

### 8.13 Length-matched rerun: length-dilution ruled out, but both diagnostics track episode length/outcome generally (2026-09-08)

**Method.** Same probe/task/battery size, `--conditions target_cue_landmark,target_cue_proximity_novel`
(both word-for-word `default`'s length). Success: `target_cue_landmark` 1/15 (6.7%, vs. real 0/10);
`target_cue_proximity_novel` 10/15 (66.7%, vs. real 5/10) — both consistent with small-n noise.

**Result A — length-dilution ruled out as a sufficient explanation.** `target_cue_landmark`, matched
length to `default`, still shows a significant whole-episode vision-attention drop *and* (unlike
`negative_contrast`) a significant resolution-layer shift:

| vs. `default` (episode-level, cont. dims) | vision_attn_last_layer | resolution_layer_frac |
|---|---|---|
| `negative_contrast` (0/15 success, +12 words) | p<0.0001, d=1.88 | p=0.12 (null) |
| `target_cue_landmark` (1/15 success, same length) | **p=0.0023, d=1.15** | **p=0.0095, d=-0.62** |
| `target_cue_proximity_novel` (10/15 success, same length) | p=0.69 (null) | p=0.25 (null) |

A length-matched condition should never show the gap if it's pure dilution — but `target_cue_landmark`
does, on both diagnostics.

**Result B — but pooled across all 4 conditions, both diagnostics track episode length/success
generally, not condition-specific content.** `corr(episode_length, vision_attn) = -0.675`; pooled
success (n=25) vision-attn mean 0.1147 vs. failure (n=35) 0.1029 (p<0.0001); resolution-layer the same
pattern, weaker. Because a failed episode runs to the 220-step cap while a success ends early, whole-
episode averages are structurally weighted toward whichever condition/episode fails — success and
episode length are mechanically coupled by the closed-loop termination rule itself, independent of
*why* a given prompt causes failure.

**Result C — the one test immune to that confound by construction still shows a small,
condition-specific signal, but borderline.** Restricted to a fixed early window (steps 0-9, present
identically regardless of outcome): `target_cue_landmark` still differs from `default` on
vision-attention (p=0.029) while `target_cue_proximity_novel` does not (p=0.19) — suggestive of finding
18 (`target_cue_landmark` reuses "next to X," a phrase bound to *other* tasks; `target_cue_proximity_novel`'s
"close to X" has no such binding), but one borderline p-value among many tests on n=15 — a hint, not a
finding.

**Reading.** These whole-episode diagnostics are substantially proxies for episode length/success,
itself determined by the closed-loop termination rule, not a clean window into *why* the policy failed.
Result C's early-window hint is the clearest remaining lead, needing either more episodes or (better) a
design that avoids the length/outcome coupling by construction — pursued next (§8.14).

Artifacts: `probe_mechanistic_localization/libero_spatial--t5--lenmatch_full--2026_09_08-04_11_24.jsonl`
(15 episodes × 2 conditions, full 220-step cap).

### 8.14 Early-window replication at n=50: signal confirmed, but more complicated (2026-09-08)

**Method.** Instead of post hoc restriction to steps 0-9 of a variable-length rollout, cap the rollout
there directly (`--max_env_steps_to_instrument 10`) — every episode is now exactly 10 steps, making the
length/outcome coupling structurally impossible rather than merely avoided, and >20x cheaper, so all 50
pre-sampled init states/task could be used. Ran `default`, `target_cue_landmark`,
`target_cue_proximity_novel` (task 5) at n=50.

**Result — both comparisons are now non-borderline, but more complicated than §8.13 suggested:**

| vs. `default` (episode-level, cont. dims, n=50) | vision_attn_last_layer | resolution_layer_frac |
|---|---|---|
| `target_cue_landmark` | **p=0.0009, d=0.62** | **p=0.019, d=0.45** |
| `target_cue_proximity_novel` | p=0.039, d=-0.42 | p=0.19 (null) |

1. **§8.13's headline result replicates and strengthens** — `target_cue_landmark`'s early-window
   vision-attention deficit goes from borderline (p=0.029, n=15) to clearly significant (p=0.0009,
   n=50, d=0.62) under a design that rules out the length/outcome confound by construction.
2. **But `target_cue_proximity_novel` is not the clean null control §8.13 expected** — at n=50 it's
   nominally significant (p=0.039), in the *opposite* direction (higher, not lower, vision-attention).
   Both off-template conditions differ from `default` in the early window, with different signs and
   magnitudes; `target_cue_landmark`'s effect is the more robust of the two.
3. **Resolution-layer's direction flips between whole-episode (§8.13: resolves *later*, d=-0.62) and
   early-window (resolves *earlier*, d=+0.45)** — since the early-window design can't be contaminated
   by length coupling, this means the two measurements pick up genuinely different phenomena, not a
   diluted/concentrated version of one. Tentative reading: `target_cue_landmark` reuses "next to X" (a
   phrase bound to *other* tasks) — an earlier, more decisive resolution could reflect the policy
   confidently committing to that other task's motion pattern (**confident misexecution**) rather than
   hesitating, with the later whole-episode average instead reflecting the long failure tail once that
   misexecution doesn't complete the task. Plausible, not confirmed — no diagnostic here directly
   measures "is the policy executing a different task's motion."

**Net effect on gap 3.** Still no diagnostic here cleanly, monotonically separates "will fail" from
"will succeed" by condition — but this upgrades the early-window signal from an unreplicated hint to a
real, replicated, more complicated pattern, and establishes a cheap, confound-free method (truncate to
the window of interest) for testing it further. Next step: a diagnostic that can distinguish
"confidently executing the wrong motion" from "confidently executing the right one," which
resolution-layer/attention-mass alone can't do — pursued in §8.15.

Artifacts: `probe_mechanistic_localization/libero_spatial--t5--earlywin_n50--2026_09_08-04_46_21.jsonl`
(50 episodes × 3 conditions × 10 steps, 1,500 records).

### 8.15 Confidence diagnostic: distinguishing confident-correct from confident-wrong motion (implemented 2026-09-09, run 2026-09-10)

**Motivation.** §8.14's flagged next step: resolution-layer and vision-attention-share are both blind
to whether the eventually-chosen action token was actually a *good* choice.

**Method.** Two cheap additions to `get_action_with_diagnostics`'s existing logit-lens pass, at zero
extra forward-pass cost: `final_layer_margin` (top1-top2 logit gap — how decisively the model
committed) and `final_layer_entropy` (full-vocab softmax entropy). Crossed against the already-recorded
`dist_by_bowl` trajectory (no rerun needed for that half — it's been logged since §8.11), this lets
analysis bucket steps by (confidence) × (movement toward target vs. distractor), which neither prior
diagnostic could do.

**Status.** Implemented 2026-09-09, run 2026-09-10 in bf16. 4-bit was ruled out because
`final_layer_margin`/`final_layer_entropy` are first-order functions of the logit values themselves,
exactly what quantization perturbs, and this checkpoint is already sensitive to far smaller numerical
perturbations (§8.10). Server setup and the `spatial/default` validation run that preceded it:
`eval_log.md` 2026-09-10 entries.

**Method for this run.** Same cheap, confound-free early-window design as §8.14
(`--max_env_steps_to_instrument 10 --num_trials 50`, task 5, `default`/`target_cue_landmark`/
`target_cue_proximity_novel`) — every episode truncated to exactly 10 instrumented steps, so
episode-length/outcome coupling (§8.13 Result B) is structurally impossible, matching the design that
already found a real vision-attention/resolution-layer signal at this n. Two analyses, both at the
episode level (matching §8.12's correction — steps within an episode are autocorrelated, episode is
the valid independent unit): (1) episode-level mean `final_layer_margin`/`final_layer_entropy` over the
6 continuous action dims, replicating §8.14's table format with the new diagnostic in place of
vision-attention/resolution-layer; (2) the test this diagnostic was actually built for — per-episode
Pearson correlation between each step's mean continuous-dim `final_layer_margin` and that step's
resulting change in distance-to-target-bowl (negative = the action moved the arm closer; this is the
direct operationalization of "was a confident step also a *correct* one").

**Result — clean null on both tests; no support for the "confident misexecution" hypothesis.**

| vs. `default` (episode-level, cont. dims, n=50) | `final_layer_margin` | `final_layer_entropy` |
|---|---|---|
| `target_cue_landmark` | p=0.15, d=0.31 (null) | p=0.15, d=-0.25 (null) |
| `target_cue_proximity_novel` | p=0.22, d=0.28 (null) | p=0.65, d=-0.00 (null) |

Raw decision confidence in this window doesn't differ by condition — unlike vision-attention/
resolution-layer at the same n (§8.14, both significant), margin/entropy show no condition-level
effect here.

| Per-episode Pearson(margin, Δdist-to-target) | mean r | % episodes r<0 (confidence tracks closing in) |
|---|--:|--:|
| `default` | −0.108 | 58.0% |
| `target_cue_landmark` | −0.070 | 52.0% |
| `target_cue_proximity_novel` | −0.218 | 76.0% |

vs. `default`: `target_cue_landmark` p=0.54, d=0.086 (null); `target_cue_proximity_novel` p=0.29,
d=-0.238 (null). All three conditions trend the same direction (confidence modestly tracks closing in
on the target, as it should under normal operation) with no significant condition-level difference —
and where there's any numeric separation, it runs *opposite* the "confident misexecution" hypothesis:
`target_cue_landmark` (the condition §8.14 flagged as possibly confidently executing the wrong motion)
has the *weakest* confidence-tracks-correctness relationship of the three, not a reversed one, and the
gap from `default` is tiny and non-significant.

**Reading.** This diagnostic does what §8.14 asked for — it can, in principle, tell confident-correct
from confident-wrong motion apart — and finds no evidence for it here: neither raw confidence nor the
confidence/correctness relationship differs by condition in this window, at n=50. Combined with §8.14's
real vision-attention/resolution-layer effects, the fullest current picture is that `target_cue_landmark`'s
early-window difference is about *where the network looks and how quickly it resolves*, not about
*whether it's more decisively wrong* — the "confident misexecution" reading from §8.14 was a plausible
story, not a confirmed one, and this is the first diagnostic built to test it directly rather than by
inference; it comes back null. Caveat: each episode's own correlation is estimated from only 9
within-episode step-pairs (sd≈0.41-0.48 across episodes), so this is a real but noisy per-episode
signal — a longer instrumented window (trading off against the length/outcome coupling §8.13 warned
about) or more episodes would sharpen it further, not yet done.

Artifacts: `openvla/experiments/logs/probe_mechanistic_localization/libero_spatial--t5--confdiag_earlywin_n50--2026_09_10-09_52_30.jsonl`
(50 episodes × 3 conditions × 10 steps, 1,500 records; server-local, gitignored); smoketest jsonl
deleted after passing. Code: `probe_mechanistic_localization.py`, committed in the `openvla` fork (PR
[#1](https://github.com/Qian-0203/openvla/pull/1), merged to `main` 2026-09-29; before that it was
fast-forwarded only into the server's local `main`, on 2026-09-10). Analysis: ad hoc stdlib-only Python (hand-implemented Mann-Whitney
U with tie correction, Pearson correlation, Cohen's d — same convention as §8.12), not checked into
either repo.
