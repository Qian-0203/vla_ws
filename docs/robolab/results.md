# RoboLab results

One section per run, newest last. Numbers come from `results/robolab/<run_id>/summary.csv`, built from
RoboLab's `episode_results.jsonl` by `robolab_lang/make_run_record.py`. Launch details are in
[`eval_log.md`](eval_log.md). Only runs with stage `main` go into result tables. Smoke, gate, screen and
exploratory runs are listed below and labelled. They are checks, not findings.

SR is RoboLab's own success flag. The 95% CI is Wilson.

## Main results

None yet.

## Checks (smoke and gate), π0.5-DROID, `default` instruction, Isaac Sim 5.0 / Isaac Lab 2.2.0, RoboLab `b6cfcf6`

| Run ID | Stage | Task | n | SR | 95% CI | mean score | episodes with `WRONG_OBJECT_GRABBED` |
|---|---|---|---:|---:|---|---:|---:|
| `20261021-1551_robolab_none_smoke_nj21` | smoke (no policy, install check) | BananaInBowlTask | 1 | — | — | — | — |
| `20261021-1552_robolab_pi05_smoke_nj21` | smoke | BananaInBowlTask | 1 | 0/1 | — | 0.00 | 0 |
| `20261021-1555_robolab_pi05_smoke_nj21` | smoke | BananaInBowlTask | 10 | 9/10 | 60–98% | 0.90 | 0 |
| `20261021-1600_robolab_pi05_gate_nj21` | gate A | BananaInBowlTask | 50 | 46/50 (92%) | 81–97% | 0.92 | 1 |
| `20261021-1600_robolab_pi05_gate_nj21` | gate A | RubiksCubeAndBananaTask | 50 | 36/50 (72%) | 58–83% | 0.84 | 6 |

Gate A was set at ≥ 85% per task. BananaInBowl passed. RubiksCubeAndBanana was accepted by the user on
2026-10-23 because its failures are grasp failures (see `eval_log.md`). The leaderboard publishes no
per-task numbers, so neither row can be compared with the board.

### Outcome fields: what they mean and their limits

Each row of `episodes.jsonl` carries, from RoboLab's HDF5:

- `lifted_objects` / `first_lifted`: objects whose root rose more than 3 cm.
- `in_container`: object centroid inside a container's bbox footprint at the last step.
- `pair_*`: mirror-pair labels for the native pair tasks.

`in_container` is looser than RoboLab's success predicate, which also requires contact and an open
gripper. In the gate run, every successful episode has its objects in the bowl. Eight failed episodes
also do: the object ended over the bowl but was still held, or rested on the rim at timeout.
