# RoboCasa365 results

One table per run, newest last. Every number traces to `results/robocasa/<run_id>/summary.csv` (small files, in the
repo), whose full run directory is backed up at `~/vla_results_backup/robocasa/<run_id>/` on the laptop and, once
synced, at HF `Qian0203/vla_ws-eval-results` under `robocasa/<run_id>/`. Launch details: `docs/robocasa/eval_log.md`.

Run ID: `YYYYMMDD-HHMM_robocasa_<model>_<stage>_<host>`. Smoke and screen runs are labelled and never enter the main
result tables (§ Main results below stays empty until a `gate`-passed `main` run exists).

## Main results

None yet.

## Gate runs

### `20261023-0927_robocasa_xr1_gate_berkeley2` — gate, **passed**

Xiaomi-Robotics-1 (HF `3a6d029`, bf16, sdpa), upstream client, published seeds, `berkeley-2` GPU 0. Rule: per task
within 14 pts and mean within 7 pts of the published results (Xiaomi-Robotics-1 `summary.json`).

| Task | n | SR | Published | Diff (pts) | Episode agreement |
|---|--:|--:|--:|--:|--:|
| NavigateKitchen | 50 | 88% | 92% | −4 | 0.88 |
| OpenStandMixerHead | 50 | 94% | 98% | −4 | 0.96 |
| PickPlaceCounterToCabinet | 50 | 88% | 86% | +2 | 0.90 |
| TurnOffStove | 50 | 58% | 60% | −2 | 0.78 |
| **Mean** | 200 | **82.0%** | **84.0%** | **−2.0** | |

Instructions and scene hashes reconstructed after the run (upstream client); no outcome fields. Sync: HF, repo,
laptop — all three. An earlier attempt, `20261023-0915_robocasa_xr1_gate_berkeley2`, aborted before any episode (port held
by a leftover process).

## Screens, smoke and exploratory runs (not results)

### `20261021-1636_robocasa_xr1_smoke_berkeley2` — smoke

Xiaomi-Robotics-1 (HF `3a6d029`, bf16, sdpa), upstream client, `berkeley-2` GPU 0. Seeds 15 and 16 (upstream seed rule
with `num_trials=2`), **not** the published seeds, so not comparable with the leaderboard.

| Task | Condition | n | Successes | SR |
|---|---|--:|--:|--:|
| NavigateKitchen | default | 2 | 1 | 50% |

Instruction and scene hash were reconstructed afterwards on the laptop (not logged by the upstream client); no
outcome fields. Sync status (2026-10-23): HF `robocasa/20261021-1636_robocasa_xr1_smoke_berkeley2/` (12 files incl. videos),
repo `results/robocasa/…`, laptop backup — all three.
