# RoboCasa365 results

One table per run, newest last. Every number traces to `results/robocasa/<run_id>/summary.csv` (small files, in the
repo), whose full run directory is backed up at `~/vla_results_backup/robocasa/<run_id>/` on the laptop and, once
synced, at HF `Qian0203/vla_ws-eval-results` under `robocasa/<run_id>/`. Launch details: `docs/robocasa/eval_log.md`.

Run ID: `YYYYMMDD-HHMM_robocasa_<model>_<stage>_<host>`. Smoke and screen runs are labelled and never enter the main
result tables (§ Main results below stays empty until a `gate`-passed `main` run exists).

## Main results

None yet.

## Gate runs

None yet.

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
