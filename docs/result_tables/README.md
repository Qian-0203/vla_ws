# Result tables, 2026-10-13

All benchmark conditions for ten checkpoints (our LoRA r32, official OpenVLA, QwenVLA, `pi05_libero`, `pi0_libero`,
OpenVLA-OFT, VLA-Adapter Spatial and Spatial-Pro, X-VLA, GR00T N1.5), built from the raw results JSONL. Analysis:
`docs/benchmark_split_result.md` §13; launch record: `docs/eval_log.md` (2026-10-12 → 2026-10-13).

| file | content |
|---|---|
| `full_table_2026-10-13.md` | condition x checkpoint success rate (mean of per-task rates; rollouts in brackets) |
| `task_condition_sr_2026-10-13.csv` | per checkpoint / condition / task: successes, trials, rate |
| `outcomes_2026-10-13.csv` | which-bowl classes per checkpoint / condition / task (only runs with outcome logging) |

Built by `scripts/build_full_data_package.py`:

    python3 scripts/build_full_data_package.py --out OUT --max-steps 230 --exclude /vla_ws_lang2/ \
      --drop-condition landmark_exchange \
      --results-dir <raw_data_library>/results --results-dir "<campaign>/*/*/results"

Rules (they decide the numbers): a cell is the mean of per-task success rates; for one (checkpoint, condition) the
main run is used, else the re-run (`*`), else a which-bowl re-run; repeated (task, episode) rows keep the first; a
failed row with `num_steps <= 10` (a client whose server was stopped under it) is dropped and counted (0 here);
`--max-steps 230` counts a success only inside 230 environment steps (X-VLA's `default` ran at an 800-step horizon;
every success is inside 230 steps, so the 220-step protocol numbers are identical).

Things to know before citing:

- Single seed (7), 50 trials per task, one run per condition; noise bands in `benchmark_split_result.md` §0.
- Policies use different protocols (action chunking, cameras, absolute vs delta actions); compare each policy with
  its own `spatial/default`. `benchmark_split_result.md` §13 lists them.
- `landmark_exchange` is an exploratory probe and is left out of these tables (`--drop-condition`); its rollouts exist
  in the raw results.
- `--exclude /vla_ws_lang2/` leaves out a duplicate copy of two pi0 `no_location` shards whose servers were killed by
  another queue (42 lost-server rows); the complete clean copy (`vla_ws_lang`, 500 rows) is used.
- Re-runs (`*`): for 11 conditions the original runs of our LoRA checkpoint are lost; the 2026-10-11 re-run is used
  (six conditions match exactly per task, the others are within 3.2 points).
- Withdrawn: `target_cue_region`, `_v2`, `_v3` (Erratum E1). The corrected `_img` conditions exist only for the
  new checkpoints.
- `pi0_fast_libero` is not included: its gate was 16.0% with an unresolved integration problem.
- `no_location`, `target_swap` and `closed_drawer_control` have no QwenVLA run; `closed_drawer_control` exists only
  for our LoRA and QwenVLA.
