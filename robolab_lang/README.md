# robolab_lang

Language stress tests on NVIDIA RoboLab, using RoboLab as an unmodified dependency. Design and status:
[`docs/robolab/plan.md`](../docs/robolab/plan.md); launches: [`docs/robolab/eval_log.md`](../docs/robolab/eval_log.md).

| File | Purpose |
|---|---|
| `frame_check.py` | Project scene objects into the policy camera (`over_shoulder_left_camera`, 224×224 letterboxed) and flag left/right words whose robot-frame and image-frame meanings differ. Run on every scene before writing a spatial instruction. CPU only. |
| `droid_vocab.py` | Count words / phrases in the DROID instruction corpus (RoboLab's `analysis/droid_dataset/tasks.jsonl`) to order lexical-ladder rungs. |
| `ladders.txt` | Draft rungs for `droid_vocab.py --ladder`. |
| `setup_robolab.sh`, `docker/Dockerfile.ffmpeg` | One-time install: RoboLab `b6cfcf6` with assets, `robolab:b6cfcf6(-ffmpeg)` images, `xuningy/openpi@aa64205`. |
| `serve_pi05.sh` | π0.5-DROID policy server. |
| `run_robolab.sh`, `run_pi0.py` | One RoboLab eval in the container; `run_pi0.py` is RoboLab's π0 client plus task loading from `tasks/`. |
| `run_option_a.sh` | Language pilot Option A (13 tasks × default/blank/no_target × 20 episodes); step-by-step guide in [`docs/robolab/run_option_a.md`](../docs/robolab/run_option_a.md). |
| `run_paper_tests.sh` | The ARR draft's tests (negation, switch, paraphrase, irrelevant content, distractor mention, spatial wording) in 3 shared scenes, 600 episodes, groups `mustard` / `cube` / `size`; guide and known gaps in [`docs/robolab/run_paper_tests.md`](../docs/robolab/run_paper_tests.md). |
| `tasks/` | Subclasses of official tasks that only add instruction keys. |
| `make_run_record.py`, `outcomes.py` | Per-run `episodes.jsonl` / `summary.csv` with which-object outcomes. |

Both scripts need only Python 3 and numpy, e.g.

```bash
python3 robolab_lang/frame_check.py <RoboLab>/assets/scenes/two_bin.usda
```

```bash
python3 robolab_lang/droid_vocab.py --tasks <RoboLab>/analysis/droid_dataset/tasks.jsonl --ladder robolab_lang/ladders.txt
```
