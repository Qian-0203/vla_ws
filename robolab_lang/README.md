# robolab_lang

Language stress tests on NVIDIA RoboLab, using RoboLab as an unmodified dependency. Design and status:
[`docs/robolab/plan.md`](../docs/robolab/plan.md); launches: [`docs/robolab/eval_log.md`](../docs/robolab/eval_log.md).

| File | Purpose |
|---|---|
| `frame_check.py` | Project scene objects into the policy camera (`over_shoulder_left_camera`, 224×224 letterboxed) and flag left/right words whose robot-frame and image-frame meanings differ. Run on every scene before writing a spatial instruction. CPU only. |
| `droid_vocab.py` | Count words / phrases in the DROID instruction corpus (RoboLab's `analysis/droid_dataset/tasks.jsonl`) to order lexical-ladder rungs. |
| `ladders.txt` | Draft rungs for `droid_vocab.py --ladder`. |

Both scripts need only Python 3 and numpy, e.g.

```bash
python3 robolab_lang/frame_check.py <RoboLab>/assets/scenes/two_bin.usda
```

```bash
python3 robolab_lang/droid_vocab.py --tasks <RoboLab>/analysis/droid_dataset/tasks.jsonl --ladder robolab_lang/ladders.txt
```
