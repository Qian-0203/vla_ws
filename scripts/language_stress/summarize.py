#!/usr/bin/env python3
"""Summarize language-stress screening runs: success rate per candidate x policy, and the change from
each policy's own same-budget `screen_default`.

  python3 scripts/language_stress/summarize.py [--results-dir DIR ...] NOTE [NOTE ...]

NOTE is a screening run note (`<RUN_NOTE>--screen`, as written by screen.sh). Several --results-dir
may be given (e.g. the Docker runs' directory and the QwenVLA host runner's). `target_swap` counts a
rollout as a success when the *named* (distractor) bowl reaches the plate; every other candidate
scores the original target. Rates are the mean of per-task rates, as in aggregate_results.py.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAND_DIR = os.path.join(ROOT, "screening", "language_stress", "candidates")


def load(results_dirs, note):
    """{condition: {task_id: [success, ...]}} for one screening run note (all shards)."""
    out = defaultdict(lambda: defaultdict(list))
    for d in results_dirs:
        for path in glob.glob(os.path.join(d, f"*--{note}*.jsonl")):
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    out[r["condition"]][r["task_id"]].append(bool(r["success"]))
    return out


def rate(per_task):
    rates = [sum(v) / len(v) for v in per_task.values() if v]
    n = sum(len(v) for v in per_task.values())
    if not rates:
        return None, 0, None
    p = sum(rates) / len(rates)
    return 100 * p, n, 100 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", action="append", default=None)
    ap.add_argument("notes", nargs="+")
    a = ap.parse_args()
    dirs = a.results_dir or [os.path.join(ROOT, "openvla", "experiments", "logs", "results")]
    order = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(CAND_DIR, "*.json")))]
    order = ["screen_default"] + [c for c in order if c != "screen_default"]
    data = {n: load(dirs, n) for n in a.notes}

    print("| candidate | " + " | ".join(a.notes) + " |")
    print("|---|" + "---:|" * len(a.notes))
    for cond in order:
        cells = []
        for n in a.notes:
            sr, k, se = rate(data[n].get(cond, {}))
            base, _, _ = rate(data[n].get("screen_default", {}))
            if sr is None:
                cells.append("—")
            elif cond == "screen_default" or base is None:
                cells.append(f"{sr:.0f}% (n={k})")
            else:
                cells.append(f"{sr:.0f}% ({sr - base:+.0f}, n={k}, ±{se:.0f})")
        print(f"| {cond} | " + " | ".join(cells) + " |")
    print("\nCells: success rate (change from that policy's screen_default, rollouts, ±1 SE). "
          "target_swap scores the named distractor bowl.")


if __name__ == "__main__":
    main()
