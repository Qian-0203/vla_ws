#!/usr/bin/env python3
"""Which-object outcomes for RoboLab mirror-pair episodes, from RoboLab's own HDF5 (plan §3.3).

For every episode in <output folder>/episode_results.jsonl whose env is listed in SPECS, reads
<env>/run_<run>.hdf5 / demo_<env_id> and labels what happened to the twin objects:

  destination pairs (one object, two identical containers): which container holds the object at the
  end (`delivered`), whether it was lifted at all;
  object pairs (two identical objects, one goes onto the other): which twin was lifted first and which
  twin ends stacked on the other.

Outcome classes (as in LIBERO's RolloutOutcomeTracker): target_delivered, wrong_delivered,
wrong_grasped (wrong twin lifted, nothing delivered), target_grasped_only, no_grasp.
`success` is RoboLab's own flag, copied unchanged.

Thresholds: lifted = root z more than 3 cm above its first recorded z; in container = object centroid
inside the container's axis-aligned bbox footprint (x, y) at the last step; stacked = top twin's
centroid within 6 cm (x, y) of the bottom twin's and at least 2 cm higher.

Usage (any Python with h5py + numpy, e.g. the RoboLab image's):
    python outcomes.py <output folder> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

import h5py
import numpy as np

LIFT = 0.03
STACK_XY = 0.06
STACK_DZ = 0.02

# env name -> spec. "dest": object + two containers; "stack": mover onto base, twins.
SPECS = {}
for prefix in ("", "Lang"):
    SPECS[f"{prefix}MustardInLeftBinTask"] = dict(kind="dest", obj="mustard",
                                                  target="grey_bin_left", other="grey_bin_right")
    SPECS[f"{prefix}MustardInRightBinTask"] = dict(kind="dest", obj="mustard",
                                                   target="grey_bin_right", other="grey_bin_left")
    SPECS[f"{prefix}BowlStackingLeftOnRightTask"] = dict(kind="stack", target="bowl_2", other="bowl_1")
    SPECS[f"{prefix}BowlStackingRightOnLeftTask"] = dict(kind="stack", target="bowl_1", other="bowl_2")

# Object tasks with one container: (requested object, other object, container) per task, or per
# (task, instruction key) where the sentence picks the object. "other" is the object that the paper's
# tests count as the violation: the excluded one under negation, the non-requested one under switch.
OBJECT_REQUESTS = {
    "LangLargerObjectRaisinBoxInBinTask": ("raisin_box", "butter", "grey_bin"),
    "LangSmallerObjectButterInBinTask": ("butter", "raisin_box", "grey_bin"),
}
for _key, _obj in (("switch_cube", "rubiks_cube"), ("switch_banana", "banana"),
                   ("negation_not_banana", "rubiks_cube"), ("negation_not_cube", "banana"),
                   ("distractor_cube", "rubiks_cube"), ("distractor_banana", "banana")):
    _other = "banana" if _obj == "rubiks_cube" else "rubiks_cube"
    OBJECT_REQUESTS[("LangRubiksCubeOrBananaTask", _key)] = (_obj, _other, "bowl")


def pos(g, name):
    return g[f"states/rigid_object/{name}/root_pose"][:, :3].astype(float)


def bbox_xy(g, name, t=-1):
    c = g[f"bbox/bbox_mm/{name}"][t].astype(float) / 1000.0
    return c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()


def first_lift(p):
    up = np.where(p[:, 2] - p[0, 2] > LIFT)[0]
    return int(up[0]) if len(up) else None


def label_dest(g, s):
    p = pos(g, s["obj"])
    lifted = first_lift(p)
    end = p[-1]
    delivered = None
    for c in (s["target"], s["other"]):
        x0, x1, y0, y1 = bbox_xy(g, c)
        if x0 <= end[0] <= x1 and y0 <= end[1] <= y1:
            delivered = c
    if delivered == s["target"]:
        outcome = "target_delivered"
    elif delivered == s["other"]:
        outcome = "wrong_delivered"
    elif lifted is not None:
        outcome = "target_grasped_only"  # only one object exists; "grasped, not delivered"
    else:
        outcome = "no_grasp"
    return dict(delivered=delivered, first_lift_step=lifted, outcome=outcome)


def label_stack(g, s):
    pt, po = pos(g, s["target"]), pos(g, s["other"])
    lt, lo = first_lift(pt), first_lift(po)
    first = None
    if lt is not None or lo is not None:
        first = s["target"] if (lo is None or (lt is not None and lt <= lo)) else s["other"]
    on_top = None
    a, b = pt[-1], po[-1]
    if np.linalg.norm(a[:2] - b[:2]) < STACK_XY and abs(a[2] - b[2]) > STACK_DZ:
        on_top = s["target"] if a[2] > b[2] else s["other"]
    if on_top == s["target"]:
        outcome = "target_delivered"
    elif on_top == s["other"]:
        outcome = "wrong_delivered"
    elif first == s["other"]:
        outcome = "wrong_grasped"
    elif first == s["target"]:
        outcome = "target_grasped_only"
    else:
        outcome = "no_grasp"
    return dict(delivered=on_top, first_lifted=first, first_lift_step=min(x for x in (lt, lo) if x is not None)
                if first else None, outcome=outcome)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    rows = []
    with open(os.path.join(args.folder, "episode_results.jsonl")) as f:
        eps = [json.loads(l) for l in f if l.strip()]
    for ep in eps:
        s = SPECS.get(ep["env_name"])
        if s is None:
            continue
        run = ep.get("run", ep["episode"] // max(1, ep.get("num_envs", 1)))
        h5 = os.path.join(args.folder, ep["env_name"], f"run_{run}.hdf5")
        with h5py.File(h5, "r") as fh:
            g = fh["data"][f"demo_{ep['env_id']}"]
            lab = label_dest(g, s) if s["kind"] == "dest" else label_stack(g, s)
        rows.append(dict(env_name=ep["env_name"], episode=ep["episode"], run=run, env_id=ep["env_id"],
                         instruction_fields=",".join(ep.get("instruction_fields_used") or ep.get("instruction_fields", [])),
                         instruction=ep["instruction"], success=ep["success"],
                         target=s["target"], other=s["other"], **lab))
    summary = collections.defaultdict(collections.Counter)
    for r in rows:
        summary[(r["env_name"], r["instruction_fields"])][r["outcome"]] += 1
        summary[(r["env_name"], r["instruction_fields"])]["success"] += int(r["success"])
        summary[(r["env_name"], r["instruction_fields"])]["n"] += 1
    cols = ["n", "success", "target_delivered", "wrong_delivered", "wrong_grasped", "target_grasped_only", "no_grasp"]
    print("| env | key | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for (env, key), c in sorted(summary.items()):
        print(f"| {env} | {key} | " + " | ".join(str(c[k]) for k in cols) + " |")
    # Sanity: RoboLab success must imply our target_delivered.
    bad = [r for r in rows if r["success"] and r["outcome"] != "target_delivered"]
    if bad:
        print(f"\nWARNING: {len(bad)} successful episode(s) not labelled target_delivered", file=sys.stderr)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
