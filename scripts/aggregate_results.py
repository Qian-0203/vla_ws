#!/usr/bin/env python3
"""
aggregate_results.py

Canonical aggregation for eval runs: reads the structured per-rollout JSONL
files written by openvla/experiments/robot/libero/run_libero_eval.py (under
openvla/experiments/logs/results/) and prints per-task + suite-wide success
rates, in the same "mean of per-task rates" convention benchmark_split_result.md uses.

Multi-GPU runs write one JSONL per shard (name ends in --shard{i}of{N}); this
merges shards of the same run back into one suite-wide result automatically.

Usage:
    python scripts/aggregate_results.py                       # every run found
    python scripts/aggregate_results.py --filter 3bowl_open    # substring filter
    python scripts/aggregate_results.py --results-dir openvla/experiments/logs/results
    python scripts/aggregate_results.py --filter negation_only --outcomes   # + per-task failure modes
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def load_records(results_dir, name_filter):
    # base_key strips the "--shard{i}of{N}" suffix so shards of one run merge.
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        base_key = re.sub(r"--shard\d+of\d+$", "", os.path.basename(path)[: -len(".jsonl")])
        if name_filter and name_filter not in base_key:
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    runs[base_key].append(json.loads(line))
    return runs


def summarize(records):
    by_task = defaultdict(list)
    for r in records:
        by_task[(r["task_id"], r["task_name"])].append(r["success"])
    per_task = {
        key: (sum(v) / len(v), len(v)) for key, v in sorted(by_task.items())
    }
    overall = sum(rate for rate, _ in per_task.values()) / len(per_task) if per_task else 0.0
    return per_task, overall


OUTCOMES = (
    "target_delivered", "wrong_bowl_delivered", "wrong_bowl_grasped", "target_grasped_only", "no_grasp", "unknown",
)


def classify_outcome(r):
    """One rollout's behavioral outcome, from the env-side fields run_libero_eval.py logs since the
    RolloutOutcomeTracker was added. Older rows (no `target_object`) are "unknown".

    target_delivered      success (the goal's own predicate held)
    wrong_bowl_delivered  failed, and a distractor bowl is On the plate at episode end
    wrong_bowl_grasped    failed, nothing wrong delivered, and the first bowl grasped (lifted >3 cm while
                          touching the gripper) was a distractor
    target_grasped_only   failed, and the first bowl grasped was the target
    no_grasp              failed without ever lifting a bowl
    """
    if r.get("target_object") is None:
        return "unknown"
    if r["success"]:
        return "target_delivered"
    if set(r.get("bowls_on_plate") or []) & set(r.get("distractor_objects") or []):
        return "wrong_bowl_delivered"
    grasped = r.get("first_bowl_grasped")
    if grasped is None:
        return "no_grasp"
    return "target_grasped_only" if grasped == r["target_object"] else "wrong_bowl_grasped"


def print_outcomes(records):
    by_task = defaultdict(list)
    for r in records:
        by_task[(r["task_id"], r["task_name"])].append(classify_outcome(r))
    print("\n| id | " + " | ".join(OUTCOMES) + " | n |")
    print("|--:|" + "--:|" * (len(OUTCOMES) + 1))
    totals = defaultdict(int)
    for (task_id, _), outs in sorted(by_task.items()):
        for o in outs:
            totals[o] += 1
        print(f"| {task_id} | " + " | ".join(str(outs.count(o)) for o in OUTCOMES) + f" | {len(outs)} |")
    print("| all | " + " | ".join(str(totals[o]) for o in OUTCOMES) + f" | {len(records)} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "openvla", "experiments", "logs", "results"),
    )
    parser.add_argument("--filter", default=None, help="Only include runs whose name contains this substring")
    parser.add_argument(
        "--outcomes", action="store_true",
        help="Also print per-task outcome counts (wrong bowl delivered/grasped, ...); see classify_outcome",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    if not os.path.isdir(results_dir):
        print(f"No results directory at {results_dir} -- nothing to aggregate yet.")
        return

    runs = load_records(results_dir, args.filter)
    if not runs:
        print(f"No matching *.jsonl results found in {results_dir}")
        return

    for run_key, records in runs.items():
        per_task, overall = summarize(records)
        n_trials = next(iter(per_task.values()))[1] if per_task else 0
        print(f"\n## {run_key}  ({len(records)} rollouts, {len(per_task)} tasks x {n_trials} trials)")
        print("| id | task | success | n |")
        print("|--:|---|--:|--:|")
        for (task_id, task_name), (rate, n) in per_task.items():
            print(f"| {task_id} | {task_name} | {rate * 100:.1f}% | {n} |")
        print(f"\n**Overall (mean of per-task rates): {overall * 100:.1f}%**")
        if args.outcomes:
            print_outcomes(records)


if __name__ == "__main__":
    main()
