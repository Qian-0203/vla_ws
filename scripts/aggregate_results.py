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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "openvla", "experiments", "logs", "results"),
    )
    parser.add_argument("--filter", default=None, help="Only include runs whose name contains this substring")
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


if __name__ == "__main__":
    main()
