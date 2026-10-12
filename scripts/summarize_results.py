#!/usr/bin/env python3
"""One table of success rates: conditions down, run notes (checkpoints) across, merged from several results dirs.

  python3 scripts/summarize_results.py --notes pi0_libero oft_libero_spatial vla_adapter_spatial \\
      --results-dir ~/vla_results_backup/libero_ckpt_campaign_2026-10-12/*/*/results

Per cell: mean of per-task success rates (the convention of aggregate_results.py and benchmark_split_result.md) and
the number of rollouts. Rows are checked on the way in and reported in a footer, because two things corrupted
numbers during the 2026-10-12 campaign and neither is visible in a success rate:
  - lost-server rows: a failed rollout with num_steps <= 10 (the exception fires at the first policy query), written
    by a client whose server was stopped under it; resume then counts it as a finished rollout;
  - duplicates: the same (task, episode) more than once (two clients on one results file).
Cells with either problem are marked with `!`. `--max-steps N` recounts a rollout as a success only if it finished
within N environment steps (settle steps included): `--max-steps 230` gives the standard 220-step protocol from a run
made with a longer horizon.
"""
import argparse
import collections
import glob
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", nargs="+", required=True, help="run_id_note of each checkpoint (a column each)")
    ap.add_argument("--results-dir", nargs="+", required=True, help="results directories (globs allowed), merged")
    ap.add_argument("--max-steps", type=int, default=0, help="count a success only if num_steps <= this (0 = off)")
    ap.add_argument("--conditions", nargs="*", default=None, help="only these suite--condition rows")
    args = ap.parse_args()

    dirs = []
    for d in args.results_dir:
        dirs += sorted(glob.glob(os.path.expanduser(d)))
    files = [f for d in dirs for f in glob.glob(os.path.join(d, "*.jsonl"))]

    cells = collections.defaultdict(lambda: collections.defaultdict(dict))  # (cond, note) -> task -> episode -> row
    flags = collections.defaultdict(lambda: {"lost": 0, "dup": 0})
    for f in files:
        parts = os.path.basename(f)[:-6].split("--")
        if len(parts) < 3:
            continue
        suite, cond, note = parts[0], parts[1], parts[2]
        if note not in args.notes:
            continue
        key = (f"{suite}--{cond}", note)
        if args.conditions and key[0] not in args.conditions:
            continue
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            ep = r["episode_idx"]
            task_rows = cells[key][r["task_id"]]
            if r["num_steps"] <= 10 and not r["success"]:
                flags[key]["lost"] += 1
            if ep in task_rows:
                flags[key]["dup"] += 1
                continue  # keep the first row of a repeated episode
            task_rows[ep] = r

    conds = sorted({c for c, _ in cells})
    print("| condition | " + " | ".join(args.notes) + " |")
    print("|---|" + "|".join(["--:"] * len(args.notes)) + "|")
    bad = []
    for c in conds:
        out = []
        for note in args.notes:
            tasks = cells.get((c, note))
            if not tasks:
                out.append("")
                continue
            rates, n = [], 0
            for rows in tasks.values():
                succ = sum(1 for r in rows.values()
                           if r["success"] and (not args.max_steps or r["num_steps"] <= args.max_steps))
                rates.append(succ / len(rows))
                n += len(rows)
            mark = "!" if flags[(c, note)]["lost"] or flags[(c, note)]["dup"] else ""
            if mark:
                bad.append((c, note, flags[(c, note)]))
            out.append(f"{100 * sum(rates) / len(rates):.1f}% ({n}){mark}")
        print(f"| {c} | " + " | ".join(out) + " |")
    if bad:
        print("\n`!` cells with problem rows (first row of a repeated episode kept):")
        for c, note, fl in bad:
            print(f"- {c} / {note}: {fl['lost']} lost-server row(s), {fl['dup']} duplicate row(s)")
    print(f"\n{len(files)} results files under {len(dirs)} directories"
          + (f"; success counted within {args.max_steps} steps" if args.max_steps else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
