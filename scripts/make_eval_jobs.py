#!/usr/bin/env python3
"""Print a jobs file for scripts/gpu_queue.py: one job per (split, task) for one checkpoint.

The split list and the per-split --task_ids subsets are read from scripts/run_benchmark.sh
(DEFAULT_SPLITS, TASK_IDS), so the queue runs exactly what `run_benchmark.sh` would. A job is one
task of one split, run as shard `task_id` of `n_tasks_in_suite` -- the same file layout
(`{suite}--{condition}--{note}--shard{i}of{n}.jsonl`) and the same `--shard` semantics that
run_libero_eval.py already has, so scripts/aggregate_results.py reads the results unchanged.

  python3 scripts/make_eval_jobs.py --note pi0_libero --family openpi \\
      --checkpoint gs://openpi-assets/checkpoints/pi0_libero --splits spatial/default,grounding/landmark_exchange

Families: `openpi` (a resident policy server per slot; {port} is filled by gpu_queue.py) and
`openvla` / `openvla_oft` (the model loads inside the eval container; CHECKPOINT is a host path).
Extra settings go in with --env KEY=VALUE (e.g. --env CENTER_CROP=False) and --extra '<flags>'.
"""
import argparse
import re
import shlex
import sys
from pathlib import Path


def parse_run_benchmark(path: Path):
    text = path.read_text()
    task_ids = dict(re.findall(r"\[([\w/]+)\]=([\d,]+)", text))
    block = re.search(r"DEFAULT_SPLITS=\((.*?)\n\)", text, re.S)
    if not block:
        sys.exit(f"DEFAULT_SPLITS not found in {path}")
    splits = [ln.split("#")[0].strip() for ln in block.group(1).splitlines()]
    return [s for s in splits if s], task_ids


def suite_sizes(libero_root: Path):
    ns = {}
    exec((libero_root / "libero/libero/benchmark/libero_suite_task_map.py").read_text(), ns)  # noqa: S102
    return {k: len(v) for k, v in ns["libero_task_map"].items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note", required=True, help="run_id_note: tags every results file of this checkpoint")
    ap.add_argument("--family", choices=["openvla", "openvla_oft", "openpi"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="", help="comma list; default = run_benchmark.sh DEFAULT_SPLITS")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--image", default="openvla-libero:blackwell")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for run_eval.sh (repeatable)")
    ap.add_argument("--extra", default="", help="extra flags appended to run_eval.sh")
    args = ap.parse_args()

    repo = Path(args.repo)
    default_splits, subset = parse_run_benchmark(repo / "scripts/run_benchmark.sh")
    splits = [s for s in args.splits.split(",") if s] or default_splits
    sys.path.insert(0, str(repo / "openvla"))
    from experiments.robot.libero.eval_registry import SPLITS  # noqa: E402
    sizes = suite_sizes(repo / "LIBERO")

    env = {
        "MACHINE_CONFIG": "config/server.env",
        "MODEL_FAMILY": args.family,
        "CHECKPOINT": args.checkpoint,
        "IMAGE_NAME": args.image,
        "OPENVLA_ATTN_IMPLEMENTATION": "sdpa",
        "LOAD_IN_4BIT": "False",
        "NUM_TRIALS_PER_TASK": str(args.trials),
        "SEED": str(args.seed),
        "GPUS": "{gpu}",
    }
    if args.family == "openpi":
        env["POLICY_PORT_BASE"] = "{port}"
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    env_str = " ".join(f"{k}={shlex.quote(v) if '{' not in v else v}" for k, v in env.items())

    n_jobs = 0
    for split in splits:
        if split not in SPLITS:
            sys.exit(f"unknown split {split!r} (not in eval_registry.SPLITS)")
        suite = SPLITS[split][0]
        n = sizes[suite]
        wanted = [int(t) for t in subset[split].split(",")] if split in subset else list(range(n))
        for tid in wanted:
            if tid >= n:
                sys.exit(f"{split}: task id {tid} outside suite {suite} (n={n})")
            shard = f"--num_shards {n} --shard_index {tid} " if n > 1 else ""
            name = f"{args.note}__{split.replace('/', '_')}__t{tid}"
            cmd = (f"cd {shlex.quote(str(repo))} && {env_str} bash docker/openvla_libero/run_eval.sh "
                   f"--split {split} --run_id_note {shlex.quote(args.note)} --task_ids {tid} "
                   f"{shard}--resume True {args.extra}").strip()
            print(f"{name}\t{cmd}")
            n_jobs += 1
    print(f"# {n_jobs} jobs, {len(splits)} splits, note={args.note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
