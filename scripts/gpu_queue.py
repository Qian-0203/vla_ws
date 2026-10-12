#!/usr/bin/env python3
"""Run a list of eval jobs on a fixed set of GPU slots, one job per slot at a time.

Why: run_eval.sh shards one split over N GPUs and waits for the slowest shard, so a 10-task split
on 8 slots idles 6 of them for a whole task. Here a job is one (checkpoint, split, task) and a slot
takes the next job as soon as it is free, so slots stay busy across splits and checkpoints.

Jobs file (from scripts/make_eval_jobs.py): one job per line, `name<TAB>shell command`. In the command,
`{gpu}` is replaced by the slot's GPU index and `{port}` by --port-base + the slot's position (for
openpi, slot i is served by the policy server started as i-th entry of GPUS, see serve_openpi.sh).

  python3 scripts/gpu_queue.py --jobs jobs.tsv --slots 0,0,1,1,2,2,3,3 --port-base 8000 \\
      --log-dir ~/logs/queue_pi0 --min-free-gb 0

Safe to re-run: a job with `<log-dir>/<name>.done` is skipped, and every eval call carries
--resume True, so a half-finished job continues. Failed jobs are recorded as `<name>.fail` and are not
retried unless --retry-failed. Nothing here kills processes that are not its own children.
On a shared GPU host, --min-free-gb makes a slot wait until its GPU has that much memory free
(use ~20 for an in-container 7B model, 0 when the policy server is already resident).
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def free_gb(gpu: str) -> float:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(out.strip().splitlines()[0]) / 1024.0


def read_jobs(path: str):
    jobs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        name, _, cmd = line.partition("\t")
        if not cmd:
            sys.exit(f"bad job line (need name<TAB>command): {line[:80]}")
        jobs.append((name.strip(), cmd.strip()))
    names = [n for n, _ in jobs]
    if len(set(names)) != len(names):
        sys.exit("duplicate job names in jobs file")
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--slots", required=True, help="GPU index per slot, repeats allowed (0,0,1,1)")
    ap.add_argument("--port-base", type=int, default=8000)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--min-free-gb", type=float, default=0.0)
    ap.add_argument("--stagger", type=float, default=45.0,
                    help="min seconds between two launches on the same GPU (model load, memory settle)")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slots = [s.strip() for s in args.slots.split(",") if s.strip()]
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = []
    for name, cmd in read_jobs(args.jobs):
        if (log_dir / f"{name}.done").exists():
            continue
        if (log_dir / f"{name}.fail").exists() and not args.retry_failed:
            continue
        pending.append((name, cmd))
    print(f"{len(pending)} job(s) to run on {len(slots)} slot(s): {','.join(slots)}", flush=True)
    if args.dry_run:
        for i, (name, cmd) in enumerate(pending):
            slot = i % len(slots)
            print(f"[{name}] slot {slot}: {cmd.replace('{gpu}', slots[slot]).replace('{port}', str(args.port_base + slot))}")
        return 0

    running = {}      # slot index -> (name, Popen, t0)
    last_launch = {}  # gpu -> time
    stop = {"flag": False}

    def _stop(signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    n_ok = n_fail = 0
    while (pending or running) and not stop["flag"]:
        for slot, (name, proc, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[slot]
            mins = (time.time() - t0) / 60
            if rc == 0:
                (log_dir / f"{name}.done").write_text(f"{time.strftime('%F %T')} {mins:.1f} min\n")
                (log_dir / f"{name}.fail").unlink(missing_ok=True)
                n_ok += 1
            else:
                (log_dir / f"{name}.fail").write_text(f"{time.strftime('%F %T')} rc={rc} {mins:.1f} min\n")
                n_fail += 1
            print(f"{time.strftime('%T')} {'done' if rc == 0 else f'FAIL rc={rc}'} {name} ({mins:.1f} min) "
                  f"[{n_ok} ok, {n_fail} failed, {len(pending)} queued, {len(running)} running]", flush=True)

        for slot, gpu in enumerate(slots):
            if not pending:
                break
            if slot in running or time.time() - last_launch.get(gpu, 0.0) < args.stagger:
                continue
            if args.min_free_gb and free_gb(gpu) < args.min_free_gb:
                continue
            name, cmd = pending.pop(0)
            cmd = cmd.replace("{gpu}", gpu).replace("{port}", str(args.port_base + slot))
            log = open(log_dir / f"{name}.log", "ab")
            proc = subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            running[slot] = (name, proc, time.time())
            last_launch[gpu] = time.time()
            print(f"{time.strftime('%T')} start {name} on GPU {gpu} slot {slot}", flush=True)
        time.sleep(args.poll)

    if stop["flag"]:
        print("stopping: terminating this scheduler's own children", flush=True)
        for _slot, (name, proc, _t0) in running.items():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return 130
    print(f"queue finished: {n_ok} ok, {n_fail} failed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
