#!/usr/bin/env python3
"""Build one data package from every results directory: a model x condition table, a per-task CSV and an outcomes CSV
in the formats of the paper-prompt files (task_condition_sr_*.csv, outcomes_*.csv), plus the checks that went into them.

  python3 scripts/build_full_data_package.py --out DIR \\
      --results-dir ~/vla_results_backup/raw_data_library_2026-10-12/results \\
      --results-dir "$HOME/vla_results_backup/libero_ckpt_campaign_2026-10-12/*/*/results"

Which rows count (the rules, because they decide the numbers):
  - a file is `{suite}--{condition}--{note}[--{extra}][--shardXofN].jsonl`; a note maps to a model (MODELS below);
    screens (extra=screen), smoke runs and unknown notes are left out;
  - for one (model, condition) the run is `main` (no extra); if there is none, the re-run (`--rerun`); a
    `--whichbowl` re-run is used only when neither exists;
  - repeated (task, episode) rows keep the first one; a failed row with num_steps <= 10 (a client whose server was
    stopped under it) is dropped and counted, never scored;
  - success = LIBERO `done`; `--max-steps N` recounts success only if num_steps <= N (X-VLA's gate splits were
    run at horizon 800; every success there is inside 230 steps, so N = 230 is the standard 220-step protocol);
  - the cell is the mean of per-task success rates, like aggregate_results.py and benchmark_split_result.md;
  - `--exclude SUBSTR` leaves out every results file whose path contains SUBSTR (a workspace known to hold a
    contaminated duplicate copy; the cells it duplicated are then served by the clean copy and the exclusion is part
    of the command that built the package);
  - `--drop-condition NAME` leaves out one canonical condition everywhere (e.g. an exploratory split that is
    not reported).
"""
import argparse
import collections
import csv
import glob
import json
import os
import sys

MODELS = {
    "openvla-7b-libero-spatial-lora-r32": "ours (OpenVLA LoRA r32)",
    "openvla-official-libero-spatial": "official OpenVLA",
    "qwenvla-78.6pct": "QwenVLA (Ken)",
    "pi05_libero": "pi05_libero",
    "pi0_libero": "pi0_libero",
    "oft_libero_spatial": "OpenVLA-OFT (spatial)",
    "vla_adapter_spatial": "VLA-Adapter (Spatial)",
    "vla_adapter_pro_spatial": "VLA-Adapter (Spatial-Pro)",
    "xvla_libero": "X-VLA (Libero)",
    "gr00t_n15_spatial": "GR00T N1.5 (spatial)",
}
MODEL_ORDER = list(MODELS.values())
CANON = {
    ("libero_spatial_3bowl_front", "default"): "irrelevant",
    ("libero_spatial_3bowl_semantic2", "default"): "semantic",
    ("libero_spatial_3bowl_hardneg", "default"): "landmark",
    ("libero_spatial_3bowl_hardneg", "hardneg"): "landmark_with_hardneg_prompt",
    ("libero_spatial_3bowl_open", "default"): "drawer_open",
    ("libero_spatial_3bowl", "default"): "closed_drawer_control (center_fixed_legacy)",
    ("libero_spatial_grounding_region_surface", "default"): "region_surface",
    ("libero_spatial_grounding_surface_landmark", "default"): "surface_landmark",
    ("libero_spatial_landmark_exchange", "default"): "landmark_exchange",
}
WITHDRAWN = {"target_cue_region", "target_cue_region_v2", "target_cue_region_v3"}  # Erratum E1
ORDER = ["default", "positive_contrast", "negative_contrast", "negation_only", "length_control_suffix",
         "length_control_infix", "paraphrase_lexical", "paraphrase_syntactic", "target_cue_landmark",
         "target_cue_proximity_novel", "target_cue_proximity_beside", "target_cue_proximity_near",
         "target_cue_proximity_adjacent", "target_cue_region_img", "target_cue_region_v2_img",
         "target_cue_region_v3_img", "irrelevant", "semantic", "landmark", "landmark_with_hardneg_prompt",
         "drawer_open", "region_surface", "surface_landmark", "landmark_exchange"]
RUN_PRIORITY = {"": 0, "rerun": 1, "whichbowl": 2}


def canon(suite, cond):
    return CANON.get((suite, cond), cond if suite == "libero_spatial" else f"{suite}/{cond}")


def parse(path):
    parts = os.path.basename(path)[:-6].split("--")  # drop .jsonl
    suite, cond = parts[0], parts[1]
    note = parts[2] if len(parts) > 2 else ""
    extra = []
    for p in parts[3:]:
        if not (p.startswith("shard") and "of" in p):
            extra.append(p)
    return suite, cond, note, "--".join(extra)


def outcome_class(r):
    """target = the bowl the goal asks for (for landmark_exchange: the bowl the sentence describes)."""
    t = r.get("target_object")
    if t is None:
        return None
    on = r.get("bowls_on_plate") or []
    if t in on:
        return "target_delivered"
    if on:
        return "other_delivered"
    g = r.get("first_bowl_grasped")
    if g == t:
        return "target_lifted_only"
    if g:
        return "other_lifted_only"
    a = r.get("first_bowl_approached")
    if a == t:
        return "no_lift_approached_target"
    if a:
        return "no_lift_approached_other"
    return "no_lift_approached_neither"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-dir", action="append", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--stamp", default="")
    ap.add_argument("--exclude", action="append", default=[], help="skip results files whose path contains this")
    ap.add_argument("--drop-condition", action="append", default=[], help="canonical condition name to leave out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = []
    for d in args.results_dir:
        for dd in sorted(glob.glob(os.path.expanduser(d))):
            files += [f for f in glob.glob(os.path.join(dd, "*.jsonl"))
                      if not any(x in f for x in args.exclude)]
    # (model, condition) -> run extra -> task -> episode -> row
    runs = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    dropped = collections.Counter()
    dups = collections.Counter()
    for f in sorted(files):
        suite, cond, note, extra = parse(f)
        if note not in MODELS or extra in ("screen",) or "smoke" in extra or "smoke" in note:
            continue
        c = canon(suite, cond)
        if cond in WITHDRAWN and suite == "libero_spatial":
            continue
        if c in args.drop_condition:
            continue
        model = MODELS[note]
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["num_steps"] <= 10 and not r["success"]:
                dropped[(model, c)] += 1
                continue
            rows = runs[(model, c)][extra][r["task_id"]]
            prev = rows.get(r["episode_idx"])
            if prev is not None:
                # the same file reached us through two copies (library + server pull): harmless. A repeated episode
                # whose outcome differs is a real conflict and is flagged.
                if (prev["success"], prev["num_steps"]) != (r["success"], r["num_steps"]):
                    dups[(model, c)] += 1
                continue
            rows[r["episode_idx"]] = r

    chosen = {}
    for key, by_extra in runs.items():
        extras = sorted(by_extra, key=lambda e: RUN_PRIORITY.get(e, 9))
        chosen[key] = (extras[0], by_extra[extras[0]])

    def ok(r):
        return r["success"] and (not args.max_steps or r["num_steps"] <= args.max_steps)

    cell = {}
    sr_rows, out_rows = [], []
    for (model, c), (extra, tasks) in chosen.items():
        rates, n = [], 0
        for tid in sorted(tasks):
            rows = list(tasks[tid].values())
            succ = sum(1 for r in rows if ok(r))
            rates.append(succ / len(rows))
            n += len(rows)
            run = extra or "main"
            sr_rows.append([model, c, run, tid, succ, len(rows), round(100.0 * succ / len(rows), 1), "raw_jsonl"])
            cls = collections.Counter(outcome_class(r) for r in rows if r.get("target_object") is not None)
            if cls and None not in cls:
                keys = ["target_delivered", "other_delivered", "target_lifted_only", "other_lifted_only",
                        "no_lift_approached_target", "no_lift_approached_other", "no_lift_approached_neither"]
                out_rows.append([model, c, run, tid, sum(cls.values())] + [cls.get(k, 0) for k in keys])
        cell[(model, c)] = (100.0 * sum(rates) / len(rates), n, extra or "main", len(tasks))

    expected = {}
    for (m, c), v in cell.items():
        expected[c] = max(expected.get(c, 0), v[1])
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in cell)]
    conds = [c for c in ORDER if any(k[1] == c for k in cell)] + sorted(
        {k[1] for k in cell} - set(ORDER))
    with open(os.path.join(args.out, "task_condition_sr.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "condition", "run", "task_id", "successes", "trials", "success_rate_percent", "source"])
        w.writerows(sorted(sr_rows, key=lambda r: (MODEL_ORDER.index(r[0]), r[1], r[3])))
    with open(os.path.join(args.out, "outcomes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "condition", "run", "task_id", "n", "target_delivered", "other_delivered",
                    "target_lifted_only", "other_lifted_only", "no_lift_approached_target",
                    "no_lift_approached_other", "no_lift_approached_neither"])
        w.writerows(sorted(out_rows, key=lambda r: (MODEL_ORDER.index(r[0]), r[1], r[3])))

    with open(os.path.join(args.out, "full_table.md"), "w") as fh:
        fh.write(f"# Success rate by condition and checkpoint {args.stamp}\n\n")
        fh.write("Cell = mean of per-task success rates (rollouts). `*` = not the main run (re-run). "
                 "`!` = a lost-server row was dropped or a repeated episode had a different outcome. "
                 "`partial` = the run is not finished. Blank = not run.\n\n")
        fh.write("| condition | " + " | ".join(models) + " |\n|---|" + "|".join(["--:"] * len(models)) + "|\n")
        for c in conds:
            cs = []
            for m in models:
                v = cell.get((m, c))
                if not v:
                    cs.append("")
                    continue
                mark = ("*" if v[2] != "main" else "") + ("!" if dropped.get((m, c)) or dups.get((m, c)) else "")
                part = f" partial of {expected[c]}" if v[1] < expected[c] else ""
                cs.append(f"{v[0]:.1f}% ({v[1]}{part}){mark}")
            fh.write(f"| {c} | " + " | ".join(cs) + " |\n")
        if dropped or dups:
            fh.write("\nDropped rows (lost server: failed with num_steps <= 10) and skipped duplicate episodes:\n\n")
            for k in sorted(set(dropped) | set(dups)):
                fh.write(f"- {k[0]} / {k[1]}: {dropped.get(k, 0)} lost-server, {dups.get(k, 0)} duplicate\n")
        if args.max_steps:
            fh.write(f"\nSuccess counted only within {args.max_steps} environment steps.\n")
    print(f"models: {len(models)}, conditions: {len(conds)}, cells: {len(cell)}, files: {len(files)}")
    print("dropped lost-server rows:", sum(dropped.values()), " conflicting repeated episodes:", sum(dups.values()))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
