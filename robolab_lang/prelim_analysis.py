#!/usr/bin/env python3
"""Preliminary, simulator-free analyses of RoboLab-120 and its leaderboard (2026-10-21).

Prints markdown tables used by docs/robolab/preliminary_analysis.md and, with --figdir, draws the
leaderboard language-axis figure (needs matplotlib; e.g. `uv run --with matplotlib python ...`).

    A  leaderboard SR under vague / default / specific instructions, with Wilson CIs and drops
    B  RoboLab-120 instruction words absent from the DROID instruction corpus, per variant
    C  scenes shared by several tasks (the scene alone cannot pick the goal)
    D  left/right words in RoboLab-120 checked against the policy camera (robot vs image frame)

Usage:
    python3 robolab_lang/prelim_analysis.py --robolab <RoboLab checkout> \
        --leaderboard docs/robolab/leaderboard_v55.json [--figdir docs/robolab/figures]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frame_check import CAMERAS, parse_usda, project, to_policy_px  # noqa: E402

VARIANTS = ["vague", "default", "specific"]
STOP = set("the a an and in on of to it them into onto put place pick up from with that is are all any "
           "one two three this other its so be at".split())


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def diff_ci(k1, n1, k2, n2, z=1.96):
    """Difference of two proportions (pts), normal approximation. Same tasks, different episodes."""
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return 100 * (p1 - p2), 100 * z * se


def part_a(lb, figdir):
    print("## A. Leaderboard SR by instruction variant\n")
    print("| Policy | vague | default | specific | default − vague (95% CI) | specific − default |")
    print("|---|---|---|---|---|---|")
    rows = []
    for e in lb["entries"]:
        r = e["results"]
        if "vague" not in r:
            continue
        cells = []
        for v in VARIANTS:
            if v in r:
                lo, hi = wilson(r[v]["successes"], r[v]["episodes"])
                cells.append(f"{r[v]['sr']:.1f} [{lo:.1f}, {hi:.1f}]")
            else:
                cells.append("—")
        d, h = diff_ci(r["default"]["successes"], r["default"]["episodes"],
                       r["vague"]["successes"], r["vague"]["episodes"])
        sd = (f"{r['specific']['sr'] - r['default']['sr']:+.1f}" if "specific" in r else "—")
        print(f"| {e['display_name']} | {' | '.join(cells)} | {d:+.1f} ± {h:.1f} | {sd} |")
        rows.append((e["display_name"], {v: r[v]["sr"] for v in VARIANTS if v in r},
                     {v: wilson(r[v]["successes"], r[v]["episodes"]) for v in VARIANTS if v in r}))
    print()
    print("By competency axis, default − vague (pts):\n")
    print("| Policy | visual | relational | procedural |")
    print("|---|---|---|---|")
    for e in lb["entries"]:
        r = e["results"]
        if "vague" not in r:
            continue
        cells = [f"{r['default']['by_axis'][a]['sr'] - r['vague']['by_axis'][a]['sr']:+.1f}"
                 for a in ("visual", "relational", "procedural")]
        print(f"| {e['display_name']} | {' | '.join(cells)} |")
    print()
    if figdir:
        draw_a(rows, figdir)


def draw_a(rows, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"vague": "#2a78d6", "default": "#eb6834", "specific": "#1baf7a"}
    offs = {"vague": -0.22, "default": 0.0, "specific": 0.22}
    rows = sorted(rows, key=lambda r: r[1]["default"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (name, sr, ci) in enumerate(rows):
        lo = min(sr.values())
        hi = max(sr.values())
        ax.plot([lo, hi], [i, i], color="#c3c2b7", lw=2, zorder=1, solid_capstyle="round")
        for v in VARIANTS:
            if v in sr:
                ax.errorbar(sr[v], i + offs[v], xerr=[[sr[v] - ci[v][0]], [ci[v][1] - sr[v]]],
                            fmt="o", ms=6, color=colors[v], ecolor=colors[v], elinewidth=1, capsize=0,
                            mec="#fcfcfb", mew=1.5, zorder=3, label=v if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], color="#0b0b0b", fontsize=9)
    ax.set_xlabel("RoboLab-120 success rate (%), 1,200 episodes per point, 95% Wilson CI",
                  color="#52514e", fontsize=9)
    ax.tick_params(axis="x", colors="#52514e", labelsize=8)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9, title="instruction", title_fontsize=9)
    ax.set_title("Leaderboard success by instruction specificity", loc="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(figdir, f"leaderboard_language_axis.{ext}"), facecolor=fig.get_facecolor())
    print(f"(figure written to {figdir}/leaderboard_language_axis.png/.svg)\n")


def part_b(meta, droid_tasks):
    corpus = [json.loads(l)["task"].lower() for l in open(droid_tasks)]
    vocab = collections.Counter(w for t in corpus if t for w in set(re.findall(r"[a-z]+", t)))
    print("## B. RoboLab-120 instruction words absent from DROID\n")
    print(f"DROID corpus: {sum(1 for t in corpus if t)} distinct task strings, {len(vocab)} word types.\n")
    print("| Variant | mean share of content words absent from DROID | tasks with ≥ 1 absent word | example absent words |")
    print("|---|---|---|---|")
    for v in VARIANTS:
        shares, absent = [], collections.Counter()
        for x in meta:
            words = [w for w in re.findall(r"[a-z]+", x["instruction_variants"][v].lower()) if w not in STOP]
            miss = [w for w in words if vocab[w] == 0]
            absent.update(miss)
            shares.append(len(miss) / max(1, len(words)))
        ex = ", ".join(w for w, _ in absent.most_common(8))
        print(f"| {v} | {100 * sum(shares) / len(shares):.1f}% | {sum(s > 0 for s in shares)} / {len(shares)} | {ex} |")
    print()
    print("Instructions that contain an exclusion (not / except / non- / leave / do not):\n")
    for x in meta:
        for v in VARIANTS:
            s = x["instruction_variants"][v]
            if re.search(r"\b(not|except|without|leave)\b|non-", s.lower()):
                print(f"- `{x['task_name']}` ({v}): \"{s}\"")
    print()


def part_c(meta):
    by_scene = collections.defaultdict(list)
    for x in meta:
        by_scene[x["scene"]].append(x["task_name"])
    shared = {s: t for s, t in by_scene.items() if len(t) > 1}
    n_tasks = sum(len(t) for t in shared.values())
    print("## C. Scenes shared by several tasks\n")
    print(f"{len(by_scene)} scenes for {len(meta)} tasks; {len(shared)} scenes host 2–7 tasks, "
          f"covering {n_tasks} tasks; {len(meta) - n_tasks} tasks have a scene of their own.\n")


# Left/right words in RoboLab-120 and the objects they name (from the task files' terminations).
SIDE_CHECKS = [
    ("MustardIn{Left,Right}BinTask", "two_bin.usda", "grey_bin_left", "grey_bin_right"),
    ("BowlStacking{LeftOnRight,RightOnLeft}Task", "bowls_2_table.usda", "bowl_2", "bowl_1"),
    ("ClampInRightBin / HammersInLeftBin / NonHammerTools / ToolOrganization*", "tools_container.usda",
     "left_bin", "right_bin"),
    ("FoodPackingByColorTask", "food_packing.usda", "bin_a06", "bin_b03"),
]
RELATION_CHECKS = [
    ("RubiksCubeLeftOfBowlTask", "rubiks_cube_banana_bowl.usda", "bowl", (0.0, 0.10), "left of"),
    ("RubiksCubeBehindBowlTask", "rubiks_cube_banana_bowl.usda", "bowl", (0.10, 0.0), "behind"),
    ("RubiksCubeInFrontOfBowlTask", "rubiks_cube_banana_bowl.usda", "bowl", (-0.10, 0.0), "in front of"),
    ("RubiksCubeRightOfBowlTask", "rubiks_cube_banana_bowl_mug_bin.usda", "bowl", (0.0, -0.10), "right of"),
]


def part_d(robolab):
    cam = CAMERAS["over_shoulder_left"]
    scenes = os.path.join(robolab, "assets", "scenes")
    print("## D. Left/right words against the policy camera\n")
    print("Policy image = `over_shoulder_left_camera`, letterboxed to 224×224. Δu > 0 means the robot-left "
          "object is drawn to the image left; Δv > 0 means it is drawn lower.\n")
    print("| Task(s) | robot-left object (y) | robot-right object (y) | Δu (px) | Δv (px) | image reading |")
    print("|---|---|---|---|---|---|")
    for task, scene, left, right in SIDE_CHECKS:
        o = parse_usda(os.path.join(scenes, scene))
        ul = to_policy_px(cam, project(cam, o[left]))
        ur = to_policy_px(cam, project(cam, o[right]))
        du, dv = ur[0] - ul[0], ul[1] - ur[1]
        reading = "consistent" if du >= 8 else ("**no horizontal separation**" if abs(du) < 8 else "**reversed**")
        print(f"| {task} | `{left}` ({o[left][1]:+.2f}) | `{right}` ({o[right][1]:+.2f}) | {du:+.1f} | {dv:+.1f} | {reading} |")
    print()
    print("Goal relations (10 cm from the reference object in the robot-frame direction):\n")
    print("| Task | relation | image direction of the goal (Δu, Δv px) |")
    print("|---|---|---|")
    for task, scene, ref, (dx, dy), rel in RELATION_CHECKS:
        o = parse_usda(os.path.join(scenes, scene))
        p = o[ref]
        a = to_policy_px(cam, project(cam, p))
        b = to_policy_px(cam, project(cam, (p[0] + dx, p[1] + dy, p[2])))
        du, dv = b[0] - a[0], b[1] - a[1]
        word = ("left" if du < 0 else "right") + " and " + ("up" if dv < 0 else "down")
        print(f"| {task} | {rel} | ({du:+.1f}, {dv:+.1f}) → image {word} |")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robolab", required=True)
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--figdir")
    args = ap.parse_args(argv)
    lb = json.load(open(args.leaderboard))
    meta = json.load(open(os.path.join(args.robolab, "robolab/tasks/_metadata/task_metadata.json")))
    part_a(lb, args.figdir)
    part_b(meta, os.path.join(args.robolab, "analysis/droid_dataset/tasks.jsonl"))
    part_c(meta)
    part_d(args.robolab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
