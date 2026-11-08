#!/usr/bin/env python3
"""Gate check for a RoboCasa run against Xiaomi-Robotics-1's published per-episode results (plan §5.3).

Rule: per task |SR - published| <= 14 pts, and the mean over tasks within 7 pts of the published mean. Also reports
episode-level agreement on identical seeds (not 100% expected: diffusion sampling is stochastic).
Usage: python gate_report.py --episodes <episodes.jsonl> [--published robocasa_lang/pilot_seeds.json] [--out gate.json]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--published", default=os.path.join(HERE, "pilot_seeds.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    pub = json.load(open(a.published))["tasks"]
    rows = [json.loads(l) for l in open(a.episodes) if "success" in l]
    out = {"tasks": {}}
    ours_m, pub_m = [], []
    for task, p in pub.items():
        mine = {r["seed"]: r["success"] for r in rows if r["task"] == task}
        if not mine:
            continue
        ps = {e["seed"]: e["success"] for e in p["episodes"]}
        common = sorted(set(mine) & set(ps))
        sr, psr = sum(mine.values()) / len(mine), sum(ps.values()) / len(ps)
        agree = sum(mine[s] == ps[s] for s in common) / len(common)
        both = sum(mine[s] and ps[s] for s in common)
        out["tasks"][task] = dict(n=len(mine), sr=round(sr * 100, 1), published_sr=round(psr * 100, 1),
                                  diff_pts=round((sr - psr) * 100, 1), pass_14pts=abs(sr - psr) * 100 <= 14,
                                  episode_agreement=round(agree, 3), both_success=both, n_common=len(common))
        ours_m.append(sr)
        pub_m.append(psr)
    m, pm = sum(ours_m) / len(ours_m) * 100, sum(pub_m) / len(pub_m) * 100
    out["mean_sr"], out["published_mean_sr"] = round(m, 1), round(pm, 1)
    out["mean_within_7pts"] = abs(m - pm) <= 7
    out["gate_pass"] = out["mean_within_7pts"] and all(t["pass_14pts"] for t in out["tasks"].values())
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
