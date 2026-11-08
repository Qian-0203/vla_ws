#!/usr/bin/env python3
"""Screen report: per-cell SR, paired language contribution vs the gate's `default`, and which-object outcomes.

Pairing: a screen episode pairs with the gate episode of the same task and seed when their scene hashes match
(RoboCasa has ~3% history-dependent seeds, plan §3.4). Language contribution = SR(default) - SR(cell) on the paired
episodes. Outcome classes (plan §3.3):
  pnp    target delivered / other delivered / other grasped (not delivered) / target grasped only / none grasped
  nav    named destination reached / other destination reached / none
  stove  named burner off only / other burner off only / both off / none off   (pair cells; single-burner: off / on)
Usage: python screen_report.py --gate <gate episodes.jsonl> --screen <screen episodes.jsonl> [--out report.json]
"""
import argparse
import json
from collections import Counter, OrderedDict


def outcome_class(r):
    o, cell = r.get("outcome") or {}, r["cell"]
    if not o:
        return None
    if "delivered" in o:  # pnp
        cands = list((o.get("candidates") or {}).keys())
        swap = r["env"] == "PnPCounterToCabinetSwap"
        target = "distr_counter" if swap else ("obj" if "obj" in cands else "obj_a")
        delivered = set(o.get("delivered") or [])
        grasped = {n for n, s in (o.get("grasp_step_by_obj") or {}).items() if s is not None}
        if target in delivered:
            return "target delivered"
        if delivered - {target}:
            return "other delivered"
        if grasped - {target}:
            return "other grasped"
        if target in grasped:
            return "target grasped only"
        return "none grasped"
    if "first_reached" in o:  # nav
        slot = (r.get("env_kwargs") or {}).get("target_slot", "a")
        reached = {s for s, t in (o.get("reached_step_by_dest") or {}).items() if t is not None}
        if slot in reached:
            return "named reached"
        if reached:
            return "other reached"
        return "none reached"
    if "lit_at_end" in o:  # stove
        cands = o.get("candidates") or {}
        slot = (r.get("env_kwargs") or {}).get("target_slot", "a")
        named = cands.get(slot)
        other = cands.get("b" if slot == "a" else "a")
        off = set(o.get("lit_at_start") or []) - set(o.get("lit_at_end") or [])
        if other is None:
            return "named off" if named in off else "named on"
        if named in off and other in off:
            return "both off"
        if named in off:
            return "named off only"
        if other in off:
            return "other off only"
        return "none off"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    gate = {(r["task"], r["seed"]): r for r in map(json.loads, open(a.gate))}
    rows = [json.loads(l) for l in open(a.screen)]
    skipped = Counter(r["cell"] for r in rows if "skipped" in r)
    rows = [r for r in rows if "success" in r]
    rep = OrderedDict()
    for cell in OrderedDict.fromkeys(r["cell"] for r in rows):
        rs = [r for r in rows if r["cell"] == cell]
        pairs = [(gate[(r["task"], r["seed"])], r) for r in rs if (r["task"], r["seed"]) in gate
                 and gate[(r["task"], r["seed"])].get("scene_hash") == r.get("scene_hash")]
        d_sr = sum(g["success"] for g, _ in pairs) / len(pairs) if pairs else None
        c_sr = sum(r["success"] for _, r in pairs) / len(pairs) if pairs else None
        rep[cell] = OrderedDict(
            task=rs[0]["task"], env=rs[0]["env"], condition=rs[0]["condition"], n=len(rs),
            sr=round(100 * sum(r["success"] for r in rs) / len(rs), 1), skipped_seeds=skipped.get(cell, 0),
            n_paired=len(pairs),
            default_sr_paired=None if d_sr is None else round(100 * d_sr, 1),
            cell_sr_paired=None if c_sr is None else round(100 * c_sr, 1),
            drop_vs_default=None if d_sr is None else round(100 * (d_sr - c_sr), 1),
            outcomes=dict(Counter(outcome_class(r) for r in rs)),
            example_instruction=rs[0]["instruction"])
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
