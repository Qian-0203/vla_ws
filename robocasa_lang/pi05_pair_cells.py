#!/usr/bin/env python3
"""A2 (2026-11-16): language cells on RoboCasa's two-object pick-and-place scene, for run_lang_eval.py --policy pi05.

In PickPlaceCounterToCabinet the requested object is the nearer one in 48/50 scenes, so a policy that ignores the
sentence and takes the nearer object passes negation_only and fails target_swap. PnPCounterToCabPair (envs.py) places
two graspable objects from the same counter region and names slot a or b, so neither the requested nor the other
object is systematically nearer. Each condition runs with slot a and slot b named; init_dist_by_obj in the outcome
records which object started nearer. Seeds: the default's 50 (split into halves _1/_2 for balance).
Usage: python pi05_pair_cells.py > cells.jsonl
"""
import json

from pi05_gate_cells import seeds

PNP = "PickPlaceCounterToCabinet"
CELLS = [  # (name, condition, target_slot)
    ("pair_default_a", "default", "a"),                       # "Pick the {a} ..."
    ("pair_default_b", "default", "b"),                       # "Pick the {b} ..." (switch relative to a)
    ("pair_negation_a", "negation_only", "a"),                # "... not the {b} ..."
    ("pair_negation_b", "negation_only", "b"),                # "... not the {a} ..."
    ("pair_contrast_a", "positive_contrast", "a"),            # distractor mention
    ("pair_lenctrl_a", "length_control_positive", "a"),       # same-length filler control
]


def cells():
    out = []
    for name, cond, slot in CELLS:
        s = seeds(PNP, 50)
        for half, ss in (("1", s[:25]), ("2", s[25:])):
            out.append(dict(cell=f"{name}_{half}", env="PnPCounterToCabPair", env_kwargs={"target_slot": slot},
                            horizon_task=PNP, family="pnp", condition=cond, seeds=ss, n=len(ss), require_valid=True))
    return out


if __name__ == "__main__":
    for c in cells():
        print(json.dumps(c))
