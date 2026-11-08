#!/usr/bin/env python3
"""Language-test cells for π0.5 on RoboCasa365 (step 3, 2026-11-08), as JSON lines for run_lang_eval.py --policy pi05.

Only the tasks whose π0.5 default is >= 50% in the gate run 20261104-1530_robocasa_pi05_gate_berkeley3:
PickPlaceCounterToCabinet (28/50) and OpenStandMixerHead (36/50). NavigateKitchen (0/50) and TurnOffStove (9/50)
are dropped. Each cell uses the same 50 seeds as that task's default cells, split into two cells of 25 (suffix _a/_b)
so that workers stay balanced; the defaults themselves come from the gate run.
Usage: python pi05_screen_cells.py > cells.jsonl
"""
import json

from pi05_gate_cells import seeds

PNP, MIX = "PickPlaceCounterToCabinet", "OpenStandMixerHead"
CELLS = [  # (name, env, task, family, condition, require_valid)
    ("pnp_target_swap", "PnPCounterToCabinetSwap", PNP, "pnp", "target_swap", True),   # switch
    ("pnp_negation_only", PNP, PNP, "pnp", "negation_only", False),                    # negation
    ("pnp_positive_contrast", PNP, PNP, "pnp", "positive_contrast", False),            # distractor mention
    ("pnp_length_control_positive", PNP, PNP, "pnp", "length_control_positive", False),  # irrelevant content
    ("pnp_paraphrase_syntactic", PNP, PNP, "pnp", "paraphrase_syntactic", False),      # paraphrase (order)
    ("pnp_verb_synonym", PNP, PNP, "pnp", "verb_synonym", False),                      # paraphrase (words)
    ("mixer_paraphrase_syntactic", MIX, MIX, "other", "paraphrase_syntactic", False),  # paraphrase (order)
]


def cells():
    out = []
    for name, env, task, fam, cond, valid in CELLS:
        s = seeds(task, 50)
        for half, ss in (("a", s[:25]), ("b", s[25:])):
            out.append(dict(cell=f"{name}_{half}", env=env, env_kwargs={}, horizon_task=task, family=fam,
                            condition=cond, seeds=ss, n=len(ss), require_valid=valid))
    return out


if __name__ == "__main__":
    for c in cells():
        print(json.dumps(c))
