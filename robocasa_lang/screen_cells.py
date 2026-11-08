#!/usr/bin/env python3
"""Write the pilot language-screen cells (plan §5.3 step 4, revised §4.0) as JSON lines for run_lang_eval.py.

20 cells x 20 episodes = 400 rollouts, on the seeds Xiaomi-Robotics-1 published (robocasa_lang/pilot_seeds.json): the
first 20 seeds of each task; cells marked require_valid walk further down the 50-seed list past invalid seeds
(non-graspable distractor, no second destination). Usage: python screen_cells.py > cells.jsonl
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
seeds = {t: [e["seed"] for e in v["episodes"]]
         for t, v in json.load(open(os.path.join(HERE, "pilot_seeds.json")))["tasks"].items()}
N = 20


def cell(name, env, task, family, condition, kwargs=None, valid=False):
    return dict(cell=name, env=env, env_kwargs=kwargs or {}, horizon_task=task, family=family,
                condition=condition, seeds=seeds[task], n=N, require_valid=valid)


NAV, PNP, STOVE, MIX = "NavigateKitchen", "PickPlaceCounterToCabinet", "TurnOffStove", "OpenStandMixerHead"
CELLS = [
    # language contribution: blank on the original scenes (default comes from the gate on the same seeds)
    cell("nav_blank", NAV, NAV, "nav", "blank"),
    cell("pnp_blank", PNP, PNP, "pnp", "blank"),
    cell("stove_blank", STOVE, STOVE, "stove", "blank"),
    cell("mixer_blank", MIX, MIX, "other", "blank"),
    # layout floor
    cell("nav_no_target", NAV, NAV, "nav", "no_target"),
    cell("pnp_no_target", PNP, PNP, "pnp", "no_target"),
    cell("stove_no_target", STOVE, STOVE, "stove", "no_target"),
    # P1 target swap on the original scenes (PnP: success on the distractor; Nav: second destination = pair slot b)
    cell("pnp_target_swap", "PnPCounterToCabinetSwap", PNP, "pnp", "target_swap", valid=True),
    cell("nav_target_swap", "NavigateKitchenPair", NAV, "nav", "default", {"target_slot": "b"}, valid=True),
    # TurnOffStove needs a second lit burner for any swap (only the target is lit in the original scene)
    cell("stovepair_a", "TurnOffStovePair", STOVE, "stove", "default", {"target_slot": "a"}),
    cell("stovepair_b", "TurnOffStovePair", STOVE, "stove", "default", {"target_slot": "b"}),
    # P3, P5 (distractor named) on original-scene PnP; Nav contrast names the pair's second destination
    cell("pnp_negation_only", PNP, PNP, "pnp", "negation_only", valid=False),
    cell("pnp_positive_contrast", PNP, PNP, "pnp", "positive_contrast"),
    cell("pnp_length_control_positive", PNP, PNP, "pnp", "length_control_positive"),
    cell("nav_positive_contrast", "NavigateKitchenPair", NAV, "nav", "positive_contrast", {"target_slot": "a"},
         valid=True),
    cell("nav_length_control_positive", "NavigateKitchenPair", NAV, "nav", "length_control_positive",
         {"target_slot": "a"}, valid=True),
    # P4 rung 1
    cell("nav_paraphrase_syntactic", NAV, NAV, "nav", "paraphrase_syntactic"),
    cell("pnp_paraphrase_syntactic", PNP, PNP, "pnp", "paraphrase_syntactic"),
    cell("stove_paraphrase_syntactic", STOVE, STOVE, "stove", "paraphrase_syntactic"),
    cell("mixer_paraphrase_syntactic", MIX, MIX, "other", "paraphrase_syntactic"),
]

if __name__ == "__main__":
    assert len(CELLS) == 20 and len({c["cell"] for c in CELLS}) == 20
    for c in CELLS:
        print(json.dumps(c))
