#!/usr/bin/env python3
"""Generate the language-stress screening candidates (one JSON per condition).

Each JSON is consumed by `run_libero_eval.py --instruction_file`, outside the canonical registry
(`eval_registry.SPLITS`), so candidates can be screened cheaply on every policy before any of them
is promoted to a real split. Every candidate keeps the stock 2-bowl `libero_spatial` scene and
init states and describes its referent truthfully. The distractor-location phrases are taken from
the already-authored `negative_contrast` prompts (instructions.py), the single source for where
each task's second bowl sits.

Usage: python3 scripts/language_stress/make_candidates.py   (writes screening/language_stress/candidates/)
"""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "screening" / "language_stress" / "candidates"
INSTRUCTIONS = ROOT / "openvla" / "experiments" / "robot" / "libero" / "instructions.py"

# task id -> (LIBERO task name, target phrase as LIBERO words it, target phrase usable after "the black bowl")
TASKS = {
    0: ("pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", "between the plate and the ramekin", "between the plate and the ramekin"),
    1: ("pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate", "next to the ramekin", "next to the ramekin"),
    2: ("pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "from table center", "at the center of the table"),
    3: ("pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", "on the cookie box", "on the cookie box"),
    4: ("pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", "in the top drawer of the wooden cabinet", "in the top drawer of the wooden cabinet"),
    5: ("pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", "on the ramekin", "on the ramekin"),
    6: ("pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", "next to the cookie box", "next to the cookie box"),
    7: ("pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", "on the stove", "on the stove"),
    8: ("pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate", "next to the plate", "next to the plate"),
    9: ("pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate", "on the wooden cabinet", "on the wooden cabinet"),
}

# Candidates promoted to real splits. run_libero_eval.py refuses a screening condition that shares a
# registry condition's name, so these are no longer written here; run them with --split instead.
PROMOTED = {"negation_only": "spatial/negation_only"}  # 2026-10-10

# Landmark-noun synonyms (the relation words stay; only the object names change).
NOUN_SYNONYMS = {"ramekin": "small baking dish", "cookie box": "biscuit box", "stove": "cooktop", "wooden cabinet": "wooden cupboard"}

# Native-speaker-checkable translations of the target phrase. \todo: have a native speaker verify each.
ZH = {0: "盘子和小烤碗之间的", 1: "小烤碗旁边的", 2: "桌子中央的", 3: "饼干盒上的", 4: "木柜最上层抽屉里的",
      5: "小烤碗上的", 6: "饼干盒旁边的", 7: "炉子上的", 8: "盘子旁边的", 9: "木柜上的"}
ES = {0: "que está entre el plato y el ramequín", 1: "que está junto al ramequín", 2: "que está en el centro de la mesa",
      3: "que está sobre la caja de galletas", 4: "que está en el cajón superior del gabinete de madera",
      5: "que está sobre el ramequín", 6: "que está junto a la caja de galletas", 7: "que está sobre la estufa",
      8: "que está junto al plato", 9: "que está sobre el gabinete de madera"}


def distractor_phrases():
    spec = importlib.util.spec_from_file_location("instructions", INSTRUCTIONS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    for tid, (name, _, _) in TASKS.items():
        m = re.search(r", not the one (.+?), and place it on the plate$", mod.LIBERO_SPATIAL_EXPLICIT_INSTRUCTIONS[name])
        assert m, f"negative_contrast prompt for task {tid} has an unexpected shape"
        out[tid] = m.group(1)
    return out


def synonymize(phrase):
    for noun, syn in NOUN_SYNONYMS.items():
        phrase = phrase.replace(noun, syn)
    return phrase


def build():
    d = distractor_phrases()
    ids = list(TASKS)
    syn_ids = [t for t in ids if synonymize(TASKS[t][2]) != TASKS[t][2]]
    c = []

    def add(condition, family, hypothesis, text_fn, task_ids=ids, swap_target=False):
        c.append({
            "condition": condition, "family": family, "hypothesis": hypothesis,
            "swap_target": swap_target, "task_ids": task_ids,
            "instructions": {TASKS[t][0]: text_fn(t) for t in task_ids},
        })

    add("screen_default", "baseline", "Same-budget baseline: LIBERO's own task language.",
        lambda t: f"pick up the black bowl {TASKS[t][1]} and place it on the plate")
    add("target_swap", "language following",
        "The instruction truthfully names the OTHER bowl and success is scored on that bowl. A policy that "
        "executes the layout's trained motion instead of reading the sentence scores near 0.",
        lambda t: f"pick up the black bowl {d[t]} and place it on the plate", swap_target=True)
    add("no_location", "language ablation",
        "The location phrase is removed; success is still scored on the original target. A high score means "
        "the layout alone identifies the task, i.e. the policy need not read the location.",
        lambda t: "pick up the black bowl and place it on the plate")
    add("negation_only", "negation",
        "The target is identified only by negating the distractor's location. Negation is weak in VLMs, and "
        "the sentence names only the distractor's place.",
        lambda t: f"pick up the black bowl that is not {d[t]} and place it on the plate")
    add("distractor_first", "reference order",
        "The distractor is named before the target (same content as negative_contrast, reversed order).",
        lambda t: f"leave the black bowl {d[t]} where it is; pick up the black bowl {TASKS[t][2]} and place it on the plate")
    add("coreference", "discourse",
        "The target is introduced in one sentence and acted on through a pronoun in the next.",
        lambda t: f"there is a black bowl {TASKS[t][2]}. pick it up and place it on the plate.")
    add("noun_synonym", "lexical (object names)",
        "Landmark nouns are replaced by synonyms; relation words stay. Tests whether robustness to relation-word "
        "paraphrase extends to the object names grounded in vision. Tasks 2 and 8 have no substitutable noun.",
        lambda t: f"pick up the black bowl {synonymize(TASKS[t][2])} and place it on the plate", task_ids=syn_ids)
    add("translate_zh", "multilingual",
        "The default instruction in Simplified Chinese. All backbones saw multilingual pretraining; fine-tuning "
        "was English only.",
        lambda t: f"拿起{ZH[t]}黑色碗，把它放到盘子上")
    add("translate_es", "multilingual", "The default instruction in Spanish.",
        lambda t: f"toma el tazón negro {ES[t]} y colócalo en el plato")
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in PROMOTED:
        (OUT / f"{name}.json").unlink(missing_ok=True)
    for cand in build():
        if cand["condition"] in PROMOTED:
            print(f"(skipped {cand['condition']}: promoted to {PROMOTED[cand['condition']]})")
            continue
        path = OUT / f"{cand['condition']}.json"
        path.write_text(json.dumps(cand, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{path.relative_to(ROOT)}  ({len(cand['task_ids'])} tasks, swap_target={cand['swap_target']})")


if __name__ == "__main__":
    main()
