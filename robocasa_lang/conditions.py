#!/usr/bin/env python3
"""Condition text rules for the RoboCasa365 language stress test (docs/robocasa/plan.md §4).

Pure Python: no simulator, no model. Each family takes the scene slots that the task's own get_ep_meta() already
computes and returns the instruction for one condition. The env-side override (plan §2.2) calls `make()` once per
episode; it must never draw from env.rng.

Families and slots:
  pnp    target, distractor (object category words from get_obj_lang), src, dst   e.g. PickPlaceCounterToCabinet
  nav    target, distractor (fixture nat_lang)                                     NavigateKitchen
  stove  target, distractor (knob names, e.g. "front_left")                        TurnOffStove(Pair)

Conditions that need two candidates in the scene (negation_*, contrast, mirror) require `distractor`.
Usage: python3 robocasa_lang/conditions.py   (runs the self-checks below)
"""

PNP = {
    "default": "Pick the {T} from the {src} and place it in the {dst}.",
    "target_swap": "Pick the {D} from the {src} and place it in the {dst}.",
    "no_target": "Pick the object from the {src} and place it in the {dst}.",
    "blank": "",
    "negation_only": "Pick the object that is not the {D} and place it in the {dst}.",
    # negates the usual object; scored on the distractor (PnPCounterToCabinetSwap), as LIBERO's negation_swap
    "negation_swap": "Pick the object that is not the {T} and place it in the {dst}.",
    "negation_except": "Pick any object on the {src} except the {D} and place it in the {dst}.",
    "negation_other": "Pick the other object, not the {D}, and place it in the {dst}.",
    "positive_contrast": "Pick the {T}, not the {D}, from the {src} and place it in the {dst}.",
    "length_control_positive": "Pick the {T}, {F}, from the {src} and place it in the {dst}.",
    "negative_contrast": "Do not pick the {D}; pick the {T} from the {src} and place it in the {dst}.",
    "length_control_negative": "{Fcap}; pick the {T} from the {src} and place it in the {dst}.",
    "paraphrase_syntactic": "From the {src}, pick the {T} and place it in the {dst}.",
    "verb_synonym": "Fetch the {T} off the {src} and stow it in the {dst}.",  # fetch/stow: not in Human300 templates
}

NAV = {
    "default": "Navigate to the {T}.",
    "target_swap": "Navigate to the {D}.",
    "no_target": "Navigate to the fixture.",
    "blank": "",
    "positive_contrast": "Navigate to the {T}, not the {D}.",
    "length_control_positive": "Navigate to the {T}, {F}.",
    "paraphrase_syntactic": "To the {T}, navigate.",
    "verb_synonym": "Go over to the {T}.",
}

STOVE = {
    "default": "Turn off the {T} burner of the stove.",
    "target_swap": "Turn off the {D} burner of the stove.",
    "no_target": "Turn off the burner of the stove.",
    "blank": "",
    "negation_only": "Turn off the burner that is not the {D} one.",
    "negation_other": "Turn off the other burner, not the {D} one.",
    "positive_contrast": "Turn off the {T} burner, not the {D} one, of the stove.",
    "length_control_positive": "Turn off the {T} burner, {F}, of the stove.",
    "paraphrase_syntactic": "Of the stove, turn off the {T} burner.",
    "verb_synonym": "Switch off the {T} burner on the stove.",
}

FAMILIES = {"pnp": PNP, "nav": NAV, "stove": STOVE}

# contrast condition -> its length control (plan §4, P5)
LENGTH_PAIRS = {
    "positive_contrast": "length_control_positive",
    "negative_contrast": "length_control_negative",
}


# Content-free fillers by word count. A length control replaces the contrast clause ("not the {D}", "Do not pick
# the {D}") with the filler of the same word count, so the gap follows the distractor's own length.
FILLERS = {
    2: "slowly now",
    3: "slowly and carefully",
    4: "slowly and very carefully",
    5: "slowly, steadily and very carefully",
    6: "slowly, steadily and very carefully now",
    7: "move slowly, steadily and very carefully now",
    8: "please move slowly, steadily and very carefully now",
}


def _words(slot):
    return slot.replace("_", " ")


def _contrast_clause_len(family, condition, distractor):
    d = len(_words(distractor).split())
    if condition == "length_control_positive":
        return d + (3 if family == "stove" else 2)  # "not the {D} one" / "not the {D}"
    if condition == "length_control_negative":
        return d + 4  # "Do not pick the {D}"
    return None


def make(family, condition, target, distractor=None, src="counter", dst="cabinet"):
    """Instruction text for one (family, condition). Raises if the condition needs a distractor and none is given."""
    tpl = FAMILIES[family][condition]
    if "{D}" in tpl and not distractor:
        raise ValueError(f"{family}/{condition} needs a distractor")
    if distractor and _words(distractor) == _words(target) and "{D}" in tpl:
        raise ValueError(f"{family}/{condition}: target and distractor have the same words ({target!r})")
    filler = ""
    n = _contrast_clause_len(family, condition, distractor) if "{F" in tpl else None
    if n is not None:
        if not distractor:
            raise ValueError(f"{family}/{condition} needs the distractor to match its length")
        if n not in FILLERS:
            raise ValueError(f"no {n}-word filler for {family}/{condition}")
        filler = FILLERS[n]
    return tpl.format(T=_words(target), D=_words(distractor or ""), src=src, dst=dst,
                      F=filler, Fcap=filler[:1].upper() + filler[1:])


def length_gap(family, contrast, **slots):
    """Word-count difference between a contrast prompt and its length control (must be <= 1).

    The token-count check (within 2 tokens under each policy's own tokenizer) needs the tokenizers and runs in the
    sim/policy environment, not here.
    """
    a = make(family, contrast, **slots).split()
    b = make(family, LENGTH_PAIRS[contrast], **slots).split()
    return abs(len(a) - len(b))


def _self_check():
    pnp = dict(target="mug", distractor="apple", src="counter", dst="cabinet")
    assert make("pnp", "default", **pnp) == "Pick the mug from the counter and place it in the cabinet."
    assert make("pnp", "target_swap", **pnp) == "Pick the apple from the counter and place it in the cabinet."
    assert make("pnp", "negation_swap", **pnp) == "Pick the object that is not the mug and place it in the cabinet."
    assert make("pnp", "blank", **pnp) == ""
    for fam, slots in [("pnp", pnp),
                       ("nav", dict(target="sink", distractor="fridge")),
                       ("stove", dict(target="front_left", distractor="rear_right"))]:
        for cond in FAMILIES[fam]:
            text = make(fam, cond, **slots)
            # the negated / contrasted / swapped condition must name the distractor; the others must not
            names_d = _words(slots["distractor"]) in text
            assert names_d == ("{D}" in FAMILIES[fam][cond]), (fam, cond, text)
        for contrast in LENGTH_PAIRS:
            if contrast in FAMILIES[fam]:
                gap = length_gap(fam, contrast, **slots)
                assert gap <= 1, (fam, contrast, gap)
    try:
        make("pnp", "negation_only", target="mug")
        raise AssertionError("missing distractor accepted")
    except ValueError:
        pass
    try:
        make("pnp", "target_swap", target="mug", distractor="mug")
        raise AssertionError("same-word distractor accepted")
    except ValueError:
        pass
    print("conditions.py self-checks passed:",
          ", ".join(f"{f}={len(c)}" for f, c in FAMILIES.items()), "conditions")


if __name__ == "__main__":
    _self_check()
