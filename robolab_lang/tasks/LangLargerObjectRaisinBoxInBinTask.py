"""LargerObjectRaisinBoxInBinTask with the language-stress instruction keys (robolab_lang; condition names follow the ARR draft's tests).

Scene, terminations and subtasks are the official ones, unchanged; the official default / vague /
specific strings are copied from the base class. Select a key with --instruction-type <key>.
`blank` is "." because RoboLab rejects empty instructions; `no_target` drops the words that pick the goal.
"""

from dataclasses import dataclass

import robolab.tasks.benchmark.bigger_object as _official


@dataclass
class LangLargerObjectRaisinBoxInBinTask(_official.LargerObjectRaisinBoxInBinTask):
    instruction = {
        **_official.LargerObjectRaisinBoxInBinTask.instruction,
        "blank": ".",
        "no_target": "Place the object in the grey bin.",
        "negation": "Place the object that is not the butter in the grey bin.",
        "distractor_mention": "Place the larger object in the grey bin, not the smaller one.",
        "irrelevant_content": "Place the larger object in the grey bin, and let go gently.",
    }
