"""LargerObjectRaisinBoxInBinTask with the language-stress instruction keys (robolab_lang, plan §4.2).

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
    }
