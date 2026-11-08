#!/usr/bin/env python3
"""Gate + default cells for π0.5 on RoboCasa365 (2026-11-04), as JSON lines for run_lang_eval.py --policy pi05.

π0.5 publishes only its Atomic-Seen mean (39.6%, robocasa 1.0.0), no per-task numbers, so the gate is on the mean:
all 18 Atomic-Seen tasks x 10 episodes, pass if the mean is within 7 pts of 39.6. The four screen tasks run 50
episodes (their default for the paired language cells; kept only if default >= 50%), split in two cells of 25 so that
workers stay balanced. The gate mean uses each task's first 10 seeds.

Seeds follow Xiaomi-Robotics-1's published rule, 7 + 50 * task_index + episode, with task_index the row order of
docs/robocasa/published_per_task_sr.csv (checked: NavigateKitchen 207, OpenStandMixerHead 357,
PickPlaceCounterToCabinet 407, TurnOffStove 707 match pilot_seeds.json).
Usage: python pi05_gate_cells.py [--smoke] > cells.jsonl
"""
import json
import sys

TARGET50_ATOMIC_SEEN = [  # first 18 rows of published_per_task_sr.csv (split atomic_seen), in order
    "CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor", "CoffeeSetupMug", "NavigateKitchen", "OpenCabinet",
    "OpenDrawer", "OpenStandMixerHead", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter", "SlideDishwasherRack",
    "TurnOffStove", "TurnOnElectricKettle", "TurnOnMicrowave", "TurnOnSinkFaucet",
]
FAMILY = {"NavigateKitchen": "nav", "PickPlaceCounterToCabinet": "pnp", "TurnOffStove": "stove",
          "OpenStandMixerHead": "other"}
GATE_N, DEFAULT_N = 10, 50


def seeds(task, n):
    i = TARGET50_ATOMIC_SEEN.index(task)
    return [7 + 50 * i + e for e in range(n)]


def cell(name, task, s):
    return dict(cell=name, env=task, env_kwargs={}, horizon_task=task, family=FAMILY.get(task, "other"),
                condition="default", seeds=s, n=len(s), require_valid=False)


def cells(smoke=False):
    if smoke:
        return [cell("smoke_pnp", "PickPlaceCounterToCabinet", seeds("PickPlaceCounterToCabinet", 1)),
                cell("smoke_nav", "NavigateKitchen", seeds("NavigateKitchen", 1))]
    out = []
    for t in TARGET50_ATOMIC_SEEN:
        if t in FAMILY:
            s = seeds(t, DEFAULT_N)
            out += [cell(f"default_{t}_a", t, s[:25]), cell(f"default_{t}_b", t, s[25:])]
        else:
            out.append(cell(f"gate_{t}", t, seeds(t, GATE_N)))
    return out


if __name__ == "__main__":
    assert seeds("NavigateKitchen", 1) == [207] and seeds("TurnOffStove", 1) == [707]
    for c in cells("--smoke" in sys.argv):
        print(json.dumps(c))
