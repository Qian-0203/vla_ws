#!/usr/bin/env python3
"""Fill instruction + scene_hash into episodes.jsonl rows written by a client that does not log them (upstream XR-1).

Replays each task's seeds in the client's own order in one env per task (the upstream client builds one env per task
and resets it seed by seed), because ~3% of RoboCasa seeds give a history-dependent scene (plan §3.4). Rows get
`instruction_source` / `scene_hash_source` = "reconstructed". No policy.

Usage (sim venv): python reconstruct_scenes.py --episodes <episodes.jsonl> [--tasks A,B]   (rewrites the file)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()
    import gymnasium as gym
    import robocasa  # noqa: F401
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    from scene_probe import kitchen_of, scene_hash
    rows = [json.loads(l) for l in open(a.episodes)]
    tasks = a.tasks.split(",") if a.tasks else sorted({r["task"] for r in rows})
    for task in tasks:
        env = gym.make(f"robocasa/{task}", split="pretrain", seed=7)
        for r in sorted((r for r in rows if r["task"] == task), key=lambda r: r["episode"]):
            obs, _ = env.reset(seed=r["seed"])
            r["instruction"] = str(obs["annotation.human.task_description"])
            r["scene_hash"] = scene_hash(kitchen_of(env))
            r["instruction_source"] = r["scene_hash_source"] = "reconstructed (same seed and order, no policy)"
        env.close()
        print(task, "done", flush=True)
    with open(a.episodes + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(a.episodes + ".tmp", a.episodes)


if __name__ == "__main__":
    main()
