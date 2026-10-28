#!/usr/bin/env python3
"""RoboLab pi0-family evaluation with robolab_lang task files.

Same as RoboLab's `policies/pi0_family/run.py` (same client, same eval loop, same CLI flags), except
that `--lang-task` takes task *files* from robolab_lang/tasks/ and registers them by path, then runs
them by class name. RoboLab's own `--task <name>` still works for official tasks, so one runner covers
both the gate runs and the language conditions.

Run inside the RoboLab image, from the RoboLab checkout:

    /workspace/isaaclab/_isaac_sim/python.sh /lang/run_pi0.py --policy pi05 --headless \
        --lang-task /lang/tasks/LangMustardInLeftBinTask.py --instruction-type p3_neg_not \
        --num-envs 10 --num-runs 2 --output-folder-name <name>
"""

import argparse
import os
import sys
import traceback

import cv2  # noqa: F401  (RoboLab: import before isaaclab)
from isaaclab.app import AppLauncher

sys.path.insert(0, os.getcwd())  # RoboLab checkout, so `policies` imports as in run.py

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]  # as in policies/pi0_family/run.py

parser = argparse.ArgumentParser(description="RoboLab pi0-family eval with robolab_lang tasks")
parser.add_argument("--policy", choices=PI0_VARIANTS, default="pi05")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000)
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None)
parser.add_argument("--lang-task", nargs="+", default=None,
                    help="robolab_lang task files (class name = file stem)")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask

if args_cli.lang_task:
    paths = [os.path.abspath(p) for p in args_cli.lang_task]
    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=paths)
    args_cli.task = [os.path.splitext(os.path.basename(p))[0] for p in paths]
else:
    auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args):
    kwargs = dict(remote_host=args.remote_host, remote_port=args.remote_port,
                  open_loop_horizon=args.open_loop_horizon, policy_variant=args.policy)
    return Pi0DroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})


def main():
    run_evaluation(args_cli, policy=args_cli.policy, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[robolab_lang] Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
