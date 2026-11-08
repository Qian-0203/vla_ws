#!/usr/bin/env python3
"""Serve the RoboCasa365 π0.5 checkpoint (robocasa-benchmark/openpi, config pi05_pretrain_human300) over websocket.

Same as the fork's scripts/serve_policy.py with `policy:checkpoint`, except that the config's data_dirs are cleared.
With data_dirs set, LeRobotRobocasaDataConfig.create() first tries to merge norm stats from the Human300 training
datasets (not downloaded here) and fails; with them cleared, create_trained_policy() loads the checkpoint's own
assets/norm_stats.json, which is what the fork intends for evaluation ("norm stats are assumed to be located in
checkpoint assets folder").

Usage (server venv = the fork's .venv): python pi05_server.py --ckpt <.../75000> --port 8000
"""
import argparse
import dataclasses
import logging

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="pi05_pretrain_human300")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    cfg = _config.get_config(a.config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dirs=None))
    policy = policy_config.create_trained_policy(cfg, a.ckpt)
    logging.info("serving %s on port %d", a.ckpt, a.port)
    websocket_policy_server.WebsocketPolicyServer(policy, host="0.0.0.0", port=a.port,
                                                  metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
