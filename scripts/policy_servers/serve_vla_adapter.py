#!/usr/bin/env python3
"""Policy server around VLA-Adapter's own LIBERO eval code, speaking openpi's websocket protocol.

Why a server: VLA-Adapter ships its own `prismatic` package (Qwen2.5 backbone, different action-token
constants) that cannot share a process with this repo's openvla fork. The server runs inside the eval
image with the VLA-Adapter checkout first on PYTHONPATH and calls its own `initialize_model` / `get_action`,
so preprocessing and action decoding are the authors' code, not a port. The eval side is
`run_libero_eval.py --model_family openpi --policy_obs raw --replan_steps 8`.

Request (msgpack-numpy):
  observation/image, observation/wrist_image : 256x256x3 uint8, rotated 180 degrees (raw LIBERO renders)
  observation/state                          : 8-d (eef pos, axis-angle, gripper qpos)
  prompt                                     : the task sentence
Reply:
  actions : (K, 7) float array in the environment's convention (gripper -1 = open, +1 = close), K <= 8

VLA-Adapter's loader rewrites a local checkpoint directory (config.json auto_map, and it overwrites
modeling_prismatic.py / configuration_prismatic.py with the repo's versions after making .back files),
and finds those repo files under ./prismatic, so run with cwd = the checkout and give --checkpoint a
private working copy (small files copied, weights symlinked). scripts/queue_vla_adapter.sh does all of
this for every slot:
  docker run --rm --network host --gpus device=0 -e PYTHONPATH=/vla_adapter -w /vla_adapter \\
      -v $HOME/vla_adapter:/vla_adapter:ro -v <this dir>:/serve:ro -v <work copy>:<work copy> \\
      -v <real ckpt dir>:<real ckpt dir>:ro openvla-libero:blackwell \\
      python3 /serve/serve_vla_adapter.py --checkpoint <work copy> --port 8000 --use_pro_version 1
"""
import argparse
import logging
import sys
import time
import traceback

import numpy as np


def _stub_client_only_imports() -> None:
    """VLA-Adapter's utils import two packages only its HTTP *client* path uses (msgpack_numpy, json_numpy; the
    latter is also patched at import). The eval image has neither, and the server never touches them."""
    import types

    for name in ("msgpack_numpy", "json_numpy"):
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            stub.patch = lambda *a, **k: None
            stub.encode = stub.decode = lambda *a, **k: None
            sys.modules[name] = stub


def build_policy(checkpoint: str, suite: str, pro: bool):
    # Imported late: PYTHONPATH must put the VLA-Adapter checkout first, and argv must mention "libero"
    # (its constants module picks the LIBERO constants from sys.argv).
    _stub_client_only_imports()
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
    from experiments.robot.openvla_utils import resize_image_for_policy
    from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action

    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name=suite,
        use_pro_version=pro,
        save_version="vla-adapter",
    )
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    logging.info("loaded %s (unnorm_key=%s, pro=%s)", type(model).__name__, cfg.unnorm_key, cfg.use_pro_version)

    def infer(obs: dict) -> dict:
        observation = {
            "full_image": resize_image_for_policy(np.ascontiguousarray(obs["observation/image"]), 224),
            "wrist_image": resize_image_for_policy(np.ascontiguousarray(obs["observation/wrist_image"]), 224),
            "state": np.asarray(obs["observation/state"], dtype=np.float64),
        }
        actions = get_action(
            cfg, model, observation, str(obs["prompt"]),
            processor=processor, action_head=action_head, proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector, use_film=cfg.use_film, use_minivlm=cfg.use_minivlm,
        )
        out = []
        for a in actions:
            a = normalize_gripper_action(np.asarray(a, dtype=np.float64).copy(), binarize=True)
            out.append(invert_gripper_action(a))  # same post-processing as the official LIBERO loop (family "openvla")
        return {"actions": np.stack(out)}

    meta = {"policy": "vla-adapter", "checkpoint": checkpoint, "unnorm_key": cfg.unnorm_key,
            "use_pro_version": bool(cfg.use_pro_version), "chunk": 8}
    return infer, meta


def serve(infer, meta, host: str, port: int) -> None:
    import websockets.exceptions
    import websockets.sync.server as ws_server
    from openpi_client import msgpack_numpy

    packer = msgpack_numpy.Packer()

    def handler(conn):
        logging.info("connection from %s", conn.remote_address)
        conn.send(packer.pack(meta))
        while True:
            try:
                msg = conn.recv()
            except websockets.exceptions.ConnectionClosed:
                break
            t0 = time.time()
            try:
                reply = infer(msgpack_numpy.unpackb(msg))
            except Exception:  # noqa: BLE001 -- keep serving; the client sees the connection close
                logging.error("infer failed:\n%s", traceback.format_exc())
                raise
            reply["server_timing"] = {"infer_ms": (time.time() - t0) * 1000}
            conn.send(packer.pack(reply))

    with ws_server.serve(handler, host, port, compression=None, max_size=None) as server:
        logging.info("serving on %s:%d", host, port)
        server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--task_suite_name", default="libero_spatial")  # also keeps "libero" in sys.argv
    ap.add_argument("--use_pro_version", type=int, required=True, help="1 for the *-Pro checkpoints, 0 otherwise")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    infer, meta = build_policy(args.checkpoint, args.task_suite_name, bool(args.use_pro_version))
    serve(infer, meta, args.host, args.port)


if __name__ == "__main__":
    main()
