#!/usr/bin/env python3
"""GR00T N1.5 policy server around NVIDIA's own `Gr00tPolicy`, speaking openpi's websocket protocol.

NVIDIA's LIBERO eval (examples/Libero/eval/run_libero_eval.py in the n1.5-release tag of Isaac-GR00T) queries the
policy at every environment step and executes only the first of the 16 predicted actions. The eval side here is
therefore the generic client: `run_libero_eval.py --model_family openpi --policy_obs raw --replan_steps 1`.
This server replicates that script's observation packing and its action post-processing; the model, the data
config (examples.Libero.custom_data_config:LiberoDataConfig), the embodiment tag (new_embodiment) and the 8
denoising steps are the ones in the checkpoint's README.

Request (msgpack-numpy): observation/image, observation/wrist_image (256x256x3 uint8, both rotated 180 degrees),
observation/state (8,) = eef pos (3), axis-angle (3), gripper qpos (2), prompt (str).
Reply: {"actions": (16, 7)} in the environment's convention (gripper binarized to -1 / +1).

  docker run --rm --network host --gpus device=0 -w /gr00t -e PYTHONPATH=/gr00t -v $HOME/Isaac-GR00T:/gr00t:ro \\
      -v <this dir>:/serve:ro -v <ckpt>:<ckpt>:ro openvla-libero:gr00t \\
      python3 /serve/serve_gr00t.py --model_path <ckpt> --port 8600
"""
import argparse
import logging
import sys
import time
import traceback

import numpy as np

STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _force_sdpa() -> None:
    """The Eagle backbone's config asks for flash-attention-2, which has no build for Blackwell; use SDPA.
    Numerically the same attention, different kernel."""
    import transformers

    original = transformers.AutoConfig.from_pretrained.__func__

    def patched(cls, *args, **kwargs):
        cfg = original(cls, *args, **kwargs)
        for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None)):
            if c is not None:
                try:
                    c._attn_implementation = "sdpa"
                except Exception:  # noqa: BLE001 -- configs without the property are left as they are
                    pass
        return cfg

    transformers.AutoConfig.from_pretrained = classmethod(patched)


def build(model_path: str, data_config: str, embodiment_tag: str, denoising_steps: int):
    _force_sdpa()
    from gr00t.experiment.data_config import load_data_config
    from gr00t.model.policy import Gr00tPolicy

    dc = load_data_config(data_config)
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(),
        embodiment_tag=embodiment_tag,
        denoising_steps=denoising_steps,
    )
    logging.info("loaded Gr00tPolicy from %s (data_config=%s, steps=%d)", model_path, data_config, denoising_steps)

    def infer(req: dict) -> dict:
        s = np.asarray(req["observation/state"], dtype=np.float64)
        obs = {
            "video.image": np.ascontiguousarray(req["observation/image"])[None],
            "video.wrist_image": np.ascontiguousarray(req["observation/wrist_image"])[None],
            "annotation.human.action.task_description": [str(req["prompt"])],
            "state.gripper": s[6:8][None],
        }
        for i, k in enumerate(STATE_KEYS):
            obs[f"state.{k}"] = np.array([[s[i]]])
        chunk = policy.get_action(obs)
        horizon = np.asarray(chunk["action.x"]).shape[0]
        actions = np.stack(
            [np.asarray(chunk[f"action.{k}"]).reshape(horizon, -1)[:, 0] for k in ACTION_KEYS], axis=1
        ).astype(np.float64)
        # NVIDIA's normalize_gripper_action(binarize=True): [0, 1] -> [+1, -1], then the sign.
        actions[:, -1] = np.sign(1.0 - 2.0 * actions[:, -1])
        return {"actions": actions}

    return infer


def serve(infer, meta: dict, host: str, port: int) -> None:
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
            except Exception:  # noqa: BLE001 -- log, then let the client see the connection close
                logging.error("infer failed:\n%s", traceback.format_exc())
                raise
            reply["server_timing"] = {"infer_ms": (time.time() - t0) * 1000}
            conn.send(packer.pack(reply))

    with ws_server.serve(handler, host, port, compression=None, max_size=None) as server:
        logging.info("serving on %s:%d", host, port)
        server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    ap.add_argument("--denoising_steps", type=int, default=8)
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    infer = build(args.model_path, args.data_config, args.embodiment_tag, args.denoising_steps)
    meta = {"policy": "gr00t-n1.5", "model_path": args.model_path, "data_config": args.data_config,
            "denoising_steps": args.denoising_steps, "chunk": 16}
    serve(infer, meta, args.host, args.port)


if __name__ == "__main__":
    main()
