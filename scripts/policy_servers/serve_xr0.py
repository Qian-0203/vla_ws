#!/usr/bin/env python3
"""Xiaomi-Robotics-0 (XiaomiRobotics/Xiaomi-Robotics-0-LIBERO) policy server, speaking openpi's websocket protocol.

Replicates the authors' LIBERO eval (eval_libero/main.py + deploy/server.py in github.com/XiaomiRobotics/
Xiaomi-Robotics-0 @ 7c31660): the base and left-wrist renders (256x256, rotated 180 degrees, no resize -- the
processor does its own), an 8-d state (eef pos, axis-angle, gripper qpos) zero-padded to 32, robot type
"libero_all" (one model for the four suites), the instruction capitalized, a per-query flow-matching seed derived
from a hash of the inputs, then "." appended. The model returns a 10-step chunk (the action mask is 10 x 7) and the
authors execute all of it (replan_steps 10), so the
eval side is `run_libero_eval.py --model_family openpi --policy_obs raw --replan_steps 10`.

Request (msgpack-numpy): observation/image, observation/wrist_image (256x256x3 uint8), observation/state (8,), prompt.
Reply: {"actions": (10, 7)} in the environment's convention (the authors step the env with them directly).

  docker run --rm --network host --gpus device=0 -v <this dir>:/serve:ro -v <ckpt>:<ckpt>:ro openvla-libero:xr0 \\
      python3 /serve/serve_xr0.py --model_path <ckpt> --port 8800
"""
import argparse
import hashlib
import json
import logging
import sys
import time
import traceback

import numpy as np

ROBOT_TYPE = "libero_all"


def hash_data_to_seed(data, max_bytes=4):
    """Verbatim from the authors' eval_libero/main.py: a stable seed from the query's inputs."""
    from PIL import Image

    def custom_encoder(obj):
        if isinstance(obj, np.ndarray):
            return {"__type__": "numpy", "dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tobytes().hex()}
        if isinstance(obj, Image.Image):
            img_hash = hashlib.md5(obj.tobytes()).hexdigest()
            return {"__type__": "PIL.Image", "mode": obj.mode, "size": obj.size, "content_hash": img_hash}
        if isinstance(obj, set):
            return sorted(list(obj))
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    json_str = json.dumps(data, default=custom_encoder, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    seed_int = int(hashlib.sha256(json_str.encode("utf-8")).hexdigest(), 16)
    if max_bytes > 0:
        seed_int = seed_int % (2 ** (8 * max_bytes))
    return seed_int


def build(model_path: str, attn: str):
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    if attn == "sdpa":
        # The remote code's two outer classes (MiBoTForActionGeneration, DiT) do not declare `_supports_sdpa`, so
        # transformers refuses SDPA. The Qwen3-VL backbone does support it, and the DiT attention always calls
        # F.scaled_dot_product_attention whatever the setting, so declaring it changes no computation.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module("modeling_mibot.MiBoTForActionGeneration", model_path)
        cls._supports_sdpa = True
        sys.modules[cls.__module__].DiT._supports_sdpa = True
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation=attn,
                                      dtype=torch.bfloat16).cuda().to(torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    action_mask = processor.get_action_mask(ROBOT_TYPE).to(model.device, model.dtype)
    logging.info("loaded %s (attn=%s)", model_path, attn)

    @torch.no_grad()
    def infer(req: dict) -> dict:
        s = np.asarray(req["observation/state"], dtype=np.float64)
        state = np.concatenate([s[:8], np.zeros(24)])
        data = {
            "task_id": ROBOT_TYPE,
            "state": state,
            "base": Image.fromarray(np.ascontiguousarray(req["observation/image"])),
            "wrist_left": Image.fromarray(np.ascontiguousarray(req["observation/wrist_image"])),
            "language": str(req["prompt"]).capitalize(),
        }
        data["seed"] = hash_data_to_seed(data)
        data["language"] = data["language"] + "."
        # deploy/server.py from here on.
        instruction = (
            "<|im_start|>user\nThe following observations are captured from multiple views.\n"
            "# Base View\n<|vision_start|><|image_pad|><|vision_end|>\n"
            "# Left-Wrist View\n<|vision_start|><|image_pad|><|vision_end|>\n"
            f"Generate robot actions for the task:\n{data['language']} /no_cot<|im_end|>\n"
            "<|im_start|>assistant\n<cot></cot><|im_end|>\n"
        )
        vl_inputs = processor(text=[instruction], images=[data["base"], data["wrist_left"]], videos=None,
                              padding=True, return_tensors="pt").to(model.device)
        data.update(vl_inputs)
        data["action_mask"] = action_mask
        data["state"] = torch.from_numpy(data["state"]).to(model.device, model.dtype).view(1, 1, -1)
        outputs = model(**data)
        action = processor.decode_action(outputs.actions, robot_type=ROBOT_TYPE).cpu()
        # main.py: client(...)[0, :, :-1], then [:, 0:7].
        actions = action[0, :, :-1][:, 0:7].numpy().astype(np.float64)
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
    ap.add_argument("--attn", default="sdpa", help="the authors use flash_attention_2 (no Blackwell build)")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    infer = build(args.model_path, args.attn)
    meta = {"policy": "xiaomi-robotics-0", "model_path": args.model_path, "robot_type": ROBOT_TYPE,
            "attn": args.attn, "chunk": 10}
    serve(infer, meta, args.host, args.port)


if __name__ == "__main__":
    main()
