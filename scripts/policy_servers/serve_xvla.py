#!/usr/bin/env python3
"""X-VLA policy server (the model only), speaking openpi's websocket protocol.

Counterpart of experiments/robot/xvla_utils.py in the openvla fork, which holds the protocol state (the proprio
chain, the queued 30-step plan, the 6D -> axis-angle conversion). This server is stateless and mirrors the
`/act` handler of X-VLA's own FastAPI server (models/modeling_xvla.py): same processor, same
`generate_actions`, no changes. It needs X-VLA's checkout (model code) and its pinned dependencies, hence its
own image (docker/openvla_libero/Dockerfile.xvla).

Request (msgpack-numpy): image0 / image1 (HxWx3 uint8), language_instruction (str), proprio (20,), domain_id (int),
steps (int). Reply: {"action": (30, 10) float32 = pos(3), rotation-6D(6), gripper(1)}.

  docker run --rm --network host --gpus device=0 -w /xvla -e PYTHONPATH=/xvla -v $HOME/X-VLA:/xvla:ro \\
      -v <this dir>:/serve:ro -v <ckpt>:<ckpt>:ro openvla-libero:xvla \\
      python3 /serve/serve_xvla.py --model_path <ckpt> --port 8500
"""
import argparse
import logging
import sys
import time
import traceback

import numpy as np


def build(model_path: str, processor_path: str | None, lora_path: str | None):
    import torch
    from PIL import Image

    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = XVLAProcessor.from_pretrained(processor_path or model_path)
    model = XVLA.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float32).to(device).to(torch.float32)
    if lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora_path, torch_dtype=torch.float32).to(device)
    model.eval()
    logging.info("loaded %s on %s (lora=%s)", type(model).__name__, device, lora_path)

    @torch.no_grad()
    def infer(req: dict) -> dict:
        images = [Image.fromarray(np.asarray(req[k]).astype(np.uint8)) for k in ("image0", "image1", "image2") if k in req]
        if not images:
            raise ValueError("no image in the request")
        inputs = processor(images, str(req["language_instruction"]))
        proprio = torch.as_tensor(np.asarray(req["proprio"]))
        domain_id = torch.tensor([int(req["domain_id"])], dtype=torch.long)
        param = next(model.parameters())

        def to_model(t):
            t = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
            return t.to(device=param.device, dtype=param.dtype) if t.is_floating_point() else t.to(device=param.device)

        inputs = {k: to_model(v) for k, v in inputs.items()}
        inputs.update({"proprio": to_model(proprio.unsqueeze(0)), "domain_id": domain_id.to(param.device)})
        action = model.generate_actions(**inputs, steps=int(req.get("steps", 10))).squeeze(0).float().cpu().numpy()
        return {"action": action.astype(np.float32)}

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
    ap.add_argument("--processor_path", default=None)
    ap.add_argument("--lora_path", default=None)
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    infer = build(args.model_path, args.processor_path, args.lora_path)
    meta = {"policy": "x-vla", "model_path": args.model_path, "lora_path": args.lora_path or "", "chunk": 30}
    serve(infer, meta, args.host, args.port)


if __name__ == "__main__":
    main()
