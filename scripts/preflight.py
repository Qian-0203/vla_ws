#!/usr/bin/env python3
"""
preflight.py

Lightweight, host-side sanity check before running eval on a new machine --
no torch/CUDA import needed (those only exist inside the Docker image), so
this runs with the system python. Reports GPU/VRAM/CUDA driver, docker +
image availability, and whether the checkpoint looks complete and fits in
VRAM at the requested precision. Does not run the model.

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --checkpoint /path/to/checkpoint --image openvla-libero:blackwell
"""
import argparse
import glob
import json
import os
import shutil
import subprocess


def run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def check_gpu():
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader"])
    if out is None:
        print("[FAIL] nvidia-smi not found or no GPU visible.")
        return None
    lines = out.splitlines()
    print(f"[OK] {len(lines)} GPU(s) detected:")
    gpus = []
    for line in lines:
        name, mem, cc, driver = [x.strip() for x in line.split(",")]
        mem_gb = float(mem.replace(" MiB", "")) / 1024
        print(f"       {name} | {mem_gb:.1f} GB VRAM | compute_cap {cc} | driver {driver}")
        gpus.append({"name": name, "vram_gb": mem_gb, "compute_cap": cc})
    return gpus


def check_docker(image_name):
    if shutil.which("docker") is None:
        print("[FAIL] docker not found on PATH.")
        return
    print("[OK] docker found:", run(["docker", "--version"]))
    images = run(["docker", "images", image_name, "--format", "{{.Repository}}:{{.Tag}}"])
    if images:
        print(f"[OK] image '{image_name}' is built locally.")
    else:
        print(f"[WARN] image '{image_name}' not found locally -- build it first (see docker/openvla_libero/).")


def check_checkpoint(checkpoint, gpus, load_in_4bit, load_in_8bit):
    if not checkpoint or not os.path.isdir(checkpoint):
        print(f"[WARN] checkpoint path does not exist locally: {checkpoint}")
        print("       (fine if it only exists on the server -- this check is host-side)")
        return
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print(f"[WARN] no model.safetensors.index.json under {checkpoint}; can't verify shard completeness.")
        return
    with open(index_path) as f:
        index = json.load(f)
    expected = int(index.get("metadata", {}).get("total_size", 0))
    actual = sum(os.path.getsize(p) for p in glob.glob(os.path.join(checkpoint, "model-*.safetensors")))
    ok = expected and actual >= expected * 0.95
    print(f"[{'OK' if ok else 'FAIL'}] checkpoint shards: {actual / 1e9:.1f} GB found vs {expected / 1e9:.1f} GB expected.")

    if not gpus:
        return
    vram_gb = max(g["vram_gb"] for g in gpus)
    if load_in_4bit:
        est_gb = expected / 1e9 / 4
    elif load_in_8bit:
        est_gb = expected / 1e9 / 2
    else:
        est_gb = expected / 1e9  # checkpoint is stored bf16; full precision load is ~same size
    headroom_gb = 2.0  # rough allowance for activations/KV cache
    fits = est_gb + headroom_gb <= vram_gb
    mode = "4-bit" if load_in_4bit else "8-bit" if load_in_8bit else "bf16"
    print(f"[{'OK' if fits else 'WARN'}] estimated {mode} weight size {est_gb:.1f} GB + ~{headroom_gb:.0f} GB headroom "
          f"vs {vram_gb:.1f} GB VRAM on largest GPU.")
    if not fits and not (load_in_4bit or load_in_8bit):
        print("       -> pass --load_in_4bit True (see config/laptop.env) to fit on a small GPU.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT"))
    parser.add_argument("--image", default=os.environ.get("IMAGE_NAME", "openvla-libero:cuda12.1"))
    parser.add_argument("--load_in_4bit", action="store_true", default=os.environ.get("LOAD_IN_4BIT") == "True")
    parser.add_argument("--load_in_8bit", action="store_true", default=os.environ.get("LOAD_IN_8BIT") == "True")
    args = parser.parse_args()

    print("=== GPU ===")
    gpus = check_gpu()
    print("\n=== Docker ===")
    check_docker(args.image)
    print("\n=== Checkpoint ===")
    check_checkpoint(args.checkpoint, gpus, args.load_in_4bit, args.load_in_8bit)


if __name__ == "__main__":
    main()
