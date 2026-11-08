#!/usr/bin/env python3
"""Run Xiaomi-Robotics-1's own policy server (deploy/server.py) with a chosen attention implementation.

The upstream server hard-codes attn_implementation="flash_attention_2"; flash-attn is unverified on Blackwell
(sm_120). With sdpa, transformers refuses to load the model because the top-level class does not declare
_supports_sdpa (first smoke, 2026-10-21); see _allow_sdpa below for why declaring it is safe. This wrapper leaves the upstream file
untouched: it overrides the keyword in AutoModel.from_pretrained, then executes deploy/server.py as __main__ with the
remaining arguments.

Usage: python xr1_server.py --repo <Xiaomi-Robotics-1 checkout> [--attn sdpa] -- <deploy/server.py args>
"""
import argparse
import os
import runpy
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--attn", default=os.environ.get("XR1_ATTN", "sdpa"))
args, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]

import transformers  # noqa: E402

_orig = transformers.AutoModel.from_pretrained


def _allow_sdpa(model_path):
    """MiBoTForActionGeneration and DiT do not declare _supports_sdpa, so transformers refuses sdpa at load time.
    Their VLM (Qwen3VLPreTrainedModel) declares it, and the DiT's attention always calls F.scaled_dot_product_attention
    whatever the setting, so the only change sdpa makes is in the VLM. Declare the flag on the two classes."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    top = get_class_from_dynamic_module(cfg.auto_map["AutoModel"], model_path)
    mod = sys.modules[top.__module__]
    for name in ("MiBoTForActionGeneration", "DiT"):
        setattr(getattr(mod, name), "_supports_sdpa", True)
    print(f"[xr1_server] declared _supports_sdpa on MiBoTForActionGeneration, DiT ({top.__module__})", flush=True)


def _from_pretrained(path, *a, **kw):
    kw["attn_implementation"] = args.attn
    if args.attn == "sdpa":
        _allow_sdpa(path)
    print(f"[xr1_server] attn_implementation={args.attn}", flush=True)
    return _orig(path, *a, **kw)


transformers.AutoModel.from_pretrained = _from_pretrained
server = os.path.join(args.repo, "deploy", "server.py")
sys.path.insert(0, os.path.dirname(server))
sys.argv = [server] + rest
runpy.run_path(server, run_name="__main__")
