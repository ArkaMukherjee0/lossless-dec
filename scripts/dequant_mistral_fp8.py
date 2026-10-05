"""Dequantize a Mistral-format FP8 checkpoint (per-tensor qscale_weight) to BF16.

MI250 (gfx90a) has no FP8 kernels in vLLM, so Mistral Small 4 must run in BF16.
W_bf16 = W_fp8.float() * qscale_weight; qscale_* tensors are dropped and the
`quantization` block is removed from params.json. Other files are copied as-is.

  GPUS= docker/run.sh python3 scripts/dequant_mistral_fp8.py <src_snapshot_dir> <dst_dir> [--only N]
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--only", type=int, default=None, help="convert only the N-th shard (testing)")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)

    shards = sorted(glob.glob(f"{args.src}/consolidated-*.safetensors")) or [f"{args.src}/consolidated.safetensors"]
    idx_path = f"{args.src}/consolidated.safetensors.index.json"
    index = json.load(open(idx_path)) if os.path.exists(idx_path) else {"weight_map": {}}
    new_map = {}
    for i, path in enumerate(shards):
        name = os.path.basename(path)
        if args.only is not None and i != args.only:
            continue
        out = {}
        with safe_open(path, "pt") as f:
            keys = set(f.keys())
            for k in sorted(keys):
                if k.endswith((".qscale_weight", ".qscale_act")):
                    continue
                t = f.get_tensor(k)
                if t.dtype == torch.float8_e4m3fn:
                    base = k[: -len(".weight")]
                    scale_key = f"{base}.qscale_weight"
                    # the scale may live in another shard
                    if scale_key in keys:
                        s = f.get_tensor(scale_key)
                    else:
                        with safe_open(f"{args.src}/{index['weight_map'][scale_key]}", "pt") as g:
                            s = g.get_tensor(scale_key)
                    t = (t.float() * s.float()).to(torch.bfloat16)
                out[k] = t
                new_map[k] = name
        save_file(out, f"{args.dst}/{name}", metadata={"format": "pt"})
        print(f"{name}: {len(out)} tensors, {sum(v.numel() * v.element_size() for v in out.values()) / 1e9:.1f} GB",
              flush=True)

    if args.only is None:
        if len(shards) > 1:
            json.dump({"metadata": {}, "weight_map": new_map},
                      open(f"{args.dst}/consolidated.safetensors.index.json", "w"))
        params = json.load(open(f"{args.src}/params.json"))
        params.pop("quantization", None)
        json.dump(params, open(f"{args.dst}/params.json", "w"), indent=2)
        for extra in os.listdir(args.src):
            # skip the duplicate HF-format (still FP8) weights; only the Mistral format is converted
            if extra.startswith(("consolidated", "model-", "model.safetensors")) or extra == "params.json":
                continue
            src = os.path.realpath(f"{args.src}/{extra}")
            if os.path.isfile(src):
                shutil.copy(src, f"{args.dst}/{extra}")
        if os.path.exists(f"{args.dst}/config.json"):  # HF config still declares fp8
            cfg = json.load(open(f"{args.dst}/config.json"))
            for c in (cfg, cfg.get("text_config", {})):
                c.pop("quantization_config", None)
            json.dump(cfg, open(f"{args.dst}/config.json", "w"), indent=2)


if __name__ == "__main__":
    main()
