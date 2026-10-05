"""Phase G: MoE expert-union vs verify window size.

Runs AR with vLLM's enable_return_routed_experts on a Spec-Bench subset, then for each
window of k consecutive output tokens (= one verify pass of k tokens) counts the distinct
experts touched per layer. Expert weight bytes read per step scale with that count.

  GPUS=0,1 docker/run.sh python3 scripts/router_union.py --model Qwen/Qwen3.6-35B-A3B
"""
import argparse
import json
import os
import sys

import numpy as np
from vllm import LLM, SamplingParams

sys.path.insert(0, "/workspace/scripts")
from bench_vllm import load_prompts  # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8, 12, 16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--per-cat", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    llm = LLM(args.model, tensor_parallel_size=args.tp, max_model_len=8192, gpu_memory_utilization=0.85,
              enable_return_routed_experts=True, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    prompts = load_prompts(args.per_cat)
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for _, p in prompts]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    union = {k: [] for k in KS}  # mean distinct experts per layer, per window
    n_experts = None
    for o in outs:
        re = o.outputs[0].routed_experts  # [seq_len, layers, topk], prompt + output
        if re is None:
            sys.exit("routed_experts missing: flag unsupported for this model")
        re = re[-len(o.outputs[0].token_ids):]  # output tokens only
        n_experts = max(n_experts or 0, int(re.max()) + 1)
        for k in KS:
            for s in range(0, len(re) - k + 1, k):
                w = re[s:s + k]  # [k, layers, topk]
                union[k].append(np.mean([len(np.unique(w[:, l])) for l in range(w.shape[1])]))
    topk = re.shape[2]
    res = {"model": args.model, "layers": int(re.shape[1]), "topk": int(topk), "n_experts_seen": n_experts,
           "mean_union": {k: float(np.mean(v)) for k, v in union.items()},
           "ratio_vs_k1": {k: float(np.mean(v) / np.mean(union[1])) for k, v in union.items()}}
    for k in KS:
        print(f"k={k:>2}: {res['mean_union'][k]:6.1f} distinct experts/layer "
              f"({res['ratio_vs_k1'][k]:.2f}x of k=1; no-overlap bound {k * topk})", flush=True)
    out_dir = f"/workspace/results/g/{args.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/router_union.json", "w"), indent=1)


if __name__ == "__main__":
    main()
