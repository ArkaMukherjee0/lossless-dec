"""Medusa step 2: final (post-norm) hidden states of the frozen target on the self-distilled data.

For each sequence prompt+output we keep h_t for t = len(prompt)-1 .. len(seq)-2 (the states that
predict output tokens) plus the token ids, in bf16 chunks under /models/medusa_data/<model>/hidden/.
With --spec-bench it instead dumps the phase-A Spec-Bench AR trajectories (for offline acceptance).

  GPUS=0,1 docker/run.sh python3 scripts/medusa_hidden.py --model Qwen/Qwen3.8-27B --shard 0
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/scripts")
from bench_vllm import load_prompts  # noqa: E402

MAX_LEN = 2048
CHUNK = 250


def load_examples(args, tok):
    name = args.model.split("/")[-1]
    if args.spec_bench:
        recs = json.load(open(f"/workspace/results/a/{name}/ar_t0.0_c1.json"))["records"]
        out = []
        for (_, p), r in zip(load_prompts(80), recs):
            text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            out.append((tok(text, add_special_tokens=False).input_ids, r["token_ids"]))
        return out
    rows = [json.loads(l) for l in open(f"/models/medusa_data/{name}/gen_{args.shard}.jsonl")]
    return [(r["prompt_ids"], r["output_ids"]) for r in rows]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--spec-bench", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    name = args.model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(args.model)
    examples = load_examples(args, tok)[: args.limit]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    base = model.model  # decoder stack; last_hidden_state is post final-norm
    dev = next(model.parameters()).device
    out_dir = f"/models/medusa_data/{name}/{'specbench' if args.spec_bench else 'hidden'}"
    os.makedirs(out_dir, exist_ok=True)

    buf, n_chunk, checked = [], 0, False
    for i, (p, o) in enumerate(examples):
        seq = (p + o)[:MAX_LEN]
        pl = len(p)
        if len(seq) <= pl + 1:
            continue
        ids = torch.tensor([seq], device=dev)
        h = base(input_ids=ids).last_hidden_state[0]
        if not checked:  # the saved state must reproduce the model's own next-token predictions
            lm = model.lm_head(h.to(model.lm_head.weight.device)).argmax(-1)
            ref = model(input_ids=ids).logits[0].argmax(-1)
            agree = (lm == ref.to(lm.device)).float().mean().item()
            print(f"sanity: lm_head(last_hidden_state) vs logits argmax agreement = {agree:.4f}", flush=True)
            assert agree > 0.99
            checked = True
        buf.append({"hidden": h[pl - 1: len(seq) - 1].to(torch.bfloat16).cpu(),
                    "tokens": torch.tensor(seq[pl:], dtype=torch.int32)})  # h[j] predicts tokens[j]
        if len(buf) == CHUNK or i == len(examples) - 1:
            torch.save(buf, f"{out_dir}/shard{args.shard}_{n_chunk:03d}.pt")
            print(f"saved chunk {n_chunk} ({i + 1}/{len(examples)})", flush=True)
            buf, n_chunk = [], n_chunk + 1
    if buf:
        torch.save(buf, f"{out_dir}/shard{args.shard}_{n_chunk:03d}.pt")


if __name__ == "__main__":
    main()
