"""Medusa step 1: self-distilled training data (target's own greedy answers to UltraChat prompts).

Shard i of N; batched (throughput mode). Output: /models/medusa_data/<model>/gen_<i>.jsonl with
prompt_ids / output_ids per example.

  GPUS=0,1 docker/run.sh python3 scripts/medusa_gen.py --model Qwen/Qwen3.8-27B --shard 0 --num-shards 4
"""
import argparse
import json
import os

from datasets import load_dataset
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--n", type=int, default=8000, help="total prompts across all shards")
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    prompts = [r["messages"][0]["content"] for r in ds.select(range(args.n))
               if r["messages"] and r["messages"][0]["role"] == "user"]
    prompts = prompts[args.shard::args.num_shards]

    llm = LLM(args.model, tensor_parallel_size=2, max_model_len=4096, gpu_memory_utilization=0.9,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) for p in prompts]
    texts = [t for t in texts if len(tok(t).input_ids) < 2048]
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens))

    out_dir = f"/models/medusa_data/{args.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/gen_{args.shard}.jsonl", "w") as f:
        for o in outs:
            f.write(json.dumps({"prompt_ids": list(o.prompt_token_ids),
                                "output_ids": list(o.outputs[0].token_ids)}) + "\n")
    print(f"shard {args.shard}: {len(outs)} examples, "
          f"{sum(len(o.outputs[0].token_ids) for o in outs)} output tokens", flush=True)


if __name__ == "__main__":
    main()
