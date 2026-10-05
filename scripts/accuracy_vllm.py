"""Losslessness at T>0: task accuracy with vs without speculative decoding (GSM8K test set).

At T>0 spec decoding should preserve the output *distribution*, so accuracy over many samples
must match AR within sampling noise. Runs batched (throughput is irrelevant here).

  GPUS=0,1 docker/run.sh python3 scripts/accuracy_vllm.py --model Qwen/Qwen3.8-27B --tag mtp3 \
      --spec '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
"""
import argparse
import json
import os
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams

PROMPT = ("Solve the following math problem step by step. "
          "End your response with a line of the form '#### <number>'.\n\n{q}")


def extract(text):
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"(-?[\d,]*\.?\d+)", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="test")
    golds = [r["answer"].split("####")[-1].strip().replace(",", "") for r in ds]
    llm = LLM(args.model, tensor_parallel_size=args.tp, max_model_len=4096, gpu_memory_utilization=0.85,
              speculative_config=json.loads(args.spec) if args.spec else None,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=r["question"])}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for r in ds]

    res = {"model": args.model, "tag": args.tag, "spec": args.spec, "temperature": args.temperature, "seeds": {}}
    for seed in map(int, args.seeds.split(",")):
        sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=seed)
        outs = llm.generate(texts, sp)
        correct = [same(extract(o.outputs[0].text), g) for o, g in zip(outs, golds)]
        res["seeds"][seed] = {"accuracy": sum(correct) / len(correct), "correct": correct,
                              "mean_len": sum(len(o.outputs[0].token_ids) for o in outs) / len(outs)}
        print(f"{args.model} {args.tag} seed={seed}: acc={res['seeds'][seed]['accuracy']:.4f} "
              f"mean_len={res['seeds'][seed]['mean_len']:.0f}", flush=True)

    out_dir = f"/workspace/results/acc/{args.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/{args.tag}_t{args.temperature}.json", "w"))


if __name__ == "__main__":
    main()
