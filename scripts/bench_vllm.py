"""bs=1 (or fixed-concurrency) vLLM benchmark on Spec-Bench with per-request spec-decode stats.

Example:
  GPUS=0,1 docker/run.sh python3 scripts/bench_vllm.py --model Qwen/Qwen3.8-27B --tp 2 \
      --spec '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' --tag mtp3 --per-cat 10
"""
import argparse
import collections
import json
import os
import random
import time

from vllm import LLM, SamplingParams

SPEC_BENCH = "/workspace/data/spec_bench.jsonl"
MT_BENCH_CATS = {"writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"}
COUNTERS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens")


def load_prompts(per_cat, seed=0):
    by_cat = collections.defaultdict(list)
    for line in open(SPEC_BENCH):
        q = json.loads(line)
        cat = "mt_bench" if q["category"] in MT_BENCH_CATS else q["category"]
        by_cat[cat].append(q["turns"][0])
    rng = random.Random(seed)
    return [(c, p) for c, ps in sorted(by_cat.items()) for p in rng.sample(ps, min(per_cat, len(ps)))]


def counters(llm):
    vals = {m.name: m.value for m in llm.get_metrics() if m.name in COUNTERS and hasattr(m, "value")}
    return [vals.get(c, 0) for c in COUNTERS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--spec", default=None, help="speculative_config JSON")
    ap.add_argument("--tag", default="ar")
    ap.add_argument("--per-cat", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (prompts are fixed)")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--mistral", action="store_true", help="Mistral-format checkpoint (consolidated + tekken)")
    ap.add_argument("--max-num-seqs", type=int, default=None, help="also caps CUDA-graph capture sizes")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--exp", default="bench", help="results/<exp>/<model>/")
    ap.add_argument("--save-gaps", action="store_true", help="store top-2 logprob gap per output token")
    args = ap.parse_args()

    spec = json.loads(args.spec) if args.spec else None
    llm = LLM(args.model, tensor_parallel_size=args.tp, speculative_config=spec,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              **({"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"}
                 if args.mistral else {}),
              enable_prefix_caching=os.environ.get("PREFIX_CACHE", "1") == "1",
              **({"attention_config": {"backend": os.environ["ATTN"]}} if os.environ.get("ATTN") else {}),
              **{k: v for k, v in {"max_num_seqs": args.max_num_seqs,
                                   "max_num_batched_tokens": args.max_num_batched_tokens}.items() if v},
              disable_log_stats=False, limit_mm_per_prompt={"image": 0, "video": 0}
              if args.mistral or any(m in args.model for m in ("Qwen3.8", "gemma-4", "Qwen3.6")) else None)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                        logprobs=2 if args.save_gaps else None)

    prompts = load_prompts(args.per_cat)
    if args.mistral:  # tekken tokenizer: render the chat to token ids
        from vllm.inputs import TokensPrompt
        texts = [TokensPrompt(prompt_token_ids=tok.apply_chat_template([{"role": "user", "content": p}]))
                 for _, p in prompts]
    else:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=args.thinking)
                 for _, p in prompts]

    # Warm up with the same batch shape we will time (vLLM JIT-compiles kernels for new shapes).
    llm.generate(texts[: args.concurrency], sp, use_tqdm=False)

    records = []
    t_all = time.perf_counter()
    for i in range(0, len(texts), args.concurrency):
        c0 = counters(llm)
        t = time.perf_counter()
        outs = llm.generate(texts[i: i + args.concurrency], sp, use_tqdm=False)
        dt = time.perf_counter() - t
        d = [b - a for a, b in zip(c0, counters(llm))]
        for j, o in enumerate(outs):
            records.append({"category": prompts[i + j][0], "n_out": len(o.outputs[0].token_ids),
                            "time": dt, "drafts": d[0], "draft_tokens": d[1], "accepted": d[2],
                            "token_ids": list(o.outputs[0].token_ids)})
            if args.save_gaps:
                records[-1]["gaps"] = [(lambda v: v[0] - v[1] if len(v) > 1 else None)(
                    sorted((x.logprob for x in lp.values()), reverse=True)) for lp in o.outputs[0].logprobs]
    t_all = time.perf_counter() - t_all

    n = sum(r["n_out"] for r in records)
    summary = {"model": args.model, "tag": args.tag, "tp": args.tp, "spec": spec,
               "temperature": args.temperature, "thinking": args.thinking,
               "concurrency": args.concurrency, "tokens": n, "seconds": t_all, "tok_per_s": n / t_all}
    by_cat = collections.defaultdict(lambda: [0, 0.0, 0, 0])
    for r in records:
        b = by_cat[r["category"]]
        b[0] += r["n_out"]; b[1] += r["time"] / args.concurrency; b[2] += r["drafts"]; b[3] += r["accepted"]
    summary["per_category"] = {c: {"tok_per_s": v[0] / v[1],
                                   "tau": (v[3] / v[2] + 1) if v[2] else None} for c, v in by_cat.items()}
    tot_drafts = sum(r["drafts"] for r in records) / args.concurrency
    tot_acc = sum(r["accepted"] for r in records) / args.concurrency
    summary["tau"] = tot_acc / tot_drafts + 1 if tot_drafts else None

    out_dir = f"/workspace/results/{args.exp}/{args.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.tag}_t{args.temperature}_c{args.concurrency}{'_think' if args.thinking else ''}" + (
        f"_s{args.seed}" if args.seed else "")
    json.dump({"summary": summary, "records": records}, open(f"{out_dir}/{name}.json", "w"))
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "per_category"}), flush=True)
    for c, v in sorted(summary["per_category"].items()):
        print(f"  {c:>15}: {v['tok_per_s']:7.1f} tok/s  tau={v['tau'] and round(v['tau'], 2)}", flush=True)


if __name__ == "__main__":
    main()
