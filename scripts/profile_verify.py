"""Where does verification time go? Profile decoding at a fixed concurrency (torch profiler, eager) and sum
GPU kernel time by kernel family, per verify step. Run once without speculation and once per draft length
on a hybrid target (Qwen3.8: Gated DeltaNet + attention) and an attention-only control (Gemma 4 31B).

  GPUS=0,1 docker/run.sh python3 scripts/profile_verify.py --model Qwen/Qwen3.8-27B --concurrency 64 \
      --spec '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' --tag mtp1
Writes results/hybrid/prof/<model>/<tag>_c<C>.json (family totals) next to the raw traces.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import shutil

from vllm import LLM, SamplingParams

from bench_vllm import counters, load_prompts

FAMILIES = [  # first match wins; anything else is "other" (the top kernels are printed to refine this)
    ("gdn_recurrent", r"recurrent|gated_delta|delta_rule|gdn"),
    ("gdn_chunk", r"chunk_|wy_fast|solve_tril|chunk_local_cumsum|kkt"),
    ("conv1d", r"conv1d|causal_conv"),
    ("attention", r"attn|attention|flash|paged|unified|fmha|mha"),
    ("moe", r"moe|expert|topk_softmax|fused_experts"),
    ("gemm", r"gemm|Cijk|matmul|wvSplitK|LLGemm|hipblaslt|rocblas|wvSpltK|skinny|mm_"),
    ("comm", r"nccl|rccl|allreduce|all_reduce|cross_device|custom_ar"),
    ("copy_index", r"index_select|index_copy|gather|scatter|copy|cat_|elementwise_kernel.*copy|fill"),
    ("norm_act_rope", r"norm|silu|gelu|act_and_mul|rotary|rope|sigmoid|softplus|l2"),
    ("sample_reject", r"sample|argmax|softmax|gumbel|rejection|topk|top_k"),
]


def family(name):
    low = name.lower()
    return next((f for f, pat in FAMILIES if re.search(pat, low)), "other")


def kernel_times(trace_dir):
    """Sum device kernel durations (us) by name from rank 0's worker trace."""
    files = sorted(glob.glob(f"{trace_dir}/**/*rank0*.json*", recursive=True)) or \
        sorted(glob.glob(f"{trace_dir}/**/*.json*", recursive=True))
    assert files, f"no trace under {trace_dir}"
    path = max(files, key=os.path.getsize)  # the worker trace is the big one
    opener = gzip.open if path.endswith(".gz") else open
    events = json.load(opener(path, "rt"))["traceEvents"]
    by_name = collections.Counter()
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            by_name[e["name"]] += e.get("dur", 0)
    return by_name, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--tag", default="ar")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    out_dir = f"/workspace/results/hybrid/prof/{args.model.split('/')[-1]}"
    trace_dir = f"{out_dir}/{args.tag}_c{args.concurrency}_trace"
    shutil.rmtree(trace_dir, ignore_errors=True)
    llm = LLM(args.model, tensor_parallel_size=args.tp, speculative_config=json.loads(args.spec) if args.spec else None,
              max_model_len=4096, gpu_memory_utilization=0.85, max_num_seqs=args.concurrency, enforce_eager=True,
              disable_log_stats=False, limit_mm_per_prompt={"image": 0, "video": 0},
              profiler_config={"profiler": "torch", "torch_profiler_dir": trace_dir,
                               "torch_profiler_with_stack": False})
    tok = llm.get_tokenizer()
    prompts = [p for _, p in load_prompts(per_cat=args.concurrency)][: args.concurrency]
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for p in prompts]
    # Prefill outside the profiled window would need a scheduler hook; instead decode long enough that the
    # (few, large) prefill kernels are a small share, and report them separately via their family.
    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
    llm.generate(texts, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True), use_tqdm=False)  # warm-up
    c0 = counters(llm)
    llm.start_profile()
    outs = llm.generate(texts, sp, use_tqdm=False)
    llm.stop_profile()
    drafts, _, accepted = [b - a for a, b in zip(c0, counters(llm))]
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    tau = accepted / drafts + 1 if drafts else 1.0
    steps = n_out / (args.concurrency * tau)  # target forward passes in the decode phase (approx.)

    by_name, path = kernel_times(trace_dir)
    fam = collections.Counter()
    for name, us in by_name.items():
        fam[family(name)] += us
    total = sum(fam.values())
    summary = {"model": args.model, "tag": args.tag, "concurrency": args.concurrency, "tau": tau,
               "steps": steps, "output_tokens": n_out, "trace": path,
               "ms_per_step": {f: round(us / 1e3 / steps, 2) for f, us in fam.most_common()},
               "ms_per_step_total": round(total / 1e3 / steps, 2),
               "top_kernels_ms_per_step": [(n[:120], round(us / 1e3 / steps, 2)) for n, us in by_name.most_common(25)]}
    json.dump(summary, open(f"{out_dir}/{args.tag}_c{args.concurrency}.json", "w"), indent=1)
    print(f"SUMMARY tau={tau:.2f} steps={steps:.0f} kernel ms/step={summary['ms_per_step_total']}")
    for f, ms in summary["ms_per_step"].items():
        print(f"  {f:>14}: {ms:8.2f} ms/step")
    print("top kernels:")
    for n, ms in summary["top_kernels_ms_per_step"][:15]:
        print(f"  {ms:8.2f}  {n}")


if __name__ == "__main__":
    main()
