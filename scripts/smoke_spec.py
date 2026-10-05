"""Smoke test: vLLM spec decoding on gfx90a — greedy outputs must match AR exactly."""
import json
import os
import sys
import time

from vllm import LLM, SamplingParams

TARGET = os.environ.get("TARGET", "Qwen/Qwen3-1.7B")
DRAFT = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Explain speculative decoding in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Repeat after me: the quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog.",
]
ATTN = os.environ.get("ATTN")  # e.g. TRITON_ATTN; default lets vLLM pick (ROCM_ATTN on gfx90a)
OUT = f"/workspace/results/smoke/{TARGET.split('/')[-1]}" + (f"-{ATTN}" if ATTN else "") + ("-eager" if os.environ.get("EAGER") == "1" else "")
SP = SamplingParams(temperature=0.0, max_tokens=256, logprobs=2)

CONFIGS = {
    "ar": None,
    "ngram": {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
    "ngram_gpu": {"method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_max": 4},
    "draft_model": {"method": "draft_model", "model": DRAFT, "num_speculative_tokens": 4},
}


def run(name, spec):
    llm = LLM(TARGET, speculative_config=spec, gpu_memory_utilization=0.6,
              max_model_len=4096, enforce_eager=os.environ.get("EAGER") == "1", disable_log_stats=False,
              **({"attention_config": {"backend": ATTN}} if ATTN else {}))
    for it in range(2):  # batch size 1; report the second pass
        t = time.perf_counter()
        outs = [llm.generate([p], SP, use_tqdm=False)[0] for p in PROMPTS]
        dt = time.perf_counter() - t
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[{name}] {n} tok in {dt:.2f}s -> {n / dt:.1f} tok/s (bs=1)", flush=True)
    for m in llm.get_metrics():
        if m.name.startswith("vllm:spec_decode_num_") and hasattr(m, "value"):
            print(f"[{name}] {m.name} = {m.value}", flush=True)
    # top-2 logprob gap per position (for diagnosing near-tie flips)
    gaps = [[(lambda v: v[0] - v[1] if len(v) > 1 else None)(sorted((x.logprob for x in lp.values()), reverse=True))
             for lp in o.outputs[0].logprobs] for o in outs]
    toks = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    return {"tokens": toks, "gaps": gaps}


if __name__ == "__main__":
    # One config per process (vLLM does not reliably free GPU memory between LLM instances).
    # `python smoke_spec.py <config>` runs one; `python smoke_spec.py compare` checks vs AR.
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1] == "compare":
        ar = json.load(open(f"{OUT}/ar.json"))
        for name in CONFIGS:
            if name != "ar" and os.path.exists(f"{OUT}/{name}.json"):
                toks = json.load(open(f"{OUT}/{name}.json"))["tokens"]
                for i, (a, b) in enumerate(zip(ar["tokens"], toks)):
                    d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
                    gap = f", AR top-2 logprob gap there = {ar['gaps'][i][d]:.4f}" if d is not None else ""
                    print(f"[{name}] prompt {i}: first divergence at {d}{gap}")
    else:
        json.dump(run(sys.argv[1], CONFIGS[sys.argv[1]]), open(f"{OUT}/{sys.argv[1]}.json", "w"))
