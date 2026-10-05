"""Per-step acceptance traces (phases E1/E2): bs=1, streaming deltas, one record per engine step.

Each streamed delta at bs=1 is one engine step, so len(delta) = accepted + 1 for spec decoding
(1 for AR). Output: results/<exp>/<model>/<tag>.json with per-request step sizes and times.
Caveat: if the consumer lags, vLLM merges deltas; the per-step timestamps let us detect that
(merged steps show up as abnormally large gaps).

  E1: --data /workspace/data/reasoning.jsonl --thinking --max-tokens 16384
  E2: --longbench --ctx-len 32768 --n 20 --max-tokens 256
"""
import argparse
import asyncio
import json
import os
import time

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


def load_jsonl(path, n):
    rows = [json.loads(l) for l in open(path)]
    return [(r["category"], r["turns"][0]) for r in rows][:n]


def load_longbench(tok, ctx_len, n):
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    out = []
    for r in ds:
        ctx_ids = tok(r["context"], add_special_tokens=False).input_ids
        if len(ctx_ids) < ctx_len:
            continue
        keep = ctx_len - 512  # room for question, choices and template
        ctx = tok.decode(ctx_ids[: keep // 2] + ctx_ids[-keep // 2:])  # middle truncation
        q = (f"{ctx}\n\nQuestion: {r['question']}\nA) {r['choice_A']}\nB) {r['choice_B']}\n"
             f"C) {r['choice_C']}\nD) {r['choice_D']}\nAnswer with the letter and a short justification.")
        out.append((f"longbench_{r['domain']}", q))
        if len(out) == n:
            break
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--longbench", action="store_true")
    ap.add_argument("--ctx-len", default="0", help="comma list for --longbench, e.g. 8192,32768")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=None, help="override the engine's max_model_len")
    args = ap.parse_args()

    ctx_lens = [int(c) for c in args.ctx_len.split(",")]
    max_len = max(max(ctx_lens) + args.max_tokens + 1024, 8192) if args.longbench else args.max_tokens + 4096
    max_len = args.max_model_len or max_len
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
        model=args.model, tensor_parallel_size=args.tp, max_model_len=max_len,
        gpu_memory_utilization=0.85, disable_log_stats=False,
        speculative_config=json.loads(args.spec) if args.spec else None,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_prefix_caching=os.environ.get("PREFIX_CACHE", "1") == "1",
        **({"attention_config": {"backend": os.environ["ATTN"]}} if os.environ.get("ATTN") else {})))
    tok = await engine.get_tokenizer() if asyncio.iscoroutinefunction(engine.get_tokenizer) else engine.get_tokenizer()

    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=0,
                        output_kind=RequestOutputKind.DELTA)

    async def run_one(rid, text):
        steps, times, t0 = [], [], time.perf_counter()
        async for out in engine.generate(text, sp, request_id=rid):
            n = len(out.outputs[0].token_ids)
            if n:
                steps.append(n)
                times.append(time.perf_counter() - t0)
        return steps, times

    for ctx in (ctx_lens if args.longbench else [0]):
        prompts = load_longbench(tok, ctx, args.n) if args.longbench else load_jsonl(args.data, args.n)
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=args.thinking)
                 for _, p in prompts]
        await run_one("warmup", texts[0])  # compile/JIT for this shape before timing
        records = []
        for i, ((cat, _), text) in enumerate(zip(prompts, texts)):
            steps, times = await run_one(f"r{i}", text)
            records.append({"category": cat, "prompt_tokens": len(tok(text).input_ids),
                            "steps": steps, "times": times})
            n = sum(steps)
            print(f"[{i}] {cat}: {n} tok, {len(steps)} steps, tau={n / max(len(steps), 1):.2f}, "
                  f"{n / times[-1] if times else 0:.1f} tok/s (incl. prefill)", flush=True)
        out_dir = f"/workspace/results/{args.exp}/{args.model.split('/')[-1]}"
        os.makedirs(out_dir, exist_ok=True)
        name = f"{args.tag}{'_ctx' + str(ctx) if args.longbench else ''}_t{args.temperature}" + (
            f"_mml{args.max_model_len}" if args.max_model_len else "") + (
            "_nopc" if os.environ.get("PREFIX_CACHE") == "0" else "") + (
            f"_{os.environ['ATTN']}" if os.environ.get("ATTN") else "")
        json.dump({"args": vars(args), "records": records}, open(f"{out_dir}/{name}.json", "w"))
    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
