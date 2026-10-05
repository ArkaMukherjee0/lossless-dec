"""Phase F part 2: teacher-forced greedy acceptance for an independent draft model (vanilla SpS).

Under greedy decoding, a chained draft proposes d_j = argmax p_draft(. | AR[:t+j]) as long as its
earlier proposals matched AR, so its accepted length at position t is the run of consecutive
positions where the draft's top-1 on the AR trajectory equals the AR token (capped at k).
One batched prompt_logprobs pass of the draft over prompt + AR output gives every top-1.

  GPUS=0,1 docker/run.sh python3 scripts/sim_draft_tf.py --target Qwen/Qwen3.8-27B --draft Qwen/Qwen3.5-0.8B
"""
import argparse
import collections
import json
import os
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

sys.path.insert(0, "/workspace/scripts")
from bench_vllm import load_prompts  # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--exp", default="a")
    ap.add_argument("--per-cat", type=int, default=80)
    args = ap.parse_args()

    name = args.target.split("/")[-1]
    recs = json.load(open(f"/workspace/results/{args.exp}/{name}/ar_t0.0_c1.json"))["records"]
    tok = AutoTokenizer.from_pretrained(args.target)
    prompts = load_prompts(args.per_cat)
    assert len(prompts) == len(recs) and all(p[0] == r["category"] for p, r in zip(prompts, recs))
    seqs, plens = [], []
    for (_, p), r in zip(prompts, recs):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        plens.append(len(ids))
        seqs.append(ids + r["token_ids"])

    llm = LLM(args.draft, max_model_len=max(map(len, seqs)) + 16, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for s in seqs],
                        SamplingParams(max_tokens=1, prompt_logprobs=1))

    per_cat = {k: collections.defaultdict(collections.Counter) for k in KS}
    for o, s, pl, r in zip(outs, seqs, plens, recs):
        # prompt_logprobs[i] is the distribution for token i given s[:i] (entry 0 is None)
        top1 = [None] + [max(lp.items(), key=lambda kv: kv[1].logprob)[0] for lp in o.prompt_logprobs[1:]]
        match = [top1[i] == s[i] for i in range(pl, len(s))]  # over AR output positions
        n = len(match)
        for k in KS:
            t = steps = 0
            while t < n:
                a = 0
                while a < k and t + a < n and match[t + a]:
                    a += 1
                steps += 1
                t += a + 1
            per_cat[k][r["category"]].update({"tokens": n, "steps": steps})

    res = {"target": args.target, "draft": args.draft, "tau": {}, "per_category_tau": {}}
    for k in KS:
        tot = sum(per_cat[k].values(), collections.Counter())
        res["tau"][k] = tot["tokens"] / tot["steps"]
        res["per_category_tau"][k] = {c: v["tokens"] / v["steps"] for c, v in sorted(per_cat[k].items())}
        print(f"{name} <- {args.draft.split('/')[-1]} k={k}: tau={res['tau'][k]:.2f} | " +
              " ".join(f"{c}={v:.2f}" for c, v in res["per_category_tau"][k].items()), flush=True)
    out_dir = f"/workspace/results/f/{name}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/sim_draft_{args.draft.split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
