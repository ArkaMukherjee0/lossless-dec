"""Phase F: offline greedy acceptance for token-only drafters, replayed on saved AR outputs.

Under greedy decoding the spec output equals the AR output, so each drafter can be replayed
against the AR trajectory: at position t it proposes a draft from context = prompt + AR[:t],
the accepted length is the matching prefix of AR[t:], and the next step starts at t + a + 1.
tau = generated tokens / target forward passes (steps without a draft count as 1-token steps).

Usage (CPU only): GPUS= docker/run.sh python3 scripts/sim_offline.py --model Qwen/Qwen3.8-27B
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/scripts")
from bench_vllm import load_prompts  # noqa: E402


def ngram_vllm(min_n, max_n, k):
    from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens

    def propose(ctx, _state):
        return list(_find_longest_matched_ngram_and_propose_tokens(np.array(ctx, dtype=np.int32),
                                                                   min_n, max_n, 1 << 30, k))
    return propose


def longest_match(k, max_n=32):
    """Longest suffix of the context that occurred earlier -> copy what followed its latest
    earlier occurrence (SAM-decoding style, match length capped at max_n)."""
    def propose(ctx, state):
        idx = state.setdefault("idx", [dict() for _ in range(max_n + 1)])  # n -> {ngram: end pos}
        # index every n-gram ending strictly before the current end of the context
        for end in range(max(state.get("n_indexed", 0), 1), len(ctx)):
            for n in range(1, min(max_n, end) + 1):
                idx[n][tuple(ctx[end - n:end])] = end
        state["n_indexed"] = len(ctx)
        for n in range(min(max_n, len(ctx) - 1), 0, -1):
            pos = idx[n].get(tuple(ctx[len(ctx) - n:]))
            if pos is not None:
                return ctx[pos:pos + k]
        return []
    return propose


class Suffix:
    """vLLM's suffix decoding (arctic_inference), incl. its cross-request response cache."""

    def __init__(self, depth=24, min_prob=0.1, factor=1.0):
        from arctic_inference.suffix_decoding import SuffixDecodingCache
        self.cache = SuffixDecodingCache(max_tree_depth=depth, max_cached_requests=10000)
        self.min_prob, self.factor, self.depth = min_prob, factor, depth

    def propose(self, ctx, state):
        rid = state["rid"]
        if rid not in self.cache.active_requests:
            self.cache.start_request(rid, state["prompt"])
            state["fed"] = len(state["prompt"])
        new = ctx[state["fed"]:]
        if new:
            self.cache.add_active_response(rid, new)
        state["fed"] = len(ctx)
        d = self.cache.speculate(rid, ctx[-self.depth:], max_spec_tokens=self.depth, max_spec_factor=self.factor,
                                 min_token_prob=self.min_prob)
        return list(d.token_ids)

    def finish(self, state):
        self.cache.stop_request(state["rid"])


def simulate(propose, prompt, out, state):
    t = steps = drafted_steps = draft_tokens = accepted = 0
    while t < len(out):
        d = propose(prompt + out[:t], state)[: len(out) - t]
        a = 0
        while a < len(d) and d[a] == out[t + a]:
            a += 1
        steps += 1
        drafted_steps += bool(d)
        draft_tokens += len(d)
        accepted += a
        t += a + 1
    return {"tokens": len(out), "steps": steps, "drafted_steps": drafted_steps,
            "draft_tokens": draft_tokens, "accepted": accepted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", default="a")
    ap.add_argument("--methods", default="ngram4,pld8,longest8,suffix")
    args = ap.parse_args()

    name = args.model.split("/")[-1]
    recs = json.load(open(f"/workspace/results/{args.exp}/{name}/ar_t0.0_c1.json"))["records"]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = load_prompts(80)
    assert len(prompts) == len(recs) and all(p[0] == r["category"] for p, r in zip(prompts, recs))
    prompt_ids = []
    for _, p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        prompt_ids.append(tok(text, add_special_tokens=False).input_ids)

    results = {}
    for method in args.methods.split(","):
        suffix = Suffix() if method == "suffix" else None
        propose = suffix.propose if suffix else {"ngram4": ngram_vllm(4, 4, 4), "pld8": ngram_vllm(1, 4, 8),
                                                 "longest8": longest_match(8)}[method]
        per_cat = collections.defaultdict(collections.Counter)
        for i, (p, r) in enumerate(zip(prompt_ids, recs)):
            state = {"rid": i, "prompt": p}
            per_cat[r["category"]].update(simulate(propose, p, r["token_ids"], state))
            if suffix:
                suffix.finish(state)
        total = sum(per_cat.values(), collections.Counter())
        results[method] = {
            "tau": total["tokens"] / total["steps"],
            "draft_rate": total["drafted_steps"] / total["steps"],
            "mean_draft_len": total["draft_tokens"] / max(total["drafted_steps"], 1),
            "per_category_tau": {c: v["tokens"] / v["steps"] for c, v in sorted(per_cat.items())},
            "totals": dict(total)}
        cats = " ".join(f"{c}={v:.2f}" for c, v in results[method]["per_category_tau"].items())
        print(f"{name} {method:>9}: tau={results[method]['tau']:.2f} "
              f"draft_rate={results[method]['draft_rate']:.2f} | {cats}", flush=True)

    os.makedirs(f"/workspace/results/f/{name}", exist_ok=True)
    json.dump(results, open(f"/workspace/results/f/{name}/sim_token_only.json", "w"), indent=1)


if __name__ == "__main__":
    main()
