"""Phase F part 2: layer-skip self-speculation (Draft & Verify / SWIFT style), teacher-forced.

The draft is the target with a subset of decoder layers skipped (identity). Under greedy decoding
its accepted length at position t is the run of consecutive positions where the skipped model's
top-1 on the AR trajectory equals the AR token (as for any chained drafter). One teacher-forced
forward per sequence per skip set; a small calibration split picks the best skip set per budget
(random search, first/last layers always kept), then it is evaluated on all prompts.

Draft cost per token ~ kept_layers / total_layers of a target decode step.

  GPUS=0,1 docker/run.sh python3 scripts/sim_layerskip.py --model google/gemma-4-31B-it
"""
import argparse
import collections
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/scripts")
from bench_vllm import load_prompts  # noqa: E402

KS = [1, 2, 3, 4, 6, 8]


def find_layers(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and len(mod) > 8 \
                and "vision" not in name and "audio" not in name:
            return mod
    raise RuntimeError("decoder layer list not found")


def run_lengths(match, k):
    t = steps = 0
    n = len(match)
    while t < n:
        a = 0
        while a < k and t + a < n and match[t + a]:
            a += 1
        steps += 1
        t += a + 1
    return n, steps


@torch.no_grad()
def top1_matches(model, seqs, plens, device):
    out = []
    for s, pl in zip(seqs, plens):
        ids = torch.tensor([s], device=device)
        logits = model(input_ids=ids, use_cache=False).logits[0]
        pred = logits[:-1].argmax(-1)  # prediction for token i+1 given s[:i+1]
        tgt = ids[0, 1:]
        out.append((pred == tgt)[pl - 1:].tolist())  # positions of AR output tokens
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", default="a")
    ap.add_argument("--per-cat", type=int, default=80)
    ap.add_argument("--calib", type=int, default=36, help="prompts used for the skip-set search")
    ap.add_argument("--trials", type=int, default=12, help="random skip sets per budget")
    ap.add_argument("--budgets", default="0.25,0.4,0.5")
    ap.add_argument("--max-seqs", type=int, default=240, help="evaluation subset size")
    args = ap.parse_args()

    name = args.model.split("/")[-1]
    recs = json.load(open(f"/workspace/results/{args.exp}/{name}/ar_t0.0_c1.json"))["records"]
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = load_prompts(args.per_cat)
    assert len(prompts) == len(recs)
    seqs, plens, cats = [], [], []
    for (c, p), r in zip(prompts, recs):
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False).input_ids
        seqs.append(ids + r["token_ids"]); plens.append(len(ids)); cats.append(c)
    rng = random.Random(0)
    order = list(range(len(seqs))); rng.shuffle(order)
    calib, evals = order[: args.calib], order[: args.max_seqs]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    layers = find_layers(model)
    L = len(layers)
    device = next(model.parameters()).device

    # Detect whether decoder layers return a tuple, so identity matches the calling convention.
    sample = {}
    def record(j):
        def hook(module, inputs, output):
            sample.setdefault(j, isinstance(output, tuple))  # must return None (non-None replaces the output)
        return hook
    hooks = [l.register_forward_hook(record(j)) for j, l in enumerate(layers)]
    base = top1_matches(model, [seqs[i] for i in calib[:2]], [plens[i] for i in calib[:2]], device)
    for h in hooks:
        h.remove()
    tuple_out = any(sample.values())
    orig = [l.forward for l in layers]
    skip = set()

    def make(i):
        def fwd(*a, **kw):
            if i in skip:
                h = a[0] if a else kw["hidden_states"]
                return (h,) if tuple_out else h
            return orig[i](*a, **kw)
        return fwd
    for i, l in enumerate(layers):
        l.forward = make(i)
    print(f"{name}: {L} layers, tuple_out={tuple_out}, full-model self-match on 2 seqs = "
          f"{sum(map(sum, base)) / sum(map(len, base)):.3f} (should be ~1.0)", flush=True)

    def tau(idx, k=4):
        m = top1_matches(model, [seqs[i] for i in idx], [plens[i] for i in idx], device)
        n = st = 0
        for mm in m:
            a, b = run_lengths(mm, k); n += a; st += b
        return n / st, m

    res = {"model": args.model, "layers": L, "budgets": {}}
    for frac in map(float, args.budgets.split(",")):
        n_skip = round(frac * L)
        best = (0, None)
        for trial in range(args.trials):
            if trial == 0:  # structured candidate: every other layer from the back, then the rest
                order_ = list(range(L - 2, 0, -2)) + list(range(L - 3, 0, -2))
                cand = order_[:n_skip]
            else:
                cand = rng.sample(range(1, L - 1), n_skip)
            skip.clear(); skip.update(cand)
            t, _ = tau(calib)
            if t > best[0]:
                best = (t, sorted(cand))
            assert len(cand) == n_skip
        skip.clear(); skip.update(best[1])
        _, m = tau(evals)
        per_k = {}
        for k in KS:
            n = st = 0
            by_cat = collections.defaultdict(lambda: [0, 0])
            for i, mm in zip(evals, m):
                a, b = run_lengths(mm, k); n += a; st += b
                by_cat[cats[i]][0] += a; by_cat[cats[i]][1] += b
            per_k[k] = {"tau": n / st, "per_category": {c: v[0] / v[1] for c, v in sorted(by_cat.items())}}
        res["budgets"][frac] = {"skip": best[1], "kept_fraction": 1 - n_skip / L, "calib_tau_k4": best[0],
                                "per_k": per_k}
        print(f"{name} skip {frac:.0%} ({n_skip}/{L}): calib tau@4={best[0]:.2f}; eval " +
              " ".join(f"k{k}:{v['tau']:.2f}" for k, v in per_k.items()), flush=True)

    out_dir = f"/workspace/results/f/{name}"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/sim_layerskip.json", "w"), indent=1)


if __name__ == "__main__":
    main()
