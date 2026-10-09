"""M2: how much do sibling samples of the same prompt know about each other's next tokens?

Input: the same prompts sampled with seeds 0..S-1 at T = 1, each run draft-logged (scripts/draftlog).
At every drafter step of every sample we compare chain proposals of K tokens from the same anchor:
  drafter        the logged drafter's top-1 at each position
  self           continuation of an earlier occurrence of the current suffix in the sample's own output
  other prompts  ... in other prompts' outputs (generic text reuse; SuffixDecoding-like control)
  sib / done     ... in the other samples of the same prompt, either only up to the same output index
                 (time-aligned: siblings decoding concurrently) or complete (finished siblings)
  oracle         the better of drafter and siblings at each step
  switch8        siblings if their suffix match is >= 8 tokens, else the drafter
Proposals are deterministic, so the realized match with the sampled continuation is an unbiased estimate
of their acceptance under lossless verification at T = 1. tau = 1 + accepted tokens per step.

  python3 scripts/sibling_predictability.py results/m2/gemma-4-31B-it dflash15
"""
import collections
import glob
import json
import random
import sys

import numpy as np

from draftlog_io import iter_steps

N_INDEX, M_MAX, K = 3, 32, 15
CATS = ["math_reasoning", "mt_bench", "qa", "rag", "summarization", "translation"]


def index(seq):
    idx = collections.defaultdict(list)
    for e in range(N_INDEX - 1, len(seq)):
        idx[tuple(seq[e - N_INDEX + 1:e + 1])].append(e)
    return idx


def best_match(ctx, cands):
    """cands: [(seq, idx, avail)]. Longest suffix match of ctx ending at e with a continuation before avail."""
    if len(ctx) < N_INDEX:
        return 0, []
    key = tuple(ctx[-N_INDEX:])
    best = (0, -1, None, 0)
    for seq, idx, avail in cands:
        for e in idx.get(key, ()):
            if e + 1 >= avail:
                continue
            m = N_INDEX
            while m < M_MAX and e - m >= 0 and m < len(ctx) and seq[e - m] == ctx[-m - 1]:
                m += 1
            if (m, e) > best[:2]:
                best = (m, e, seq, avail)
    m, e, seq, avail = best
    return (m, seq[e + 1:min(e + 1 + K, avail)]) if seq is not None else (0, [])


def accepted(prop, truth):
    n = 0
    for p, t in zip(prop, truth):
        if p != t:
            break
        n += 1
    return n


def main():
    root, tag = sys.argv[1], sys.argv[2]
    runs = {}
    for s in range(64):
        bench = glob.glob(f"{root}/{tag}_s{s}_t1.0_c1*.json")
        if bench:
            runs[s] = (f"{root}/{tag}_s{s}_drafts.jsonl", bench[0])
    seeds = sorted(runs)
    outs = {s: [r["token_ids"] for r in json.load(open(runs[s][1]))["records"]] for s in seeds}
    cats = [r["category"] for r in json.load(open(runs[seeds[0]][1]))["records"]]
    idx = {s: [index(o) for o in outs[s]] for s in seeds}
    rng = random.Random(0)

    rows = []  # one per drafter step
    for s in seeds:
        others_same_seed = None
        for st in iter_steps(*runs[s]):
            req, out = st["req"], outs[s][st["req"]]
            a = st["idx"][0] - 1  # output index of the anchor (last token before the draft)
            ctx, truth = out[:a + 1], out[a + 1:a + 1 + K]
            row = {"cat": cats[req], "drafter": accepted(list(st["top_ids"][:, 0]), truth)}
            row["self"] = accepted(best_match(ctx, [(out, idx[s][req], a + 1)])[1], truth)
            if others_same_seed is None:
                others_same_seed = [(outs[s][q], idx[s][q], len(outs[s][q])) for q in range(len(outs[s]))]
            row["other_prompts"] = accepted(best_match(ctx, [c for q, c in enumerate(others_same_seed) if q != req])[1],
                                            truth)
            sibs = [t for t in seeds if t != s and req < len(outs[t])]
            rng.shuffle(sibs)
            for g in sorted({1, 2, 4, len(sibs)}):
                for var, avail in (("sib", lambda t: a + 1), ("done", lambda t: len(outs[t][req]))):
                    m, prop = best_match(ctx, [(outs[t][req], idx[t][req], avail(t)) for t in sibs[:g]])
                    row[f"{var}{g}_m"], row[f"{var}{g}"] = m, accepted(prop, truth)
            row["G"] = len(sibs)
            rows.append(row)

    G = max(r["G"] for r in rows)
    groups = sorted({1, 2, 4, G})

    def tau(sub, key):
        return round(1 + float(np.mean([r[key] for r in sub])), 3)

    def report(sub):
        res = {"steps": len(sub)}
        for k in ("drafter", "self", "other_prompts"):
            res[k] = tau(sub, k)
        for var in ("sib", "done"):  # siblings so far (time-aligned) / finished siblings
            for g in groups:
                res[f"{var}{g}"] = tau(sub, f"{var}{g}")
                res[f"{var}{g}_coverage"] = round(float(np.mean([r[f"{var}{g}_m"] >= N_INDEX for r in sub])), 3)
                res[f"{var}{g}_oracle"] = round(1 + float(np.mean([max(r["drafter"], r[f"{var}{g}"]) for r in sub])), 3)
                res[f"{var}{g}_switch8"] = round(1 + float(np.mean(
                    [r[f"{var}{g}"] if r[f"{var}{g}_m"] >= 8 else r["drafter"] for r in sub])), 3)
        return res

    result = {"seeds": seeds, "K": K, "all": report(rows)}
    for c in CATS:
        sub = [r for r in rows if r["cat"] == c]
        if sub:
            result[c] = report(sub)
    json.dump(result, open(f"{root}/{tag}_siblings.json", "w"), indent=1)

    cols = ["all"] + [c for c in CATS if c in result]
    keys = ["steps", "drafter", "self", "other_prompts"] + [f"{var}{g}{suf}" for var in ("sib", "done")
                                                           for suf in ("", "_coverage", "_oracle", "_switch8")
                                                           for g in groups]
    print(f"seeds {seeds}; tau = 1 + accepted tokens per step (K={K}); sibN / doneN = N siblings, time-aligned / finished;"
          " oracle = better of drafter and siblings per step; switch8 = siblings if suffix match >= 8 else drafter")
    print(f"{'':16}" + "".join(f"{c[:11]:>12}" for c in cols))
    for k in keys:
        print(f"{k:16}" + "".join(f"{str(result[c][k]):>12}" for c in cols))


if __name__ == "__main__":
    main()
