"""M1: where do drafts get rejected? Acceptance by the target's own confidence (top-2 logprob margin at
that token) and by task category, from a greedy (T = 0) draft-logged run with --save-gaps.

At T = 0 a drafted token is accepted iff it equals the target's argmax. A rejection where the target was
near-tied (small margin) is a tie-break the drafter could not know; one where the target was confident
is a plain modelling miss. Only positions the chain actually reached count (all earlier drafts accepted).

  python3 scripts/rejection_anatomy.py results/m1/gemma-4-31B-it/dflash15
      (reads <prefix>_drafts.jsonl and <prefix>_t0.0_c1.json; writes <prefix>_anatomy.json)
"""
import collections
import json
import sys

import numpy as np

from draftlog_io import NOT_FOUND, iter_steps

EDGES = [0, 0.25, 0.5, 1, 2, 4, 8, float("inf")]  # target top-1 minus top-2 logprob, nats
NEAR_TIE = 1.0
CATS = ["math_reasoning", "mt_bench", "qa", "rag", "summarization", "translation"]


def reached_positions(prefix):
    rows = []  # (category, depth, margin, rank, drafter top-1 prob)
    for st in iter_steps(f"{prefix}_drafts.jsonl", f"{prefix}_t0.0_c1.json"):
        gaps, out = st["record"]["gaps"], st["record"]["token_ids"]
        for j, (k, r) in enumerate(zip(st["idx"], st["ranks"])):
            if k >= len(out) or gaps[k] is None:
                break
            rows.append((st["record"]["category"], j + 1, gaps[k], r, float(np.exp(st["top_lp"][j][0]))))
            if r != 0:  # the chain stops at its first rejection
                break
    return rows


def summarize(rows):
    acc = [r[3] == 0 for r in rows]
    rej = [r for r in rows if r[3] != 0]
    tie = [r for r in rows if r[2] < NEAR_TIE]
    alpha = float(np.mean(acc))
    out = {"positions": len(rows), "alpha": round(alpha, 3),
           "near_tie_share_of_positions": round(len(tie) / len(rows), 3),
           "alpha_near_tie": round(float(np.mean([r[3] == 0 for r in tie])), 3) if tie else None,
           "alpha_confident": round(float(np.mean([r[3] == 0 for r in rows if r[2] >= NEAR_TIE])), 3),
           "near_tie_share_of_rejections": round(sum(r[2] < NEAR_TIE for r in rej) / len(rej), 3) if rej else None,
           # if every near-tie rejection were accepted instead (upper bound on what tie-breaking is worth)
           "alpha_if_ties_fixed": round(1 - sum(r[2] >= NEAR_TIE for r in rej) / len(rows), 3),
           "rejected_truth_in_top2": round(float(np.mean([1 <= r[3] <= 1 for r in rej])), 3) if rej else None,
           "rejected_truth_in_top4": round(float(np.mean([1 <= r[3] <= 3 for r in rej])), 3) if rej else None,
           "rejected_truth_not_in_topM": round(float(np.mean([r[3] >= NOT_FOUND for r in rej])), 3) if rej else None,
           "drafter_p1_accepted": round(float(np.mean([r[4] for r in rows if r[3] == 0])), 3),
           "drafter_p1_rejected": round(float(np.mean([r[4] for r in rej])), 3) if rej else None}
    bins = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        b = [r for r in rows if lo <= r[2] < hi]
        if b:
            bins.append({"margin": f"{lo}-{hi}", "share_of_positions": round(len(b) / len(rows), 3),
                         "alpha": round(float(np.mean([r[3] == 0 for r in b])), 3),
                         "share_of_rejections": round(sum(r[3] != 0 for r in b) / max(1, len(rej)), 3)})
    out["by_margin"] = bins
    return out


def main():
    prefix = sys.argv[1]
    rows = reached_positions(prefix)
    result = {"all": summarize(rows)}
    for c in CATS:
        sub = [r for r in rows if r[0] == c]
        if sub:
            result[c] = summarize(sub)
    shallow, deep = [r for r in rows if r[1] <= 2], [r for r in rows if r[1] >= 3]
    result["depth_1_2"], result["depth_3_plus"] = summarize(shallow), summarize(deep)
    json.dump(result, open(f"{prefix}_anatomy.json", "w"), indent=1)

    keys = ["positions", "alpha", "alpha_confident", "alpha_near_tie", "near_tie_share_of_positions",
            "near_tie_share_of_rejections", "alpha_if_ties_fixed", "rejected_truth_in_top2", "rejected_truth_in_top4",
            "rejected_truth_not_in_topM", "drafter_p1_accepted", "drafter_p1_rejected"]
    cols = ["all"] + [c for c in CATS if c in result] + ["depth_1_2", "depth_3_plus"]
    print(f"{'':30}" + "".join(f"{c[:11]:>12}" for c in cols))
    for k in keys:
        print(f"{k:30}" + "".join(f"{str(result[c][k]):>12}" for c in cols))
    print(f"\nby target margin (nats), all categories:")
    for b in result["all"]["by_margin"]:
        print(f"  {b['margin']:>10}: {b['share_of_positions']:6.1%} of positions, alpha {b['alpha']:.2f}, "
              f"{b['share_of_rejections']:6.1%} of rejections")


if __name__ == "__main__":
    main()
