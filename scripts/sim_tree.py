"""Offline: how many tokens per verify step a token TREE built from a parallel drafter's per-position
candidates would accept, vs the chain the drafter verifies today, at equal verify budget N (nodes).

Input: a draft log from scripts/draftlog (per drafter step: the drafted positions and the top-M candidate
ids / logprobs at each) and the bench_vllm.py result JSON of the same run (output token ids).
Greedy (T=0) acceptance. A parallel drafter's position-j candidates do not depend on which token was
picked at j-1, so every tree is a set of rank paths (r_1..r_d), r_j = rank of the token at depth j in
the position-j candidate list; the tree accepts depth d iff the true tokens' rank path (r*_1..r*_d) is
in it. Expected accepted length = sum over tree nodes of P(rank path prefix).

Trees compared at each budget N (draft nodes, excluding the bonus token):
  chain    the top-1 path of length N (what DFlash/DSpark verify)
  static   the N rank paths with the highest empirical frequency (one tree for all steps, Sequoia-like)
  dynamic  per step, the N paths with the highest drafter-estimated probability (prod of q_j, EAGLE-2-like)
  oracle   per step, the true path truncated to N (upper bound: no tree of N nodes accepts more)

  python3 scripts/sim_tree.py --log results/tree/gemma-4-31B-it/dflash15_drafts.jsonl \
      --bench results/tree/gemma-4-31B-it/dflash15_log_t0.0_c1.json
"""
import argparse
import collections
import heapq
import json

import numpy as np

from draftlog_io import NOT_FOUND, iter_steps


def load_steps(log_path, bench_path):
    """Yield (rank vector of the true continuation, candidate logprobs [K, M]) per drafter step."""
    for st in iter_steps(log_path, bench_path):
        yield st["ranks"], st["top_lp"]


def accepted(ranks, tree):
    """Draft tokens accepted from a tree given as a set of rank-path tuples."""
    d = 0
    while d < len(ranks) and tuple(ranks[:d + 1]) in tree:
        d += 1
    return d


def best_paths(score_child, root_children, n):
    """Best-first: the n highest-scoring nodes, where a node's score never exceeds its parent's."""
    heap = [(-s, p) for p, s in root_children]
    heapq.heapify(heap)
    tree = set()
    while heap and len(tree) < n:
        neg, p = heapq.heappop(heap)
        tree.add(p)
        for c, s in score_child(p, -neg):
            heapq.heappush(heap, (-s, c))
    return tree


def dynamic_tree(lp, n, width):
    """EAGLE-2-style: score a path by the drafter's own probability, prod_j q_j(token at rank r_j)."""
    K, M = lp.shape
    w = min(width, M)
    q = np.exp(lp[:, :w])

    def children(path, score):
        j = len(path)
        return [] if j >= K else [(path + (r,), score * q[j, r]) for r in range(w)]
    return best_paths(children, [((r,), q[0, r]) for r in range(w)], n)


def static_tree(steps, n, max_depth):
    """The n most frequent true rank-path prefixes over all steps (prefix-closed by construction)."""
    freq = collections.Counter()
    for ranks, _ in steps:
        for d in range(1, max_depth + 1):
            if ranks[d - 1] >= NOT_FOUND:
                break
            freq[tuple(ranks[:d])] += 1
    return {p for p, _ in freq.most_common(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--bench", required=True, help="bench_vllm.py result JSON of the logged run")
    ap.add_argument("--budgets", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--width", type=int, default=4, help="max children per node for the dynamic tree")
    ap.add_argument("--json", default=None, help="also write the table here")
    args = ap.parse_args()

    steps = list(load_steps(args.log, args.bench))
    assert steps, "no drafter steps"
    top1 = np.mean([r[0] == 0 for r, _ in steps])
    print(f"alignment check: drafter top-1 matches the output at the first drafted position in {top1:.0%} "
          "of steps (near 0% means the positions are off)")
    K = len(steps[0][0])
    budgets = [int(b) for b in args.budgets.split(",")]
    # Fit the static tree on even steps, score it on odd ones, so it is not graded on its own data.
    fit, test = steps[0::2], steps[1::2]

    rows = []
    for n in budgets:
        chain = {tuple([0] * d) for d in range(1, min(n, K) + 1)}
        stat = static_tree(fit, n, K)
        res = collections.defaultdict(list)
        for ranks, lp in test:
            res["chain"].append(accepted(ranks, chain))
            res["static"].append(accepted(ranks, stat))
            res["dynamic"].append(accepted(ranks, dynamic_tree(lp, n, args.width)))
            true_depth = next((d for d, r in enumerate(ranks) if r >= NOT_FOUND), K)
            res["oracle"].append(min(true_depth, n))
        # tau = accepted drafts + the bonus token the target always adds
        row = {"budget": n, **{k: round(1 + float(np.mean(v)), 3) for k, v in res.items()}}
        rows.append(row)
        print(f"N={n:3d}  " + "  ".join(f"{k} {row[k]:.2f}" for k in ("chain", "static", "dynamic", "oracle")))

    # Where do chains lose? Rank of the true token at the first position the top-1 path misses.
    miss = collections.Counter()
    for ranks, _ in steps:
        first = next((r for r in ranks if r != 0), None)
        if first is not None:
            miss["not in top-M" if first >= NOT_FOUND else f"rank {first}"] += 1
    total = sum(miss.values())
    print("first miss of the top-1 chain: " + ", ".join(f"{k} {v / total:.1%}" for k, v in miss.most_common(6)))
    print(f"{len(steps)} steps ({len(test)} scored), K={K}, tau = 1 + accepted drafts")
    if args.json:
        json.dump({"rows": rows, "first_miss": dict(miss), "steps": len(steps), "K": K}, open(args.json, "w"),
                  indent=1)


if __name__ == "__main__":
    main()
