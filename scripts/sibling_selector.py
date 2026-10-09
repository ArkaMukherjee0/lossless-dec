"""Can a cheap learned rule decide, per drafter step, whether to verify the drafter's chain or a chain copied
from sibling samples? Offline, on the M2 logs (see sibling_predictability.py for the setup).

Per step and sibling variant (concurrent = siblings so far, finished = complete siblings) we build:
  best       the continuation after the longest suffix match in any sibling
  consensus  token-by-token majority over the continuations of every sibling that matches (>= 3 tokens)
and features the drafter already has or that cost nothing to compute: suffix-match length, number of
matching siblings and how many agree on the next token, the drafter's top-1 probabilities and its own
expected accepted length, and how many leading tokens drafter and sibling proposals share.
A cost-sensitive logistic regression (weights = |accepted difference|) picks sibling vs drafter; it is
trained and scored with 4-fold cross-validation split by prompt.
Policies are compared by tau = 1 + accepted tokens per step; "tree" verifies both chains (= oracle) and is
reported with the extra verified tokens it costs.

  python3 scripts/sibling_selector.py results/m2/gemma-4-31B-it dflash15
"""
import collections
import glob
import json
import random
import sys

import numpy as np
from scipy.optimize import minimize

from draftlog_io import iter_steps
from sibling_predictability import K, N_INDEX, accepted, best_match, index

CATS = ["math_reasoning", "mt_bench", "qa", "rag", "summarization", "translation"]
FEATURES = ["match_len", "n_matching", "next_token_votes", "drafter_p1", "drafter_p2", "drafter_exp_len",
            "agree_len", "sib_prop_len", "log_pos"]


def consensus(props):
    """Prefix-consistent majority vote over candidate continuations."""
    out, live = [], [p for p in props if p]
    for j in range(K):
        votes = collections.Counter(p[j] for p in live if len(p) > j)
        if not votes:
            break
        tok, n = votes.most_common(1)[0]
        if n < 1:
            break
        out.append(tok)
        live = [p for p in live if len(p) > j and p[j] == tok]
    return out


def lead(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def collect(root, tag, g_max):
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
    rows = []
    for s in seeds:
        for st in iter_steps(*runs[s]):
            req, out = st["req"], outs[s][st["req"]]
            a = st["idx"][0] - 1
            ctx, truth = out[:a + 1], out[a + 1:a + 1 + K]
            drafter = list(st["top_ids"][:, 0])
            p1 = np.exp(st["top_lp"][:, 0])
            base = {"req": req, "cat": cats[req], "acc_drafter": accepted(drafter, truth),
                    "drafter_p1": float(p1[0]), "drafter_p2": float(p1[1]) if len(p1) > 1 else 0.0,
                    "drafter_exp_len": float(np.cumprod(p1).sum()), "log_pos": float(np.log1p(a))}
            sibs = [t for t in seeds if t != s and req < len(outs[t])]
            rng.shuffle(sibs)
            sibs = sibs[:g_max]
            for var, avail in (("concurrent", lambda t: a + 1), ("finished", lambda t: len(outs[t][req]))):
                per = [best_match(ctx, [(outs[t][req], idx[t][req], avail(t))]) for t in sibs]
                matched = [(m, p) for m, p in per if m >= N_INDEX and p]
                m_best, p_best = max(matched, key=lambda x: x[0]) if matched else (0, [])
                cons = consensus([p for _, p in matched])
                rows.append(dict(base, variant=var, match_len=min(m_best, 32), n_matching=len(matched),
                                 next_token_votes=sum(p[0] == p_best[0] for _, p in matched) if p_best else 0,
                                 agree_len=lead(drafter, p_best), sib_prop_len=len(p_best),
                                 acc_best=accepted(p_best, truth), acc_cons=accepted(cons, truth),
                                 len_best=len(p_best), len_cons=len(cons)))
    return rows


def fit_logistic(X, y, w, l2=1e-2):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])

    def loss(b):
        z = Z @ b
        return np.sum(w * (np.logaddexp(0, z) - y * z)) / w.sum() + l2 * np.sum(b[:-1] ** 2)
    b = minimize(loss, np.zeros(Z.shape[1]), method="L-BFGS-B").x
    return lambda Xn: 1 / (1 + np.exp(-(np.hstack([(Xn - mu) / sd, np.ones((len(Xn), 1))]) @ b))), b, mu, sd


def evaluate(rows, sib_key):
    """Cross-validated learned choice between drafter and a sibling proposal (acc_best / acc_cons)."""
    reqs = sorted({r["req"] for r in rows})
    folds = [set(reqs[i::4]) for i in range(4)]
    choice = np.zeros(len(rows), dtype=bool)
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    d = np.array([r["acc_drafter"] for r in rows], dtype=float)
    sib = np.array([r[sib_key] for r in rows], dtype=float)
    has = np.array([r["n_matching"] > 0 for r in rows])
    coef = []
    for test in folds:
        tr = np.array([r["req"] not in test for r in rows]) & has & (sib != d)
        te = np.array([r["req"] in test for r in rows]) & has
        predict, b, _, _ = fit_logistic(X[tr], (sib[tr] > d[tr]).astype(float), np.abs(sib[tr] - d[tr]))
        choice[te] = predict(X[te]) > 0.5
        coef.append(b[:-1])
    return choice, np.mean(coef, axis=0)


def main():
    root, tag = sys.argv[1], sys.argv[2]
    result = {}
    for g in (2, 7):
        rows = collect(root, tag, g)
        for var in ("concurrent", "finished"):
            sub = [r for r in rows if r["variant"] == var]
            d = np.array([r["acc_drafter"] for r in sub], dtype=float)
            res_v = {}
            for name, key in (("best", "acc_best"), ("consensus", "acc_cons")):
                s_ = np.array([r[key] for r in sub], dtype=float)
                choice, coef = evaluate(sub, key)
                learned = np.where(choice, s_, d)
                switch = np.where(np.array([r["match_len"] >= 8 for r in sub]), np.array(
                    [r["acc_best"] for r in sub], dtype=float), d)
                extra = np.array([r["len_best" if key == "acc_best" else "len_cons"] for r in sub], dtype=float)
                pol = {"drafter": d, f"siblings_{name}": s_, "switch8": switch, "learned": learned,
                       "oracle_tree": np.maximum(d, s_)}
                res_v[name] = {"overall": {k: round(1 + float(v.mean()), 3) for k, v in pol.items()},
                               "tree_extra_verified_tokens": round(float(extra.mean()), 2),
                               "learned_picks_sibling": round(float(choice.mean()), 3),
                               "coef": dict(zip(FEATURES, np.round(coef, 3).tolist()))}
                res_v[name]["by_category"] = {
                    c: {k: round(1 + float(v[[r["cat"] == c for r in sub]].mean()), 3) for k, v in pol.items()}
                    for c in CATS}
            result[f"G{g}_{var}"] = res_v
            print(f"\n=== G={g} siblings, {var}")
            for name, rv in res_v.items():
                o = rv["overall"]
                print(f"  {name:9}: " + "  ".join(f"{k} {v:.2f}" for k, v in o.items())
                      + f"  | learned picks sibling {rv['learned_picks_sibling']:.0%}, tree verifies "
                        f"+{rv['tree_extra_verified_tokens']} tokens")
    json.dump(result, open(f"{root}/{tag}_selector.json", "w"), indent=1)
    print("\nlearned-rule weights (standardized; > 0 favours the sibling), G=7 finished, consensus:")
    for f, c in result["G7_finished"]["consensus"]["coef"].items():
        print(f"  {f:18} {c:+.2f}")


if __name__ == "__main__":
    main()
