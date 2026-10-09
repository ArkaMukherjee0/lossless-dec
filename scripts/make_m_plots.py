"""Figures for the two acceptance measurements (lit/l6_measurements.*).

(a) M1: where greedy drafts get rejected, by the target's top-2 margin at that token (rejection_anatomy.py)
(b) M2: tau vs number of sibling samples available, T = 1 (sibling_predictability.py)
(c) M2: per category, drafter alone vs drafter + 7 finished siblings (switch and oracle)

  GPUS= docker/run.sh python3 scripts/make_m_plots.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_depth_plot import C, GRID, INK, INK2, NEUTRAL, SURFACE, SHORT  # noqa: E402  (also sets rcParams)

R, OUT = "/workspace/results", "/workspace/lit"
M1 = [("gemma-4-31B-it", "dflash15"), ("gemma-4-26B-A4B-it", "dflash15"), ("Qwen3.8-27B", "dflash7")]
CAT_LBL = {"math_reasoning": "math", "mt_bench": "chat", "qa": "QA", "rag": "RAG", "summarization": "summ.",
           "translation": "transl."}


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.1, 1, 1.15]})

    # (a) share of rejections by target margin
    ax = axes[0]
    models = [(m, json.load(open(f"{R}/m1/{m}/{t}_anatomy.json"))) for m, t in M1
              if os.path.exists(f"{R}/m1/{m}/{t}_anatomy.json")]
    bins = [b["margin"] for b in models[0][1]["all"]["by_margin"]]
    width = 0.8 / len(models)
    shades = [INK, NEUTRAL, "#b9b8b3"]
    for i, (m, res) in enumerate(models):
        ys = [b["share_of_rejections"] for b in res["all"]["by_margin"]]
        ax.bar([x + (i - (len(models) - 1) / 2) * width for x in range(len(bins))], ys, width * 0.92,
               color=shades[i], label=f"{SHORT[m]} (alpha {res['all']['alpha']:.2f})", zorder=2)
    ax.axvspan(-0.5, 2.5, color=GRID, zorder=0)
    top = max(b["share_of_rejections"] for _, r in models for b in r["all"]["by_margin"])
    ax.set_ylim(0, top * 1.35)  # headroom for the legend
    ax.text(1, top * 0.55, "near-tie\n(< 1 nat)", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(bins)), [b.replace("-inf", "+").replace("-", "–") for b in bins], fontsize=7.5)
    ax.set(xlabel="target top-1 minus top-2 logprob at the rejected token (nats)", ylabel="share of rejections")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("(a) DFlash misses tokens the target is sure of;\n     the stronger DFlash2 mostly misses near-ties",
                 fontsize=9.5, loc="left", color=INK)

    # (b) tau vs siblings
    ax = axes[1]
    sib = json.load(open(f"{R}/m2/gemma-4-31B-it/dflash15_siblings.json"))["all"]
    gs = sorted(int(k[3:]) for k in sib if k.startswith("sib") and k[3:].isdigit())
    ax.axhline(sib["drafter"], color=C["DFlash"], lw=1.8, zorder=2)
    ax.text(gs[-1], sib["drafter"] - 0.08, "DFlash alone", ha="right", va="top", fontsize=7.5, color=INK2)
    series = [("done{}_oracle", "drafter or finished sibling (oracle)", INK, "-", "o"),
              ("done{}_switch8", "switch to finished sibling (match >= 8)", INK, "--", "s"),
              ("done{}", "finished siblings alone", NEUTRAL, "-", "^"),
              ("sib{}_oracle", "drafter or concurrent sibling (oracle)", INK2, ":", "o"),
              ("sib{}", "concurrent siblings alone", "#b9b8b3", "-", "v")]
    for key, lbl, col, ls, mk in series:
        ys = [sib[key.format(g)] for g in gs]
        ax.plot(gs, ys, color=col, ls=ls, lw=1.6, marker=mk, ms=4, zorder=3)
        ax.annotate(lbl, (gs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7,
                    color=INK2)
    ax.set_xticks(gs)
    ax.set(xlabel="sibling samples of the same prompt", ylabel="tau (tokens per verify step)", xlim=(0.7, 13.5))
    ax.set_title("(b) Gemma 4 31B, T = 1: acceptance rises with siblings", fontsize=9.5, loc="left", color=INK)

    # (c) per category at G = 7
    ax = axes[2]
    res = json.load(open(f"{R}/m2/gemma-4-31B-it/dflash15_siblings.json"))
    g = gs[-1]
    cats = [c for c in CAT_LBL if c in res]
    for i, (key, lbl, col) in enumerate([("drafter", "DFlash alone", C["DFlash"]),
                                         (f"done{g}_switch8", "+ switch (finished)", NEUTRAL),
                                         (f"done{g}_oracle", "+ oracle (finished)", INK)]):
        ax.bar([x + (i - 1) * 0.27 for x in range(len(cats))], [res[c][key] for c in cats], 0.25, color=col,
               label=lbl, zorder=2)
    ax.set_xticks(range(len(cats)), [CAT_LBL[c] for c in cats])
    ax.set(ylabel=f"tau with {g} siblings")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("(c) the gain holds in every category", fontsize=9.5, loc="left", color=INK)

    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(f"{OUT}/l6_measurements.{ext}")
    print("wrote l6_measurements.svg/.pdf")


if __name__ == "__main__":
    main()
