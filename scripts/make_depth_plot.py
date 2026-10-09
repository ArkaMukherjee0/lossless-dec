"""Acceptance by draft depth and the acceptance needed for long drafts to pay under load (lit/l5_depth.*).

(a) conditional acceptance P(position j accepted | j-1 accepted) along the draft, 4 requests in flight
(b) per-position acceptance alpha* above which a 16-token verify beats 4- and 8-token ones, by concurrency
(c) best speedup over k in {3, 7, 15} if every position were accepted with probability alpha (Gemma 4 31B)
Step costs come from the measured DFlash k = 3 / 7 / 15 runs (results/dk_grid), drafter included.

  GPUS= docker/run.sh python3 scripts/make_depth_plot.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = "/workspace/results/dk_grid"
OUT = "/workspace/lit"
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
C = {"MTP": "#2a78d6", "DFlash": "#eb6834"}
SHORT = {"gemma-4-31B-it": "Gemma 4 31B", "gemma-4-26B-A4B-it": "Gemma 4 26B-A4B",
         "Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B", "Qwen3.8-27B": "Qwen3.8-27B"}
CURVES = [("DFlash", m, "dflash15") for m in ("gemma-4-31B-it", "gemma-4-26B-A4B-it", "Qwen3.6-35B-A3B")] + \
         [("DFlash", "Qwen3.8-27B", "dflash7")] + [("MTP", m, "mtp6") for m in SHORT]
CONCS = (4, 16, 64, 128)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8})


def run(model, tag, c):
    return json.load(open(f"{R}/{model}/{tag}_c{c}.json"))


def step_costs(model, c):
    """Verify-step cost (in plain-decoding steps) for 4, 8 and 16 tokens per request."""
    ar = run(model, "ar", c)["output_throughput"]
    out = {}
    for k in (3, 7, 15):
        r = run(model, f"dflash{k}", c)
        out[k + 1] = r["spec_decode_acceptance_length"] / (r["output_throughput"] / ar)
    return out


def tau(n, a):
    return sum(a ** d for d in range(n))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # (a) conditional acceptance along the draft
    ax = axes[0]
    for method, model, tag in CURVES:
        cum = run(model, tag, 4)["spec_decode_per_position_acceptance_rates"]
        cond = [cum[0]] + [cum[j] / cum[j - 1] for j in range(1, len(cum))]
        xs = range(1, len(cond) + 1)
        ax.plot(xs, cond, color=C[method], lw=1.6, marker="o", ms=3.2, alpha=0.9, zorder=3)
        if method == "DFlash":  # the four MTP lines end together at j = 6; the legend names them
            ax.annotate(SHORT[model].replace("Gemma 4 ", "G").replace("Qwen", "Q"), (len(cond), cond[-1]),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK2)
    ax.axhline(0.9, color=NEUTRAL, lw=1, ls="--", zorder=1)
    ax.text(1, 0.905, "0.9: what 16-token drafts need under load", fontsize=7.5, color=INK2, va="bottom")
    ax.set(xlabel="draft position j", ylabel="P(j accepted | j-1 accepted)", ylim=(0.5, 1.0), xlim=(0.5, 17.5))
    ax.set_xticks([1, 4, 8, 12, 15])
    ax.legend(handles=[plt.Line2D([], [], color=C[m], lw=1.6, marker="o", ms=3.2, label=lbl)
                       for m, lbl in (("DFlash", "DFlash (k=15; Q3.8: k=7)"), ("MTP", "MTP (k=6), all 4 models"))],
              loc="lower left", frameon=False, fontsize=8)
    ax.set_title("(a) acceptance does not decay with depth: it compounds", fontsize=9.5, loc="left", color=INK)

    # (b) alpha* by concurrency
    ax = axes[1]
    ax.axhspan(0.65, 0.82, color=GRID, zorder=0)
    ax.text(4.3, 0.66, "today's drafters", fontsize=7.5, color=INK2, va="bottom")
    for model, col, mk in (("gemma-4-31B-it", INK, "o"), ("gemma-4-26B-A4B-it", NEUTRAL, "s")):
        ys = []
        for c in CONCS:
            cost = step_costs(model, c)
            ys.append(next(a / 1000 for a in range(500, 1000)
                           if max(cost, key=lambda n: tau(n, a / 1000) / cost[n]) == 16))
        ax.plot(CONCS, ys, color=col, lw=1.8, marker=mk, ms=4.5, zorder=3)
        ax.annotate(SHORT[model], (CONCS[-1], ys[-1]), xytext=(-4, 6 if mk == "s" else -10),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set(xscale="log", xlabel="requests in flight", ylabel="alpha* for a 16-token draft to win", ylim=(0.6, 1.0))
    ax.set_xticks(CONCS, [str(c) for c in CONCS])
    ax.minorticks_off()
    ax.set_title("(b) acceptance needed rises with load", fontsize=9.5, loc="left", color=INK)

    # (c) best speedup vs alpha, Gemma 4 31B
    ax = axes[2]
    model = "gemma-4-31B-it"
    alphas = [a / 100 for a in range(60, 97)]
    shades = ["#a3a29d", "#7a7974", "#4a4945", "#0b0b0b"]  # one hue, light to dark = low to high load
    ax.axvspan(0.65, 0.82, color=GRID, zorder=0)
    for c, col in zip(CONCS, shades):
        cost = step_costs(model, c)
        ys = [max(tau(n, a) / cost[n] for n in cost) for a in alphas]
        ax.plot(alphas, ys, color=col, lw=1.8, zorder=3)
        ax.annotate(f"{c} req.", (alphas[-1], ys[-1]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK2)
    ax.axhline(1.0, color=NEUTRAL, lw=1, ls="--", zorder=1)
    ax.set(xlabel="per-position acceptance alpha", ylabel="best speedup over k in {3, 7, 15}", xlim=(0.6, 1.0))
    ax.set_title("(c) Gemma 4 31B: speedup if alpha were raised", fontsize=9.5, loc="left", color=INK)

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(f"{OUT}/l5_depth.{ext}")
    print("wrote l5_depth.svg/.pdf")


if __name__ == "__main__":
    main()
