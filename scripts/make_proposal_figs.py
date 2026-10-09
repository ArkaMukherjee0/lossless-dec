"""Figures for the sibling-conditioned drafting proposal (lit/l7_selector.*, lit/s4_proposal.*).

  GPUS= docker/run.sh python3 scripts/make_proposal_figs.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

import make_depth_plot as style  # noqa: E402  (chart rcParams and palette)

OUT = "/workspace/lit"
SEL = "/workspace/results/m2/gemma-4-31B-it/dflash15_selector.json"


def selector_figure():
    res = json.load(open(SEL))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    groups = [("G2_concurrent", "2 siblings\nconcurrent"), ("G7_concurrent", "7 siblings\nconcurrent"),
              ("G2_finished", "2 siblings\nfinished"), ("G7_finished", "7 siblings\nfinished")]
    bars = [("drafter", "DFlash alone", style.C["DFlash"]), ("switch8", "switch (match >= 8)", "#b9b8b3"),
            ("learned", "learned rule (9 features)", style.INK2), ("oracle_tree", "oracle", style.INK)]
    for i, (key, lbl, col) in enumerate(bars):
        xs = [x + (i - 1.5) * 0.2 for x in range(len(groups))]
        ys = [res[g]["best"]["overall"][key] for g, _ in groups]
        ax.bar(xs, ys, 0.18, color=col, label=lbl, zorder=2)
        if key == "learned":
            for x, y in zip(xs, ys):
                ax.text(x, y + 0.06, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=style.INK2)
    ax.set_xticks(range(len(groups)), [lbl for _, lbl in groups])
    ax.set(ylabel="tau (tokens per verify step)", ylim=(0, 6.6))
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("(a) a cheap learned rule recovers most of the oracle gain", fontsize=9.5, loc="left",
                 color=style.INK)

    ax = axes[1]
    coef = res["G7_finished"]["best"]["coef"]
    names = {"match_len": "suffix-match length", "n_matching": "siblings that match", "next_token_votes":
             "siblings agreeing on next token", "drafter_p1": "drafter p(top-1), pos. 1", "drafter_p2":
             "drafter p(top-1), pos. 2", "drafter_exp_len": "drafter's expected accepted length",
             "agree_len": "drafter-sibling agreement", "sib_prop_len": "sibling proposal length",
             "log_pos": "position in output (log)"}
    items = sorted(coef.items(), key=lambda kv: kv[1])
    ax.barh(range(len(items)), [v for _, v in items], 0.6,
            color=[style.INK if v > 0 else style.C["DFlash"] for _, v in items], zorder=2)
    ax.set_yticks(range(len(items)), [names[k] for k, _ in items], fontsize=8)
    ax.axvline(0, color=style.NEUTRAL, lw=1)
    ax.set(xlabel="weight (standardized); > 0 favours the sibling")
    ax.set_title("(b) what the rule looks at (7 finished siblings)", fontsize=9.5, loc="left", color=style.INK)
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(f"{OUT}/l7_selector.{ext}")


def proposal_schematic():
    """Correlated requests of one prompt share a group memory; the drafter of each request reads it."""
    import make_schematics as sch  # monochrome schematic style (resets rcParams)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0.1, 4.4); ax.axis("off")
    ink, ink2 = sch.INK, sch.INK2

    # left: one prompt, G samples at different progress
    sch.node(ax, 0.1, 1.75, 1.2, 0.95, "prompt", "GRPO group /\nbest-of-N")
    rows = [(3.5, 1.0, "sample 1: finished"), (2.75, 0.75, "sample 2"), (2.0, 0.45, "sample 3"),
            (1.25, 0.6, "sample i (drafting now)")]
    for y, frac, lbl in rows:
        ax.add_patch(Rectangle((1.9, y - 0.14), 2.6, 0.28, facecolor="white", edgecolor=ink, lw=0.8))
        ax.add_patch(Rectangle((1.9, y - 0.14), 2.6 * frac, 0.28, facecolor=sch.GROUP, edgecolor="none"))
        ax.text(1.95, y, lbl, va="center", fontsize=8.5, color=ink)
        sch.edge(ax, (1.3, 2.2), (1.9, y))
    ax.text(3.2, 0.95, "same prompt, T = 1: siblings write similar content", ha="center", va="top", fontsize=8,
            color=ink2, style="italic")

    # middle: group memory
    ax.add_patch(Rectangle((4.95, 0.75), 2.3, 3.2, facecolor="#e3f4ec", edgecolor="none", zorder=0))
    ax.text(5.03, 3.85, "group memory", ha="left", va="top", fontsize=9, color="#13805a", style="italic")
    sch.node(ax, 5.15, 2.75, 1.9, 0.7, "sibling tokens", "suffix index")
    sch.node(ax, 5.15, 1.6, 1.9, 0.7, "sibling states", "target hidden / KV")
    for y in (3.5, 2.75, 2.0):
        sch.edge(ax, (4.5, y), (5.15, 3.1 if y > 2.5 else 1.95))

    # right: drafter + verifier for sample i
    sch.node(ax, 7.75, 2.05, 1.45, 1.1, "drafter", "block drafter +\nsibling cross-attn")
    sch.node(ax, 9.65, 2.05, 1.25, 1.1, "target", "verifies k+1\n(lossless)")
    for a_, b_ in (((7.05, 3.1), (7.75, 2.85)), ((7.05, 1.95), (7.75, 2.35))):  # what the drafter reads from siblings
        ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>", mutation_scale=9, lw=1.4, color="#1baf7a", zorder=1))
    # sample i's own context goes around (below) the memory, straight to its drafter
    ax.plot([4.5, 4.75, 4.75, 8.47, 8.47], [1.25, 1.25, 0.45, 0.45, 1.85], color=ink, lw=0.9, zorder=1)
    sch.edge(ax, (8.47, 1.85), (8.47, 2.05))
    ax.text(6.6, 0.35, "sample i's own context (target features)", fontsize=8, color=ink2, style="italic",
            ha="center", va="top")
    sch.edge(ax, (9.2, 2.6), (9.65, 2.6))
    ax.text(9.42, 2.9, "k drafts", fontsize=8, color=ink2, ha="center", style="italic")
    ax.add_patch(FancyArrowPatch((10.27, 3.15), (6.1, 3.45), connectionstyle="arc3,rad=0.35", arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color=ink, ls=(0, (3, 2)), zorder=1))
    ax.text(8.3, 4.25, "accepted tokens and their target states join the memory", fontsize=8, color=ink2,
            style="italic", ha="center")
    fig.savefig(f"{OUT}/s4_proposal.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}/s4_proposal.pdf", bbox_inches="tight")


if __name__ == "__main__":
    selector_figure()
    proposal_schematic()
    print("wrote l7_selector, s4_proposal")
