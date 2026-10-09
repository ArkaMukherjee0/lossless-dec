"""Intuition figures for the sibling-drafting proposal.

  lit/m0_story.*      the motivation in four steps, each panel from our own measurements
  lit/m1_mechanism.*  one real draft step, token by token: the drafter alone vs a draft read from a sibling

  GPUS= docker/run.sh python3 scripts/make_story_figs.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import make_depth_plot as style  # noqa: E402  (chart rcParams and palette)

R, OUT = "/workspace/results", "/workspace/lit"
SIB = "#1baf7a"        # siblings: the retrieval colour of the site's palette
SIB_LIGHT = "#d6f0e5"
DFL = style.C["DFlash"]
DFL_LIGHT = "#fbe0d4"
MODEL = "gemma-4-31B-it"


def run(tag, c):
    return json.load(open(f"{R}/dk_grid/{MODEL}/{tag}_c{c}.json"))


def story():
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8), gridspec_kw={"wspace": 0.42})
    concs = [4, 16, 64, 128]

    # 1. load raises cost, not acceptance
    ax = axes[0]
    tau = [run("dflash15", c)["spec_decode_acceptance_length"] for c in concs]
    cost = [t / (run("dflash15", c)["output_throughput"] / run("ar", c)["output_throughput"]) for t, c in zip(tau, concs)]
    ax.plot(concs, [t / tau[0] for t in tau], color=DFL, lw=2, marker="o", ms=5, zorder=3)
    ax.plot(concs, [c / cost[0] for c in cost], color=style.INK, lw=2, marker="s", ms=5, zorder=3)
    ax.annotate("accepted tokens per step", (128, tau[-1] / tau[0]), xytext=(-2, 9), textcoords="offset points",
                ha="right", fontsize=8.5, color=DFL)
    ax.annotate("cost of verifying\n16 tokens", (64, cost[2] / cost[0]), xytext=(4, -14), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=style.INK)
    ax.set(xscale="log", xlabel="requests in flight", ylabel="relative to 4 requests", ylim=(0, 3.8))
    ax.set_xticks(concs, [str(c) for c in concs])
    ax.minorticks_off()
    ax.set_title("① Load raises the cost of a draft,\n    not its acceptance", fontsize=10.5, loc="left", color=style.INK)

    # 2. acceptance compounds
    ax = axes[1]
    cum = run("dflash15", 4)["spec_decode_per_position_acceptance_rates"]
    js = range(1, len(cum) + 1)
    for a, col, lbl in ((0.73, DFL, "today: α ≈ 0.73 per token → τ ≈ 3.6"), (0.9, SIB, "goal: α = 0.9 → τ ≈ 8.1")):
        ys = [a ** j for j in js]
        ax.fill_between(js, ys, color=col, alpha=0.15, lw=0)
        ax.plot(js, ys, color=col, lw=2, label=lbl)
    ax.plot(js, cum, "o", color=DFL, ms=4, mec="white", mew=0.6, zorder=4, label="measured (DFlash, Gemma 4 31B)")
    ax.set(xlabel="draft position j", ylabel="P(token j is accepted)", ylim=(0, 1.0), xlim=(0.5, 15.5))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=style.GRID))
    labels.append("shaded area = extra tokens per step")
    ax.legend(handles, labels, frameon=False, fontsize=7.8, loc="upper right")
    ax.set_title("② Acceptance compounds, so long drafts\n    only pay if α is high", fontsize=10.5, loc="left",
                 color=style.INK)

    # 3. why drafts fail
    ax = axes[2]
    rows = [("Gemma 4 31B\n+ DFlash", "gemma-4-31B-it/dflash15"), ("Gemma 4 26B-A4B\n+ DFlash", "gemma-4-26B-A4B-it/dflash15"),
            ("Qwen3.8-27B\n+ DFlash2", "Qwen3.8-27B/dflash7")]
    segs = [((0, 1), "target near-tied\n(< 1 nat)", "#c9c8c3"), ((1, 8), "target sure\n(1–8 nats)", "#6b6a66"),
            ((8, 1e9), "target very sure\n(≥ 8 nats)", style.INK)]
    for i, (lbl, path) in enumerate(rows):
        bins = json.load(open(f"{R}/m1/{path}_anatomy.json"))["all"]["by_margin"]
        left = 0
        for (lo, hi), _, col in segs:
            share = sum(b["share_of_rejections"] for b in bins
                        if lo <= float(b["margin"].split("-")[0]) < hi)
            ax.barh(i, share, left=left, color=col, height=0.6, zorder=2)
            if share > 0.08:
                ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if col != "#c9c8c3" else style.INK)
            left += share
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="share of rejected draft tokens")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in segs], labels=[s[1] for s in segs],
              frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("③ Drafts fail where the target is sure:\n    the drafter lacks the content", fontsize=10.5,
                 loc="left", color=style.INK)

    # 4. siblings already wrote it
    ax = axes[3]
    sel = json.load(open(f"{R}/m2/{MODEL}/dflash15_selector.json"))
    sib = json.load(open(f"{R}/m2/{MODEL}/dflash15_siblings.json"))["all"]
    gs = [1, 2, 4, 7]
    ax.axhline(sib["drafter"], color=DFL, lw=2)
    ax.text(7, sib["drafter"] - 0.1, "drafter alone", ha="right", va="top", fontsize=8.5, color=DFL)
    for var, key, col, mk in (("finished", "done", SIB, "o"), ("concurrent", "sib", "#7fcfae", "s")):
        ax.plot(gs, [sib[f"{key}{g}_oracle"] for g in gs], color=col, lw=1.4, ls="--", marker=mk, ms=4, zorder=3)
        ax.plot([2, 7], [sel[f"G{g}_{var}"]["best"]["overall"]["learned"] for g in (2, 7)], color=col, lw=2.4,
                marker=mk, ms=6, zorder=4)
        ax.annotate(f"{var} siblings", (7, sib[f"{key}7_oracle"]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=col)
    ax.plot([], [], color=style.INK2, lw=2.4, label="learned rule (cross-validated)")
    ax.plot([], [], color=style.INK2, lw=1.4, ls="--", label="best possible choice (oracle)")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.set_xticks(gs)
    ax.set(xlabel="sibling samples of the same prompt", ylabel="τ (tokens per verify step)", ylim=(3, 6.4),
           xlim=(0.6, 9.8))
    ax.set_title("④ Siblings already wrote it: acceptance\n    rises with the size of the group", fontsize=10.5,
                 loc="left", color=style.INK)

    for i, txt in enumerate(["so per-token acceptance\nmust rise", "why do drafts\nfail?", "where is the\nmissing content?"]):
        x0 = axes[i].get_position().x1 + 0.004
        x1 = axes[i + 1].get_position().x0 - 0.035
        fig.add_artist(FancyArrowPatch((x0, 0.885), (x1 + 0.03, 0.885), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=12, lw=1.2, color=style.NEUTRAL))
        fig.text((x0 + x1 + 0.03) / 2, 0.9, txt, ha="center", va="bottom", fontsize=8, color=style.INK2,
                 style="italic")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.72, bottom=0.18)
    for ext in ("svg", "pdf"):
        fig.savefig(f"{OUT}/m0_story.{ext}")


def token_row(ax, x, y, toks, face="white", edge=style.GRID, color=style.INK, marks=None, outline=()):
    """Draw tokens as boxes starting at x; returns the x after the last box and each box's (x0, x1)."""
    spans = []
    for i, t in enumerate(toks):
        s = t.replace("\n", "⏎").replace(" ", "·") if t.strip() == "" else t.replace("\n", "⏎")
        w = 0.085 * max(len(s), 1) + 0.16
        ec = SIB if i in outline else edge
        ax.add_patch(FancyBboxPatch((x, y - 0.17), w, 0.34, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=face if not callable(face) else face(i), edgecolor=ec,
                                    lw=1.8 if i in outline else 0.8, zorder=2))
        ax.text(x + w / 2, y, s, ha="center", va="center", fontsize=9, color=color if not callable(color) else color(i),
                zorder=3, family="DejaVu Sans")
        if marks:
            m = marks(i)
            if m:
                ax.text(x + w / 2, y + 0.3, m, ha="center", va="center", fontsize=9.5, zorder=3,
                        color=SIB if m == "✓" else DFL, fontweight="bold")
        spans.append((x, x + w))
        x += w + 0.04
    return x, spans


def mechanism():
    ex = next(e for e in json.load(open(f"{OUT}/sibling_examples.json"))["examples"] if e["category"] == "qa")
    m = ex["match_len"]
    fig, ax = plt.subplots(figsize=(22, 5.6))
    ax.set_xlim(0, 22); ax.set_ylim(0, 5.6); ax.axis("off")
    lab = dict(ha="right", va="center", fontsize=9.5, color=style.INK)
    sub = dict(ha="right", va="center", fontsize=8, color=style.INK2, style="italic")

    # sibling row: its own wording, the matched suffix, then what it wrote next (green)
    x_match_end = 1.95 + sum(0.085 * max(len(t.replace("\n", "⏎")), 1) + 0.2 for t in ex["sibling_before_tokens"])
    sb = ex["sibling_before_tokens"]
    xs, _ = token_row(ax, 1.95, 4.75, sb, outline=range(len(sb) - m, len(sb)))
    x_right, _ = token_row(ax, xs, 4.75, ex["sibling_tokens"], face=SIB_LIGHT, edge=SIB)
    ax.text(1.8, 4.85, "a sibling sample", **lab)
    ax.text(1.8, 4.6, "(finished earlier)", **sub)

    # sample i so far, right-aligned so its last tokens sit under the sibling's matched suffix
    ctx = ex["context_tokens"]
    width = sum(0.085 * max(len(t.replace("\n", "⏎")), 1) + 0.2 for t in ctx)
    x0 = x_match_end - width
    xe, _ = token_row(ax, x0, 3.55, ctx, outline=range(len(ctx) - m, len(ctx)))
    ax.text(min(x0, 1.95) - 0.15, 3.65, "sample i so far", **lab)
    ax.text(min(x0, 1.95) - 0.15, 3.4, "(being drafted now)", **sub)
    ax.annotate(f"same last {m} tokens: a suffix match", ((x_match_end + x0) / 2 + 2.2, 4.2), fontsize=8.5,
                color=SIB, ha="center", style="italic")

    # drafter alone: its first guess is already wrong, so the whole chain is thrown away
    d = ex["drafter_tokens"][:10]
    token_row(ax, xe, 2.35, d, face=lambda i: DFL_LIGHT if i == 0 else "#f3f3f1",
              edge=DFL, color=lambda i: style.INK if i == 0 else "#a9a8a3",
              marks=lambda i: "✗" if i == 0 else "")
    ax.text(xe - 0.15, 2.45, "drafter alone", **lab)
    ax.text(xe - 0.15, 2.2, "1 token this step", **sub)

    # group-aware draft: reads the sibling's continuation; every token survives verification
    s = ex["sibling_tokens"]
    acc = ex["sibling_accepted"]
    xend, spans = token_row(ax, xe, 1.2, s, face=SIB_LIGHT, edge=SIB, marks=lambda i: "✓" if i < acc else "✗")
    ax.text(xe - 0.15, 1.3, "group-aware draft", **lab)
    ax.text(xe - 0.15, 1.05, f"{acc + 1} tokens this step", **sub)
    ax.add_patch(FancyArrowPatch((xe + 1.0, 4.5), (xe + 1.0, 1.55), arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                 color=SIB, connectionstyle="arc3,rad=-0.15", zorder=1))
    ax.text(xe + 1.35, 3.0, "read the\nsibling's\ncontinuation", fontsize=8.5, color=SIB, style="italic", va="center")

    ax.text(0.2, 0.35, "The target verifies every draft exactly as in ordinary speculative decoding (✓ = matches what the "
                       "target produces), so the output is unchanged; only the number of tokens produced per target "
                       "pass grows.\nReal step from our Gemma 4 31B runs (Spec-Bench QA, T = 1). The sibling worded the "
                       "sentence differently (\"represented\"), but the fact it wrote next is exactly what sample i needs.",
            fontsize=8.5, color=style.INK2, va="center")
    width = max(x_right, xend) + 0.3  # canvas fits the longest token row (1 data unit = 1 inch)
    ax.set_xlim(0, width)
    fig.set_size_inches(width, 5.6)
    for ext in ("svg", "pdf"):
        fig.savefig(f"{OUT}/m1_mechanism.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    story()
    mechanism()
    print("wrote m0_story, m1_mechanism")
