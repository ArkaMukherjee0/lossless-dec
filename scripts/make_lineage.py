"""Lineage tree of lossless speculative decoding (website/plots/s3_lineage.svg).

Boxes are papers. Each junction (dot + italic text) is one limitation of the parent; the papers to its
right are the ones that set out to fix it. Same academic style as make_schematics.py.

  docker/run.sh python3 scripts/make_lineage.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

from make_schematics import GROUP, INK, INK2, OUT  # noqa: E402  (also applies the shared rcParams)


def P(name, venue, *branches, ours=False):
    """Paper node. ours=True: evaluated in this study (drawn with a heavy border)."""
    return {"name": name, "venue": venue, "branches": list(branches), "ours": ours}


def B(problem, *kids):
    """Junction: a limitation of the parent, and the papers that address it."""
    return (problem, list(kids))


PRECURSOR = P("Blockwise parallel", "NeurIPS '18")
PRECURSOR_EDGE = "needs a fine-\ntuned base model;\ngreedy only"
ROOT = ("Speculative decoding", "ICML '23 · SpS arXiv '23")

BANDS = [
    ("what drafts", [
        B("a second model to train\nand host next to the target",
          P("Draft & Verify", "ACL '24",
            B("skip set found by slow offline\nsearch, fixed per task", P("SWIFT", "ICLR '25")),
            B("skipped-layer drafts are\nweak without training", P("LayerSkip", "ACL '24"), P("Kangaroo", "NeurIPS '24")),
            B("layer skipping is coarse; at\nbatch 1 weight reads dominate", P("Cassandra", "ISCA '26")),
            ours=True),
          P("Medusa", "ICML '24",
            B("each head guesses t+k blind\nto the drafted tokens before it",
              P("Hydra", "COLM '24"),
              P("EAGLE", "ICML '24",
                B("static draft tree ignores\ncontext-dependent confidence",
                  P("EAGLE-2", "EMNLP '24",
                    B("feature-regression loss caps\ngains from more data",
                      P("EAGLE-3", "NeurIPS '25",
                        B("head still drafts k tokens\nin k sequential passes", P("P-EAGLE", "arXiv '26")),
                        ours=True)),
                    B("trained on target features,\ndrafts on its own (exposure bias)",
                      P("HASS", "ICLR '25"), P("GRIFFIN", "NeurIPS '25"))))),
              P("ReDrafter", "arXiv '24")),
            B("heads bolted onto a model\npretrained for one token", P("MTP", "ICML '24 · DSv3", ours=True)),
            ours=True)),
        B("drafter must share the\ntarget's vocabulary", P("Hetero-vocab SD", "ICML '25")),
        B("outputs often copy spans\nalready in the context",
          P("LLMA", "arXiv '23",
            B("needs a reference to copy;\nopen-ended text has none",
              P("REST", "NAACL '24",
                B("static datastore: big, domain-\nbound; misses the app's reuse",
                  P("SuffixDecoding", "NeurIPS '25", ours=True), P("SAM Decoding", "ACL '25", ours=True)),
                B("exact-suffix lookup often\nfinds no match", P("LogitSpec", "ACL Findings '26"))),
              P("Lookahead", "ICML '24"))),
          P("Prompt lookup", "code '23", ours=True)),
        B("drafter disagrees with the\ntarget: low acceptance",
          P("DistillSpec", "ICLR '24",
            B("offline distillation misses the\nlive query distribution", P("Online SD", "ICML '24")))),
        B("the drafter is autoregressive\ntoo: k serial draft passes",
          P("Staged SD", "ES-FoMo '23",
            B("same drafter for every position;\nlate tokens are rarely kept", P("Cascade SD", "NeurIPS '24"))),
          P("ParallelSpec", "arXiv '24",
            B("parallel drafter retrained\nfor every target", P("PARD", "ICLR '26"))),
          P("SpecDiff", "NAACL '25",
            B("standalone diffusion LM: large,\nblind to the target's state",
              P("DFlash", "ICML '26",
                B("positions drafted independently\nconflict; fixed block length",
                  P("DSpark", "arXiv '26")),
                ours=True))),
          ),
    ]),
    ("draft shape & verification", [
        B("one draft chain: the first\nrejection discards the rest",
          P("SpecInfer", "ASPLOS '24",
            B("hand-designed tree shape does\nnot scale with budget/hardware",
              P("Sequoia", "NeurIPS '24",
                B("trees sized for GPUs; offloaded\ntargets can verify 1000s",
                  P("SpecExec", "NeurIPS '24"))))),
          P("SpecTr", "NeurIPS '23",
            B("OT selection is approximate;\ngap to optimum unknown", P("Multi-draft opt.", "ICLR '25")))),
        B("token-by-token checks reject\nmore than necessary",
          P("Block verif.", "ICLR '25",
            B("single chain only; no trees\nor multiple drafts",
              P("Traversal verif.", "NeurIPS '25"), P("HSD", "ICLR '26")))),
        B("fixed draft length k: too long\nwastes, too short under-uses",
          P("SpecDec++", "COLM '25")),
    ]),
    ("systems & serving", [
        B("draft and target idle while\nthe other runs (mutual waiting)",
          P("PEARL", "ICLR '25"),
          P("AMUSD", "ISCAS '25",
            B("after a rejection, drafting is\nback on the critical path", P("SSD", "ICLR '26"))),
          P("DSI", "ICLR '25")),
        B("long context: KV reads dominate,\nsmall drafters can't keep up",
          P("TriForce", "COLM '24",
            B("batch 1 only; SD assumed\nuseless at large batch",
              P("MagicDec", "ICLR '25",
                B("sparse-KV self-draft: low\nacceptance, extra memory", P("QuantSpec", "ICML '25")),
                B("static sparsity fits long\nreasoning traces poorly", P("SparseSpec", "MLSys '26")))),
            B("drafting with the full target\n(sparse KV) is still heavy",
              P("LongSpec", "ACL '26"), P("SpecExtend", "ACL Findings '26")))),
        B("large batches: verification turns\ncompute-bound, speedup vanishes",
          P("SmartSpec", "arXiv '24",
            B("optimises throughput,\nignores latency SLOs", P("AdaSpec", "SoCC '25")))),
    ]),
]

# Geometry in inches; the axes map 1 data unit to 1 inch, with y growing downward.
FS_NAME, FS_VENUE, FS_PROB, FS_BAND = 9.5, 7.5, 7.8, 9
BOX_H, PITCH, FAMILY_GAP, BAND_GAP = 0.38, 0.47, 0.16, 0.42
PAD, TRUNK, BUS, LABEL_PAD = 0.1, 0.1, 0.16, 0.07
MARGIN = 0.15


def text_w(s, size, style="normal"):
    prop = FontProperties(family="DejaVu Serif", style=style)
    return 1.06 * max(TextPath((0, 0), line, size=size, prop=prop).get_extents().width
                      for line in s.split("\n")) / 72


def walk(n, d=0):
    n["d"] = d
    yield n
    for _, kids in n["branches"]:
        for k in kids:
            yield from walk(k, d + 1)


def layout(root):
    nodes = list(walk(root))
    depth = max(n["d"] for n in nodes) + 1
    colw = [max(max(text_w(n["name"], FS_NAME), text_w(n["venue"], FS_VENUE, "italic")) + 2 * PAD
                for n in nodes if n["d"] == d) for d in range(depth)]
    gapw = [max([text_w(p, FS_PROB, "italic") for n in nodes if n["d"] == d for p, _ in n["branches"]], default=0)
            + TRUNK + 2 * LABEL_PAD + BUS for d in range(depth - 1)]
    xs = [MARGIN]
    for d in range(depth - 1):
        xs.append(xs[-1] + colw[d] + gapw[d])
    for n in nodes:
        n["x"], n["w"] = xs[n["d"]], colw[n["d"]]
    return xs[-1] + colw[-1] + MARGIN


def place(n, cur):
    """Leaves take consecutive rows; a parent sits midway between its first and last junction."""
    if not n["branches"]:
        n["y"] = cur[0]
        cur[0] += PITCH
        return n["y"]
    n["stems"] = []
    for _, kids in n["branches"]:
        ys = [place(k, cur) for k in kids]
        n["stems"].append((ys[0] + ys[-1]) / 2)
    n["y"] = (n["stems"][0] + n["stems"][-1]) / 2
    return n["y"]


def draw_node(ax, n):
    lw = 1.7 if n["ours"] else 0.8
    ax.add_patch(Rectangle((n["x"], n["y"] - BOX_H / 2), n["w"], BOX_H, facecolor="white", edgecolor=INK,
                           linewidth=lw, zorder=3))
    cx = n["x"] + n["w"] / 2
    ax.text(cx, n["y"] - 0.065, n["name"], ha="center", va="center", fontsize=FS_NAME, color=INK, zorder=4)
    ax.text(cx, n["y"] + 0.095, n["venue"], ha="center", va="center", fontsize=FS_VENUE, color=INK2,
            style="italic", zorder=4)


def line(ax, xs, ys, **kw):
    ax.plot(xs, ys, color=INK, lw=0.8, solid_capstyle="butt", zorder=1, **kw)


def draw_edges(ax, n):
    if not n["branches"]:
        return
    xr = n["x"] + n["w"]
    kid_x = n["branches"][0][1][0]["x"]
    xb = kid_x - BUS
    xs = xr
    if len(n["stems"]) > 1:
        xs = xr + TRUNK
        line(ax, [xr, xs], [n["y"], n["y"]])
        line(ax, [xs, xs], [n["stems"][0], n["stems"][-1]])
    for (prob, kids), y in zip(n["branches"], n["stems"]):
        line(ax, [xs, xb], [y, y])
        ax.plot([xb], [y], "o", ms=3.2, color=INK, zorder=2)
        ax.text(xs + LABEL_PAD, y - 0.035, prob, ha="left", va="bottom", fontsize=FS_PROB, color=INK2,
                style="italic", linespacing=1.15, zorder=2)
        ky = [k["y"] for k in kids]
        line(ax, [xb, xb], [min(ky + [y]), max(ky + [y])])
        for k in kids:
            line(ax, [xb, kid_x], [k["y"], k["y"]])
            draw_edges(ax, k)


def lineage():
    root = P(*ROOT, *[b for _, bs in BANDS for b in bs], ours=True)
    width = layout(root)

    # Bands top to bottom with a little air between families; the root sits level with its first
    # junction so the tree reads from the top-left, and the precursor sits just above it.
    cur = [MARGIN + PITCH / 2 + 0.12]
    spans, stems = [], []
    for band, branches in BANDS:
        y0 = cur[0]
        for problem, kids in branches:
            ys = [place(k, cur) for k in kids]
            stems.append((ys[0] + ys[-1]) / 2)
            cur[0] += FAMILY_GAP
        spans.append((band, y0, cur[0] - PITCH - FAMILY_GAP))
        cur[0] += BAND_GAP
    root["stems"] = stems
    root["y"] = stems[0]
    bottom = cur[0] - BAND_GAP - PITCH + BOX_H / 2
    height = bottom + 0.55

    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis("off")

    band_x = root["x"] + root["w"] + TRUNK + 0.03
    for band, y0, y1 in spans:
        ax.add_patch(Rectangle((band_x, y0 - PITCH / 2 - 0.12), width - MARGIN - band_x, y1 - y0 + PITCH + 0.2,
                               facecolor=GROUP, edgecolor="none", zorder=0))
        ax.text(width - MARGIN - 0.08, y0 - PITCH / 2 - 0.04, band, ha="right", va="top", fontsize=FS_BAND,
                color=INK2, style="italic")

    nodes = list(walk(root))
    draw_edges(ax, root)
    for n in nodes:
        draw_node(ax, n)

    # Precursor sits above the root, joined by a dashed edge.
    pre = dict(PRECURSOR, x=root["x"], w=root["w"], y=root["y"] - 1.1)
    draw_node(ax, pre)
    cx = root["x"] + root["w"] / 2
    ax.plot([cx, cx], [pre["y"] + BOX_H / 2, root["y"] - BOX_H / 2], color=INK, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(cx + LABEL_PAD, (pre["y"] + root["y"]) / 2, PRECURSOR_EDGE, ha="left", va="center", fontsize=FS_PROB,
            color=INK2, style="italic", linespacing=1.15)

    ax.text(MARGIN, bottom + 0.3,
            "● junction = one limitation of the parent; papers to its right address it.   "
            "Heavy border: evaluated in this study.   Venue = first peer-reviewed venue (arXiv/code otherwise).",
            ha="left", va="center", fontsize=8, color=INK2)
    return fig


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig = lineage()
    fig.savefig(f"{OUT}/s3_lineage.svg")
    plt.close(fig)
    print("wrote s3_lineage.svg")
