"""Experimental-design schematics for the write-up (website/plots/s*.svg).

Academic style: monochrome, thin strokes, few boxes, booktabs-style table. SVG with real text.

  GPUS= docker/run.sh python3 scripts/make_schematics.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = "/workspace/website/plots"
INK, INK2, RULE, GROUP = "#111111", "#555555", "#111111", "#f2f2f2"
plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none", "font.size": 10,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


def node(ax, x, y, w, h, text, sub=None):
    ax.add_patch(Rectangle((x, y), w, h, fill=True, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=10.5, color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.5, color=INK2,
                style="italic", zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5, color=INK, zorder=3)


def edge(ax, a, b, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color=INK,
                                 linestyle=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0, zorder=1))


def group(ax, x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=GROUP, edgecolor="none", zorder=0))
    ax.text(x + 0.08, y + h - 0.1, label, ha="left", va="top", fontsize=9, color=INK2, style="italic")


def schematic_design():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")

    # (1) inputs
    node(ax, 0.1, 2.1, 1.95, 0.75, "Target models", "dense · MoE · hybrid")
    node(ax, 0.1, 0.9, 1.95, 0.75, "Workloads", "4 suites")
    # (2) online measurement
    group(ax, 2.45, 0.55, 3.3, 2.85, "online")
    node(ax, 2.65, 1.45, 1.45, 0.95, "vLLM 0.30", "MI250 · TP2 / TP8")
    node(ax, 4.45, 2.05, 1.15, 0.7, "speedup")
    node(ax, 4.45, 0.85, 1.15, 0.7, "τ / step")
    # (3) offline analysis
    group(ax, 6.1, 0.55, 3.8, 2.85, "offline")
    node(ax, 6.3, 1.45, 1.45, 0.95, "saved outputs", "plain decoding")
    node(ax, 8.1, 2.35, 1.65, 0.62, "simulated τ")
    node(ax, 8.1, 1.6, 1.65, 0.62, "divergence")
    node(ax, 8.1, 0.85, 1.65, 0.62, "expert union")

    edge(ax, (2.05, 2.47), (2.65, 2.1))
    edge(ax, (2.05, 1.27), (2.65, 1.75))
    edge(ax, (4.1, 2.1), (4.45, 2.4))
    edge(ax, (4.1, 1.75), (4.45, 1.2))
    ax.plot([3.38, 3.38, 7.02], [1.45, 0.3, 0.3], color=INK, lw=0.9, zorder=1)  # elbow, one arrowhead
    edge(ax, (7.02, 0.3), (7.02, 1.45))
    for y in (2.66, 1.91, 1.16):
        edge(ax, (7.75, 1.92), (8.1, y))
    edge(ax, (8.1, 2.75), (5.6, 2.5), dashed=True)  # simulated τ -> projected speedup
    ax.text(3.6, 0.18, "tokens, log-probs, expert IDs", fontsize=8.5, color=INK2, style="italic", va="top")
    fig.savefig(f"{OUT}/s1_design.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote s1_design.svg")


def schematic_coverage():
    cols = ["1 request", "draft len.", "concurrency", "T = 1", "long ctx", "reasoning", "lossless"]
    F, H, O, N = "●", "◐", "○", "–"  # all 4 models run · subset run · offline simulation · not run
    groups = [
        ("vLLM runs", [
            ("MTP", [F, F, F, F, H, F, F]),
            ("DFlash", [F, F, F, F, H, F, F]),
            ("Suffix†", [F, N, N, N, H, N, F]),
            ("n-gram†", [H, N, N, N, N, N, F]),
            ("Medusa", [H, N, N, N, N, N, H]),
        ]),
        ("offline simulation", [
            ("Draft model", [O, O, N, N, N, N, N]),
            ("Layer skip", [O, O, N, N, N, N, N]),
            ("Prompt lookup", [O, N, N, N, N, N, N]),
        ]),
    ]
    nrows = sum(len(r) for _, r in groups) + len(groups)
    fig, ax = plt.subplots(figsize=(9.8, 0.33 * nrows + 1.2))
    x0, dx = 2.3, 1.2
    width = x0 + dx * len(cols) - 0.4
    ax.set_xlim(0, width); ax.set_ylim(nrows + 0.9, -1.0); ax.axis("off")

    ax.plot([0, width], [-0.95, -0.95], color=RULE, lw=1.1)  # top rule
    for j, c in enumerate(cols):
        ax.text(x0 + j * dx, -0.45, c, ha="center", va="center", fontsize=9.5, color=INK)
    ax.plot([0, width], [0.0, 0.0], color=RULE, lw=0.6)  # mid rule
    y = 0.6
    for g, rows in groups:
        ax.text(0.0, y, g, ha="left", va="center", fontsize=9, color=INK2, style="italic")
        y += 0.85
        for name, cells in rows:
            ax.text(0.2, y, name, ha="left", va="center", fontsize=10, color=INK)
            for j, c in enumerate(cells):
                ax.text(x0 + j * dx, y, c, ha="center", va="center", fontsize=12 if c != N else 10,
                        color=INK if c != N else "#b5b5b5")
            y += 0.85
    ax.plot([0, width], [y - 0.35, y - 0.35], color=RULE, lw=1.1)  # bottom rule
    ax.text(0.0, y + 0.25, f"{F} all 4 models   {H} subset   {O} offline simulation   {N} not run   "
            "† wrong output on hybrid Qwen models", ha="left", va="center", fontsize=8.5, color=INK2)
    ax.set_ylim(y + 0.6, -1.0)
    fig.savefig(f"{OUT}/s2_coverage.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote s2_coverage.svg")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    schematic_design()
    schematic_coverage()
