"""Build website/amo/: figures + index.html summarizing Amo (Agentic Model Optimization) results so far.

Numbers are transcribed from the Amo repo (/home/arkamukh/amo); each block cites its source file.
  GPUS= docker/run.sh python3 website/amo/build.py
"""
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/plots"
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"

# ── data ─────────────────────────────────────────────────────────────────────────────────────────
# runs/throughput-qwen38-27b-ci/status.md (Result) and report.md: IX output tok/s, c=64 1k:1k, 3-run bootstrap CI
CHAIN_27B = [("baseline", 689.2, 684.8, 695.2),
             ("int8 state\n+ split-KV", 762.5, 746.9, 781.5),
             ("+ small-tile\nGEMM", 905.1, 898.4, 915.3),
             ("+ DFlash2\nspec (K=3)", 1050.6, 1015.9, 1071.1)]
# runs/throughput-qwen36-35b-a3b-ci/report.md (Final result)
CHAIN_35B = [("baseline", 1079.8, 1059.3, 1100.0),
             ("MoE\ntables", 1254.3, 1243.6, 1265.8),
             ("+ MTP\nspec (K=4)", 2434.6, 2358.6, 2512.2),
             ("+ int8\nexperts", 2562.3, 2519.3, 2628.9),
             ("+ fused\nprefill", 2689.5, 2640.7, 2718.3),
             ("+ kernel\nstack", 2918.7, 2836.6, 3008.4),
             ("+ hybrid\ndrafter", 3413.4, 3368.1, 3480.2)]
# runs/throughput-qwen38-27b-ci/status.md: winner vs baseline grid, single runs
GRID_27B = {"1k:1k": (2.14, 2.25, 1.40), "8k:1k": (2.59, 1.41, 1.21), "1k:8k": (3.06, 3.54, 2.23)}
# runs/mi250x_qwen3.8_report.md §4 (Qwen3.8-27B, TP=2, c=64 1k:1k); verdict: ok | loss | bad
FEAS = [("fp8 per-token-head KV", 1.02, 1.93, "+0.01%", "ok"),
        ("int8 per-token-head KV", 0.89, 1.93, "−0.01%", "ok"),
        ("fp8 e4m3 KV", 0.85, 1.96, "0.00%", "ok"),
        ("turboquant k8v4", 0.87, 2.55, "+0.05%", "loss"),
        ("turboquant 4bit_nc", 0.83, 3.70, "+0.05%", "loss"),
        ("turboquant k3v4_nc", 0.82, 4.16, "+0.05%", "loss"),
        ("turboquant 3bit_nc", 0.70, 4.79, "+0.08%", "loss"),
        ("gpu-mem-util 0.95", 0.98, 1.08, "0.00%", "ok"),
        ("AITER", 1.01, 1.00, "0.00%", "ok"),
        ("FP8 block weights", 0.13, 1.44, "+0.30%", "bad"),
        ("MXFP4 weights", 0.52, 1.67, "+10.5%", "bad")]
# docs/amo-findings.md (MI355X, Aug 2026)
MI355 = [("Qwen3-14B", "throughput", "SGLang + torch-compile", "+0.6–4.6%", "—"),
         ("Qwen3-14B", "KV capacity", "fp8 weights + turboquant k8v4 KV", "tie-break", "2.39×"),
         ("Qwen3-14B", "memory + concurrency", "4bit_nc KV + fp8 weights + mem-util 0.95", "tie-break", "3.35×"),
         ("Qwen3-14B", "throughput + memory", "fp8 KV + fp8 weights + mem-util 0.95", "1.54× @ c512", "2.25×"),
         ("GPT-OSS-120B", "throughput + memory", "fp8 e4m3 KV (throughput at HW ceiling)", "flat", "2.00×")]

sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "grid.color": GRID, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 9, "legend.frameon": False,
    "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ── figures ──────────────────────────────────────────────────────────────────────────────────────
def fig_chains():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), gridspec_kw={"width_ratios": [4, 7]})
    for ax, (title, chain) in zip(axes, [("Qwen3.8-27B (dense, hybrid)", CHAIN_27B),
                                         ("Qwen3.6-35B-A3B (MoE, hybrid)", CHAIN_35B)]):
        base = chain[0][1]
        xs = range(len(chain))
        ys = [c[1] for c in chain]
        err = [[c[1] - c[2] for c in chain], [c[3] - c[1] for c in chain]]
        ax.bar(xs, ys, 0.62, color=[NEUTRAL] + [BLUE] * (len(chain) - 1), zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
        for x, y in zip(xs, ys):
            ax.text(x, y * 1.02 + 20, f"{y / base:.2f}×", ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(list(xs), [c[0] for c in chain])
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("Output tok/s (c=64, 1k:1k)")
        ax.set_ylim(0, max(c[3] for c in chain) * 1.15)
    save(fig, "amo01_kept_chains")


def fig_grid():
    import numpy as np
    shapes = list(GRID_27B)
    data = np.array([GRID_27B[s] for s in shapes])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    sns.heatmap(data, ax=ax, cmap=sns.light_palette(BLUE, as_cmap=True), vmin=1, vmax=3.6, annot=True,
                fmt=".2f", annot_kws={"fontsize": 10}, linewidths=2, linecolor=SURFACE,
                xticklabels=["c=4", "c=16", "c=64"], yticklabels=[f"ISL:OSL {s}" for s in shapes],
                cbar_kws={"label": "Speedup vs baseline"})
    ax.tick_params(length=0)
    save(fig, "amo02_grid_27b")


LABEL_POS = {"AITER": (0.0, 0.05, "center"), "gpu-mem-util 0.95": (0.07, -0.03, "left"),
             "int8 per-token-head KV": (-0.07, 0.0, "right"), "fp8 e4m3 KV": (0.07, -0.035, "left"),
             "fp8 per-token-head KV": (0.07, 0.02, "left"), "turboquant k8v4": (0.0, 0.045, "center"),
             "turboquant 4bit_nc": (0.0, 0.045, "center"), "turboquant k3v4_nc": (0.0, -0.045, "center"),
             "turboquant 3bit_nc": (0.0, -0.045, "center")}


def fig_feasibility():
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    style = {"ok": (BLUE, "o", "works, no accuracy cost"), "loss": (YELLOW, "s", "works, small accuracy cost"),
             "bad": (RED, "X", "not viable")}
    for name, tp, kv, _, v in FEAS:
        col, mk, _ = style[v]
        ax.scatter(kv, tp, s=70, color=col, marker=mk, zorder=3, edgecolor=SURFACE, linewidth=1)
        dx, dy, ha = LABEL_POS.get(name, (0.07, 0.0, "left"))
        ax.annotate(name, (kv, tp), xytext=(kv + dx, tp + dy), fontsize=8, color=INK2, va="center", ha=ha)
    ax.axhline(1.0, color=NEUTRAL, ls="--", lw=1.1, zorder=1)
    ax.axvline(1.0, color=NEUTRAL, ls="--", lw=1.1, zorder=1)
    ax.set_xlabel("KV-cache capacity vs BF16 baseline (×)")
    ax.set_ylabel("Throughput vs BF16 baseline (×)")
    ax.set_xlim(0.8, 5.6)
    ax.set_ylim(0, 1.15)
    ax.legend(handles=[plt.Line2D([], [], color=c, marker=m, ls="", ms=8, label=l) for c, m, l in style.values()],
              loc="lower right")
    save(fig, "amo03_gfx90a_feasibility")


def fig_mi355():
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    rows = [("Qwen3-14B\nthroughput", 1.046, None), ("Qwen3-14B\nKV capacity", None, 2.39),
            ("Qwen3-14B\nmem. + conc.", None, 3.35), ("Qwen3-14B\ncombined", 1.54, 2.25),
            ("GPT-OSS-120B\ncombined", 1.0, 2.00)]
    w = 0.36
    for i, (lbl, tp, kv) in enumerate(rows):
        if tp:
            ax.bar(i - w / 2, tp, w * 0.9, color=BLUE, zorder=2)
            ax.text(i - w / 2, tp + 0.05, f"{tp:.2f}×" if lbl != "Qwen3-14B\nthroughput" else "≤1.05×",
                    ha="center", fontsize=8, color=INK2)
        if kv:
            ax.bar(i + w / 2, kv, w * 0.9, color=ORANGE, zorder=2)
            ax.text(i + w / 2, kv + 0.05, f"{kv:.2f}×", ha="center", fontsize=8, color=INK2)
    ax.axhline(1.0, color=NEUTRAL, ls="--", lw=1.1, zorder=1)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylabel("Gain vs baseline (×)")
    ax.set_ylim(0, 3.8)
    ax.legend(handles=[Patch(color=BLUE, label="throughput"), Patch(color=ORANGE, label="KV capacity")],
              loc="upper left")
    save(fig, "amo04_mi355x")


def schematic_loop():
    plt.rcParams.update({"svg.fonttype": "none"})
    with plt.rc_context({"font.family": "DejaVu Serif"}):
        fig, ax = plt.subplots(figsize=(10, 2.9))
        ax.set_xlim(0, 10); ax.set_ylim(0, 2.9); ax.axis("off")

        def node(x, y, w, h, t, s=None):
            ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#111111", lw=0.9, zorder=2))
            ax.text(x + w / 2, y + h * (0.62 if s else 0.5), t, ha="center", va="center", fontsize=10.5, zorder=3)
            if s:
                ax.text(x + w / 2, y + h * 0.28, s, ha="center", va="center", fontsize=8.5, color="#555555",
                        style="italic", zorder=3)

        def edge(a, b, ls="-"):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#111111",
                                         linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))
        ax.add_patch(Rectangle((2.55, 0.35), 5.25, 2.3, facecolor="#f2f2f2", edgecolor="none", zorder=0))
        ax.text(2.63, 2.55, "round (repeats)", fontsize=9, color="#555555", style="italic", va="top")
        node(0.1, 1.05, 2.0, 0.85, "run config", "objective · gate · budget")
        node(2.8, 1.05, 1.35, 0.85, "explorer", "rank actions")
        node(4.45, 1.05, 1.6, 0.85, "executors", "parallel on GPUs")
        node(6.35, 1.05, 1.25, 0.85, "CI gate", "keep / reject")
        node(8.1, 1.05, 1.8, 0.85, "finalize", "best config")
        edge((2.1, 1.47), (2.8, 1.47))
        edge((4.15, 1.47), (4.45, 1.47))
        edge((6.05, 1.47), (6.35, 1.47))
        edge((7.6, 1.47), (8.1, 1.47))
        ax.plot([6.97, 6.97, 3.47], [1.05, 0.55, 0.55], color="#111111", lw=0.9, zorder=1)
        edge((3.47, 0.55), (3.47, 1.05))
        ax.text(5.2, 0.45, "search tree: tried · frontier · deferred", fontsize=8.5, color="#555555",
                style="italic", ha="center", va="top")
        fig.savefig(f"{OUT}/amo00_loop.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote amo00_loop.svg")


# ── page ─────────────────────────────────────────────────────────────────────────────────────────
def table(head, rows):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def fig_html(tag, img, title, take, data=""):
    det = f"<details><summary>Data</summary>{data}</details>" if data else ""
    return (f'<section class="fig"><h3><span class="fn">{tag}</span> {title}</h3><p class="take">{take}</p>'
            f'<img src="plots/{img}" alt="{html.escape(title)}" loading="lazy">{det}</section>')


def page():
    chain_rows = lambda ch: [(c[0].replace("\n", " "), f"{c[1]:.1f}", f"[{c[2]:.1f}, {c[3]:.1f}]",  # noqa: E731
                              f"{c[1] / ch[0][1]:.2f}×") for c in ch]
    feas_rows = [(n, f"{t:.2f}×", f"{k:.2f}×", w, {"ok": "works", "loss": "small accuracy cost",
                                                     "bad": "not viable"}[v]) for n, t, k, w, v in FEAS]
    feas_rows.append(("fp8 e5m2 KV", "—", "1.96×", "×18.7 ppl", "silently corrupt output"))
    grid_rows = [(s,) + tuple(f"{v:.2f}×" for v in GRID_27B[s]) for s in GRID_27B]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amo Results</title>
<style>
:root {{ color-scheme: light; --bg:#f6f5f1; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7974;
  --line:#e4e3df; --accent:#2a78d6; --code:#efeee9; --warn:#8a5a00; --warnbg:#fff6e0; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --bg:#141413; --surface:#1d1d1b;
  --ink:#f4f3ee; --ink2:#c3c2b7; --muted:#9a998f; --line:#34332f; --accent:#3987e5; --code:#262623; --warn:#f0c46b; --warnbg:#2b2416; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --bg:#141413; --surface:#1d1d1b; --ink:#f4f3ee; --ink2:#c3c2b7; --muted:#9a998f;
  --line:#34332f; --accent:#3987e5; --code:#262623; --warn:#f0c46b; --warnbg:#2b2416; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:40px 16px 80px; }}
h1 {{ font-size:2rem; line-height:1.2; margin:0 0 6px; }}
h2 {{ font-size:1.35rem; margin:44px 0 12px; padding-top:8px; border-top:1px solid var(--line); }}
h3 {{ font-size:1.05rem; margin:0 0 4px; }}
.fn {{ color:var(--muted); font-weight:500; margin-right:4px; }}
p, li {{ color:var(--ink2); }} strong {{ color:var(--ink); }}
code {{ background:var(--code); padding:1px 5px; border-radius:4px; font-size:.88em; }}
a {{ color:var(--accent); }}
.conf {{ display:inline-block; font-size:.8rem; font-weight:600; letter-spacing:.04em; color:var(--warn); background:var(--warnbg);
  border-radius:4px; padding:2px 8px; margin-bottom:12px; }}
nav {{ display:flex; flex-wrap:wrap; gap:6px 16px; margin:18px 0 0; font-size:.92rem; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:12px; margin:20px 0; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.card .big {{ font-size:1.6rem; font-weight:650; line-height:1.1; color:var(--ink); }}
.card .lbl {{ font-size:.85rem; color:var(--ink2); margin-top:4px; }}
.fig {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px; margin:18px 0; }}
.fig img {{ width:100%; height:auto; display:block; margin:10px 0 6px; border-radius:6px; background:#fcfcfb; }}
.take {{ margin:0; }}
details summary {{ cursor:pointer; color:var(--accent); font-size:.9rem; margin-top:6px; }}
.tbl {{ overflow-x:auto; margin:10px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:.86rem; background:var(--surface); }}
th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--ink2); font-weight:600; white-space:nowrap; }} td {{ color:var(--ink); }}
.note {{ font-size:.88rem; color:var(--muted); }}
</style></head>
<body><main>
<span class="conf">AMD CONFIDENTIAL · INTERNAL</span>
<h1>Amo Results</h1>
<p>Agentic Model Optimization: results so far. An agent loop searches serving configurations and kernel- or weight-level changes for the fastest setup that doesn’t measurably lose accuracy. MI355X runs from August 2026; MI250X runs from September 23–25, 2026.</p>
<nav><a href="#summary">Summary</a><a href="#how">How Amo works</a><a href="#mi250x">MI250X runs</a><a href="#feas">gfx90a feasibility</a>
<a href="#eval">Evaluation reliability</a><a href="#mi355x">MI355X runs</a><a href="#caveats">Caveats</a><a href="#sources">Sources</a>
<a href="../index.html">← Lossless decoding study</a></nav>

<h2 id="summary">Summary</h2>
<div class="cards">
  <div class="card"><div class="big">3.16×</div><div class="lbl">Qwen3.6-35B-A3B on MI250X, throughput at c=64 (CI 3.09–3.24×).</div></div>
  <div class="card"><div class="big">1.52×</div><div class="lbl">Qwen3.8-27B on MI250X, throughput at c=64 (CI 1.48–1.56×).</div></div>
  <div class="card"><div class="big">1.54× · 2.25×</div><div class="lbl">Qwen3-14B on MI355X: throughput (c=512) and KV capacity, combined objective.</div></div>
  <div class="card"><div class="big">1.93×</div><div class="lbl">KV capacity on gfx90a at no throughput or accuracy cost (fp8 per-token-head KV).</div></div>
</div>
<ul>
  <li><strong>On MI250X, the throughput wins come from code the agents wrote</strong>, not from flags or checkpoints. The biggest step on both models is lossless speculative decoding plus fixes for the hybrid models’ recurrent state. Every off-the-shelf quantization tested on gfx90a costs speed.</li>
  <li><strong>All MI250X winners pass the gated accuracy check</strong> (gsm8k, aime) against the baseline. The Qwen3.6 winner shows a reproducible ~3-point MMLU drop (report-only), so the runner-up (2.70×, no drop) is the safer deployment.</li>
  <li><strong>On MI355X, the wins are mostly capacity:</strong> fp8 KV appears in every capacity or combined winner. GPT-OSS-120B was already at the hardware throughput ceiling.</li>
  <li><strong>Evaluation noise drove a protocol change.</strong> A 100-item, 1% gate couldn’t separate candidates from noise, so the MI250X runs use 3 fresh servers per config and non-overlapping 95% bootstrap CIs.</li>
</ul>

<h2 id="how">How Amo works</h2>
<section class="fig"><h3><span class="fn">Schematic</span> The optimization loop</h3>
<img src="plots/amo00_loop.svg" alt="Amo loop: run config, then rounds of explorer ranking, parallel executors and CI gate, then finalize" loading="lazy"></section>
<ul>
  <li><strong>Explorer:</strong> each round, re-diagnoses the bottleneck from what was measured and writes a ranked queue of actions (the <em>frontier</em>) into a shared search tree.</li>
  <li><strong>Executors:</strong> one sub-agent per action implements and screens it on a free GPU set; candidates run in parallel.</li>
  <li><strong>Gate (MI250X runs):</strong> 3 fresh-server runs per config; keep only if the throughput CI clears the current best and no gated accuracy CI falls entirely below it. MI355X runs used a 2% relative per-task tolerance instead.</li>
</ul>

<h2 id="mi250x">MI250X runs: throughput objective</h2>
<p>8× MI250X GCDs (gfx90a), vLLM 0.28.0, TP=2 (4 parallel slots). Engine and TP were fixed by the user. Metric: output tok/s from InferenceX at 64 concurrent requests, 1k input / 1k output tokens. Gated accuracy: gsm8k and aime at N=100 with a 16k-token answer cap.</p>
{fig_html("Fig. 1", "amo01_kept_chains.png", "Kept configurations, step by step (95% CI over 3 servers)",
          "Each bar is a change the gate kept. Speculative decoding with a lossless spec bundle (deferred-commit recurrent state, packed verify attention, scheduler fixes) is the largest step on both models.",
          table(["config", "tok/s", "95% CI", "× baseline"], chain_rows(CHAIN_27B)) +
          table(["config", "tok/s", "95% CI", "× baseline"], chain_rows(CHAIN_35B)))}
{fig_html("Fig. 2", "amo02_grid_27b.png", "Qwen3.8-27B winner across shapes and concurrency",
          "Wins at every point, most at low concurrency and decode-heavy shapes. Single runs, not CIs. No baseline grid exists for Qwen3.6.",
          table(["shape", "c=4", "c=16", "c=64"], grid_rows))}
<h3 style="margin-top:22px">Accuracy at the final configs</h3>
{table(["model", "metric", "baseline", "winner", "verdict"], [
    ("Qwen3.8-27B", "gsm8k strict / flex", "0.707 / 0.770", "0.697 / 0.757", "pass (CI overlap)"),
    ("Qwen3.8-27B", "aime (16k cap)", "0.687", "0.693", "pass"),
    ("Qwen3.8-27B", "mmlu · hellaswag · wikitext ppl", "0.8432 · 0.61/0.75 · 9.2943", "0.8447 · 0.61/0.75 · 9.2946", "unchanged (report-only)"),
    ("Qwen3.6-35B-A3B", "gsm8k strict / flex", "0.970 / 0.980", "0.980 / 0.990", "pass"),
    ("Qwen3.6-35B-A3B", "aime (16k cap)", "0.043", "0.033", "pass (floor-bound)"),
    ("Qwen3.6-35B-A3B", "mmlu (report-only)", "0.8435", "0.816 / 0.810", "~3-pt drop, reproducible"),
])}
<p class="note">Qwen3.6 AIME scores sit near the floor because its verbose thinking rarely finishes inside the 16k cap. The MMLU drop was localized to the hybrid-drafter step (not the int8 experts); the runner-up config without it reaches 2.70× [2.62, 2.79].</p>

<h2 id="feas">gfx90a feasibility of known levers</h2>
{fig_html("Fig. 3", "amo03_gfx90a_feasibility.png", "Throughput vs KV capacity per lever (Qwen3.8-27B, MI250X)",
          "fp8 per-token-head KV is the one free lever: 1.93× capacity at unchanged speed. Turboquant trades 13–30% speed for 2.6–4.8× capacity. FP8 and MXFP4 weights have no native path on gfx90a.",
          table(["lever", "throughput", "KV capacity", "wikitext ppl Δ", "verdict"], feas_rows))}
<p class="note">On MI355X, turboquant cost ~4× throughput; on gfx90a it costs only 0.70–0.87×. fp8 e5m2 KV passed the smoke test’s confidence check while producing garbage, and was caught only by the keyword probe.</p>

<h2 id="eval">Evaluation reliability</h2>
<ul>
  <li><strong>100-item gates are inside the noise.</strong> An unchanged Qwen3.8-27B moved 4 gsm8k questions across server restarts and 7 aime questions across repeats. One question is worth 1.3–2.6% relative, so a 1% tolerance fails any single lost question.</li>
  <li><strong>Answer caps decide aime.</strong> At a 2048-token cap, 54% of Qwen3.8 aime answers were truncated (aime 0.42); at 16k the same model scores 0.69.</li>
  <li><strong>Wikitext perplexity is the stable signal</strong> (±0.004% across servers) and separated lossy from lossless changes in every case.</li>
  <li><strong>Harness bug found and fixed:</strong> lm-eval’s <code>"Question:"</code> stop sequence truncated Qwen3.6’s thinking, scoring gsm8k 0.00. It was fixed before any comparison.</li>
  <li><strong>Host noise:</strong> NUMA auto-balancing on the host triggered ~10 s GPU queue evictions that stalled speculative-decoding runs. A per-container NUMA-local wrapper removed them, without inflating the baseline (682 vs 689 tok/s).</li>
</ul>

<h2 id="mi355x">MI355X runs (August 2026)</h2>
<p>AMD MI355X (gfx950), vLLM 0.21.0 / SGLang 0.5.12, TP=1. Guardrail: 2% relative per-task tolerance on gsm8k, mmlu, hellaswag and wikitext.</p>
{fig_html("Fig. 4", "amo04_mi355x.png", "Gains per model and objective",
          "Combined = throughput + memory. Capacity objectives win through KV precision; throughput gains are small except in the combined run, where fp8 KV plus fp8 weights give 1.54× at c=512.",
          table(["model", "objective", "winning config", "throughput", "KV capacity"], MI355))}

<h2 id="caveats">Caveats</h2>
<ul>
  <li>MI250X speedups are <strong>best within a fixed serving config</strong> (engine and TP pinned) at one benchmark point (c=64, 1k:1k), with a greedy client.</li>
  <li>The Qwen3.8-27B CI run ended early: a sub-agent’s <code>pgrep -f</code> also matched its own shell and killed the orchestrator. Six queued directions and the final stack never ran; the result was finalized afterwards by the user agent.</li>
  <li>Three Qwen3.6 non-spec winners (int8 experts, MoE launch bundle, int8 GDN state) have no accuracy verdict.</li>
  <li>MI355X gsm8k baselines vary between 0.63 and 0.68 across runs of the same model, consistent with the eval noise above; gains there (e.g. +28.7% gsm8k) should not be read as quality improvements.</li>
  <li>An earlier Qwen3.8-27B run with a 1% / 100-item gate reached 1.109×; it is superseded by the CI-gated run.</li>
</ul>

<h2 id="sources">Sources</h2>
<p class="note">All numbers transcribed from <code>/home/arkamukh/amo</code>: <code>docs/amo-findings.md</code> (MI355X), <code>runs/mi250x_qwen3.8_report.md</code> (feasibility, noise study), <code>runs/throughput-qwen38-27b-ci/{{report,status}}.md</code>, <code>runs/throughput-qwen36-35b-a3b-ci/report.md</code>. Page generated by <code>website/amo/build.py</code>.</p>
</main></body></html>"""


if __name__ == "__main__":
    fig_chains()
    fig_grid()
    fig_feasibility()
    fig_mi355()
    schematic_loop()
    open(f"{HERE}/index.html", "w").write(page())
    print("wrote index.html")
