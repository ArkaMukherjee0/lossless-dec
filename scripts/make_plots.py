"""Figures + summary tables for the write-up (website/). Runs on CPU from results/.

  GPUS= docker/run.sh python3 scripts/make_plots.py
"""
import glob
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

R = "/workspace/results"
OUT = "/workspace/website/plots"
os.makedirs(OUT, exist_ok=True)

# Reference categorical palette (validated, fixed order); color follows the method everywhere.
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
C = {"MTP": "#2a78d6", "DFlash": "#eb6834", "Suffix": "#1baf7a", "Medusa": "#eda100", "n-gram": "#e87ba4"}
MODELS = [("Qwen3.8-27B", "Qwen3.8-27B\n(dense, hybrid)"), ("gemma-4-31B-it", "Gemma 4 31B\n(dense)"),
          ("Qwen3.6-35B-A3B", "Qwen3.6-35B-A3B\n(MoE, hybrid)"), ("gemma-4-26B-A4B-it", "Gemma 4 26B-A4B\n(MoE)")]
SHORT = {m: l.split("\n")[0] for m, l in MODELS}
MTP_TAG = {"Qwen3.8-27B": "mtp3", "Qwen3.6-35B-A3B": "mtp3", "gemma-4-31B-it": "mtp4", "gemma-4-26B-A4B-it": "mtp4"}
CATS = ["math_reasoning", "mt_bench", "qa", "rag", "summarization", "translation"]
CAT_LBL = {"math_reasoning": "math", "mt_bench": "chat", "qa": "QA", "rag": "RAG",
           "summarization": "summ.", "translation": "transl."}
TABLES = {}
# vLLM's model-free proposers corrupt output on the hybrid (Gated-DeltaNet) Qwen models:
# >100/480 prompts diverge where plain decoding was confident (>1 nat); see fig13 / tables.json.
WRONG_OUTPUT = {("Qwen3.8-27B", "Suffix"), ("Qwen3.6-35B-A3B", "Suffix"),
                ("Qwen3.8-27B", "n-gram"), ("Qwen3.6-35B-A3B", "n-gram")}

sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})


def load(p):
    return json.load(open(p))


def summ(p):
    return load(p)["summary"] if os.path.exists(p) else None


def method_legend(ax, names, wrong=True, **kw):
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C.get(n, NEUTRAL), label=n) for n in names]
    if wrong:
        handles.append(Patch(facecolor=SURFACE, edgecolor=INK2, hatch="///", label="wrong output"))
    ax.legend(handles=handles, **kw)


COMPACT = {"Qwen3.8-27B": "Qwen3.8\n27B", "gemma-4-31B-it": "Gemma 4\n31B",
           "Qwen3.6-35B-A3B": "Qwen3.6\n35B-A3B", "gemma-4-26B-A4B-it": "Gemma 4\n26B-A4B"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def baseline(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(1.0, color=NEUTRAL, lw=1.2, ls="--", zorder=1)


def trace_stats(path):
    """Decode tok/s (excl. first step = prefill) and tokens/step from a streaming trace."""
    recs = load(path)["records"]
    toks = sum(sum(r["steps"]) for r in recs)
    steps = sum(len(r["steps"]) for r in recs)
    dec = sum(r["times"][-1] - r["times"][0] for r in recs if len(r["times"]) > 1)
    return (toks - len(recs)) / dec, toks / steps


# 1 ── headline: speedup at batch size 1, full Spec-Bench, T=0 ──────────────────────────────────
def fig_headline():
    methods = [("MTP", None), ("DFlash", "dflash7"), ("Suffix", "suffix"), ("Medusa", "medusa4"), ("n-gram", "ngram4")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w, rows = 0.16, []
    # runs made later on another node; each is compared with plain decoding from that same run set
    extra = {("Qwen3.6-35B-A3B", "n-gram"): "a_ngram_q35"}
    for i, (m, lbl) in enumerate(MODELS):
        for j, (name, tag) in enumerate(methods):
            d = f"{R}/{extra.get((m, name), 'a')}/{m}"
            ar = summ(f"{d}/ar_t0.0_c1.json")["tok_per_s"]
            s = summ(f"{d}/{tag or MTP_TAG[m]}_t0.0_c1.json")
            if not s:
                continue
            sp = s["tok_per_s"] / ar
            x = i + (j - 2) * w
            bad = (m, name) in WRONG_OUTPUT
            ax.bar(x, sp, w * 0.88, color=C[name], label=name if i == 0 or name in ("Medusa", "n-gram") else None,
                   hatch="///" if bad else None, edgecolor=SURFACE if bad else None, zorder=2)
            if bad:
                ax.annotate("wrong\noutput", (x, sp), xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK2)
            rows.append({"model": SHORT[m], "method": name, "speedup": round(sp, 2),
                         "tau": round(s["tau"], 2) if s["tau"] else None, "tok_s": round(s["tok_per_s"], 1),
                         "ar_tok_s": round(ar, 1), "note": "wrong output" if bad else ""})
    baseline(ax)
    ax.set_xticks(range(len(MODELS)), [l for _, l in MODELS])
    ax.set_ylabel("Speedup vs plain decoding")
    method_legend(ax, [k for k in C], ncol=6, loc="upper right")
    ax.set_ylim(0, 2.45)
    save(fig, "fig01_speedup_bs1")
    TABLES["headline"] = rows


# 2 ── concurrency ────────────────────────────────────────────────────────────────────────────────
def fig_concurrency():
    cs = [4, 16, 64, 128]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    rows = []
    for ax, (m, lbl) in zip(axes, MODELS):
        d = f"{R}/d/{m}"
        ar = {c: load(f"{d}/ar_c{c}.json")["output_throughput"] for c in cs}
        for name, tag in [("MTP", MTP_TAG[m]), ("DFlash", "dflash7")]:
            ys = [load(f"{d}/{tag}_c{c}.json")["output_throughput"] / ar[c] for c in cs]
            ax.plot(cs, ys, color=C[name], lw=2, marker="o", ms=5, label=name, zorder=3)
            ax.annotate(f"{ys[-1]:.2f}", (cs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
            rows += [{"model": SHORT[m], "method": name, "concurrency": c, "speedup": round(y, 2)}
                     for c, y in zip(cs, ys)]
        baseline(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(cs, [str(c) for c in cs])
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.set_xlabel("Concurrent requests")
    axes[0].set_ylabel("Speedup vs plain decoding")
    axes[0].legend(loc="lower left")
    save(fig, "fig02_concurrency")
    TABLES["concurrency"] = rows


# 3 ── draft length × concurrency ────────────────────────────────────────────────────────────────
def fig_k_by_conc():
    cs = [4, 16, 64, 128]
    panels = [("Qwen3.8-27B", "MTP", ["mtp1", "mtp3", "mtp6"]),
              ("gemma-4-26B-A4B-it", "MTP", ["mtp4", "mtp6"]),
              ("gemma-4-26B-A4B-it", "DFlash", ["dflash7", "dflash15"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (m, name, tags) in zip(axes, panels):
        d = f"{R}/d/{m}"
        ar = {c: load(f"{d}/ar_c{c}.json")["output_throughput"] for c in cs}
        shades = sns.light_palette(C[name], n_colors=len(tags) + 2)[2:]
        for tag, col in zip(tags, shades):
            ys = [load(f"{d}/{tag}_c{c}.json")["output_throughput"] / ar[c] for c in cs]
            k = tag.replace("mtp", "").replace("dflash", "")
            ax.plot(cs, ys, color=col, lw=2, marker="o", ms=5, label=f"k={k}", zorder=3)
        baseline(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(cs, [str(c) for c in cs])
        ax.set_title(f"{SHORT[m]} · {name}", fontsize=10)
        ax.set_xlabel("Concurrent requests")
        ax.legend(loc="lower left", title="draft length", title_fontsize=8)
    axes[0].set_ylabel("Speedup vs plain decoding")
    save(fig, "fig03_draftlen_x_concurrency")


# 4/5 ── draft-length sweep and verify cost ──────────────────────────────────────────────────────
def fig_draft_length():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    fig2, axes2 = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    rows = []
    for ax, ax2, (m, lbl) in zip(axes, axes2, MODELS):
        d = f"{R}/c/{m}"
        ar = summ(f"{d}/ar_t0.0_c1.json")["tok_per_s"]
        for name, pre in [("MTP", "mtp"), ("DFlash", "dflash")]:
            pts = []
            for p in glob.glob(f"{d}/{pre}[0-9]*_t0.0_c1.json"):
                k = int(os.path.basename(p).split("_")[0][len(pre):])
                s = summ(p)
                pts.append((k, s["tok_per_s"] / ar, s["tau"]))
            pts.sort()
            ks, sp, tau = zip(*pts)
            ax.plot(ks, sp, color=C[name], lw=2, marker="o", ms=5, label=name, zorder=3)
            ax2.plot([k + 1 for k in ks], [t / s for t, s in zip(tau, sp)], color=C[name], lw=2, marker="o",
                     ms=5, label=name, zorder=3)
            rows += [{"model": SHORT[m], "method": name, "k": k, "speedup": round(s, 2), "tau": round(t, 2),
                      "step_cost": round(t / s, 2)} for k, s, t in pts]
        baseline(ax)
        baseline(ax2)
        for a in (ax, ax2):
            a.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.set_xlabel("Draft length k")
        ax2.set_xlabel("Tokens verified per step (k+1)")
    axes[0].set_ylabel("Speedup vs plain decoding")
    axes2[0].set_ylabel("Step time / plain-decoding step")
    axes[0].legend(loc="upper left")
    axes2[0].legend(loc="upper left")
    save(fig, "fig04_draft_length")
    save(fig2, "fig05_verify_cost")
    TABLES["draft_length"] = rows


# 6 ── GEMM latency vs tokens (the MI250 verify tax) ─────────────────────────────────────────────
def fig_gemm():
    # scripts/gemm_latency.py on gfx90a (2026-09-25): µs per Qwen3-8B layer (qkv+o+gate_up+down)
    M = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128]
    stock = [301.7, 434.2, 1019.8, 386.9, 2461.6, 484.9, 485.7, 493.3, 498.4, 511.6, 563.6, 568.8, 652.3]
    patched = [303.4, 440.3, 497.5, 386.9, 502.7, 506.3, 505.4, 503.7, 506.8, 525.3, 532.2, 579.7, 666.1]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(M, [t / stock[0] for t in stock], color=C["DFlash"], lw=2, marker="o", ms=5, label="vLLM 0.30 stock")
    ax.plot(M, [t / patched[0] for t in patched], color=C["MTP"], lw=2, marker="o", ms=5, label="patched (ours)")
    baseline(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(M[::2] + [128], [str(m) for m in M[::2]] + ["128"])
    ax.set_yticks([1, 2, 4, 8], ["1×", "2×", "4×", "8×"])
    ax.set_xlabel("Tokens per forward pass")
    ax.set_ylabel("GEMM time / 1-token time")
    ax.legend(loc="upper left")
    save(fig, "fig06_gemm_verify_tax")


# 7 ── MoE expert union ──────────────────────────────────────────────────────────────────────────
def fig_moe_union():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rows = []
    for (m, _), col in zip([MODELS[2], MODELS[3]], [C["MTP"], C["DFlash"]]):
        r = load(f"{R}/g/{m}/router_union.json")
        ks = sorted(int(k) for k in r["ratio_vs_k1"])
        ys = [r["ratio_vs_k1"][str(k)] for k in ks]
        ax.plot(ks, ys, color=col, lw=2, marker="o", ms=5, zorder=3,
                label=f"{SHORT[m]} ({r['n_experts_seen']} experts, top-{r['topk']})")
        ax.annotate(f"{ys[-1]:.1f}×", (ks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
        rows += [{"model": SHORT[m], "window": k, "experts_vs_1token": round(y, 2)} for k, y in zip(ks, ys)]
    ax.plot([1, 16], [1, 16], color=NEUTRAL, lw=1, ls=":", label="no overlap (upper bound)")
    ax.set_ylim(0, 8)
    ax.set_xlabel("Tokens verified together")
    ax.set_ylabel("Distinct experts / 1-token step")
    ax.legend(loc="upper left")
    save(fig, "fig07_moe_expert_union")
    TABLES["moe_union"] = rows


# 8 ── long context ──────────────────────────────────────────────────────────────────────────────
def fig_long_context():
    ctxs = [8192, 32768, 65536]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    rows = []
    for ax, (m, lbl) in zip(axes, MODELS[:2]):
        ar = {c: trace_stats(f"{R}/e2t/{m}/ar_ctx{c}_t0.0_TRITON_ATTN.json")[0] for c in ctxs}
        series = [("MTP", f"{R}/e2t/{m}/{MTP_TAG[m]}_ctx{{c}}_t0.0_TRITON_ATTN.json"),
                  ("DFlash", f"{R}/e2t_nopc/{m}/dflash7_ctx{{c}}_t0.0_nopc_TRITON_ATTN.json"),
                  ("Suffix", f"{R}/e2t/{m}/suffix_ctx{{c}}_t0.0_TRITON_ATTN.json")]
        for name, pat in series:
            xs, ys = [], []
            for c in ctxs:
                p = pat.format(c=c)
                if os.path.exists(p):
                    xs.append(c // 1024); ys.append(trace_stats(p)[0] / ar[c])
            bad = (m, name) in WRONG_OUTPUT
            ax.plot(xs, ys, color=C[name], lw=2, marker="o", ms=5, label=name, zorder=3, ls="--" if bad else "-")
            if bad:
                ax.annotate("wrong output", (xs[-1], ys[-1]), xytext=(0, -12), textcoords="offset points",
                            ha="right", fontsize=7.5, color=INK2)
            rows += [{"model": SHORT[m], "method": name, "context_k": x, "speedup": round(y, 2),
                      "note": "wrong output" if bad else ""} for x, y in zip(xs, ys)]
        baseline(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 32, 64], ["8k", "32k", "64k"])
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.set_xlabel("Prompt length")
    axes[0].set_ylabel("Decode speedup vs plain decoding")
    axes[0].legend(loc="upper right")
    save(fig, "fig08_long_context")
    TABLES["long_context"] = rows


# 9 ── temperature 0 vs 1 (3 seeds) ──────────────────────────────────────────────────────────────
def fig_temperature():
    fig, ax = plt.subplots(figsize=(8, 3.8))
    rows, y, yt = [], 0, []
    for m, lbl in MODELS:
        ar0 = summ(f"{R}/a/{m}/ar_t0.0_c1.json")["tok_per_s"]
        for name, tag in [("MTP", MTP_TAG[m]), ("DFlash", "dflash7")]:
            t0 = summ(f"{R}/a/{m}/{tag}_t0.0_c1.json")["tok_per_s"] / ar0
            t1 = []
            for sd in (0, 1, 2):
                suf = f"_s{sd}" if sd else ""
                a, s = summ(f"{R}/b/{m}/ar_t1.0_c1{suf}.json"), summ(f"{R}/b/{m}/{tag}_t1.0_c1{suf}.json")
                if a and s:
                    t1.append(s["tok_per_s"] / a["tok_per_s"])
            mu, sd_ = st.mean(t1), st.stdev(t1)
            ax.plot([t0, mu], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
            ax.scatter([t0], [y], s=60, color=C[name], zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.errorbar([mu], [y], xerr=[sd_], fmt="o", ms=8, mfc=SURFACE, mec=C[name], mew=2, ecolor=C[name],
                        zorder=3)
            yt.append((y, f"{SHORT[m]} · {name}"))
            rows.append({"model": SHORT[m], "method": name, "T0": round(t0, 2), "T1_mean": round(mu, 2),
                         "T1_std": round(sd_, 2), "seeds": len(t1)})
            y += 1
        y += 0.6
    baseline(ax, horizontal=False)
    ax.set_yticks([p for p, _ in yt], [l for _, l in yt])
    ax.invert_yaxis()
    ax.set_xlabel("Speedup vs plain decoding")
    ax.scatter([], [], s=60, color=INK2, label="T = 0 (greedy)")
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=INK2, linewidth=2, label="T = 1 (mean ± sd, 3 seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    save(fig, "fig09_temperature")
    TABLES["temperature"] = rows


# 10 ── per-category heatmap ─────────────────────────────────────────────────────────────────────
def fig_categories():
    rows, labels = [], []
    for m, _ in MODELS:
        ar = summ(f"{R}/a/{m}/ar_t0.0_c1.json")["per_category"]
        for name, tag in [("MTP", MTP_TAG[m]), ("DFlash", "dflash7"), ("Suffix", "suffix")]:
            s = summ(f"{R}/a/{m}/{tag}_t0.0_c1.json")["per_category"]
            rows.append([s[c]["tok_per_s"] / ar[c]["tok_per_s"] for c in CATS])
            labels.append(f"{SHORT[m]} · {name}")
    data = np.array(rows)
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f1efe9", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    sns.heatmap(data, ax=ax, cmap=cmap, center=1.0, vmin=0.4, vmax=3.2, annot=True, fmt=".2f",
                annot_kws={"fontsize": 8}, linewidths=2, linecolor=SURFACE,
                xticklabels=[CAT_LBL[c] for c in CATS], yticklabels=labels,
                cbar_kws={"label": "Speedup vs plain decoding", "shrink": 0.8, "pad": 0.14})
    for i, l in enumerate(labels):
        m_short, name = l.split(" · ")
        if any(SHORT[m] == m_short and (m, name) in WRONG_OUTPUT for m in SHORT):
            ax.add_patch(plt.Rectangle((0, i), len(CATS), 1, fill=False, hatch="///", edgecolor=INK2,
                                       linewidth=0, alpha=0.35, zorder=3))
            ax.text(len(CATS) + 0.08, i + 0.5, "wrong output", va="center", fontsize=7.5, color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    save(fig, "fig10_categories")
    TABLES["categories"] = [{"row": l, **{CAT_LBL[c]: round(v, 2) for c, v in zip(CATS, r)}}
                            for l, r in zip(labels, rows)]


# 11 ── accepted tokens per step by drafter family ───────────────────────────────────────────────
def fig_acceptance():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True)
    rows = []
    fams = ["DFlash (k=7)", "MTP (k=4)", "Draft model (k=4)", "Layer skip 25% (k=4)", "Medusa (4 heads)",
            "Suffix (retrieval)", "Prompt lookup"]
    for ax, (m, lbl) in zip(axes, MODELS):
        f = f"{R}/f/{m}"
        tok = load(f"{f}/sim_token_only.json")
        draft = load(glob.glob(f"{f}/sim_draft_*.json")[0])
        ls = load(f"{f}/sim_layerskip.json")
        med = summ(f"{R}/a/{m}/medusa4_t0.0_c1.json")
        vals = {"DFlash (k=7)": summ(f"{R}/a/{m}/dflash7_t0.0_c1.json")["tau"],
                "MTP (k=4)": summ(f"{R}/c/{m}/mtp4_t0.0_c1.json")["tau"],
                "Draft model (k=4)": draft["tau"]["4"],
                "Layer skip 25% (k=4)": ls["budgets"]["0.25"]["per_k"]["4"]["tau"],
                "Medusa (4 heads)": med["tau"] if med else None,
                "Suffix (retrieval)": tok["suffix"]["tau"], "Prompt lookup": tok["pld8"]["tau"]}
        ys = [vals[k] if vals[k] else 0 for k in fams]
        ax.barh(range(len(fams)), ys, 0.7, color=C["MTP"], zorder=2)
        for i, v in enumerate(ys):
            ax.text(v + 0.06, i, f"{v:.2f}" if v else "n/a", va="center", fontsize=8, color=INK2)
        ax.axvline(1.0, color=NEUTRAL, lw=1.2, ls="--", zorder=1)
        ax.set_yticks(range(len(fams)), fams if ax is axes[0] else [""] * len(fams))
        ax.invert_yaxis()
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.set_xlim(0, 5.2)
        ax.set_xlabel("Tokens per verify step")
        ax.grid(axis="y", visible=False)
        rows += [{"model": SHORT[m], "drafter": k, "tokens_per_step": round(v, 2) if v else None}
                 for k, v in vals.items()]
    save(fig, "fig11_acceptance_by_drafter")
    TABLES["acceptance"] = rows


# 12 ── acceptance vs output position (long reasoning) ───────────────────────────────────────────
def fig_reasoning_drift():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    edges = [0, 1000, 2000, 4000, 6000, 8000, 12288]
    mids = [0.5, 1.5, 3, 5, 7, 10]
    for ax, (m, lbl) in zip(axes, MODELS):
        for name, tag in [("MTP", MTP_TAG[m]), ("DFlash", "dflash7")]:
            recs = load(glob.glob(f"{R}/e1/{m}/{tag}_*.json")[0])["records"]
            n, s = np.zeros(len(mids)), np.zeros(len(mids))
            for r in recs:
                pos = 0
                for x in r["steps"]:
                    b = np.searchsorted(edges, pos, side="right") - 1
                    if 0 <= b < len(mids):
                        n[b] += x; s[b] += 1
                    pos += x
            ok = s > 50
            ax.plot(np.array(mids)[ok], (n / np.maximum(s, 1))[ok], color=C[name], lw=2, marker="o", ms=5,
                    label=name, zorder=3)
        ax.set_title(lbl.replace("\n", " "), fontsize=10)
        ax.set_xlabel("Output position (k tokens)")
        ax.set_ylim(0, 6.5)
    axes[0].set_ylabel("Tokens per verify step")
    axes[0].legend(loc="lower left")
    save(fig, "fig12_reasoning_drift")


# 13 ── losslessness: divergence vs noise floor; batch invariance ─────────────────────────────────
def first_div(a, b):
    d = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), None)
    return d if d is not None or len(a) == len(b) else min(len(a), len(b))


def divergence(ar_path, sp_path):
    ar, sp = load(ar_path)["records"], load(sp_path)["records"]
    firsts = [first_div(a["token_ids"], b["token_ids"]) for a, b in zip(ar, sp)]
    conf = sum(1 for a, d in zip(ar, firsts) if d is not None and "gaps" in a and d < len(a["gaps"])
               and a["gaps"][d] is not None and a["gaps"][d] > 1.0)
    return 100 * sum(d is not None for d in firsts) / len(firsts), conf, len(firsts)


def fig_lossless():
    fig, (ax, axc, ax2) = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [3, 3, 1.1]})
    series = [("AR repeat", "arrepeat", NEUTRAL), ("MTP", None, C["MTP"]), ("DFlash", "dflash7", C["DFlash"]),
              ("Suffix", "suffix", C["Suffix"]), ("n-gram", "ngram4", C["n-gram"])]
    # runs that live outside results/a (each compared with plain decoding from its own run set)
    src = {("Qwen3.8-27B", "n-gram"): "pilot/bench", ("Qwen3.6-35B-A3B", "n-gram"): "a_ngram_q35"}
    w, rows = 0.16, []
    for i, (m, lbl) in enumerate(MODELS):
        for j, (name, tag, col) in enumerate(series):
            d = f"{R}/{src.get((m, name), 'a')}/{m}"
            p = f"{d}/{tag or MTP_TAG[m]}_t0.0_c1.json"
            if not os.path.exists(p):
                continue
            pct, conf, n = divergence(f"{d}/ar_t0.0_c1.json", p)
            conf_pct = 100 * conf / n
            bad = (m, name) in WRONG_OUTPUT
            x = i + (j - 2) * w
            for a_, v in ((ax, pct), (axc, conf_pct)):
                a_.bar(x, v, w * 0.88, color=col, zorder=2, hatch="///" if bad else None,
                       edgecolor=SURFACE if bad else None, label=name if (m, name) == (MODELS[0][0], name) else None)
            if pct == 0:  # exact reproduction: make the zero-height bar visible
                ax.text(x, 1.5, "0", ha="center", fontsize=7.5, color=INK2)
            rows.append({"model": SHORT[m], "method": name, "diverged_pct": round(pct, 1),
                         "confident_gt_1nat": conf, "confident_pct": round(conf_pct, 1), "prompts": n,
                         "note": "wrong output" if bad else ""})
    for a_, ylab, top in ((ax, "Prompts diverging (%)", 100), (axc, "Prompts diverging at confident tokens (%)", 40)):
        a_.set_xticks(range(len(MODELS)), [COMPACT[m] for m, _ in MODELS])
        a_.set_ylabel(ylab)
        a_.set_ylim(0, top)
    ax.set_title("(a) any divergence", fontsize=10)
    axc.set_title("(b) divergence at confident tokens (gap > 1 nat)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=NEUTRAL, label="AR repeat (noise floor)")] +
              [Patch(facecolor=C[n], label=n) for n in ("MTP", "DFlash", "Suffix", "n-gram")] +
              [Patch(facecolor=SURFACE, edgecolor=INK2, hatch="///", label="wrong output")],
              ncol=3, loc="upper left")
    # batch-invariant kernels, Gemma 4 31B, MTP k=4, 60 prompts
    vals = []
    for mode in ("off", "on"):
        d = f"{R}/bi_{mode}/gemma-4-31B-it"
        pct, _, n = divergence(f"{d}/ar_t0.0_c1.json", f"{d}/mtp4_t0.0_c1.json")
        vals.append((mode, pct, summ(f"{d}/ar_t0.0_c1.json")["tok_per_s"], summ(f"{d}/mtp4_t0.0_c1.json")["tok_per_s"]))
    ax2.bar([0, 1], [v[1] for v in vals], 0.55, color=C["MTP"], zorder=2)
    for x, v in enumerate(vals):
        ax2.text(x, v[1] + 2, f"{v[1]:.0f}%", ha="center", fontsize=9, color=INK)
    ax2.set_xticks([0, 1], ["standard\nkernels", "batch-invariant\nkernels"])
    ax2.set_ylim(0, 100)
    ax2.set_title("(c) Gemma 4 31B · MTP", fontsize=10)
    save(fig, "fig13_losslessness")
    TABLES["lossless"] = rows
    TABLES["batch_invariance"] = [{"kernels": v[0], "diverged_pct": round(v[1], 1), "ar_tok_s": round(v[2], 1),
                                   "mtp_tok_s": round(v[3], 1)} for v in vals]


# ── tables that are not figures ──────────────────────────────────────────────────────────────────
def extra_tables():
    acc = []
    for p in sorted(glob.glob(f"{R}/acc/*/*.json")):
        r = load(p)
        a = [v["accuracy"] for v in r["seeds"].values()]
        acc.append({"model": r["model"].split("/")[-1], "method": r["tag"], "gsm8k_acc": round(100 * st.mean(a), 2),
                    "sd": round(100 * st.stdev(a), 2)})
    TABLES["accuracy_t1"] = acc
    large = []
    fn = f"{R}/l/Qwen3.8-Flash-Next"
    ar = summ(f"{fn}/ar_t0.0_c1.json")["tok_per_s"]
    for k in (1, 2, 3, 4):
        s = summ(f"{fn}/mtp{k}_t0.0_c1.json")
        large.append({"model": "Qwen3.8-Flash-Next (TP=8)", "method": f"MTP k={k}", "tok_s": round(s["tok_per_s"], 1),
                      "speedup": round(s["tok_per_s"] / ar, 2), "tau": round(s["tau"], 2)})
    ms = f"{R}/l/mistral-small-4-bf16"
    a, e = summ(f"{ms}/ar_t0.0_c1.json"), summ(f"{ms}/eagle3_t0.0_c1.json")
    large.append({"model": "Mistral Small 4 (TP=8)", "method": "EAGLE k=3", "tok_s": round(e["tok_per_s"], 1),
                  "speedup": round(e["tok_per_s"] / a["tok_per_s"], 2), "tau": round(e["tau"], 2)})
    TABLES["large_moe"] = large
    TABLES["large_moe_ar"] = {"Qwen3.8-Flash-Next": round(ar, 1), "Mistral Small 4": round(a["tok_per_s"], 1)}
    tri = []
    for m, _ in MODELS:
        a0, a1 = summ(f"{R}/a/{m}/ar_t0.0_c1.json")["tok_per_s"], summ(f"{R}/a_triton/{m}/ar_t0.0_c1.json")["tok_per_s"]
        for name, tag in [("AR", "ar"), ("MTP", MTP_TAG[m]), ("DFlash", "dflash7"), ("Suffix", "suffix")]:
            s0, s1 = summ(f"{R}/a/{m}/{tag}_t0.0_c1.json"), summ(f"{R}/a_triton/{m}/{tag}_t0.0_c1.json")
            if s0 and s1:
                tri.append({"model": SHORT[m], "method": name, "default_tok_s": round(s0["tok_per_s"], 1),
                            "triton_tok_s": round(s1["tok_per_s"], 1),
                            "speedup_default": round(s0["tok_per_s"] / a0, 2), "speedup_triton": round(s1["tok_per_s"] / a1, 2)})
    TABLES["triton"] = tri


if __name__ == "__main__":
    for f in [fig_headline, fig_concurrency, fig_k_by_conc, fig_draft_length, fig_gemm, fig_moe_union,
              fig_long_context, fig_temperature, fig_categories, fig_acceptance, fig_reasoning_drift,
              fig_lossless, extra_tables]:
        f()
    json.dump(TABLES, open("/workspace/website/tables.json", "w"), indent=1)
    print("wrote tables.json")
