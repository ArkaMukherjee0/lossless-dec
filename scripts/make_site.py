"""Render website/index.html from website/tables.json and website/plots/*.png (run after make_plots.py).

  python3 scripts/make_site.py
"""
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = json.load(open(f"{ROOT}/website/tables.json"))


def rng(rows, key, pred):
    v = [r[key] for r in rows if pred(r) and r[key] is not None]
    return f"{min(v):.2f}–{max(v):.2f}×" if len(v) > 1 else f"{v[0]:.2f}×"


def table(rows, cols=None, fmt=None):
    cols = cols or list(rows[0])
    head = "".join(f"<th>{html.escape(c.replace('_', ' '))}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            txt = f"{v:.2f}" if isinstance(v, float) and not (fmt and c in fmt) else (fmt[c](v) if fmt and c in fmt else v)
            cls = ' class="num"' if isinstance(v, (int, float)) else ""
            cells.append(f"<td{cls}>{html.escape(str(txt))}</td>")
        body += "<tr>" + "".join(cells) + "</tr>"
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def pivot(rows, row_key, col_key, val_key):
    rks = list(dict.fromkeys(r[row_key] for r in rows))
    cks = list(dict.fromkeys(r[col_key] for r in rows))
    out = []
    for rk in rks:
        d = {row_key: rk}
        for ck in cks:
            v = [r[val_key] for r in rows if r[row_key] == rk and r[col_key] == ck]
            d[str(ck)] = v[0] if v else ""
        out.append(d)
    return out, [row_key] + [str(c) for c in cks]


def fig(n, img, title, takeaway, data_html):
    return f"""
<section class="fig" id="fig{n}">
  <h3><span class="fn">Fig. {n}</span> {title}</h3>
  <p class="take">{takeaway}</p>
  <img src="plots/{img}.png" alt="{html.escape(title)}" loading="lazy">
  <details><summary>Data</summary>{data_html}</details>
</section>"""


# ── derived numbers ──────────────────────────────────────────────────────────────────────────────
H = T["headline"]
conc = T["concurrency"]
lc = T["long_context"]
temp = T["temperature"]
is_ = lambda meth: (lambda r: r["method"] == meth)  # noqa: E731
at = lambda c: (lambda r: r["concurrency"] == c)  # noqa: E731

mtp_bs1 = rng(H, "speedup", is_("MTP"))
dfl_bs1 = rng(H, "speedup", is_("DFlash"))
suf_bs1 = rng(H, "speedup", is_("Suffix"))
mtp_c128 = rng(conc, "speedup", lambda r: r["method"] == "MTP" and r["concurrency"] == 128)
dfl_c128 = rng(conc, "speedup", lambda r: r["method"] == "DFlash" and r["concurrency"] == 128)
mtp_t1 = rng(temp, "T1_mean", is_("MTP"))
dfl_t1 = rng(temp, "T1_mean", is_("DFlash"))
lc64 = lambda m: rng(lc, "speedup", lambda r: r["method"] == m and r["context_k"] == 64)  # noqa: E731
acc_rows = T["acceptance"]
tau = lambda d: rng(acc_rows, "tokens_per_step", lambda r: r["drafter"] == d).replace("×", "")  # noqa: E731
med = {r["model"]: r["speedup"] for r in H if r["method"] == "Medusa"}
ngr = {r["model"]: r["speedup"] for r in H if r["method"] == "n-gram"}
bi = {r["kernels"]: r for r in T["batch_invariance"]}
fn = T["large_moe"]
ms = [r for r in fn if r["model"].startswith("Mistral")][0]
fn_best = max((r for r in fn if r["model"].startswith("Qwen")), key=lambda r: r["speedup"])

gap = [
    ("MTP (native)", "ships with model", mtp_bs1, mtp_c128, mtp_t1, lc64("MTP"), "1.43× vs 1.96× (Qwen)", "✓"),
    ("DFlash / DFlash2", "separate drafter", dfl_bs1, dfl_c128, dfl_t1, lc64("DFlash") + "†", "1.15× vs 2.09× (Qwen)", "✓"),
    ("Draft model (SpS)", "none (small model)", f"τ {tau('Draft model (k=4)')} (sim.)", "—", "—", "—", "—", "vLLM crashes‡"),
    ("Layer skip (D&V/SWIFT)", "skip-set search", f"τ {tau('Layer skip 25% (k=4)')} (sim.)", "—", "—", "—", "—", "—"),
    ("Medusa", "train heads", f"{min(med.values()):.2f}–{max(med.values()):.2f}×", "—", "—", "—",
     f"{med.get('Gemma 4 26B-A4B', 0):.2f}× on MoE", "✓"),
    ("Suffix decoding", "none", suf_bs1, "—", "—", lc64("Suffix"), "0.51× on Qwen MoE", "✗ corrupts output"),
    ("n-gram / prompt lookup", "none", f"{min(ngr.values()):.2f}–{max(ngr.values()):.2f}×", "—", "—", "—", "—",
     "✗ corrupts output"),
    ("EAGLE (Mistral Small 4)", "separate drafter", f"{ms['speedup']:.2f}× (TP=8)", "—", "—", "—", "—", "n/a"),
]
gap_html = '<div class="tbl"><table class="gap"><thead><tr>' + "".join(
    f"<th>{h}</th>" for h in ["Method", "Setup cost", "Batch size 1", "128 in flight", "Temperature 1",
                              "64k context", "MoE penalty", "Hybrid models"]) + "</tr></thead><tbody>"
for row in gap:
    gap_html += "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
gap_html += "</tbody></table></div>"

cat_rows = T["categories"]
cat_cols = list(cat_rows[0])

# ── page ─────────────────────────────────────────────────────────────────────────────────────────
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lossless Decoding on MI250</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f5f1; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #7a7974;
  --line: #e4e3df; --accent: #2a78d6; --good: #1f7a4d; --bad: #b3261e; --code: #efeee9;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #141413; --surface: #1d1d1b; --ink: #f4f3ee; --ink2: #c3c2b7; --muted: #9a998f;
    --line: #34332f; --accent: #3987e5; --good: #4fbf85; --bad: #ef6f67; --code: #262623;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #141413; --surface: #1d1d1b; --ink: #f4f3ee; --ink2: #c3c2b7; --muted: #9a998f;
  --line: #34332f; --accent: #3987e5; --good: #4fbf85; --bad: #ef6f67; --code: #262623;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 40px 16px 80px; }}
header h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 6px; letter-spacing: -0.01em; }}
header p {{ color: var(--ink2); margin: 0; }}
h2 {{ font-size: 1.35rem; margin: 48px 0 12px; padding-top: 8px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 1.05rem; margin: 0 0 4px; }}
.fn {{ color: var(--muted); font-weight: 500; margin-right: 4px; }}
p, li {{ color: var(--ink2); }}
strong {{ color: var(--ink); }}
code {{ background: var(--code); padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }}
a {{ color: var(--accent); }}
nav {{ display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 20px 0 0; font-size: 0.92rem; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; margin: 20px 0; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }}
.card .big {{ font-size: 1.6rem; font-weight: 650; color: var(--ink); line-height: 1.1; }}
.card .lbl {{ font-size: 0.85rem; color: var(--ink2); margin-top: 4px; }}
.fig {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 18px; margin: 18px 0; }}
.fig img {{ width: 100%; height: auto; display: block; margin: 10px 0 6px; border-radius: 6px; background: #fcfcfb; }}
.take {{ margin: 0; }}
details summary {{ cursor: pointer; color: var(--accent); font-size: 0.9rem; margin-top: 6px; }}
.tbl {{ overflow-x: auto; margin: 10px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; background: var(--surface); }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }}
th {{ color: var(--ink2); font-weight: 600; }}
td {{ color: var(--ink); }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.gap td {{ white-space: normal; min-width: 90px; }}
table.gap td:first-child {{ font-weight: 600; }}
.note {{ font-size: 0.88rem; color: var(--muted); }}
ol li, ul li {{ margin: 4px 0; }}
</style>
</head>
<body>
<main>
<header>
  <h1>Lossless Decoding on MI250</h1>
  <p>Where speculative decoding methods fall short on 2026 dense, MoE and hybrid models, measured on 4× AMD MI250 (8 GCDs, gfx90a) with vLLM 0.30. September 25–30, 2026.</p>
  <nav>
    <a href="#summary">Summary</a><a href="#setup">Setup</a><a href="#speed">Speed</a><a href="#serving">Serving</a>
    <a href="#why">Why</a><a href="#robust">Robustness</a><a href="#lossless">Losslessness</a>
    <a href="#gaps">Gap matrix</a><a href="#vllm">vLLM issues</a><a href="#caveats">Caveats</a><a href="#repro">Reproduce</a>
  </nav>
</header>

<h2 id="summary">Summary</h2>
<div class="cards">
  <div class="card"><div class="big">{mtp_bs1}</div><div class="lbl">Native MTP, 1 request at a time. The most reliable method.</div></div>
  <div class="card"><div class="big">{dfl_c128}</div><div class="lbl">DFlash with 128 requests in flight. Fast at low load, it collapses under load.</div></div>
  <div class="card"><div class="big">{lc64('MTP')}</div><div class="lbl">MTP at 64k context. Every method is below plain decoding.</div></div>
  <div class="card"><div class="big">{suf_bs1}</div><div class="lbl">Suffix decoding. Model-free drafters lose on MI250.</div></div>
</div>
<ol>
  <li><strong>Native MTP is the most robust method.</strong> It keeps a speedup at every setting except long context and very high concurrency. DFlash beats it only at low concurrency, and only on some models.</li>
  <li><strong>Verification is the bottleneck on MI250.</strong> Checking 2–16 tokens costs a fixed ~1.6–2× a decoding step on dense models, and up to 3.5× on the Qwen MoE. Only drafters that get about 3 or more tokens accepted per step come out ahead.</li>
  <li><strong>Speedups shrink under realistic conditions:</strong> many concurrent requests, temperature 1, long context, and MoE targets. Acceptance barely changes with concurrency; the loss comes entirely from verify cost.</li>
  <li><strong>The optimal draft length depends on load.</strong> Long drafts win with few requests in flight and lose with many. No method adapts to this.</li>
  <li><strong>“Lossless” holds only up to floating-point effects.</strong> Outputs diverge from plain decoding on 37–65% of prompts, but task accuracy is unchanged. Batch-invariant kernels remove the divergence entirely, at the cost of the whole speedup on MI250. vLLM’s model-free proposers (n-gram, suffix) produce genuinely wrong output on both hybrid Qwen models.</li>
</ol>

<h2 id="setup">Setup</h2>
<ul>
  <li><strong>Hardware:</strong> 4× MI250 = 8 GCDs × 64 GB, gfx90a (no FP8). Each 27–35B model runs in BF16 split across 2 GCDs (tensor parallel), so 4 runs go in parallel. Running 4 at once changes speed by only 1.2%.</li>
  <li><strong>Engine:</strong> vLLM 0.30.0 (ROCm 7.2) in Docker, plus 4 patches (see <a href="#vllm">vLLM issues</a>).</li>
  <li><strong>Models:</strong> Qwen3.8-27B (dense, 48 Gated-DeltaNet + 16 attention layers), Gemma 4 31B (dense), Qwen3.6-35B-A3B (MoE, hybrid), Gemma 4 26B-A4B (MoE). On all 8 GCDs: Qwen3.8-Flash-Next (125B total, 6B active, 512 experts) and Mistral Small 4 (119B, converted from FP8 to BF16).</li>
  <li><strong>Methods:</strong> native MTP, DFlash/DFlash2, suffix decoding, n-gram, Medusa (heads trained here), EAGLE (Mistral). Simulated offline from saved plain-decoding outputs (validated to within 1.5% of real runs): prompt lookup, SAM-style longest match, draft model, layer skip.</li>
  <li><strong>Workloads:</strong> Spec-Bench (480 prompts, 6 categories, up to 1024 output tokens, thinking off); AIME-2025 + MATH500 L5 with thinking on (up to 12k tokens); LongBench-v2 truncated to 8k / 32k / 64k tokens; GSM8K for accuracy.</li>
  <li><strong>Speedup</strong> = tokens/s ÷ plain-decoding tokens/s in the same engine and setup. <strong>τ</strong> = tokens produced per target forward pass.</li>
</ul>

<section class="fig" id="schA">
  <h3><span class="fn">Schematic A</span> Experimental design</h3>
  <img src="plots/s1_design.svg" alt="Experimental design: workloads and models run on 2-GCD or 8-GCD slots through vLLM; saved plain-decoding outputs feed offline analyses" loading="lazy">
</section>
<section class="fig" id="schB">
  <h3><span class="fn">Schematic B</span> Coverage: method × setting</h3>
  <img src="plots/s2_coverage.svg" alt="Coverage map of which methods were run in which settings, and on how many of the 4 main models" loading="lazy">
</section>

<h2 id="speed">Speed with one request at a time</h2>
{fig(1, "fig01_speedup_bs1", "Speedup vs plain decoding (Spec-Bench, T=0)",
     "MTP and DFlash reach about 1.9–2.1× on the dense models; the Qwen MoE gets much less. Suffix, n-gram and Medusa are at or below plain decoding. Hatched bars: wrong output.",
     table(H))}
{fig(2, "fig10_categories", "Speedup by task category",
     "Math is always the easiest category. Summarization and QA are the hardest for trained drafters. Suffix loses in every cell. Hatched rows: wrong output.",
     table(cat_rows, cat_cols))}
{fig(3, "fig04_draft_length", "Speedup vs draft length k",
     "Gains flatten at k≈3–7. k=1 is never a good choice. On Gemma MoE, DFlash gets worse past k=7.",
     table(T["draft_length"], ["model", "method", "k", "speedup", "tau", "step_cost"]))}

<h2 id="serving">Serving under load</h2>
{fig(4, "fig02_concurrency", "Speedup vs concurrent requests (continuous arrivals)",
     "Most configurations fall below plain decoding by 64–128 requests in flight. Gemma 4 26B-A4B with MTP is the only one that holds (1.60× at 128).",
     table(*pivot([dict(r, config=f"{r['model']} · {r['method']}") for r in conc],
                  "config", "concurrency", "speedup")))}
{fig(5, "fig03_draftlen_x_concurrency", "Draft length × concurrency",
     "Longer drafts win with 4 requests in flight and lose with 64+. The best k depends on load.",
     "<p class='note'>Raw results: <code>results/d/</code>.</p>")}
{fig(6, "fig09_temperature", "Temperature 0 vs 1 (3 seeds at T=1)",
     "Sampling cuts acceptance. Qwen drafters lose the most, and DFlash on the Qwen MoE drops below plain decoding.",
     table(temp))}

<h2 id="why">Why: verify cost on MI250</h2>
{fig(7, "fig06_gemm_verify_tax", "Matrix-multiply time vs tokens per forward pass (one Qwen3-8B layer)",
     "Stock vLLM is up to 8× slower at 3 and 5 tokens (a skinny-kernel misroute; we patched it). Even patched, verifying 3–16 tokens costs ~1.6–1.7× a 1-token step.",
     "<p class='note'>From <code>scripts/gemm_latency.py</code> on gfx90a.</p>")}
{fig(8, "fig05_verify_cost", "Step time vs tokens verified (measured end to end)",
     "Dense models pay ~1.5–2.2× per step. On the Qwen MoE, DFlash steps cost ~3.4× because its drafter and the extra experts add up.",
     table(T["draft_length"], ["model", "method", "k", "step_cost"]))}
{fig(9, "fig07_moe_expert_union", "MoE: distinct experts touched vs tokens verified",
     "Verifying 16 tokens touches 6.7× the experts of one token on the 256-expert Qwen MoE and 5.0× on the 128-expert Gemma MoE. That explains the larger Qwen MoE penalty.",
     table(T["moe_union"]))}
{fig(10, "fig11_acceptance_by_drafter", "Tokens accepted per verify step, by drafter",
     "Only DFlash, MTP and a separate draft model reach about 3 tokens per step. Retrieval drafters stay at 1.3–1.4, which is below break-even on MI250.",
     table(acc_rows))}

<h2 id="robust">Robustness</h2>
{fig(11, "fig08_long_context", "Speedup vs prompt length (Triton attention)",
     "Every method drops below plain decoding by 64k context on both models.",
     table(lc) + "<p class='note'>DFlash is run with prefix caching off (see vLLM issues).</p>")}
{fig(12, "fig12_reasoning_drift", "Acceptance vs output position, long reasoning (thinking on)",
     "No drift: acceptance stays flat across 10k+ tokens of reasoning on every model.",
     "<p class='note'>Raw traces: <code>results/e1/</code>.</p>")}

<h3 style="margin-top:28px">Large MoE on 8 GCDs</h3>
<p>Plain decoding: Qwen3.8-Flash-Next {T['large_moe_ar']['Qwen3.8-Flash-Next']} tok/s, Mistral Small 4 {T['large_moe_ar']['Mistral Small 4']} tok/s.
Flash-Next with MTP reaches {fn_best['speedup']:.2f}× ({fn_best['method']}), better than the smaller Qwen MoE, because communication between cards dominates each step and hides the extra expert loads.
Mistral's official EAGLE drafter is slower than plain decoding ({ms['speedup']:.2f}×).</p>
{table(fn)}

<h2 id="lossless">Losslessness</h2>
{fig(13, "fig13_losslessness", "Output divergence from plain decoding (T=0)",
     "(a) Every method diverges from plain decoding on many prompts, mostly at near-tied tokens; plain decoding reproduces exactly on 3 of 4 models. (b) Only n-gram and suffix on the hybrid Qwen models diverge where the model was confident (21–27% of prompts): wrong output. (c) Batch-invariant kernels remove all divergence.",
     table(T["lossless"]) + "<p class='note'>Qwen3.8 n-gram: pilot run, 30 prompts. Qwen3.6 n-gram: full run on a second node, compared with plain decoding from the same node. n-gram and suffix re-emit earlier text at confident tokens (gaps up to 18.6 nats), consistent with recurrent state not being rolled back after rejected drafts.</p>")}
<p><strong>Cost of exact losslessness (Gemma 4 31B, MTP):</strong> with batch-invariant kernels, plain decoding falls from {bi['off']['ar_tok_s']} to {bi['on']['ar_tok_s']} tok/s, and MTP from {bi['off']['mtp_tok_s']} to {bi['on']['mtp_tok_s']} tok/s, about the speed of plain decoding with standard kernels.</p>
<p><strong>Task accuracy at T=1 (GSM8K, 1,319 problems × 3 seeds):</strong> every speculative method is within ±0.5 points of plain decoding.</p>
{table(T["accuracy_t1"], fmt={"gsm8k_acc": lambda v: f"{v:.2f}%", "sd": lambda v: f"±{v:.2f}"})}

<h2 id="gaps">Gap matrix</h2>
<p>Speedup vs plain decoding (ranges across models) unless marked τ. “—” means not measured.</p>
{gap_html}
<p class="note">† prefix caching off. ‡ vLLM loads Qwen3.5-family draft models with the MTP-head class and crashes; numbers are offline simulations. Simulations measure acceptance only; each draft-model or layer-skip step also pays for k sequential draft passes.</p>

<h3 style="margin-top:24px">Openings for a new method</h3>
<ol>
  <li><strong>Draft length that adapts to load:</strong> long drafts when the GPU is memory-bound, short or none when compute-bound. The best k changes with concurrency (Fig. 5), and vLLM uses a fixed k.</li>
  <li><strong>MoE-aware verification:</strong> choose or order draft tokens to limit how many distinct experts one verify pass touches (Fig. 9), or verify expert by expert.</li>
  <li><strong>Long-context verification:</strong> verify cost grows with context faster than the savings. Sparse or KV-compressed verification (TriForce/MagicDec style) was not tested here.</li>
  <li><strong>Hybrid-safe drafting and trees:</strong> recurrent-state rollback is fragile (n-gram and suffix corrupt output); tree verification over recurrent layers is unexplored.</li>
  <li><strong>Cheaper exact losslessness:</strong> fast batch-invariant kernels for gfx90a, and a skinny GEMM for 3–16 tokens that would lower verify cost for every method.</li>
</ol>

<h2 id="vllm">vLLM 0.30 issues found on MI250</h2>
<ul>
  <li><strong>Skinny-GEMM misroute:</strong> 3- and 5-token matrix multiplies go to <code>wvSplitK</code>, 2.4× / 5.7× slower. This made every method slower than plain decoding until patched (Fig. 7).</li>
  <li><strong>Default attention backend:</strong> <code>ROCM_ATTN</code> falls back to a slow path on gfx90a. Plain decoding at 32k: 6.8 vs 31.2 tok/s with <code>TRITON_ATTN</code>. Short prompts: ~0–12% faster with Triton, same method ranking.</li>
  <li><strong>DFlash + prefix caching:</strong> at 6–8k context, acceptance collapses to ~1.0 (vs 3–4.7 with caching off).</li>
  <li><strong>n-gram and suffix on hybrid models:</strong> wrong output on Qwen3.8 and Qwen3.6 (103–118 of 480 prompts diverge at confident tokens). MTP, DFlash and Medusa are fine on the same models.</li>
  <li><strong>Setup bugs patched in the image:</strong> Pixtral import (Mistral), a missing attribute in the Mistral EAGLE drafter, Medusa <code>vocab_size</code> on multimodal configs, no FP8 kernels for gfx90a, and the draft-model method crashing with Qwen3.5 drafts (not patched).</li>
</ul>

<h2 id="caveats">Caveats</h2>
<ul>
  <li>The main T=0 figures use vLLM’s default attention backend. The Triton rerun changes absolute speed by up to 12% but not the method ranking (table below).</li>
  <li>T=0 runs are single runs; plain-decoding repeats give the noise floor. The T=1 speedups use 3 seeds.</li>
  <li>Draft model, layer skip, prompt lookup and SAM-style results are offline acceptance simulations, validated against real runs to within 1.5% (n-gram, suffix, draft model).</li>
  <li>Medusa heads were trained for 2 epochs on 6–8k self-distilled examples, lighter than the paper's recipe.</li>
  <li>Not covered: EAGLE-3 on the main models (no drafters available), tree verification, TriForce/LongSpec/PEARL/AMUSD ports.</li>
</ul>
<details><summary>Default vs Triton attention</summary>{table(T["triton"])}</details>

<h2 id="repro">Reproduce</h2>
<ul>
  <li>Image: <code>docker/Dockerfile</code> (vLLM 0.30 ROCm + patches); run with <code>GPUS=0,1 docker/run.sh …</code>.</li>
  <li>Benchmarks: <code>scripts/bench_vllm.py</code> (batch size 1), <code>concurrency_bench.sh</code>, <code>trace_vllm.py</code> (streaming acceptance), <code>accuracy_vllm.py</code>, <code>router_union.py</code>.</li>
  <li>Offline: <code>sim_offline.py</code>, <code>sim_draft_tf.py</code>, <code>sim_layerskip.py</code>; Medusa: <code>medusa_gen.py</code> → <code>medusa_hidden.py</code> → <code>medusa_train.py</code> (heads in <code>results/medusa_heads/</code>).</li>
  <li>This page: <code>scripts/make_plots.py</code> then <code>scripts/make_site.py</code>. Raw results: <code>results/</code>.</li>
</ul>
</main>
</body>
</html>
"""
open(f"{ROOT}/website/index.html", "w").write(page)
print("wrote website/index.html")
