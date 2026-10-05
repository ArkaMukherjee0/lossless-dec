# 40-Hour Experiment Plan (revision 2, after pilots on 2026-09-25)

Budget: **40 h wall-clock on 8 GCDs = 4 TP=2 slots = 160 slot-hours.**
Long-term plan and background: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). This file supersedes its §6–§8 for the next 40 h.

---

## 1. Pilot results (30 Spec-Bench prompts, 5/category, bs=1, T=0, thinking off, 2 GCDs)

| Model | AR tok/s | MTP | DFlash | ngram |
|---|---|---|---|---|
| Qwen3.8-27B (dense, 48 GDN + 16 attn) | 32.5 | k=3: **1.93×** (τ 3.25) | k=7: **2.13×** (τ 4.51) | k=4: **0.69×**, corrupts output |
| Qwen3.6-35B-A3B (MoE + GDN) | 75.3 | k=3: **1.47×** (τ 3.16) | k=7: 1.21×, k=15: 1.34× | — |
| Gemma 4 31B (dense) | 30.4 | k=4: **2.00×** (τ 3.62) | — | — |
| Gemma 4 26B-A4B (MoE) | 80.2 | k=4: **1.86×** (τ 3.50) | — | — |
| Qwen3-8B (1 GCD, 3 toy prompts) | 72.3 | — | — | 1.10× (draft_model 0.6B: 1.13×) |

Concurrency (Qwen3.8-27B, MTP k=3 vs AR): c=1 **1.93×**, c=4 **1.81×**, c=16 **1.80×**. MI250 still has compute headroom at 64 verified tokens per step, so D must go to c=64 and above to find the crossover.

Per-category spread is large for every method. For example, DFlash on Qwen3.8-27B reaches 96 tok/s on math but 52 on QA. QA and summarization are the weakest categories for every method, and math the strongest.

## 2. What the pilots showed about the original plan

### Confirmed
- **MoE penalty (H2) exists, but depends on the architecture.** At equal τ, MTP's speedup drops from 1.93× to 1.47× between Qwen3.8-27B and Qwen3.6-35B-A3B, but only from 2.00× to 1.86× between the two Gemma 4 models. The size of the penalty is the interesting variable. It likely depends on expert count, top-k, or hybrid layers. → G (expert-union measurement).
- **Hybrid/recurrent targets are a problem (H3), and a worse one than expected.** vLLM's `ngram` / `ngram_gpu` on Qwen3.8-27B produce **wrong tokens**: they diverge where AR's top-2 logprob gap is up to 14.4 nats, and the output visibly re-emits earlier text. The same method on the non-hybrid Qwen3-8B was fine. The likely cause is recurrent (GDN) state not rolled back on the V1-runner path.
- **Category variance (H4)** is large for trained drafters too, not only for retrieval methods.

### Wrong or needs changing
1. **Throughput assumptions were too low.** Dense 27–31B at TP=2 runs 30–33 tok/s (assumed 20–28); MoE runs 75–80 (assumed 50–75). Every cost estimate shrinks.
2. **H1 was backwards.** On gfx90a, verify cost is a *step function*. One token uses a fast skinny GEMM (~1.27 TB/s effective). Anything from 2 to about 16 tokens costs roughly the same, ~1.6–1.7× an AR step. MTP k=1 and k=3 have *identical* step times (~52 ms) on Qwen3.8-27B. So once you verify at all, verify wide. That favours block/tree drafters, and k=1 is the worst setting. The real MI250 gap is the **verify tax**, not tree width.
3. **"Lossless" can't be measured as exact match.** Two identical AR runs diverge on **47% (27B) and 63% (MoE)** of prompts, always at BF16 ties. The metric has to be: divergence rate *relative to the AR-vs-AR noise floor*, plus the AR top-2 logprob gap at each divergence (a gap above ~1 nat is a real error).
4. **Wall-clock from a custom eager harness would be misleading.** Eager mode adds large fixed per-step overhead (Qwen3-8B AR: 45 vs 72 tok/s), which inflates spec speedups. So harness-only methods should report **τ**, and speedup should be *projected* with the measured verify cost curve, not timed.
5. **The engine confounds the algorithm.** `ngram` runs on vLLM's V1 runner, has no async scheduling, and ran 35% slower than AR *even when it proposed nothing* (the translation category). `draft_model` crashes for Qwen3.5-family drafts (vLLM loads the draft with the MTP-head class). Comparisons must separate "algorithm τ" from "engine overhead".
6. **τ isn't comparable across vLLM methods.** `ngram_gpu` counts every step as a draft (τ 1.11), while `ngram` counts only steps that had a match (τ 2.35), for the same accepted tokens. Compute τ ourselves from outputs and traces.
7. **Offline simulation works and is cheap.** Under greedy decoding, acceptance can be computed by replaying the drafter against saved AR outputs. For vLLM's ngram this matched the engine to within 1.5%. For any drafter that conditions only on tokens (draft model, layer-skip self-spec), greedy acceptance is the run length of consecutive teacher-forced top-1 matches. That's **one batched forward pass per (target, drafter)**, with no spec loop.
8. **Training-heavy and port-heavy methods don't fit in 40 h.** That covers Medusa/Kangaroo/EAGLE-3 training and TriForce/LongSpec/PEARL/AMUSD/Cassandra ports. They're deferred (§6). Public MTP and DFlash drafters cover the trained-drafter family (F3) for every target.
9. **The Vicuna reproduction gate is dropped.** Checking the simulator against the engine (7) validates the tooling more cheaply and on the models we care about.
10. **Benchmark hygiene** (fixed in the scripts): warm up on the timed batch shapes; use a separate vLLM compile cache per GPU set (concurrent runs raced on one cache); apply the chat template and set thinking explicitly. **Still untested: interference between the 4 concurrent slots.** All pilots ran 4 jobs at once. → A0.
11. **The concurrency harness was wrong.** `bench_vllm.py --concurrency` submits a batch and waits for its longest request, so slots sit idle. AR at c=4 reached only 2.1× its c=1 throughput. D must use continuous arrivals with a fixed number in flight (`vllm serve` + `vllm bench serve --max-concurrency`).

## 3. The 40-hour experiment set

Costs in slot-hours (1 slot = 2 GCDs) use the measured speeds. The full Spec-Bench suite is 480 prompts, max 1024 tokens, ~160k output tokens. A run costs about 1.5 h for dense AR, 0.8 h for dense spec, 0.6 h for MoE AR and 0.4 h for MoE spec, including load.

| ID | Experiment | Configs | Slot-h |
|---|---|---|---|
| **A0** | Slot-interference check: Qwen3.8-27B AR alone vs with 3 other slots busy | 2 runs, lite | 0.5 |
| **A** | **Main engine matrix**, full Spec-Bench, T=0, bs=1, saving logprob gaps. Per model: AR, AR-repeat (noise floor), MTP (best k), DFlash (k=7 and k=15), suffix; ngram on Gemma only (control for the hybrid bug); EAGLE-3 community drafter on Qwen3.8-27B if it loads | 4 models × ~7 | ~22 |
| **B** | **Sampling T=1**: AR, MTP, DFlash per model (τ drop at T>0; lossless check = task accuracy on GSM8K/HumanEval subsets vs AR) | 4 × 3 | ~11 |
| **C** | **k-sweep and verify cost curve**: MTP k ∈ {1,2,3,4,6}, DFlash k ∈ {3,7,11,15}; derive step time vs verified tokens per model. Feeds the cost model | 4 × 9, lite (120 prompts) | ~10 |
| **D** | **Concurrency** (continuous arrivals via `vllm bench serve`): AR, MTP, DFlash × c ∈ {4, 16, 64, 128} per model (where does spec fall below 1×?) | 4 × 12, lite | ~10 |
| **E1** | **Long generation / reasoning**: thinking on, AIME-2025 (30) + MATH500-L5 (20), cap 16k. AR, MTP, DFlash on all 4 models. Streaming per-step acceptance trace → **α vs output position** (drift) | 4 × 3 | ~16 |
| **E2** | **Long context**: 20 LongBench-v2 prompts at 8k/32k/64k, 256 out. AR, MTP, DFlash, suffix on Qwen3.8-27B (hybrid, small KV) vs Gemma 4 31B (SWA+global) | 2 × 4 × 3 | ~6 |
| **F** | **Offline acceptance simulation** (mostly CPU, on A's AR outputs): PLD/LLMA, REST (UltraChat/ShareGPT datastore per tokenizer), SAM, suffix tree, LogitSpec (from saved top-k logprobs). Teacher-forced GPU passes for SpS (Qwen3.5-0.8B → Qwen3.8-27B, where the engine is broken; Gemma E2B → 31B) and layer-skip self-spec (D&V/SWIFT-style skip-set search, cheap offline). Projected speedup = τ × measured cost curve (C). Validated against the real ngram/suffix/draft runs | 4 models | ~5 |
| **G** | **MoE expert-union**: hook routers in a teacher-forced pass and count distinct experts per layer for windows of 1…16 tokens, Qwen3.6-35B-A3B vs Gemma 26B-A4B. Explains the 1.47× vs 1.86× gap | 2 models | ~1 |
| **L** | **Large-MoE probe** (TP=8, blocks all slots): Mistral Small 4 and/or Qwen3.8-Flash-Next, AR + native MTP, lite suite. Stop early if the architecture doesn't load | ≤ 4 node-h | 16 |
| | **Subtotal** | | **~98** |
| | Re-run buffer (+25%) | | ~25 |
| | **Total** | | **~123 slot-h ≈ 31 h wall on 4 slots** |

That leaves ~10 h of the 40 h window for the offline/CPU work, analysis and fixing failures.

### Schedule (wall-clock)
| Hours | Slot 1 | Slot 2 | Slot 3 | Slot 4 | CPU / engineering (in parallel) |
|---|---|---|---|---|---|
| 0–1 | A0 | A0 | idle→A | A | per-step trace script, simulator core |
| 1–7 | A Qwen3.8-27B | A Gemma 31B | A Qwen3.6-35B-A3B → A Gemma 26B | C (MoE models) | REST datastores, SAM, cost model |
| 7–11 | B dense | B dense | B MoE → C dense | C dense | F on A's outputs as each model finishes |
| 11–16 | D | D | G + F teacher-forced | E2 | |
| 16–24 | E1 Qwen3.8-27B | E1 Gemma 31B | E1 MoE ×2 | E2 / reruns | analysis: gap matrix v1 |
| 24–28 | **L (all 8 GCDs)** | | | | |
| 28–40 | reruns / buffer | | | | final analysis |

### Outputs
1. **Gap matrix**: method × {τ, speedup at c=1/4/16/64, T=0 vs T=1, worst category, % prompts slower than AR, α drift over long outputs, long-ctx scaling, MoE penalty, hybrid correctness, engine support on gfx90a}.
2. **Verify cost curves** per model on MI250 (step time vs verified tokens), and the cost model used for projections.
3. **Losslessness report**: divergence vs noise floor, plus confident-divergence count (the hybrid ngram bug).
4. Candidate directions for the new method, ranked by the gaps observed.

## 4. Engineering needed during the run (no GPU)
- `bench_vllm.py`: add streaming mode (AsyncLLM, output delta per step) to record accepted tokens per step. Used only in E1/E2, so A's timing stays clean.
- `sim/`: token-only simulators (PLD, REST, SAM, suffix, LogitSpec) and a teacher-forced runner (HF transformers, device_map over 2 GCDs) with router hooks for G.
- `cost_model.py`: projected speedup = τ / (t_draft(k) + t_verify(k)) / t_AR, with values fitted from C.
- Data: AIME-2025, MATH500-L5, LongBench-v2 subsets, UltraChat for REST.

## 5. Risks inside the 40 h
- Concurrent slots may interfere (A0 decides; if they do, A/B/C timings run with at most 2 slots busy, which costs about 8 extra hours).
- `suffix` and EAGLE-3 may also use the V1 runner or be broken on hybrid models. Treat this as a finding and check divergence before trusting timings.
- L may not load at all (new architectures on gfx90a). The 16 slot-h then go to the buffer.

## 6. Deferred beyond 40 h
Medusa/Kangaroo/EAGLE-3 training; TriForce/LongSpec/SpecExtend ports; PEARL/AMUSD (an analytic upper bound from C's draft/verify times is included in the cost model instead); Cassandra; AdaSpec; Traversal Verification (needs tree sampling in the engine); a gfx90a skinny GEMM for 3–16 tokens (would lower the verify tax for every method); fixing the hybrid rollback bug in vLLM.
