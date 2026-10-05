# Lossless Decoding — Comparative Experiment Plan (MI250)

Goal: implement and compare lossless acceleration methods on 2026 backbones (dense, MoE, hybrid linear-attention) and **find where each one fails**. The point is to find gaps for a new method, not to rank the methods.

---

## 0. Hardware & constraints (measured on `smci250-ccs-aus-c08-28`, 2026-09-25)

| Item | Value | Consequence |
|---|---|---|
| GPUs | 4× MI250 = **8 GCDs** (gfx90a / CDNA2), 64 GB HBM2e each, ~1.6 TB/s per GCD, ~180 TFLOPS BF16 peak per GCD | Each GCD is a separate device |
| Free GCDs | **8** (all free as of 2026-09-25; 512 GB total) | 4 × TP=2 slots, **or** 8 × single-GCD slots, **or** 1 × TP=8 for the large-MoE tier |
| Precision | **No FP8 on CDNA2** | FP8 checkpoints unusable; 27–35B models run BF16 at TP=2 |
| Compute/BW ridge | ~113 FLOP/byte per GCD (H100 ≈ 295) | The "free" verify budget is smaller than on H100, so big trees cost more (H1) |
| ROCm | 7.1.2, Docker available (`rocm/pytorch:rocm7.2…pytorch_2.9.1` already pulled) | vLLM ROCm image must be verified on gfx90a |
| CPU / RAM | 64 cores / 2 TB | REST/SAM datastores can be built and held in RAM |
| Storage | `/home` 400 GB free (99% full); local NVMe `/` 1.1 TB free; `/mnt/groupstorage` 9.8 TB free | Weights (~400 GB) → local NVMe; hidden-state caches (~1–2 TB) → groupstorage |

**Choosing TP pairs:** pair GCDs on the same package (probably (0,1), (2,3), (4,5), (6,7)). Confirm with `rccl-tests all_reduce_perf` before Phase 3, because inter-card xGMI is slower.

### Environment status (2026-09-25)

- **Image:** `lossless-dec:vllm0.30` ([docker/Dockerfile](docker/Dockerfile)), based on official `vllm/vllm-openai-rocm:v0.30.0` (ROCm 7.2.3, torch 2.12, vLLM kernels built for gfx90a). Launch with `GPUS=0,1 docker/run.sh <cmd>`. It runs as your user, with the model cache at `/var/tmp/arkamukh` → `/models`.
- **Spec methods available in vLLM 0.30:** ngram, ngram_gpu, suffix, draft_model, medusa, eagle, eagle3, dflash, dspark, gemma4_mtp, qwen3_5_mtp, qwen4_exp_mtp, nemotron_h_mtp, …
- **gfx90a GEMM bug (patched in the image):** vLLM routes 1–5-token GEMMs to the `wvSplitK` skinny kernel, which is 2.4× (3 tokens) and 5.7× (5 tokens) slower than hipBLASLt on gfx90a. Unpatched, every spec method ran 3–4× *slower* than AR (verify = 1 + 4 draft = 5 tokens). The patch keeps `wvSplitK` only for 1, 2 and 4 tokens.
- **Measured (Qwen3-8B, 1 GCD, bs=1, patched):** AR 72 tok/s (~1.14 TB/s effective, matches §8 assumptions). ngram 79 tok/s (1.10×), draft_model Qwen3-0.6B 82 tok/s (1.13×), 3 prompts only.
- **MI250 "verify tax" (new gap to note):** a 1-token decode GEMM runs at ~1.27 TB/s (skinny kernel), but 3–16-token verify GEMMs run at only ~0.77 TB/s (hipBLASLt). So one verify step costs ~1.6× an AR step, not ~1×. Every spec method on MI250 pays this. A gfx90a-tuned skinny GEMM for 3–16 tokens would raise all of them (an engineering win, and relevant to H1). Also: vLLM's hipBLASLt path is ~13% slower than raw `F.linear` at 3+ tokens (not yet investigated).
- **Greedy losslessness:** spec vs AR outputs diverge only at BF16 exact or near ties (top-2 logprob gap 0 or 0.125 = 1 BF16 ulp), never early. This is the E10 metric: report the divergence rate, not "exact".
- **Benchmark hygiene:** warm up on the same batch shapes you time; vLLM JIT-compiles Triton kernels for new shapes mid-run.

---

## 1. Models (2026 releases + reproduction anchors)

| ID | Model | Type | Size (BF16) | Placement | Why |
|---|---|---|---|---|---|
| A0 | Vicuna-7B-v1.3 | dense, full attn | 14 GB | 1 GCD | Reproduction gate: every legacy method (Medusa, EAGLE, Kangaroo, REST, Lookahead, PLD, SpS-68m) has public artifacts and published Spec-Bench numbers |
| A1 | Llama-3.1-8B-Instruct | dense, full attn, 128K | 16 GB | 1 GCD | Modern anchor with public EAGLE-3 / DFlash drafters; long-context baseline for TriForce/LongSpec |
| **M1** | **Qwen3.8-27B** (Aug 14 2026, Apache-2.0) | dense **hybrid**: 48 Gated-DeltaNet + 16 gated-attn layers; built-in MTP | 54 GB | TP=2 | Main dense target; recurrent state stresses tree verification and rollback |
| **M2** | **Gemma 4 31B** (Apr 2 2026, Apache-2.0) | dense, sliding-window + global attn; official MTP drafter | 62 GB | TP=2 | Main "conventional" dense target |
| **M3** | **Gemma 4 26B-A4B** | MoE, 3.8B active | 50 GB | TP=2 | Clean dense-vs-MoE pair with M2 (same family/tokenizer) |
| **M4** | **Qwen3.6-35B-A3B** | MoE (+ hybrid attn, verify) | 70 GB | TP=2 | MoE + hybrid; pairs with M1 |
| M5 | NVIDIA Nemotron hybrid Mamba-2/MoE (~31.6B / A3.6B, Aug 2026) | SSM hybrid MoE | ~63 GB | TP=2 (4th slot) | Third architecture class, promoted from optional now that a 4th slot is free; **risk:** Mamba-2 kernels on gfx90a |

**Large-MoE tier (needs the whole node, TP=8, 512 GB).** Now feasible because all 8 GCDs are free:

| ID | Model | Size (BF16) | Fit at TP=8 | Why |
|---|---|---|---|---|
| L1 | **Qwen3.8-Flash-Next** (Aug 26 2026; 125B / A6B, 512 experts, GDN + sparse attn, 4B MTP module) | ~360 GB stored (incl. 51B n-gram embedding table) | yes, ~150 GB left for KV and activations. The n-gram embedding is a lookup table and could move to CPU RAM (2 TB), leaving ~260 GB on GPU | MoE at a scale where the union of experts is large, plus hybrid attention and native MTP. Stress test for H2 and H3 |
| L2 | **Mistral Small 4** (Mar 2026; 119B / A6.5B) | ~238 GB | yes, comfortable (TP=4 is too tight for KV cache) | Conventional-attention large MoE; control for L1 |

Caveats: (a) bs=1 decode at TP=8 crosses cards over xGMI, so all-reduce latency takes a large share of each step. Estimate 25–40 tok/s AR. (b) vLLM/SGLang support for L1's new architecture on gfx90a is unverified; that's the highest porting risk. (c) Check sizes against the HF configs before downloading (~600 GB total → groupstorage, not the local NVMe).

**Still excluded in BF16:** DeepSeek-V4-Flash (284B → 568 GB), Inkling-Small (276B), GLM-5.3 (320B), Kimi K2.6/K3, Qwen3.8-2.4T. Possible stretch: DeepSeek-V4-Flash with FP8 weights dequantized in-kernel (W8A16), following the MI250/SGLang port in arXiv 2609.15627. It's high risk and not scheduled.

**Drafters per model** (check that each exists and matches the tokenizer before Phase 2):

| | SpS draft model | Native MTP | DFlash | EAGLE-3 |
|---|---|---|---|---|
| M1 | smallest Qwen3.6/3.8 with the same tokenizer | built-in (multi-step) | `z-lab/Qwen3.8-27B-DFlash2` (1.93B, vLLM ≥ 0.28) | only community Qwen3.6-27B version → test transfer, or train |
| M2 | Gemma 4 E2B | Google Gemma 4 MTP drafter | public (JarvisLabs benchmark) | train (optional) |
| M3 | Gemma 4 E2B | Google Gemma 4 MTP drafter | public | — |
| M4 | small Qwen3.6 | built-in (verify) | check z-lab | — |

---

## 2. Methods → families

Your literature table, grouped by where the draft comes from. **Bold** marks a modern reference added because a new method must beat it.

| Family | Methods | Training? | Notes |
|---|---|---|---|
| F1 Model-free / retrieval | LLMA (= prompt-lookup/PLD with reference copy), REST, SAM Decoding, LogitSpec, **Lookahead**, **SuffixDecoding** | none (datastore for REST) | One REST datastore per tokenizer family (Llama/Qwen/Gemma) |
| F2 Independent draft model | Speculative Decoding (SpS), PEARL, AMUSD, AdaSpec, Traversal Verification (a verify rule applied on top of tree drafts, matters at T>0) | none (reuse small model) | PEARL/AMUSD put the drafter on its own GCD |
| F3 Trained heads / modules | Blockwise Parallel Decoding (covered by Medusa chain mode), Medusa, ParallelSpec (**P-EAGLE** as proxy if no code), DFlash, **EAGLE-3**, **native MTP** | yes | Use public checkpoints where they exist; train only Medusa for new models |
| F4 Self-speculative | Draft & Verify (layer skip), Kangaroo, Cassandra, **SWIFT** (training-free layer skip) | D&V: layer search; Kangaroo: adapter | |
| F5 Long-context | TriForce, LongSpec, SpecExtend (add-on to F2/F3), (**MagicDec**) | varies | Only in E4 |

Dropped: Staged SD (targets <1B on-device models, already covered by TriForce's hierarchy).

---

## 3. Implementation stack

Two layers, so the algorithm comparison stays fair and the wall-clock numbers stay realistic:

1. **Unified research harness** (PyTorch + HF transformers on ROCm, static cache, `torch.compile` where it works):
   - One `Target` wrapper with **chain verify**, **tree verify** (tree attention mask), and **state rollback** for recurrent layers (GDN/Mamba: snapshot the state at each accepted prefix).
   - Each method only implements `propose(ctx) → draft tree`. Verification and acceptance are shared: greedy match, speculative sampling, and traversal verification.
   - Instrumentation: per-position acceptance, τ per verify step, draft vs verify time, MoE expert IDs per verify batch, peak memory.
   - Start from Spec-Bench (hemingkx/Spec-Bench) code for legacy methods and port each to the new architectures.
2. **Serving engine** (vLLM ROCm ≥ 0.28, fall back to SGLang): AR, ngram/PLD, suffix, draft-model, MTP, DFlash, EAGLE-3. Used for concurrency and wall-clock results. Check that harness τ matches engine τ.

---

## 4. Metrics (every run)

- **Speed:** wall-clock speedup vs AR in the same stack, tokens/s, TTFT (retrieval and prefill overhead)
- **Algorithm:** τ (mean accepted length incl. bonus token), acceptance α, **acceptance by draft position**, **acceptance vs output position** (drift over long outputs)
- **Cost:** draft time share, verify latency(k), extra memory (GB), setup cost (GPU-hours to train, datastore GB)
- **Robustness (gap finders):** speedup variance across categories, **fraction of prompts slower than AR**, worst-category speedup
- **Losslessness:** at T=0, exact token match vs AR on the first 512 tokens (report the divergence rate; BF16 tree-verify numerics can flip near-ties). At T>0, task accuracy over 3 seeds plus a distribution test on 5 prompts × 1000 samples of the first 8 tokens.

## 5. Workloads

| Suite | Content | Output tokens / run |
|---|---|---|
| **Core** | Spec-Bench (480 prompts: MT-bench, translation, summarization, QA, math, RAG; max 1024) + HumanEval (164) + MBPP (100), thinking **off** | ~270k |
| **Lite** | 40 per Spec-Bench category + 80 code | ~120k |
| **Long-ctx** | 40 prompts (PG-19 continuation + LongBench-v2 QA) at 4k / 16k / 64k context (+128k for A1), 256 out | ~10k per length |
| **Reasoning** | AIME-2025 (30) + MATH500 L5 (10), thinking **on**, cap 16k | ~360k |

---

## 6. Hypotheses → experiments

| Hypothesis (candidate gap) | Experiment |
|---|---|
| **H1** On MI250 the flat part of latency(k) is short, so large trees and wide blocks (Medusa, EAGLE trees, DFlash k=8+) lose their edge | E0.1, E8 |
| **H2 MoE penalty:** verifying k tokens activates the *union* of their experts, so verification is not ~free and speedups are lower and optimal γ smaller than on dense | E1 (M2 vs M3), E6 |
| **H3 Hybrid/recurrent targets** break tree methods (state forking) and make rollback cost significant. They also shrink the KV cache, which weakens the premise of KV-compression drafters (TriForce, MagicDec) | E4, E7 |
| **H4** Model-free methods have bimodal gains: large on RAG, summarization and code, ~1× on chat and translation | E1 per-category |
| **H5** Trained drafters lose acceptance at T=1, out of domain (translation), and deep into long reasoning traces | E1 T-sweep, E5 |
| **H6** Speedups shrink with concurrency and many fall below 1× by c=64; adaptive methods (AdaSpec) only partly recover | E3 |
| **H7** Pipelined drafting (PEARL, AMUSD) gains depend on inter-GCD link latency; same-card vs cross-card placement matters | E9 |

## 7. Experiments

- **E0 Hardware characterization.** (0.1) latency(k) of a single target forward for k ∈ {1…128} on every model, which gives the free-verify budget. (0.2) AR baselines on harness and vLLM, which calibrate every estimate below. (0.3) RCCL pair bandwidth test.
- **P1 Reproduction gate.** All F1–F4 methods on A0/A1 with the Core suite at T=0. Pass criterion: τ within ~10% of Spec-Bench and paper numbers. Speedups will differ on MI250, which is expected; record by how much.
- **P2 Artifacts.** REST datastores (3 tokenizers + code corpus). Self-distilled data (20k UltraChat/ShareGPT conversations regenerated by each target). A hidden-state cache shared by Medusa and Kangaroo. Medusa heads (M1–M4), Kangaroo adapters (M1–M4). D&V layer search. Optional EAGLE-3 training for M2.
- **E1 Main matrix.** Every feasible method × M1–M4. Core suite at T=0, Lite suite at T=1 (and T=0.6 for 5 top methods). Record N/A with the reason; incompatibilities are findings too.
- **E1b Engine validation.** vLLM-supported methods × M1–M4, Core suite at T=0.
- **E3 Concurrency.** vLLM, {AR, ngram, draft, MTP, DFlash, (EAGLE-3)} × M1–M4 × c ∈ {1, 4, 16, 64}.
- **E4 Long context.** {AR, PLD, SpS, MTP, DFlash, TriForce, LongSpec, SpecExtend} × {A1, M1, M2} × {4k, 16k, 64k}.
- **E5 Reasoning / long generation.** {AR, PLD, SAM, SpS, MTP, DFlash, Cassandra, SpecExtend} × {M1, M3}. Key plot: α vs output position.
- **E6 MoE verification cost.** Hook the router: distinct experts activated vs k, and latency(k) for M3/M4 vs M2/M1. Derive the speedup ceiling and compare it with measured speedups.
- **E7 Architecture compatibility.** Method × {full-attn, SWA+global, GDN hybrid, Mamba hybrid}: does it run, how much work did it take, and what does rollback cost.
- **E8 Draft-length / tree-budget ablation.** {SpS, Medusa/EAGLE-3, MTP, DFlash} × γ ∈ {1, 2, 3, 4, 6, 8} (+ tree sizes 8/16/32/64) × {M2, M3}, Lite subset.
- **E9 Multi-device.** {SpS, PEARL, AMUSD} × {A1, M1} × {same-card, cross-card} drafter placement.
- **E10 Losslessness.** T=0 exact-match check comes free with E1. T>0 distribution tests on 2 models (M2, M3).

---

## 8. Time estimates

### Throughput assumptions (to replace with E0.2 numbers)
BF16, batch size 1. Weight-streaming bound at ~65% of HBM bandwidth, plus TP all-reduce overhead.

| Model | Placement | vLLM AR tok/s | Harness AR tok/s (~0.5×) |
|---|---|---|---|
| A0/A1 (7–8B) | 1 GCD | 40–50 | 25–30 |
| M1 Qwen3.8-27B | TP=2 | 22–28 (GDN kernels on gfx90a may cost more) | 11–14 |
| M2 Gemma 4 31B | TP=2 | 20–26 | 10–13 |
| M3/M4 MoE A3–4B | TP=2 | 50–75 (Triton fused-MoE on gfx90a is not tuned) | 25–35 |

Run time = tokens / (AR tok/s × speedup) + ~15 min for load and warmup. Assumed speedups: ~2× on dense, ~1.5× on MoE.

### Compute budget (8 GCDs = 4 TP=2 slots; per-phase wall-clock below was computed for 3 slots, totals are updated for 4)

| Phase | Slot-hours | Wall-clock (GPU) | Engineering (calendar) |
|---|---|---|---|
| E0 hardware characterization | ~15 | 0.5 day | 3–4 days (containers, gfx90a kernel issues, weights ~400 GB) |
| P1 reproduction gate (1-GCD slots, 6-way parallel) | ~30 GCD-h | 0.5 day | **2 weeks** (harness + ~20 method ports) |
| P2 artifacts (data regen 3.5 h, hidden dump 1.5 h, Medusa 5 h, Kangaroo 5 h, D&V search 5 h per model × 4) | ~120 GCD-h | 1–1.5 days | 1 week (overlaps P1) |
| └ optional EAGLE-3 training for M2 | 60–120 GCD-h | +1–1.5 days | |
| E1 main matrix (≈14 methods × 4 models; Core at T=0 + Lite at T=1) | ~210 | **~3 days** | port methods to hybrid/MoE: 1–1.5 weeks |
| E1b engine validation | ~36 | 0.5 day | |
| E3 concurrency | ~32 | 0.5 day | |
| E4 long context | ~30 | 0.5 day | TriForce/LongSpec ports: 1 week |
| E5 reasoning | ~47 | 0.7 day | Cassandra implementation: 3–5 days |
| E6 + E7 MoE / architecture analysis | ~10 | 0.3 day | 2–3 days of analysis |
| E8 ablations | ~29 | 0.4 day | |
| E9 multi-device | ~18 (3 GCDs per run) | 0.4 day | PEARL/AMUSD ports: 3–4 days |
| E10 sampling losslessness | ~8 | 0.3 day | |
| EL large-MoE tier (L1, L2 at TP=8; AR, PLD/SAM, SpS, MTP, DFlash/EAGLE-3 if public; Lite suite at T=0; E6 router analysis; short concurrency sweep) | ~16 node-hours (blocks all slots) | 0.7 day | port L1 architecture on gfx90a: 1 week (high risk) |
| **Total** | **~600 slot-h + ~16 node-h** | **~7.5 days (+30% for re-runs → ~10–11 days)** | **~7 weeks** (≈6 without EL) |

**The critical path is engineering, not GPU time.** Run each phase on the GPUs as soon as its ports land, while porting the next ones. Schedule EL in a single block (e.g., over a weekend) so it doesn't break up the TP=2 runs.

### Suggested calendar
- **Week 1:** E0; container; harness skeleton; PLD/SpS/MTP/DFlash in vLLM on M1–M4 (early signal)
- **Weeks 2–3:** port legacy methods, P1 reproduction gate; P2 artifacts running in the background
- **Weeks 3–4:** E1 + E1b + E6/E7 (first gap readout: MoE and hybrid)
- **Weeks 5–6:** E3, E4, E5, E8, E9, E10; EL (whole node) in one block once L1/L2 load in vLLM or SGLang
- **Week 7:** analysis, method-by-axis gap matrix, choose the new method's direction

---

## 9. Risks

1. **vLLM ≥ 0.28 on gfx90a:** AITER and many new spec-decode kernels target gfx942/950. DFlash2 needs ≥ 0.28. Fallbacks: Triton attention backend, SGLang (DeepSeek-V4 has been reported running on MI250 via SGLang), or the harness only.
2. **Kernels:** Gated-DeltaNet (Triton/FLA) and Mamba-2 on gfx90a; Gemma 4 global-attention head dim vs CK flash-attn limits on CDNA2.
3. **BF16 near-tie flips** make "lossless" slightly inexact at T=0. Report the divergence rate instead of assuming 0.
4. **Storage:** keep weights off `/home`. The groupstorage hidden-state cache needs about 0.6 TB per model for EAGLE-3 (3 layers) and 0.2 TB for Medusa.
5. **Shared node:** all 8 GCDs were free as of 2026-09-25, but other users may come back. Pin `HIP_VISIBLE_DEVICES` per slot and record co-tenancy in run metadata, because noise affects timing.

## 10. Output: the gap matrix

For each method, one row across these axes: training-free · extra memory · speedup at c=1 · at c=64 · at T=1 · worst category · % prompts slower than AR · long-ctx scaling · long-generation α drift · MoE penalty · hybrid-arch support · MI250 vs published H100 ratio. Cells where every method is weak are where a new method should aim.
