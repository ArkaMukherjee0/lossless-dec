# Gap analysis, Oct 2026: raising acceptance for long drafts and under load

Sources: the 95-paper 2025–26 catalogue (`lit/papers_2026.md`), the lineage map now at 102 papers (`lit/l1_tree.pdf` … `lit/l4_gaps.pdf`), and our own MI250 measurements (`lit/l5_depth.pdf`, `results/dk_grid*`).

## 1. The two research lines, restated with data

**Acceptance does not depend on load.** τ is identical from 4 to 128 concurrent requests for every model and drafter we ran (e.g. DFlash k=15 on Gemma 4 31B: 3.58 / 3.62 / 3.58 / 3.58). The literature agrees: AdaFlash (Table 1, 1–128 requests), Beyond Parallel Blindness (5.18 at c=1 and c=32), MoESD, AngelSpec, LiLiCorr (−1.4% to +0.5%). Load only raises the price of each verified token.

**Acceptance does not decay with depth; it compounds.** The conditional rate P(position j accepted | j−1 accepted) is roughly constant along the draft: DFlash 0.65–0.80, MTP 0.76–0.84 (MTP's first position is a little higher because it sees the target's real hidden state). Beyond Parallel Blindness reports the same shape (DFlash reached-path risk flat at 0.17–0.22). With a constant rate α, accepted length is capped near 1/(1−α): 4 at α = 0.75, 10 at α = 0.9.

So both lines reduce to one quantity, per-position acceptance α, plus how much a verified token costs:

| | 4 requests | 16 | 64 | 128 |
|---|---|---|---|---|
| α needed for a 16-token draft to beat 4 and 8 (Gemma 4 31B, MI250) | 0.77 | 0.88 | 0.94 | 0.93 |
| same, Gemma 4 26B-A4B (MoE) | 0.85 | 0.82 | 0.885 | 0.96 |
| best speedup today (α ≈ 0.75, Gemma 4 31B) | 2.28× | 2.02× | 1.28× | 1.03× |
| best speedup at α = 0.9 | 4.60× | 3.38× | 1.61× | 1.36× |

MI300/MI355 have flatter verify costs, so their thresholds are lower.

## 2. What 2025–26 tried (mechanism families)

| Family | Papers | Best reported | Limitation / saturation signal |
|---|---|---|---|
| **Predecessor conditioning inside parallel drafters** | DSpark, Domino, xPress, DSpine, JetSpec, DFlash 2, LiLiCorr, DBLAST, AngelSpec, Weaver, TreeFlash, Parallel Blindness | xPress τ 10.11 vs DFlash 6.48 (Qwen3-8B, GSM8K, block 16, greedy; α ≈ 0.93) | 12 papers in five months; LiLiCorr's controlled comparison puts four of them within 3%. Mostly math, batch ≤ 32. |
| **Path-conditioned trees** | PCTree, DARTree, DominoTree, TreeSpark, TAPS, GRAFT, PRESTO, JetSpec | DARTree τ 12.97, DominoTree 12.86 (64 nodes) | Batch 1 and greedy for most. Trees shrink to chains at batch ≥ 16 unless one budget is shared across requests (CAST +7–14%, ECHO +8–14% at 256). |
| **Acceptance-aligned training objectives** | LK, EAL/WTV, BV loss, VAT, GTO, EDR, PARD-2, SpecDiff-2 | BV loss +13–21% τ | Gains of 7–21%, near zero under greedy for strong drafters. None combined with each other. |
| **On-policy / online / test-time adaptation** | Draft-OPD, AdaFlash, TTS, TLT, ReSpec, Online SD | AdaFlash τ 7.09 → 9.83; TTS +41–72% on long reasoning | Largest where the drafter is most stale. None applied to conditioned drafters. |
| **Richer target conditioning** | DFlare, KVShot, Carryover (+ DFlow, ReTrace), D²SD, H-Spec | D²SD +33% τ by re-anchoring a second drafter at likely rejections | Carryover / DFlow / ReTrace (reusing rejected-position target states) all appeared in Aug–Sep 2026. |
| **Horizon extension** | DBloom (block 16 → 24), Attention Drift (post-norm past TTT depth) | DBloom τ 10.54 at block 24 (MATH-500) | Plain drafters only; naive widening erodes the front positions. |
| **Sequence-level verification** | HSD, Block / Traversal verification, GBV, UniVer | +2–9% at T = 1 | Cannot add anything under greedy decoding. |
| **Retrieval for long runs** | SuffixDecoding hybrid, DAS, Graft, WhiFlash, APEX | SuffixDecoding τ 7.8–17 on agentic text | Only on repetitive text. |
| **Load-aware verify budgets** (cost side; stack with all of the above) | DSpark adaptive verification, D-cut, DScale, AdaFlash length head, CAST, ECHO, TreeSpark, DARTree | DSpark in production at ~200 concurrent requests | Decide how much to verify, never raise acceptance. No paper tests drafts of 8 or more beyond 128 requests. |

## 3. What remains open, ranked by headroom × openness

**G1. Open-ended content.** The α ≈ 0.93 regime exists only on math. On chat-like text the best drafters stay at τ 3.5 (DFlash, MT-Bench, block 16) to 5.4 (DBloom, block 24), and our Spec-Bench mix gives α 0.65–0.80. Almost every 2026 paper headlines GSM8K / MATH-500, and none targets the positions that dominate rejections on open-ended text. *Unknown:* whether chat rejections concentrate at near-tied or high-entropy target tokens. If they do, the problem is matching the target's choice at uncertain tokens, not depth. → measurement M1.

**G2. Correlated concurrency.** Acceptance is independent of load because requests are independent. Batches of correlated requests are the one regime where acceptance can *rise* with concurrency: GRPO rollout groups, best-of-N / self-consistency, agent fan-out over a shared context. Known prior art is model-free: DAS's suffix trees over recent rollouts, RhymeRL, SuffixDecoding's global tree. The catalogue has no neural drafter that conditions on concurrent sibling streams, but that has not been novelty-checked yet. → measurement M2.

**G3. Drafts beyond 24 tokens at α ≥ 0.95.** About half of DARTree's GSM8K rounds accept the whole 16-token block (DBloom calls this "stranded speed-up"), so the block is the cap on easy content. Nobody combines conditioned drafters with long blocks (0 in the gap grid). On MI250, long drafts under load need α ≥ 0.93–0.96.

**G4. Combinations for conditioned drafters.** The gap grid has zeros for conditioned parallel drafters × online adaptation, × RL rollouts, × re-anchoring, × target KV reuse, and × long block. Each is a natural next paper, but incremental and quickly contested.

**G5. Long-output drift.** TTS reports DFlash τ falling 15 → 1.7 within 25K reasoning tokens (Qwen3.6-35B, AIME). Our Fig. 12 found no drift up to about 12K. Either the collapse starts later, or it depends on setup (we saw DFlash acceptance collapse with prefix caching at 6–8K). Cheap to settle with our harness.

## 4. Suggested next measurements (before choosing)

- **M1, where do rejections happen?** Log DFlash and MTP drafts (scripts/draftlog) together with the target's top-2 margin (`--save-gaps`) on Spec-Bench math vs chat categories. Measure rejection rate by target margin and entropy, and the drafter's top-k coverage at rejected positions. This tells whether G1 is a near-tie problem.
- **M2, how much do siblings know?** n = 8 samples at T = 1 per prompt (MATH, GSM8K, chat). Compare the oracle accepted length from the best-matching sibling continuation with DFlash/MTP acceptance and with suffix-tree retrieval over siblings. This gives an upper bound for G2.
- **Novelty check for G2** (neural drafting conditioned on concurrent requests), as was done for the earlier candidates.

Both measurements reuse existing scripts and take about 1–2 GPU-hours each.

## 5. Measurements (Oct 8, 2026)

Scripts: `scripts/rejection_anatomy.py` (M1), `scripts/sibling_predictability.py` (M2), figure `lit/l6_measurements.pdf`. Raw data: `results/m1/`, `results/m2/`.

### M1: where greedy drafts are rejected (T = 0, Spec-Bench, 120 prompts, 22–25k reached draft positions per model)

| | Gemma 4 31B + DFlash | Gemma 4 26B-A4B + DFlash | Qwen3.8-27B + DFlash2 |
|---|---|---|---|
| per-position acceptance α (math / chat / QA) | 0.73 (0.86 / 0.74 / 0.62) | 0.68 (0.83 / 0.69 / 0.56) | 0.82 (0.92 / 0.83 / 0.75) |
| share of rejections at near-tied target tokens (< 1 nat) | 18% | 18% | 44% |
| share of rejections where the target was very sure (≥ 8 nats) | 24% | 29% | 4% |
| α if every near-tie rejection were fixed | 0.78 | 0.74 | 0.90 |
| rejected token absent from the drafter's top 8 (QA) | 23% (34%) | 27% (38%) | 14% (20%) |
| at near-tie rejections: target token = drafter's #2 / in its top 4 | 29% / 54% | 28% / 50% | 37% / 62% |

- **The weaker DFlash drafter mostly misses content the target is sure of** (82% of its rejections are at margins ≥ 1 nat; on QA, a third of rejected tokens aren't even in its top 8). That's a knowledge or information gap, not tie-breaking.
- **The stronger, conditioned DFlash2 almost never misses confident tokens.** Its residual rejections shift to near-ties (44%), and fixing those alone would lift α to 0.90, roughly what long drafts need under load. The target and drafter both differ between these models, so the shift can't be pinned to the drafter alone.
- **Near-tie misses are not narrow ones.** The target's choice is the drafter's #2 only about a third of the time, so branching at likely ties recovers at most about half of them. The drafter does not model the target's uncertain set.
- **Depth makes no difference here either:** α at positions 1–2 vs 3+ is 0.74 / 0.73 (Gemma 4 31B) and 0.83 / 0.82 (Qwen3.8).

### M2: what sibling samples know (Gemma 4 31B + DFlash k=15, T = 1, 48 prompts × 8 seeds, 20k drafter steps)

Proposals are deterministic chains of 15 tokens from the same anchor; their realized match with the sampled continuation is an unbiased estimate of lossless acceptance at T = 1. Oracle = the better of drafter and sibling proposal at each step (selection upper bound); switch = the sibling proposal when its suffix match is ≥ 8 tokens, else the drafter.

| τ | 1 sibling | 2 | 4 | 7 |
|---|---|---|---|---|
| DFlash top-1 chain (no siblings) | 3.66 | | | |
| siblings so far (concurrent, time-aligned) | 1.46 | 1.71 | 2.00 | 2.25 |
| oracle: drafter or concurrent sibling | 3.87 | 3.98 | 4.13 | 4.25 (+16%) |
| finished siblings | 2.76 | 3.40 | 4.08 | 4.62 |
| oracle: drafter or finished sibling | 4.67 | 5.06 | 5.50 | 5.85 (+60%) |
| switch to finished sibling | 4.21 | 4.44 | 4.77 | 5.06 (+38%) |
| controls: own earlier output / other prompts' outputs | 1.06 / 1.04 | | | |

- **Acceptance rises with the number of correlated requests, on top of a strong neural drafter.** The 7-sibling oracle gains +37% on math, +48% on chat, +65% on QA, +75% on RAG, +108% on summarization and +63% on translation.
- **The information is specific to siblings,** not generic text reuse: both controls stay at τ ≈ 1.05.
- **Sibling progress matters:** concurrent siblings at the same point give +16%; finished siblings give +60%. Real concurrent decoding with uneven progress, and RL's long tail, fall in between.
- **Simple selection leaves most of the concurrent-sibling gain and a quarter of the finished-sibling gain on the table:** the switch gets 3.74 vs the oracle's 4.25 (concurrent) and 5.06 vs 5.85 (finished). Model-free sibling drafting (Seer, SRT, multi-sample SD) is published; a learned drafter that conditions on siblings is not (novelty check, Oct 8).

### What this means for the two lines

1. **Correlated concurrency (G2) has a measured, large, category-general headroom,** and the open part is exactly the learned combination. Next step that needs no GPU: fit a per-step or per-position selector on the M2 logs (features: suffix-match length, how many siblings agree, drafter confidence, position) and see how much of the oracle gap a cheap learned rule closes. Then a sibling-attending drafter (DFlash-style, cross-attention to sibling continuations) against a Seer-style "DFlash + group suffix candidates" baseline at G = 8–16, T = 1.
2. **Open-ended content (G1) has two regimes.** Weaker drafters lack information (siblings and retrieval help there); strong conditioned drafters are capped by near-tied target tokens they model poorly. The second suggests training drafters specifically on the target's uncertain set, which overlaps with the acceptance-aligned losses (LK, EAL/WTV, BV) and needs a closer novelty look before it counts as a direction.
