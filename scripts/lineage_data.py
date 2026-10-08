"""Catalogue of lossless speculative-decoding papers behind the lineage figures (make_lineage.py).

PROBLEMS  id -> (band, text). A limitation of one or more papers. The same id under different parents
          means the same problem came back (the problem timeline merges them into one lane).
PAPERS    depth-first tree order. parents[0] = (tree parent, problem it fixes); further entries are
          extra (parent, problem) links that only the problem timeline draws.
          skill = what the paper brings; new / uses = design-space traits (ids from AXES):
          new = where the paper's own contribution lies, uses = inherited (best-effort reading).
Dates are the first arXiv version (code release for prompt lookup); venues are the first
peer-reviewed venue, checked Oct 2026.
"""

BANDS = ["what drafts", "draft shape & verification", "systems & serving"]

PROBLEMS = {
    "bpd_limits": (-1, "needs a fine-tuned base\nmodel; greedy only"),
    # what drafts
    "second_model": (0, "a second model to train\nand host next to the target"),
    "skip_search": (0, "skip set found by slow offline\nsearch, fixed per task"),
    "weak_exit": (0, "skipped-layer drafts are\nweak without training"),
    "coarse_skip": (0, "layer skipping is coarse; at\nbatch 1 weight reads dominate"),
    "blind_heads": (0, "each head guesses t+k blind\nto the drafted tokens before it"),
    "feat_regress": (0, "feature-regression loss caps\ngains from more data"),
    "exposure": (0, "trained on target features,\ndrafts on its own (exposure bias)"),
    "bolted_heads": (0, "heads bolted onto a model\npretrained for one token"),
    "vocab": (0, "drafter must share the\ntarget's vocabulary"),
    "copy": (0, "a model drafts even text\nalready in the context"),
    "no_reference": (0, "needs text to copy; open-\nended output has none"),
    "static_store": (0, "static datastore: big, domain-\nbound; misses the app's reuse"),
    "no_match": (0, "exact-suffix lookup often\nfinds no match"),
    "misaligned": (0, "drafter disagrees with the\ntarget: low acceptance"),
    "offline_kd": (0, "offline distillation misses the\nlive query distribution"),
    "serial_draft": (0, "drafting is autoregressive too:\nk serial passes per draft"),
    "same_drafter": (0, "same drafter for every position;\nlate tokens are rarely kept"),
    "per_target": (0, "parallel drafter retrained\nfor every target"),
    "diffusion_blind": (0, "standalone diffusion LM: large,\nblind to the target's state"),
    "block_conflict": (0, "block positions drafted apart\nconflict; fixed block length"),
    # draft shape & verification
    "single_chain": (1, "one draft chain: the first\nrejection discards the rest"),
    "static_tree": (1, "fixed tree shape ignores\nbudget, hardware and context"),
    "small_trees": (1, "trees sized for GPUs; offloaded\ntargets can verify 1000s"),
    "ot_approx": (1, "OT selection is approximate;\ngap to optimum unknown"),
    "tokenwise": (1, "token-by-token checks reject\nmore than necessary"),
    "chain_only": (1, "single chain only; no trees\nor multiple drafts"),
    "fixed_k": (1, "fixed draft length k: too long\nwastes, too short under-uses"),
    # systems & serving
    "mutual_wait": (2, "draft and target idle while\nthe other runs (mutual waiting)"),
    "rejection_stall": (2, "after a rejection, drafting is\nback on the critical path"),
    "long_ctx": (2, "long context: KV reads dominate,\nsmall drafters can't keep up"),
    "batch1": (2, "batch 1 only; SD assumed\nuseless at large batch"),
    "sparse_kv_weak": (2, "sparse-KV self-draft: low\nacceptance, extra memory"),
    "static_sparse": (2, "static sparsity fits long\nreasoning traces poorly"),
    "heavy_selfdraft": (2, "drafting with the full target\n(sparse KV) is still heavy"),
    "big_batch": (2, "large batches: verification turns\ncompute-bound, speedup vanishes"),
    "no_slo": (2, "optimises throughput,\nignores latency SLOs"),
}

AXES = [
    ("draft source", [("lm", "separate small LM"), ("self", "target's own layers / weights"),
                      ("heads", "parallel heads on target"), ("arhead", "AR head on target features"),
                      ("ctx", "lookup: prompt / references"), ("store", "lookup: datastore / history"),
                      ("jacobi", "Jacobi n-gram pool"), ("par", "parallel / diffusion drafter")]),
    ("drafter training", [("free", "training-free"), ("trained", "trained on fixed data"),
                          ("distill", "distilled from target"), ("owndraft", "trained on own drafts"),
                          ("pretrain", "built into target training")]),
    ("draft shape", [("chain", "single chain"), ("tree", "static tree"), ("dyntree", "dynamic / optimised tree"),
                     ("multidraft", "multiple drafts"), ("adapt", "adaptive length")]),
    ("verification", [("greedy", "greedy match only"), ("token", "token-wise rejection"),
                      ("coupling", "multi-draft coupling"), ("block", "block / sequence-level"),
                      ("xvocab", "cross-vocabulary")]),
    ("memory", [("sparsekv", "sparse / streaming KV"), ("quant", "quantised weights / KV"),
                ("constmem", "constant-memory drafter"), ("offload", "offloaded target")]),
    ("execution", [("hier", "hierarchy of drafters"), ("overlap", "overlap draft & verify"),
                   ("multidev", "drafter on other devices"), ("load", "load-aware speculation"),
                   ("slo", "latency-SLO-aware")]),
]


def paper(key, name, venue, date, skill, parents=(), new="", uses="", ours=False):
    """ours=True: run or simulated in this repo."""
    return {"key": key, "name": name, "venue": venue, "date": date, "skill": skill, "parents": list(parents),
            "new": new.split(), "uses": uses.split(), "ours": ours}


PAPERS = [
    paper("bpd", "Blockwise parallel", "NeurIPS '18", "2018-11", "k parallel heads, greedy check",
          new="heads trained greedy", uses="chain"),
    paper("sd", "Speculative decoding", "ICML '23 · SpS arXiv '23", "2022-11",
          "draft, then verify by\nrejection sampling", [("bpd", "bpd_limits")],
          new="lm chain token", uses="free", ours=True),

    # ---- what drafts
    paper("dv", "Draft & Verify", "ACL '24", "2023-09", "skip the target's own layers",
          [("sd", "second_model")], new="self free", uses="chain token", ours=True),
    paper("swift", "SWIFT", "ICLR '25", "2024-10", "pick skipped layers on the fly",
          [("dv", "skip_search")], new="self free", uses="chain token"),
    paper("layerskip", "LayerSkip", "ACL '24", "2024-04", "layer dropout + early-exit loss",
          [("dv", "weak_exit")], new="pretrain", uses="self chain token"),
    paper("kangaroo", "Kangaroo", "NeurIPS '24", "2024-04", "shallow exit + adapter; 2nd exit",
          [("dv", "weak_exit")], new="self trained adapt", uses="token"),
    paper("cassandra", "Cassandra", "ISCA '26", "2026-05", "pruned, low-bit self-draft (HW)",
          [("dv", "coarse_skip"), ("swift", "coarse_skip"), ("magicdec", "coarse_skip")],
          new="quant", uses="self free chain token"),
    paper("medusa", "Medusa", "ICML '24", "2024-01", "extra decoding heads + tree",
          [("sd", "second_model")], new="heads trained", uses="tree token", ours=True),
    paper("hydra", "Hydra", "COLM '24", "2024-02", "sequentially dependent heads",
          [("medusa", "blind_heads")], new="arhead", uses="trained distill tree token"),
    paper("eagle", "EAGLE", "ICML '24", "2024-01", "autoregress on top-layer features",
          [("medusa", "blind_heads")], new="arhead", uses="trained tree token"),
    paper("eagle2", "EAGLE-2", "EMNLP '24", "2024-06", "confidence-driven dynamic tree",
          [("eagle", "static_tree"), ("medusa", "static_tree")], new="dyntree", uses="arhead trained token"),
    paper("eagle3", "EAGLE-3", "NeurIPS '25", "2025-03", "fused multi-layer features;\ntrain on own drafts",
          [("eagle2", "feat_regress"), ("eagle2", "exposure")], new="arhead owndraft",
          uses="dyntree token", ours=True),
    paper("peagle", "P-EAGLE", "arXiv '26", "2026-02", "mask tokens: k drafts in 1 pass",
          [("eagle3", "serial_draft")], new="par", uses="arhead owndraft token"),
    paper("hass", "HASS", "ICLR '25", "2024-08", "multi-step context alignment",
          [("eagle2", "exposure")], new="owndraft distill", uses="arhead dyntree token"),
    paper("griffin", "GRIFFIN", "NeurIPS '25", "2025-02", "token-aligned training + head",
          [("eagle2", "exposure"), ("hass", "exposure")], new="arhead owndraft", uses="dyntree token"),
    paper("redrafter", "ReDrafter", "arXiv '24", "2024-03", "RNN draft head + beam search",
          [("medusa", "blind_heads"), ("medusa", "static_tree")], new="arhead distill dyntree", uses="token"),
    paper("mtp", "MTP", "ICML '24 · DSv3", "2024-04", "multi-token pretraining heads",
          [("medusa", "bolted_heads")], new="heads pretrain", uses="chain token", ours=True),
    paper("hetero", "Hetero-vocab SD", "ICML '25", "2025-01", "string-level / shared-vocab verify",
          [("sd", "vocab")], new="xvocab", uses="lm free chain"),
    paper("llma", "LLMA", "arXiv '23", "2023-04", "copy spans from reference text",
          [("sd", "copy")], new="ctx free", uses="chain greedy"),
    paper("rest", "REST", "NAACL '24", "2023-11", "retrieve from a datastore",
          [("llma", "no_reference"), ("sd", "second_model")], new="store free", uses="tree token"),
    paper("suffix", "SuffixDecoding", "NeurIPS '25", "2024-11", "suffix trees over past outputs",
          [("rest", "static_store"), ("pld", "static_store"), ("sd", "fixed_k")],
          new="store dyntree adapt", uses="ctx free token", ours=True),
    paper("sam", "SAM Decoding", "ACL '25", "2024-11", "suffix automaton, corpus + output",
          [("rest", "static_store"), ("pld", "static_store")], new="store", uses="ctx free chain token",
          ours=True),
    paper("logitspec", "LogitSpec", "ACL Findings '26", "2025-07", "next-next-token speculation",
          [("rest", "no_match"), ("pld", "no_match"), ("lookahead", "no_match")], new="store",
          uses="ctx free tree token"),
    paper("lookahead", "Lookahead", "ICML '24", "2024-02", "Jacobi iterations → n-gram pool",
          [("llma", "no_reference"), ("sd", "second_model")], new="jacobi free", uses="multidraft token"),
    paper("pld", "Prompt lookup", "code '23", "2023-11", "n-gram lookup in the prompt",
          [("sd", "copy")], new="ctx free", uses="chain token", ours=True),
    paper("distillspec", "DistillSpec", "ICLR '24", "2023-10", "on-policy distillation",
          [("sd", "misaligned")], new="distill", uses="lm chain token"),
    paper("osd", "Online SD", "ICML '24", "2023-10", "distil on live queries",
          [("distillspec", "offline_kd"), ("sd", "misaligned")], new="distill", uses="lm chain token"),
    paper("staged", "Staged SD", "ES-FoMo '23", "2023-08", "speculate the drafter; tree batch",
          [("sd", "serial_draft")], new="hier tree", uses="lm token"),
    paper("cascade", "Cascade SD", "NeurIPS '24", "2023-12", "vertical + horizontal cascades",
          [("staged", "same_drafter"), ("sd", "serial_draft")], new="hier", uses="lm chain token"),
    paper("parallelspec", "ParallelSpec", "arXiv '24", "2024-10", "mask tokens: k drafts in 1 pass",
          [("sd", "serial_draft"), ("eagle", "serial_draft")], new="par", uses="distill token"),
    paper("pard", "PARD", "ICLR '26", "2025-04", "one parallel drafter per family",
          [("parallelspec", "per_target")], new="par trained", uses="chain token"),
    paper("specdiff", "SpecDiff", "NAACL '25", "2024-08", "diffusion LM drafts a block",
          [("sd", "serial_draft")], new="par", uses="chain token"),
    paper("dflash", "DFlash", "ICML '26", "2026-02", "block diffusion on target features",
          [("specdiff", "diffusion_blind"), ("pard", "diffusion_blind"), ("eagle3", "serial_draft")],
          new="par", uses="distill chain token", ours=True),
    paper("dspark", "DSpark", "arXiv '26", "2026-07", "correction head; confidence-\nscheduled length",
          [("dflash", "block_conflict")], new="par adapt", uses="chain token"),

    # ---- draft shape & verification
    paper("specinfer", "SpecInfer", "ASPLOS '24", "2023-05", "token tree + tree attention",
          [("sd", "single_chain")], new="tree multidraft coupling", uses="lm distill"),
    paper("sequoia", "Sequoia", "NeurIPS '24", "2024-02", "DP-optimal, hardware-aware tree",
          [("specinfer", "static_tree")], new="dyntree coupling", uses="lm offload"),
    paper("specexec", "SpecExec", "NeurIPS '24", "2024-06", "huge trees for offloaded targets",
          [("sequoia", "small_trees"), ("specinfer", "small_trees")], new="dyntree offload", uses="lm token"),
    paper("spectr", "SpecTr", "NeurIPS '23", "2023-10", "k drafts via optimal transport",
          [("sd", "single_chain")], new="multidraft coupling", uses="lm"),
    paper("mdopt", "Multi-draft opt.", "ICLR '25", "2024-10", "optimal multi-draft acceptance",
          [("spectr", "ot_approx")], new="coupling", uses="lm multidraft"),
    paper("blockv", "Block verif.", "ICLR '25", "2024-03", "verify the block jointly",
          [("sd", "tokenwise")], new="block", uses="lm chain"),
    paper("traversal", "Traversal verif.", "NeurIPS '25", "2025-05", "leaf-to-root sequence verify",
          [("blockv", "chain_only"), ("specinfer", "tokenwise")], new="block", uses="lm tree"),
    paper("hsd", "HSD", "ICLR '26", "2026-01", "hierarchical branch resampling",
          [("blockv", "chain_only")], new="block", uses="arhead dyntree"),
    paper("specdecpp", "SpecDec++", "COLM '25", "2024-05", "learned stop-drafting head",
          [("sd", "fixed_k")], new="adapt", uses="lm trained chain token"),

    # ---- systems & serving
    paper("pearl", "PEARL", "ICLR '25", "2024-08", "pre-/post-verify overlap",
          [("sd", "mutual_wait"), ("sd", "fixed_k")], new="overlap adapt", uses="lm chain token"),
    paper("amusd", "AMUSD", "ISCAS '25", "2024-10", "draft & verify on separate GPUs",
          [("sd", "mutual_wait")], new="multidev", uses="lm chain token overlap"),
    paper("ssd", "SSD", "ICLR '26", "2026-03", "pre-speculate likely outcomes",
          [("amusd", "rejection_stall"), ("pearl", "rejection_stall")], new="overlap",
          uses="lm multidev token"),
    paper("dsi", "DSI", "ICLR '25", "2024-05", "speculation parallelism (multi-GPU)",
          [("sd", "mutual_wait")], new="multidev", uses="lm overlap token"),
    paper("triforce", "TriForce", "COLM '24", "2024-04", "sparse-KV self-draft, hierarchy",
          [("sd", "long_ctx")], new="self sparsekv", uses="lm hier chain token"),
    paper("magicdec", "MagicDec", "ICLR '25", "2024-08", "sparse-KV draft at large batch",
          [("triforce", "batch1"), ("smartspec", "big_batch")], new="load", uses="self sparsekv token"),
    paper("quantspec", "QuantSpec", "ICML '25", "2025-02", "hierarchical quantised-KV draft",
          [("magicdec", "sparse_kv_weak"), ("triforce", "sparse_kv_weak")], new="quant",
          uses="self free token"),
    paper("sparsespec", "SparseSpec", "MLSys '26", "2025-12", "dynamic sparse-attention draft",
          [("magicdec", "static_sparse")], new="sparsekv", uses="self free token"),
    paper("longspec", "LongSpec", "ACL '26", "2025-02", "constant-memory long-ctx drafter",
          [("triforce", "heavy_selfdraft"), ("magicdec", "heavy_selfdraft")], new="constmem",
          uses="arhead trained tree token"),
    paper("specextend", "SpecExtend", "ACL Findings '26", "2025-05", "target-guided draft KV retrieval",
          [("triforce", "heavy_selfdraft"), ("magicdec", "heavy_selfdraft")], new="sparsekv",
          uses="lm free tree token"),
    paper("smartspec", "SmartSpec", "arXiv '24", "2024-06", "goodput-driven draft length",
          [("sd", "big_batch"), ("sd", "fixed_k")], new="load", uses="lm ctx adapt token"),
    paper("adaspec", "AdaSpec", "SoCC '25", "2025-03", "SLO-aware adaptive draft length",
          [("smartspec", "no_slo")], new="slo", uses="load adapt token"),
]

ROOT, PRECURSOR = "sd", "bpd"
BY_KEY = {p["key"]: p for p in PAPERS}
TRAITS = {t: (axis, label) for axis, vals in AXES for t, label in vals}

assert len(BY_KEY) == len(PAPERS)
for p in PAPERS:
    assert all(par in BY_KEY and prob in PROBLEMS for par, prob in p["parents"]), p["key"]
    assert all(t in TRAITS for t in p["new"] + p["uses"]), p["key"]
