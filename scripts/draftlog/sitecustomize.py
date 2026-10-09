"""Draft logger for scripts/sim_tree.py. Loaded through PYTHONPATH, so it patches every vLLM process.

Does nothing unless DRAFT_LOG=<jsonl path> is set. Run vLLM with enforce_eager so drafting runs in
Python each step (a CUDA-graph replay would skip the hooks).
  main process   {"req": i} before and {"req_done": i, "prompt_len": [...]} after each LLM.generate call
                 (call 0 is the warm-up: req -1)
  worker rank 0  one record per drafter step: drafted positions, the drafter's picks, and the top-M
                 candidate ids / logprobs at each position (DSpark: from its base logits, before the
                 Markov bias, plus its confidences)

  docker/run.sh env PYTHONPATH=/workspace/scripts/draftlog DRAFT_LOG=/workspace/results/tree/x.jsonl \
      python3 scripts/bench_vllm.py ... --enforce-eager
"""
import importlib.abc
import importlib.util
import json
import os
import sys

PATH = os.environ.get("DRAFT_LOG")
TOP_M = int(os.environ.get("DRAFT_LOG_TOPM", "8"))


def _write(obj):
    with open(PATH, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _rank0():
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


def _topk(logits):
    v, i = logits.float().log_softmax(-1).topk(TOP_M, dim=-1)
    return i.tolist(), [[round(x, 4) for x in row] for row in v.tolist()]


def _patch_llm(mod):
    orig, calls = mod.LLM.generate, [0]

    def generate(self, *a, **kw):
        req = calls[0] - 1
        calls[0] += 1
        _write({"req": req})
        outs = orig(self, *a, **kw)
        # Async scheduling can land a request's last drafter step after the next marker, so the
        # reader also needs each request's prompt length to tell the requests' records apart.
        _write({"req_done": req, "prompt_len": [len(o.prompt_token_ids) for o in outs]})
        return outs
    mod.LLM.generate = generate


def _patch_speculator(mod):
    """DFlash and other draft-model speculators: sample_draft knows the positions, and every sampling
    branch hands the full draft logits to _maybe_predict_acceptance, so log there (no recompute)."""
    cls = mod.DraftModelSpeculator
    orig_sample, orig_predict = cls.sample_draft, cls._maybe_predict_acceptance

    def sample_draft(self, hidden_states, sample_src_positions, *a, **kw):
        self._draftlog_pos = sample_src_positions + 1  # positions of the drafted tokens
        out = orig_sample(self, hidden_states, sample_src_positions, *a, **kw)
        if _rank0() and getattr(self, "_draftlog_rec", None) is not None:
            self._draftlog_rec["draft"] = out.tolist()
            _write(self._draftlog_rec)
        self._draftlog_pos = self._draftlog_rec = None
        return out

    def predict(self, logits, idx_mapping, draft_step):
        # Only calls made from sample_draft; DSpark calls this per position from its own loop.
        if logits is not None and getattr(self, "_draftlog_pos", None) is not None and _rank0():
            ids, lp = _topk(logits)
            self._draftlog_rec = {"pos": self._draftlog_pos.tolist(), "top_ids": ids, "top_lp": lp}
        return orig_predict(self, logits, idx_mapping, draft_step)

    cls.sample_draft, cls._maybe_predict_acceptance = sample_draft, predict


def _patch_dspark(mod):
    """DSpark: base logits are computed inside the sequential Markov loop; recompute them on every
    rank (the logits gather is collective) and log rank 0's top-M in target-vocab ids."""
    cls = mod.DSparkSpeculator if hasattr(mod, "DSparkSpeculator") else next(
        v for k, v in vars(mod).items() if k.endswith("Speculator") and hasattr(v, "_sample_sequential"))

    def wrap(orig):
        def run(self, num_reqs, head_hidden):
            n = num_reqs * self.num_speculative_steps
            base = self.model.compute_draft_logits(head_hidden[self.sample_indices[:n]])
            orig(self, num_reqs, head_hidden)
            if base is not None and _rank0():
                v, i = base.float().log_softmax(-1).topk(TOP_M, dim=-1)
                rec = {"pos": self.sample_pos[:n].tolist(),
                       "top_ids": self.model.map_draft_to_target(i).tolist(),
                       "top_lp": [[round(x, 4) for x in row] for row in v.tolist()],
                       "draft": self.draft_tokens[:num_reqs].flatten().tolist()}
                if getattr(self, "use_confidence_head", False):
                    rec["conf"] = self.draft_token_confidence_probs[:num_reqs].flatten().tolist()
                _write(rec)
        return run
    for name in ("_sample_sequential", "_sample_sequential_topk"):
        if hasattr(cls, name):
            setattr(cls, name, wrap(getattr(cls, name)))


def _patch_dflash2(mod):
    """DFlash2 drafts by walking a pairwise selector over each position's top-k candidates, so the drafted
    token need not be the drafter's top candidate. Log the candidates in unary-logit order with the drafted
    token moved to the front (rank 0 = what was proposed); logprobs are over the k candidates only."""
    cls = mod.DFlash2Speculator
    orig = cls._generate_draft

    def generate(self, num_reqs, *a, **kw):
        stash, model = {}, self.model
        compute = model.compute_candidates

        def capture(h):
            stash["c"] = compute(h)
            return stash["c"]
        model.compute_candidates = capture  # instance attribute shadows the method for this call only
        try:
            orig(self, num_reqs, *a, **kw)
        finally:
            del model.compute_candidates
        if "c" in stash and _rank0():
            n = num_reqs * self.num_speculative_steps
            cand, unary = stash["c"]
            order = unary.float().argsort(dim=-1, descending=True)
            ids = cand.gather(-1, order).tolist()
            lps = unary.float().gather(-1, order).log_softmax(-1).tolist()
            draft = self.draft_tokens[:num_reqs].flatten().tolist()
            for row, lp, d in zip(ids, lps, draft):
                j = row.index(d) if d in row else len(row) - 1
                row.insert(0, row.pop(j) if d in row else d)
                lp.insert(0, lp.pop(j))
            _write({"pos": self.sample_pos[:n].tolist(), "top_ids": ids,
                    "top_lp": [[round(x, 4) for x in r] for r in lps], "draft": draft})
    cls._generate_draft = generate


TARGETS = {
    "vllm.entrypoints.llm": _patch_llm,
    "vllm.v1.worker.gpu.spec_decode.dflash2.speculator": _patch_dflash2,
    "vllm.v1.worker.gpu.spec_decode.speculator": _patch_speculator,
    "vllm.v1.worker.gpu.spec_decode.dspark.speculator": _patch_dspark,
}


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        sys.meta_path.remove(self)  # resolve the real module without finding ourselves again
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            TARGETS[name](module)
        spec.loader.exec_module = exec_and_patch
        return spec


if PATH:
    sys.meta_path.insert(0, _PatchAfterImport())
