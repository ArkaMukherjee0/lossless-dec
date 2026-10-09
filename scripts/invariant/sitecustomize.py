"""Token-count-invariance knobs for the exactness ablation. Loaded through PYTHONPATH in every vLLM process.

Plain decoding runs every op with 1 token per sequence; speculative verification with k+1. On ROCm the
GEMM dispatch picks a different kernel family by token count, and Triton attention uses a split-KV "3D"
kernel only when every query has length 1, so a token's logits can differ in the last bits between the
two paths. These switches pin one path for both:
  INV_GEMM_PAD=16    linears with <= 16 rows are padded to 16 rows (one kernel, one reduction order)
  INV_GEMM_TORCH=32  linears with <= 32 rows go to torch F.linear (hipBLASLt), which gfx90a computes
                     row-identically for 1..32 rows (scripts/gemm_invariance.py), instead of vLLM's
                     ROCm dispatch, which switches kernels at 3 and 5+ rows
  INV_ATTN_2D=1      Triton attention never takes the split-KV 3D path (decode uses the verify kernel)

Run with --enforce-eager: torch.compile may trace only one side of the row-count branch.

  docker/run.sh env PYTHONPATH=/workspace/scripts/invariant INV_GEMM_PAD=16 INV_ATTN_2D=1 \
      python3 scripts/bench_vllm.py ... --enforce-eager
"""
import importlib.abc
import importlib.util
import os
import sys

PAD = int(os.environ.get("INV_GEMM_PAD", "0"))
TORCH_MAX = int(os.environ.get("INV_GEMM_TORCH", "0"))
ATTN_2D = os.environ.get("INV_ATTN_2D") == "1"


def _patch_gemm(mod):
    import torch.nn.functional as F
    orig = mod.rocm_unquantized_gemm

    def gemm(layer, x, weight, bias=None):
        n = x.numel() // x.size(-1)
        if n <= TORCH_MAX:
            return F.linear(x, weight, bias)
        if n >= PAD:
            return orig(layer, x, weight, bias)
        flat = x.reshape(n, x.size(-1))
        out = orig(layer, F.pad(flat, (0, 0, 0, PAD - n)), weight, bias)[:n]
        return out.reshape(*x.shape[:-1], out.size(-1))
    mod.rocm_unquantized_gemm = gemm  # dispatch_unquantized_gemm() reads the module global at call time


def _patch_attn(mod):
    cls = mod.TritonAttentionMetadataBuilder
    orig = cls.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        self.seq_threshold_3D = 0  # num_seqs > 0 always, so unified_attention never picks 3D
    cls.__init__ = init


TARGETS = {}
if PAD or TORCH_MAX:
    TARGETS["vllm.model_executor.layers.utils"] = _patch_gemm
if ATTN_2D:
    TARGETS["vllm.v1.attention.backends.triton_attn"] = _patch_attn


class _PatchAfterImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        sys.meta_path.remove(self)
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


if TARGETS:
    sys.meta_path.insert(0, _PatchAfterImport())
