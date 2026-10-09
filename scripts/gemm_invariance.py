"""Is a GEMM row's result independent of how many rows are computed with it (M-invariance)? And what does
an M-invariant choice cost? Speculative verification runs every linear layer with M = draft tokens + 1
rows per sequence, plain decoding with M = 1; if row i's bits depend on M, outputs can diverge at
near-tied tokens even though the method is "lossless".

For each path and M: share of rows bit-identical to the same row computed alone (M = 1), the max abs
difference, and latency summed over one Gemma 4 31B layer's linears (TP=2 shards). "pad16" always runs
the M = 16 shape and keeps the first M rows, so every row sees one fixed kernel and reduction order.

  GPUS=0 docker/run.sh python3 scripts/gemm_invariance.py
"""
import json

import torch
import torch.nn.functional as F

SHAPES = {"qkv": (8192, 5376), "o": (5376, 4096), "gate_up": (21504, 5376), "down": (5376, 10752)}  # (out, in)
MS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64]
PAD = 16

try:
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    _vllm = dispatch_unquantized_gemm()
    vllm_gemm = lambda x, w: _vllm(None, x, w, None)  # noqa: E731
except Exception as e:  # noqa: BLE001
    print("vLLM gemm dispatch unavailable:", e)
    vllm_gemm = None


def bench(fn, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


def main():
    torch.manual_seed(0)
    paths = {"torch": lambda x, w: F.linear(x, w)}
    if vllm_gemm is not None:
        paths["vllm"] = vllm_gemm
    paths["vllm_pad16" if vllm_gemm else "torch_pad16"] = None  # filled below
    base = vllm_gemm or paths["torch"]
    pad_name = "vllm_pad16" if vllm_gemm else "torch_pad16"

    W = {s: torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for s, (n, k) in SHAPES.items()}
    X = {s: torch.randn(max(MS), k, device="cuda", dtype=torch.bfloat16) for s, (n, k) in SHAPES.items()}

    rows = []
    for name in paths:
        for m in MS:
            same = total = 0
            maxdiff = 0.0
            t = 0.0
            for s in SHAPES:
                w, x = W[s], X[s]
                if name == pad_name:
                    if m > PAD:
                        continue
                    fn = lambda: base(x[:PAD], w)  # noqa: E731
                    y = fn()[:m]
                    ref = torch.stack([base(x[:PAD].roll(-i, 0), w)[0] for i in range(m)])  # row i placed first
                else:
                    gemm = paths[name]
                    fn = lambda: gemm(x[:m], w)  # noqa: E731
                    y = fn()
                    ref = torch.cat([gemm(x[i:i + 1], w) for i in range(m)])
                same += sum(torch.equal(y[i], ref[i]) for i in range(m))
                total += m
                maxdiff = max(maxdiff, (y.float() - ref.float()).abs().max().item())
                t += bench(fn)
            if total:
                rows.append({"path": name, "M": m, "rows_bit_identical": same / total, "max_abs_diff": maxdiff,
                             "us_per_layer": round(t, 1)})
                r = rows[-1]
                print(f"{name:>11} M={m:3d}  bit-identical rows {r['rows_bit_identical']:6.1%}  "
                      f"max|diff| {maxdiff:.2e}  {r['us_per_layer']:8.1f} us/layer", flush=True)
    json.dump({"device": torch.cuda.get_device_properties(0).gcnArchName, "rows": rows},
              open("/workspace/results/hybrid/gemm_invariance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
