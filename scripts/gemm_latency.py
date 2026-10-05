"""E0.1 micro-benchmark: BF16 linear latency vs number of tokens M on the current GPU.

Uses Qwen3-8B layer shapes (hidden 4096, qkv 6144, mlp 2*12288 up/gate, 12288 down).
Compares torch F.linear (hipBLASLt/rocBLAS) against vLLM's ROCm GEMM dispatch.
"""
import torch
import torch.nn.functional as F

SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (24576, 4096), "down": (4096, 12288)}
MS = [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 64, 128]

try:
    from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
    vllm_gemm = dispatch_unquantized_gemm()
except Exception as e:  # noqa: BLE001
    print("vLLM gemm dispatch unavailable:", e)
    vllm_gemm = None


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # us


print(f"device: {torch.cuda.get_device_properties(0).gcnArchName}")
print(f"{'M':>4} | {'torch us/layer':>15} {'eff TB/s':>9} | {'vllm us/layer':>14} {'eff TB/s':>9}")
for m in MS:
    t_torch = t_vllm = 0.0
    nbytes = 0
    for n, k in SHAPES.values():
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        nbytes += w.numel() * 2
        t_torch += bench(lambda: F.linear(x, w))
        if vllm_gemm is not None:
            t_vllm += bench(lambda: vllm_gemm(None, x, w, None))
    row = f"{m:>4} | {t_torch:>15.1f} {nbytes / t_torch / 1e6:>9.2f} |"
    if vllm_gemm is not None:
        row += f" {t_vllm:>14.1f} {nbytes / t_vllm / 1e6:>9.2f}"
    print(row, flush=True)
