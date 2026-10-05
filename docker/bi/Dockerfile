# Batch-invariant variant: vLLM's VLLM_BATCH_INVARIANT only installs Triton matmul overrides
# under is_cuda(); on ROCm GEMMs stay shape-dependent. Route ROCm through the same override
# path as SM80 (Triton persistent matmul for mm/addmm/matmul/linear).
FROM lossless-dec:vllm0.30
USER root
RUN f=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/model_executor/determinism/batch_invariant.py \
 && grep -q "        if current_platform.is_device_capability_family(80):" "$f" \
 && python3 - "$f" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    if current_platform.is_cuda():\n        if current_platform.is_device_capability_family(80):"
new = "    if current_platform.is_cuda_alike():\n        if current_platform.is_rocm() or current_platform.is_device_capability_family(80):"
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
PY
ENV VLLM_BATCH_INVARIANT=1 VLLM_ROCM_USE_SKINNY_GEMM=0
