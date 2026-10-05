# Project image: official vLLM ROCm release (built for gfx90a/MI250) + experiment tooling.
# Bump VLLM_TAG to upgrade vLLM; check `strings _C.abi3.so | grep gfx90a` after each bump.
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai-rocm:${VLLM_TAG}

RUN pip install --no-cache-dir pandas matplotlib seaborn jsonlines arctic-inference==0.3.0

# gfx90a workaround: wvSplitK skinny GEMM is 2.4x/5.7x slower than hipBLASLt at 3/5 tokens
# (see scripts/gemm_latency.py), which cripples spec-decode verify (1 + 4 draft = 5 tokens)
# and any 3/5-request decode batch. Keep it only where it wins (1, 2, 4 tokens).
RUN f=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/model_executor/layers/utils.py \
 && grep -q "if m > 8 and 0 < n <= 5:" "$f" \
 && sed -i 's/if m > 8 and 0 < n <= 5:/if m > 8 and n in (1, 2, 4):  # lossless-dec gfx90a patch/' "$f"

ENV HF_HOME=/models/hf \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]

# transformers >= 5.1x renamed/removed helpers that vLLM 0.30's pixtral.py imports
# (PixtralRotaryEmbedding, position_ids_in_meshgrid). They are only used by the HF-format Pixtral
# vision tower, not by Mistral-format checkpoints (Mistral Small 4), so fall back to stubs.
COPY patch_pixtral.py /tmp/patch_pixtral.py
RUN python3 /tmp/patch_pixtral.py

# vLLM 0.30 bug: EAGLE drafter for Mistral Small 4 misses is_fused_shared_expert_enabled.
COPY patch_eagle_mistral.py /tmp/patch_eagle_mistral.py
RUN python3 /tmp/patch_eagle_mistral.py

# vLLM 0.30 bug: Medusa setup reads vocab_size from multimodal hf_config instead of text config.
COPY patch_medusa_vocab.py /tmp/patch_medusa_vocab.py
RUN python3 /tmp/patch_medusa_vocab.py
