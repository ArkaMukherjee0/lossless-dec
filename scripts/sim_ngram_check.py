"""Validity check for offline acceptance simulation.

Replays vLLM's own n-gram proposer against saved greedy AR outputs and compares the
resulting draft/accept counts with what vLLM measured during real spec decoding.
"""
import json
import sys

import numpy as np
from transformers import AutoTokenizer
from vllm.v1.spec_decode.ngram_proposer import _find_longest_matched_ngram_and_propose_tokens

sys.path.insert(0, "/workspace/scripts")
from smoke_spec import PROMPTS  # noqa: E402

TARGET, K, N = "Qwen/Qwen3-8B", 4, 4
tok = AutoTokenizer.from_pretrained(TARGET)
ar = json.load(open("/workspace/results/smoke/Qwen3-8B/ar.json"))["tokens"]

drafts = draft_tokens = accepted = 0
for prompt, out in zip(PROMPTS, ar):
    ids = tok(prompt).input_ids
    t = 0
    while t < len(out) - 1:  # the last token is never verified against a draft
        ctx = np.array(ids + out[:t], dtype=np.int32)
        k = min(K, len(out) - 1 - t)
        d = _find_longest_matched_ngram_and_propose_tokens(ctx, N, N, 1 << 20, k)
        a = 0
        if len(d):
            drafts += 1
            draft_tokens += len(d)
            while a < len(d) and d[a] == out[t + a]:
                a += 1
            accepted += a
        t += a + 1
print(f"simulated (1 pass): drafts={drafts} draft_tokens={draft_tokens} accepted={accepted}")
print("vLLM measured (per pass, from 2-pass totals 226/904/674): drafts=113 draft_tokens=452 accepted=337")
