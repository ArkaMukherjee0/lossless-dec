"""Read a scripts/draftlog run back: one dict per verified drafter step, aligned to the output tokens."""
import collections
import json

import numpy as np

NOT_FOUND = 10 ** 6


def iter_steps(log_path, bench_path):
    """Yield, per drafter step: request index, its bench record, the output index of each drafted token,
    the rank of the true token in the drafter's top-M at each position (NOT_FOUND if absent or past the
    end of the output), and the drafter's top-M ids / logprobs."""
    records = json.load(open(bench_path))["records"]
    segments, prompt_len, req = collections.defaultdict(list), {}, None
    for line in open(log_path):
        s = json.loads(line)
        if "req" in s:  # LLM.generate markers; req -1 is the warm-up
            req = s["req"]
        elif "req_done" in s:
            prompt_len[s["req_done"]] = s["prompt_len"][0]
        elif req is not None:
            segments[req].append(s)
    for req, recs in sorted(segments.items()):
        if not 0 <= req < len(records) or req not in prompt_len:
            continue
        rec, L = records[req], prompt_len[req]
        out = rec["token_ids"]
        # Async scheduling can put the previous request's last step after this request's marker, and
        # the drafter runs once more after the final token: keep this request's own, verified steps.
        start = next((i for i, r in enumerate(recs) if r["pos"][0] == L + 1), None)  # drafts after token 0
        if start is None:
            continue
        for r in recs[start:]:
            if r["pos"][0] - L >= len(out):
                continue
            # bs = 1 runs: some speculators pad the batch with a copy of the request; keep the first block
            n = next((j for j in range(1, len(r["pos"])) if r["pos"][j] <= r["pos"][j - 1]), len(r["pos"]))
            idx = [p - L for p in r["pos"][:n]]  # output index of the token drafted at each position
            ids = np.asarray(r["top_ids"][:n])
            ranks = []
            for j, k in enumerate(idx):
                hit = np.nonzero(ids[j] == out[k])[0] if k < len(out) else []
                ranks.append(int(hit[0]) if len(hit) else NOT_FOUND)
            yield {"req": req, "record": rec, "idx": idx, "ranks": ranks, "top_ids": ids,
                   "top_lp": np.asarray(r["top_lp"][:n])}
