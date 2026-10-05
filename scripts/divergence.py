"""E10: greedy divergence of spec-decode outputs vs AR (same prompts, same order)."""
import glob
import json
import os
import statistics
import sys

for model_dir in [os.path.normpath(p) for p in sys.argv[1:]]:
    ar = json.load(open(f"{model_dir}/ar_t0.0_c1.json"))["records"]
    for path in sorted(glob.glob(f"{model_dir}/*_t0.0_c1.json")):  # includes ar-repeat runs
        tag = os.path.basename(path).split("_t0.0")[0]
        if tag == "ar":
            continue
        recs = json.load(open(path))["records"]
        firsts = []
        for a, b in zip(ar, recs):
            x, y = a["token_ids"], b["token_ids"]
            d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
            if d is None and len(x) != len(y):
                d = min(len(x), len(y))
            firsts.append(d)
        div = [d for d in firsts if d is not None]
        gaps = [a["gaps"][d] for a, d in zip(ar, firsts) if d is not None and "gaps" in a and d < len(a["gaps"])]
        print(f"{os.path.basename(model_dir):>18} {tag:>10}: diverged {len(div)}/{len(firsts)} prompts; "
              f"median first-divergence token = {statistics.median(div) if div else '-'}; "
              f"<32 tokens: {sum(d < 32 for d in div)}"
              + (f"; AR top-2 gap at divergence: max={max(gaps):.3f}, >0.25: {sum(g > 0.25 for g in gaps)}" if gaps else ""))
