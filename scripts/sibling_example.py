"""Worked examples for the proposal write-up: steps where the drafter's chain dies early but a finished
sibling's continuation runs on, decoded to text (writes lit/sibling_examples.json).

  MODELS=/dev/shm/arkamukh GPUS= docker/run.sh python3 scripts/sibling_example.py results/m2/gemma-4-31B-it dflash15
"""
import glob
import json
import random
import sys

from transformers import AutoTokenizer

from draftlog_io import iter_steps
from sibling_predictability import K, accepted, best_match, index

TOKENIZER = "google/gemma-4-31B-it"


def main():
    root, tag = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    runs = {s: (f"{root}/{tag}_s{s}_drafts.jsonl", glob.glob(f"{root}/{tag}_s{s}_t1.0_c1*.json")[0])
            for s in range(8) if glob.glob(f"{root}/{tag}_s{s}_t1.0_c1*.json")}
    outs = {s: [r["token_ids"] for r in json.load(open(runs[s][1]))["records"]] for s in runs}
    cats = [r["category"] for r in json.load(open(runs[0][1]))["records"]]
    idx = {s: [index(o) for o in outs[s]] for s in runs}
    found = []
    for s in (0, 1):
        for st in iter_steps(*runs[s]):
            req, out = st["req"], outs[s][st["req"]]
            a = st["idx"][0] - 1
            ctx, truth = out[:a + 1], out[a + 1:a + 1 + K]
            drafter = [int(t) for t in st["top_ids"][:, 0]]
            per = [(best_match(ctx, [(outs[t][req], idx[t][req], len(outs[t][req]))]), t) for t in runs if t != s]
            (m, prop), sib = max(per, key=lambda x: x[0][0])
            acc_d, acc_s = accepted(drafter, truth), accepted(prop, truth)
            if acc_d <= 1 and acc_s >= 8 and cats[req] in ("mt_bench", "qa", "summarization", "translation"):
                seq = outs[sib][req]  # locate the match in that sibling to show what it had written
                e = next(e for e in range(m - 1, len(seq)) if seq[e - m + 1:e + 1] == ctx[-m:]
                         and seq[e + 1:e + 1 + len(prop)] == prop)
                pieces = lambda ids: [tok.decode([i]) for i in ids]  # noqa: E731
                found.append({"category": cats[req], "seed": s, "request": req, "sibling_seed": sib, "match_len": m,
                              "context_tail": tok.decode(ctx[-40:]), "drafter": tok.decode(drafter),
                              "sibling": tok.decode(prop), "truth": tok.decode(truth),
                              "drafter_accepted": acc_d, "sibling_accepted": acc_s,
                              "context_tokens": pieces(ctx[-14:]), "sibling_before_tokens": pieces(seq[max(0, e - 13):e + 1]),
                              "drafter_tokens": pieces(drafter), "sibling_tokens": pieces(prop),
                              "truth_tokens": pieces(truth)})
    # the four cases quoted in lit/proposal_sibling_drafting.md, located by the text they end with
    wanted = ["represent the **seven", "Police discovered the incident", "stoichiometry:**\n*   **1",
              "descriptions of the person (app"]
    picks = [next(f for f in found if f["context_tail"].endswith(w)) for w in wanted
             if any(f["context_tail"].endswith(w) for f in found)]
    random.Random(1).shuffle(found)
    for f in found:  # fill any that are missing with one example per remaining category
        if len(picks) < 4 and f["category"] not in {p["category"] for p in picks}:
            picks.append(f)
    json.dump({"total_such_steps": len(found), "examples": picks}, open("/workspace/lit/sibling_examples.json", "w"),
              indent=1, ensure_ascii=False)
    for f in picks:
        print(json.dumps(f, ensure_ascii=False, indent=1))
    print(len(found), "steps (seeds 0-1) where the drafter accepted <= 1 and a finished sibling >= 8")


if __name__ == "__main__":
    main()
