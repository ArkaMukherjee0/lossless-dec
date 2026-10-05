"""Generate jobs for the 40 h plan (PLAN_40H.md) and append new ones to jobs/queue.tsv.

Usage: python3 scripts/make_queue.py <phase> [<phase> ...]   (phases: a0 a b c_moe c_dense d e1 e2)
Jobs already present in the queue (by id) are not re-added.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUEUE = f"{ROOT}/jobs/queue.tsv"

MODELS = {
    "q27": {"name": "Qwen/Qwen3.8-27B", "dense": True,
            "mtp": {"method": "qwen3_5_mtp"}, "mtp_k": 3,
            "dflash": "z-lab/Qwen3.8-27B-DFlash2", "dflash_ks": [3, 5, 7], "ngram": False},
    "g31": {"name": "google/gemma-4-31B-it", "dense": True,
            "mtp": {"method": "gemma4_mtp", "model": "google/gemma-4-31B-it-assistant"}, "mtp_k": 4,
            "dflash": "z-lab/gemma-4-31B-it-DFlash", "dflash_ks": [3, 7, 11, 15], "ngram": True},
    "q35": {"name": "Qwen/Qwen3.6-35B-A3B", "dense": False,
            "mtp": {"method": "qwen3_5_mtp"}, "mtp_k": 3,
            "dflash": "z-lab/Qwen3.6-35B-A3B-DFlash", "dflash_ks": [3, 7, 11, 15], "ngram": False},
    "g26": {"name": "google/gemma-4-26B-A4B-it", "dense": False,
            "mtp": {"method": "gemma4_mtp", "model": "google/gemma-4-26B-A4B-it-assistant"}, "mtp_k": 4,
            "dflash": "z-lab/gemma-4-26B-A4B-it-DFlash", "dflash_ks": [3, 7, 11, 15], "ngram": True},
}
ORDER = ["q27", "g31", "q35", "g26"]  # interleave dense and MoE so waves mix


def mtp(m, k):
    return dict(m["mtp"], num_speculative_tokens=k)


def dflash(m, k):
    return {"method": "dflash", "model": m["dflash"], "num_speculative_tokens": k}


def bench(m, exp, tag, spec=None, per_cat=80, temperature=0.0, gaps=False):
    cmd = (f"docker/run.sh python3 scripts/bench_vllm.py --model {m['name']} --tp 2 --exp {exp} "
           f"--tag {tag} --per-cat {per_cat} --max-tokens 1024 --temperature {temperature}")
    if gaps:
        cmd += " --save-gaps"
    if spec:
        cmd += f" --spec '{json.dumps(spec)}'"
    return cmd


def jobs_a0():
    m = MODELS["q27"]
    return [("A0-alone", "excl", 1, bench(m, "a0_alone", "ar", per_cat=20)),
            ("A0-shared", "1", 1, bench(m, "a0_shared", "ar", per_cat=20))]


def jobs_a():
    per_model = {}
    for key in ORDER:
        m = MODELS[key]
        t = 4 if m["dense"] else 2
        js = [(f"A-{key}-ar", t, bench(m, "a", "ar", gaps=True)),
              (f"A-{key}-arrepeat", t, bench(m, "a", "arrepeat", gaps=True)),
              (f"A-{key}-mtp{m['mtp_k']}", t, bench(m, "a", f"mtp{m['mtp_k']}", mtp(m, m["mtp_k"]))),
              (f"A-{key}-dflash7", t, bench(m, "a", "dflash7", dflash(m, 7)))]
        if 15 in m["dflash_ks"]:
            js.append((f"A-{key}-dflash15", t, bench(m, "a", "dflash15", dflash(m, 15))))
        js.append((f"A-{key}-suffix", t, bench(m, "a", "suffix", {"method": "suffix"})))
        if m["ngram"]:
            js.append((f"A-{key}-ngram4", t, bench(m, "a", "ngram4",
                                                  {"method": "ngram", "num_speculative_tokens": 4,
                                                   "prompt_lookup_max": 4})))
        per_model[key] = js
    out = []  # round-robin across models
    while any(per_model.values()):
        for key in ORDER:
            if per_model[key]:
                jid, t, cmd = per_model[key].pop(0)
                out.append((jid, "1", t, cmd))
    return out


def jobs_b():
    out = []
    for key in ORDER:
        m = MODELS[key]
        for tag, spec in [("ar", None), (f"mtp{m['mtp_k']}", mtp(m, m["mtp_k"])), ("dflash7", dflash(m, 7))]:
            out.append((f"B-{key}-{tag}", "1", 3, bench(m, "b", tag, spec, per_cat=40, temperature=1.0)))
    return out


def jobs_c(keys):
    out = []
    for key in keys:
        m = MODELS[key]
        out.append((f"C-{key}-ar", "1", 2, bench(m, "c", "ar", per_cat=20)))
        for k in [1, 2, 3, 4, 6]:
            out.append((f"C-{key}-mtp{k}", "1", 2, bench(m, "c", f"mtp{k}", mtp(m, k), per_cat=20)))
        for k in m["dflash_ks"]:
            out.append((f"C-{key}-dflash{k}", "1", 2, bench(m, "c", f"dflash{k}", dflash(m, k), per_cat=20)))
    return out


def jobs_d():
    out = []
    for key in ORDER:
        m = MODELS[key]
        for tag, spec in [("ar", None), (f"mtp{m['mtp_k']}", mtp(m, m["mtp_k"])), ("dflash7", dflash(m, 7))]:
            s = f"'{json.dumps(spec)}'" if spec else "''"
            out.append((f"D-{key}-{tag}", "1", 3,
                        f"docker/run.sh bash scripts/concurrency_bench.sh {m['name']} {tag} {s} 4,16,64,128"))
    return out


def trace(m, exp, tag, spec, extra):
    cmd = f"docker/run.sh python3 scripts/trace_vllm.py --model {m['name']} --exp {exp} --tag {tag} {extra}"
    return cmd + (f" --spec '{json.dumps(spec)}'" if spec else "")


def jobs_e1():
    # thinking on, sampled (T=0.6 as recommended for reasoning; greedy thinking tends to loop)
    extra = "--data /workspace/data/reasoning30.jsonl --thinking --max-tokens 12288 --temperature 0.6"
    out = []
    for key in ORDER:
        m = MODELS[key]
        for tag, spec in [("ar", None), (f"mtp{m['mtp_k']}", mtp(m, m["mtp_k"])), ("dflash7", dflash(m, 7))]:
            out.append((f"E1-{key}-{tag}", "1", 6 if m["dense"] else 3, trace(m, "e1", tag, spec, extra)))
    return out


def jobs_e2():
    extra = "--longbench --ctx-len 8192,32768,65536 --n 20 --max-tokens 256"
    out = []
    for key in ["q27", "g31"]:
        m = MODELS[key]
        for tag, spec in [("ar", None), (f"mtp{m['mtp_k']}", mtp(m, m["mtp_k"])), ("dflash7", dflash(m, 7)),
                          ("suffix", {"method": "suffix"})]:
            out.append((f"E2-{key}-{tag}", "1", 3, trace(m, "e2", tag, spec, extra)))
    return out


PHASES = {"a0": jobs_a0, "a": jobs_a, "b": jobs_b,
          "d": jobs_d, "e1": jobs_e1, "e2": jobs_e2, "c_moe": lambda: jobs_c(["q35", "g26"]), "c_dense": lambda: jobs_c(["q27", "g31"])}

if __name__ == "__main__":
    os.makedirs(os.path.dirname(QUEUE), exist_ok=True)
    existing = set()
    if os.path.exists(QUEUE):
        existing = {line.split("\t", 1)[0] for line in open(QUEUE) if line.strip() and not line.startswith("#")}
    added = 0
    with open(QUEUE, "a") as f:
        for phase in sys.argv[1:]:
            for jid, slots, timeout_h, cmd in PHASES[phase]():
                if jid not in existing:
                    f.write(f"{jid}\t{slots}\t{timeout_h}\t{cmd}\n")
                    existing.add(jid)
                    added += 1
    print(f"added {added} jobs; queue now has {len(existing)}")
