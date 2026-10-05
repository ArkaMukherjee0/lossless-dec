"""Medusa step 3: train Medusa-1 heads (frozen backbone + frozen lm_head), export for vLLM,
and report offline greedy acceptance on the Spec-Bench AR trajectories.

Head i (0-based) maps h_t -> token t+2+i through a residual block x + SiLU(Wx) and the shared
lm_head (h_t itself predicts token t+1 via the base lm_head). Checkpoint layout follows vLLM's
Medusa model: blocks.{i}.layers.0.weight + lm_head.weight, config model_type=medusa,
original_lm_head=true.

  GPUS=0,1 docker/run.sh python3 scripts/medusa_train.py --model Qwen/Qwen3.8-27B
"""
import argparse
import glob
import json
import os
import random

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


def load_lm_head(model, device):
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
    keys = [k for k in idx if k.endswith("lm_head.weight")]
    if not keys:  # tied embeddings (e.g. Gemma): lm_head == language-model input embeddings
        keys = [k for k in idx if k.endswith("embed_tokens.weight") and "vision" not in k and "audio" not in k]
    key = keys[0]
    with safe_open(f"{path}/{idx[key]}", "pt") as f:
        return f.get_tensor(key).to(device)


def load_chunks(pattern):
    data = []
    for f in sorted(glob.glob(pattern)):
        data.extend(torch.load(f))
    return data


def batches(data, num_heads, tokens_per_batch, rng):
    """Yield (h [B,H], targets [B, num_heads]) with -100 where t+2+i runs past the sequence."""
    order = list(range(len(data)))
    rng.shuffle(order)
    hs, ts, n = [], [], 0
    for i in order:
        h, tok = data[i]["hidden"], data[i]["tokens"].long()
        L = len(tok)
        tgt = torch.full((L, num_heads), -100, dtype=torch.long)
        for k in range(num_heads):
            if L > k + 1:
                tgt[: L - k - 1, k] = tok[k + 1:]
        hs.append(h); ts.append(tgt); n += L
        if n >= tokens_per_batch:
            yield torch.cat(hs), torch.cat(ts)
            hs, ts, n = [], [], 0
    if hs:
        yield torch.cat(hs), torch.cat(ts)


class Heads(torch.nn.Module):
    def __init__(self, hidden, num_heads):
        super().__init__()
        self.w = torch.nn.ModuleList(torch.nn.Linear(hidden, hidden, bias=False) for _ in range(num_heads))
        for lin in self.w:
            torch.nn.init.zeros_(lin.weight)  # start as identity (Medusa-1 init)

    def forward(self, h):
        return [h + F.silu(lin(h)) for lin in self.w]


@torch.no_grad()
def offline_tau(heads, lm_head, data, device, ks=(1, 2, 3, 4)):
    """Greedy chain acceptance: at position t the base token t+1 is accepted (it is the target's own
    prediction), heads propose t+2.. ; tau = 1 + consecutive head matches (capped at k)."""
    res = {}
    preds = []
    for ex in data:
        h = ex["hidden"].to(device, torch.bfloat16)
        preds.append(torch.stack([(o.float() @ lm_head.float().T).argmax(-1).cpu() for o in heads(h)], 1))
    for k in ks:
        n = steps = 0
        for ex, p in zip(data, preds):
            tok = ex["tokens"].long()
            L = len(tok)
            t = 0
            while t < L:  # h index t predicts tok[t]; heads at t predict tok[t+1+i]
                a = 0
                while a < k and t + 1 + a < L and p[t, a] == tok[t + 1 + a]:
                    a += 1
                steps += 1
                t += a + 1
            n += L
        res[k] = n / steps
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tokens-per-batch", type=int, default=4096)
    ap.add_argument("--decay", type=float, default=0.8, help="loss weight decay per head (Medusa)")
    args = ap.parse_args()

    name = args.model.split("/")[-1]
    root = f"/models/medusa_data/{name}"
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    rng = random.Random(0)
    lm_head = load_lm_head(args.model, dev).to(torch.bfloat16)
    vocab, hidden = lm_head.shape
    train = load_chunks(f"{root}/hidden/shard*.pt")
    evals = load_chunks(f"{root}/specbench/shard*.pt")
    print(f"train {len(train)} seqs / {sum(len(x['tokens']) for x in train)} tokens; eval {len(evals)} seqs; "
          f"hidden={hidden} vocab={vocab}", flush=True)

    heads = Heads(hidden, args.num_heads).to(dev, torch.bfloat16)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.epochs * (sum(len(x["tokens"]) for x in train) // args.tokens_per_batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    print("before training (identity heads):", offline_tau(heads, lm_head, evals, dev), flush=True)
    step = 0
    for ep in range(args.epochs):
        for h, tgt in batches(train, args.num_heads, args.tokens_per_batch, rng):
            h, tgt = h.to(dev, torch.bfloat16), tgt.to(dev)
            loss, accs = 0.0, []
            for i, o in enumerate(heads(h)):
                logits = (o @ lm_head.T).float()
                loss = loss + (args.decay ** i) * F.cross_entropy(logits, tgt[:, i], ignore_index=-100)
                m = tgt[:, i] != -100
                accs.append((logits.argmax(-1)[m] == tgt[m, i]).float().mean().item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.3f} top1 " +
                      " ".join(f"h{i}={a:.3f}" for i, a in enumerate(accs)), flush=True)
        print(f"after epoch {ep}: offline tau", offline_tau(heads, lm_head, evals, dev), flush=True)

    out = f"/models/medusa/{name}-medusa{args.num_heads}"
    os.makedirs(out, exist_ok=True)
    state = {f"blocks.{i}.layers.0.weight": lin.weight.detach().to(torch.bfloat16).cpu().contiguous()
             for i, lin in enumerate(heads.w)}
    state["lm_head.weight"] = lm_head.cpu().contiguous()
    save_file(state, f"{out}/model.safetensors")
    json.dump({"model_type": "medusa", "architectures": ["MedusaModel"], "hidden_size": hidden,
               "vocab_size": vocab, "num_heads": args.num_heads, "num_hidden_layers": 1,
               "original_lm_head": True, "truncated_vocab_size": vocab, "torch_dtype": "bfloat16"},
              open(f"{out}/config.json", "w"), indent=1)
    json.dump({"offline_tau": offline_tau(heads, lm_head, evals, dev), "args": vars(args)},
              open(f"{out}/train_result.json", "w"), indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
