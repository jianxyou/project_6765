#!/usr/bin/env python3
"""Cross-dataset transfer: train learned projection on one dataset, evaluate on all 4."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from docpruner_replicate import (
    train_learned_projection, learned_projection_compress,
    pad_3d, maxsim_retrieval, evaluate_ndcg5, set_seed,
    VIDORE_V2_DATASETS,
)

set_seed(0)
cache_dir = Path("outputs_replicate/cache")

CONFIGS = [
    {"k1": 1.0, "k2": 0, "top_k_ratio": 0.25, "label": "~80%"},
    {"k1": 1.0, "k2": 0, "top_k_ratio": 0.1, "label": "~84%"},
]

def load_ds(short):
    pfx = f"{short}_vidore_colqwen2.5-v0.2"
    emb = torch.load(cache_dir / f"{pfx}_emb.pt", weights_only=False)
    attn = torch.load(cache_dir / f"{pfx}_attn.pt", weights_only=False)
    mask = torch.load(cache_dir / f"{pfx}_imgpad_mask.pt", weights_only=False)
    qdata = torch.load(cache_dir / f"{pfx}_qemb.pt", weights_only=False)
    return emb, attn, mask, qdata

shorts = [ds.split("/")[-1] for ds in VIDORE_V2_DATASETS]

for train_short in shorts:
    print(f"\n{'='*70}")
    print(f"TRAIN on: {train_short}")
    print(f"{'='*70}")

    t_emb, t_attn, t_mask, t_qdata = load_ds(train_short)

    for cfg in CONFIGS:
        k1, k2, tkr = cfg["k1"], cfg["k2"], cfg["top_k_ratio"]
        print(f"\n  Config: k1={k1}, k2={k2}, top_k={tkr} ({cfg['label']})")

        proj = train_learned_projection(
            t_emb, t_attn, t_mask,
            t_qdata["q_emb"], t_qdata["qrels"], t_qdata["qids"], t_qdata["cids"],
            k1=k1, k2=k2, top_k_ratio=tkr, epochs=30,
        )

        for eval_short in shorts:
            e_emb, e_attn, e_mask, e_qdata = load_ds(eval_short)

            pruned = []
            for i in range(len(e_emb)):
                r = learned_projection_compress(
                    e_emb[i], e_attn[i], proj,
                    k1=k1, k2=k2, top_k_ratio=tkr, imgpad_mask=e_mask[i],
                )
                pruned.append(r.vectors)

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            run = maxsim_retrieval(
                e_qdata["q_emb"], pruned, e_qdata["qids"], e_qdata["cids"], device=device,
            )
            ndcg = evaluate_ndcg5(run, e_qdata["qrels"])
            same = "★" if eval_short == train_short else " "
            print(f"    {same} {eval_short}: nDCG@5 = {ndcg:.4f}")
