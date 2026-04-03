#!/usr/bin/env python3
"""
Quick verification: reproduce DocPruner paper's numbers by running
identity + DocPruner k=-0.25 on ViDoRe-V2 with ALL languages (not just English).

Paper reports (ColQwen2.5 + ViDoRe-V2, all languages):
  - Identity baseline: 0.5508
  - DocPruner k=-0.25: 0.5470 (↓0.69%, 51.55% storage reduction)

Usage:
  CUDA_VISIBLE_DEVICES=0 python verify_multilang.py
"""

import json
import os
import sys

os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/active_work/environment/.cache/huggingface")

from benchmark.data import VIDORE_V2_DATASETS, load_vidore_v2
from benchmark.model import load_colqwen25, encode_corpus, encode_queries
from benchmark.methods import get_method
from benchmark.eval import maxsim_retrieval, evaluate_ndcg5
from benchmark.utils import set_seed, pick_device
from pathlib import Path
import torch
from tqdm import tqdm

OUTDIR = "/active_work/environment/benchmark_outputs"
CACHE_DIR = Path(OUTDIR) / "cache"

set_seed(0)
device = pick_device("cuda:0")

# Load model once
print("Loading ColQwen2.5...")
model, processor = load_colqwen25("vidore/colqwen2.5-v0.2", device=device, need_attention=False)

all_results = []

for ds_name in VIDORE_V2_DATASETS:
    ds_short = ds_name.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_short} (ALL languages)")
    print(f"{'='*60}")

    # Load dataset with language=None to include all languages
    corpus_ids, corpus_images, query_ids, query_texts, qrels = \
        load_vidore_v2(ds_name, split="test", language=None)

    print(f"  Queries: {len(query_ids)} (all languages)")

    # Load cached embeddings
    model_short = "vidore_colqwen2.5-v0.2"
    cache_emb = CACHE_DIR / f"{ds_short}_{model_short}_emb.pt"
    cache_attn = CACHE_DIR / f"{ds_short}_{model_short}_attn.pt"
    cache_mask = CACHE_DIR / f"{ds_short}_{model_short}_imgpad_mask.pt"

    doc_emb_list = torch.load(cache_emb, weights_only=False)
    doc_attn_list = torch.load(cache_attn, weights_only=False)
    doc_mask_list = torch.load(cache_mask, weights_only=False)
    print(f"  Loaded {len(doc_emb_list)} docs from cache")

    # Encode queries
    print("  Encoding queries...")
    q_emb_list = encode_queries(model, processor, query_texts, device, batch_size=8)

    for pruner_name, k_val in [("identity", None), ("docpruner", -0.25)]:
        compress_fn, _ = get_method(pruner_name)

        # Prune
        pruned_docs = []
        for i in range(len(doc_emb_list)):
            mask_i = doc_mask_list[i] if i < len(doc_mask_list) else None
            attn_i = doc_attn_list[i] if doc_attn_list[i] is not None else torch.zeros(doc_emb_list[i].shape[0])
            kwargs = {}
            if k_val is not None:
                kwargs["k"] = k_val
            result = compress_fn(doc_emb_list[i], attn_i, imgpad_mask=mask_i, **kwargs)
            pruned_docs.append(result.vectors)

        avg_prune = sum(1.0 - (p.shape[0] / doc_emb_list[j].shape[0])
                        for j, p in enumerate(pruned_docs)) / len(pruned_docs)

        # Retrieve
        run = maxsim_retrieval(
            Q_list=q_emb_list, D_list=pruned_docs,
            query_ids=query_ids, corpus_ids=corpus_ids,
            device=device, batch_q=4, batch_d=16,
        )
        ndcg5 = evaluate_ndcg5(run, qrels)

        print(f"  {pruner_name:>12} k={k_val}: nDCG@5={ndcg5:.4f}, pruning={avg_prune*100:.1f}%")
        all_results.append({
            "dataset": ds_name, "pruner": pruner_name, "k": k_val,
            "ndcg@5": round(ndcg5, 4), "avg_pruning": round(avg_prune, 4),
            "num_queries": len(query_ids), "languages": "all",
        })

        del pruned_docs
        torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print("SUMMARY (all languages)")
print(f"{'='*60}")

datasets = [d.split("/")[-1] for d in VIDORE_V2_DATASETS]
for pruner in ["identity", "docpruner"]:
    scores = [r["ndcg@5"] for r in all_results if r["pruner"] == pruner]
    avg = sum(scores) / len(scores)
    print(f"  {pruner:>12}: avg nDCG@5 = {avg:.4f}  (per-dataset: {scores})")

print(f"\nPaper reports: identity=0.5508, docpruner k=-0.25=0.5470")

# Save
Path(OUTDIR).joinpath("multilang_verify.json").write_text(
    json.dumps(all_results, indent=2))
print(f"Saved to {OUTDIR}/multilang_verify.json")
