#!/usr/bin/env python3
"""
Quick verification: run identity baseline with ColQwen2.5-v0.1
to check if paper's 0.5508 matches v0.1.

Usage:
  CUDA_VISIBLE_DEVICES=0 python verify_v01.py
"""

import json
import os
import torch
from pathlib import Path

os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/active_work/environment/.cache/huggingface")

from benchmark.data import VIDORE_V2_DATASETS, load_vidore_v2
from benchmark.eval import maxsim_retrieval, evaluate_ndcg5
from benchmark.utils import set_seed, pick_device
from tqdm import tqdm

MODEL_NAME = "vidore/colqwen2.5-v0.1"
OUTDIR = "/active_work/environment/benchmark_outputs"

set_seed(0)
device = pick_device("cuda:0")

# Load model
print(f"Loading {MODEL_NAME}...")
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

model = ColQwen2_5.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16,
    device_map=str(device),
).eval()
processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)

results = []

for ds_name in VIDORE_V2_DATASETS:
    ds_short = ds_name.split("/")[-1]

    # All languages
    corpus_ids, corpus_images, query_ids, query_texts, qrels = \
        load_vidore_v2(ds_name, split="test", language=None)

    print(f"\n{ds_short}: {len(corpus_ids)} docs, {len(query_ids)} queries (all langs)")

    # Encode corpus
    print("  Encoding corpus...")
    doc_emb_list = []
    batch_size = 4
    for i in tqdm(range(0, len(corpus_images), batch_size), desc="  Docs"):
        batch_imgs = corpus_images[i:i+batch_size]
        batch = processor.process_images(batch_imgs).to(device)
        with torch.no_grad():
            embs = model(**batch)
        for j in range(embs.shape[0]):
            doc_emb_list.append(embs[j].detach().cpu())

    # Encode queries
    print("  Encoding queries...")
    q_emb_list = []
    for i in tqdm(range(0, len(query_texts), 8), desc="  Queries"):
        batch_q = query_texts[i:i+8]
        batch = processor.process_queries(batch_q).to(device)
        with torch.no_grad():
            embs = model(**batch)
        for j in range(embs.shape[0]):
            q_emb_list.append(embs[j].detach().cpu())

    # Retrieve (identity, no pruning)
    run = maxsim_retrieval(
        Q_list=q_emb_list, D_list=doc_emb_list,
        query_ids=query_ids, corpus_ids=corpus_ids,
        device=device, batch_q=4, batch_d=16,
    )
    ndcg5 = evaluate_ndcg5(run, qrels)
    print(f"  {ds_short} identity (v0.1): nDCG@5 = {ndcg5:.4f}")
    results.append({"dataset": ds_name, "ndcg@5": round(ndcg5, 4), "model": MODEL_NAME})

    del doc_emb_list, q_emb_list
    torch.cuda.empty_cache()

# Summary
print(f"\n{'='*50}")
print(f"v0.1 identity baseline (all languages):")
avg = sum(r["ndcg@5"] for r in results) / len(results)
for r in results:
    print(f"  {r['dataset'].split('/')[-1]}: {r['ndcg@5']:.4f}")
print(f"  Average: {avg:.4f}")
print(f"\nPaper claims: 0.5508")
print(f"Our v0.2 avg: 0.6012")

Path(OUTDIR).joinpath("v01_baseline.json").write_text(json.dumps(results, indent=2))
print("DONE")
