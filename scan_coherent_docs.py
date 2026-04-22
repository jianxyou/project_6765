#!/usr/bin/env python3
"""
Scan ESG V2 corpus for documents with visually coherent merge groups.

Uses cached embeddings (no GPU needed). For each doc, runs DocPruner +
DP-PostMerge, then scores the top-3 largest merge clusters by pixel coherence
(mean pairwise cosine of downsampled RGB thumbnails). Outputs top N candidates
ranked by max-group coherence * max-group-size.
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

from benchmark.methods.docpruner import docpruner_compress
from benchmark.methods.dp_postmerge import dp_postmerge_compress
from benchmark.visualize import get_patch_grid


CACHE_DIR = Path("/active_work/environment/benchmark_outputs/cache")
DATASET = "vidore/esg_reports_v2"
MODEL_SHORT = "vidore_colqwen2.5-v0.2"
K = 0.0
THRESHOLD = 0.93
MIN_CLUSTER_SIZE = 3   # only consider clusters with >= N members
TOP_N_DOCS = 15        # output top N candidates


def group_pixel_coherence(img_arr, positions, h_tokens, w_tokens):
    """Mean pairwise cosine similarity of 8x8 downsampled RGB thumbnails."""
    H0, W0 = img_arr.shape[:2]
    cell_h0 = H0 / h_tokens
    cell_w0 = W0 / w_tokens
    thumbs = []
    for pos in positions:
        r = pos // w_tokens
        c = pos % w_tokens
        y0, y1 = int(r * cell_h0), int((r + 1) * cell_h0)
        x0, x1 = int(c * cell_w0), int((c + 1) * cell_w0)
        patch = img_arr[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        ph, pw = patch.shape[:2]
        bh, bw = max(1, ph // 8), max(1, pw // 8)
        patch_small = patch[: bh*8, : bw*8].reshape(
            8, bh, 8, bw, 3).mean(axis=(1, 3))
        thumbs.append(patch_small.flatten())
    if len(thumbs) < 2:
        return 0.0
    X = np.stack(thumbs)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sim = X_norm @ X_norm.T
    n = sim.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


def main():
    print(f"Loading cache from {CACHE_DIR}")
    emb_list = torch.load(CACHE_DIR / f"esg_reports_v2_{MODEL_SHORT}_emb.pt",
                          weights_only=False)
    attn_list = torch.load(CACHE_DIR / f"esg_reports_v2_{MODEL_SHORT}_attn.pt",
                           weights_only=False)
    mask_list = torch.load(CACHE_DIR / f"esg_reports_v2_{MODEL_SHORT}_imgpad_mask.pt",
                           weights_only=False)
    print(f"Cached {len(emb_list)} docs")

    print("Loading images + processor (slow, ~30s)...")
    from colpali_engine.models import ColQwen2_5_Processor
    processor = ColQwen2_5_Processor.from_pretrained("vidore/colqwen2.5-v0.2")
    ds = load_dataset(DATASET, "corpus", split="test")
    print(f"Corpus: {len(ds)} images")

    n_docs = min(len(emb_list), len(ds))
    candidates = []

    for doc_idx in tqdm(range(n_docs), desc="Scanning"):
        emb = emb_list[doc_idx]
        attn = attn_list[doc_idx]
        mask = mask_list[doc_idx]

        dp_res = docpruner_compress(emb, attn, imgpad_mask=mask, k=K)
        pm_res = dp_postmerge_compress(emb, attn, imgpad_mask=mask,
                                        k=K, merge_threshold=THRESHOLD)

        labels = pm_res.cluster_labels.numpy()
        cluster_sizes = np.bincount(labels)
        multi = [(c, s) for c, s in enumerate(cluster_sizes) if s >= MIN_CLUSTER_SIZE]
        if not multi:
            continue

        # Sort by size, take top 3
        multi.sort(key=lambda x: -x[1])
        top_multi = multi[:3]

        # Need image + patch grid for coherence
        img = ds[doc_idx]["image"]
        try:
            h_tokens, w_tokens = get_patch_grid(processor, img)
        except Exception:
            continue
        img_arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0

        dp_kept_local = dp_res.kept_indices.numpy()

        coh_scores = []
        for cid, size in top_multi:
            members = np.where(labels == cid)[0]
            positions = dp_kept_local[members]
            coh = group_pixel_coherence(img_arr, positions, h_tokens, w_tokens)
            coh_scores.append((cid, size, coh))

        # Score: best cluster's (size * coherence), tiebreaker mean coh
        best = max(coh_scores, key=lambda x: x[1] * x[2])
        mean_coh = np.mean([c[2] for c in coh_scores])

        candidates.append({
            "doc_idx": doc_idx,
            "best_cluster_size": int(best[1]),
            "best_cluster_coh": round(float(best[2]), 4),
            "mean_top3_coh": round(float(mean_coh), 4),
            "score": round(float(best[1] * best[2]), 4),
            "top3": [(int(c), int(s), round(float(h), 4)) for c, s, h in coh_scores],
        })

    # Rank by score
    candidates.sort(key=lambda x: -x["score"])
    top = candidates[:TOP_N_DOCS]

    print(f"\nTop {TOP_N_DOCS} most visually coherent docs (k={K}, θ={THRESHOLD}):")
    print(f"{'doc_idx':>8} {'best_size':>10} {'best_coh':>9} {'mean3_coh':>10} {'score':>8}")
    for c in top:
        print(f"{c['doc_idx']:>8} {c['best_cluster_size']:>10} "
              f"{c['best_cluster_coh']:>9.3f} {c['mean_top3_coh']:>10.3f} "
              f"{c['score']:>8.3f}")

    out_path = Path("/active_work/environment/benchmark_outputs/coherent_docs_scan.json")
    out_path.write_text(json.dumps(top, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
