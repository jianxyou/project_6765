#!/usr/bin/env python3
"""
Scan V2 docs to find candidates with VISUALLY-COHERENT, ATTENTION-POSITIVE
merge groups.

Scoring: we want groups where
  - Patches look visually alike (pixel cosine ≥ 0.9 on 8x8 thumbnails)
  - Patches have non-trivial attention (mean attention > 1‰ — not just noise)
  - Cluster is sizeable (>= 4)
  - Spread across the page (spatial spread bonus) — this gives the "look,
    the method found repeated elements all over the page" moment.

Output: top candidates ranked by story potential.
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from benchmark.methods.docpruner import docpruner_compress
from benchmark.methods.dp_postmerge import dp_postmerge_compress


CACHE_DIR = Path("/active_work/environment/benchmark_outputs/cache")
DATASETS = [
    "esg_reports_v2",
    "biomedical_lectures_eng_v2",
    "economics_reports_v2",
    "esg_reports_human_labeled_v2",
]

K = 0.0
THRESHOLD = 0.93
MIN_CLUSTER_SIZE = 4
MIN_ATTN_MEAN_MILLI = 1.0  # ‰
MIN_VIS_COHERENCE = 0.90


def best_group_stats(emb, attn, mask, img):
    dp_res = docpruner_compress(emb, attn, imgpad_mask=mask, k=K)
    pm_res = dp_postmerge_compress(emb, attn, imgpad_mask=mask,
                                   k=K, merge_threshold=THRESHOLD)
    if pm_res.cluster_labels is None:
        return None

    labels = pm_res.cluster_labels.numpy()
    cluster_sizes = np.bincount(labels)
    multi = [(int(c), int(s)) for c, s in enumerate(cluster_sizes) if s >= MIN_CLUSTER_SIZE]
    if not multi:
        return None

    img_idx = mask.nonzero(as_tuple=True)[0].numpy()
    dp_kept_local = dp_res.kept_indices.numpy()
    # Estimate grid from img size vs image_pad count — assume 2-to-1 spatial merge
    # We need processor to know grid exactly; approximate via sqrt-ish.
    # For scan purposes we only need (r, c) positions within a grid.
    # Use image aspect ratio + n_img to factor.
    n_img = dp_res.num_before
    W, H = img.size
    ar = W / H
    # Find (h_tokens, w_tokens) s.t. h*w = n_img and w/h ≈ ar
    # Brute force candidate pairs
    best_pair = None
    best_diff = float("inf")
    for h in range(8, 60):
        if n_img % h != 0:
            continue
        w = n_img // h
        diff = abs((w / h) - ar)
        if diff < best_diff:
            best_diff = diff
            best_pair = (h, w)
    if best_pair is None:
        return None
    h_tokens, w_tokens = best_pair
    if h_tokens * w_tokens != n_img:
        return None

    img_arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    cell_h = img_arr.shape[0] / h_tokens
    cell_w = img_arr.shape[1] / w_tokens

    scored = []
    for cid, size in multi:
        members = np.where(labels == cid)[0]
        positions = dp_kept_local[members]
        rows = positions // w_tokens
        cols = positions % w_tokens
        # Spread: range of rows × range of cols / grid_size
        spread = ((rows.max() - rows.min()) / h_tokens
                  + (cols.max() - cols.min()) / w_tokens) / 2

        # Attention
        global_idx = img_idx[positions]
        attn_vals = attn[global_idx].numpy()
        mean_attn_milli = float(attn_vals.mean() * 1000)

        if mean_attn_milli < MIN_ATTN_MEAN_MILLI:
            continue

        # Visual coherence (8x8 RGB thumbnails)
        thumbs = []
        for pos in positions:
            r = pos // w_tokens
            c = pos % w_tokens
            y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            patch = img_arr[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]
            if ph < 1 or pw < 1:
                continue
            bh, bw = max(1, ph // 8), max(1, pw // 8)
            patch_small = patch[:bh*8, :bw*8].reshape(8, bh, 8, bw, 3).mean(axis=(1, 3))
            thumbs.append(patch_small.flatten())
        if len(thumbs) < 2:
            continue
        X = np.stack(thumbs)
        X_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sim = X_n @ X_n.T
        n = sim.shape[0]
        coh = (sim.sum() - n) / (n * (n - 1))

        if coh < MIN_VIS_COHERENCE:
            continue

        # Story score: size * coherence * spread * log1p(attention)
        score = size * coh * (1 + spread) * np.log1p(mean_attn_milli)
        scored.append({
            "cluster_id": cid, "size": size, "mean_attn": mean_attn_milli,
            "coherence": coh, "spread": spread, "score": float(score),
            "rows": rows.tolist(), "cols": cols.tolist(),
        })

    if not scored:
        return None
    scored.sort(key=lambda x: -x["score"])
    return scored[0], dp_res.num_after, pm_res.num_after


def main():
    all_results = []
    for ds_short in DATASETS:
        emb_path = CACHE_DIR / f"{ds_short}_vidore_colqwen2.5-v0.2_emb.pt"
        attn_path = CACHE_DIR / f"{ds_short}_vidore_colqwen2.5-v0.2_attn.pt"
        mask_path = CACHE_DIR / f"{ds_short}_vidore_colqwen2.5-v0.2_imgpad_mask.pt"
        if not emb_path.exists():
            continue
        print(f"\nScanning {ds_short}...")

        # Also load image corpus for pixel extraction
        ds_full = f"vidore/{ds_short}"
        try:
            corpus = load_dataset(ds_full, "corpus", split="test")
        except Exception as e:
            print(f"  skip (corpus load failed): {e}")
            continue

        embs = torch.load(emb_path, weights_only=False)
        attns = torch.load(attn_path, weights_only=False)
        masks = torch.load(mask_path, weights_only=False)

        for i, (emb, attn, mask) in enumerate(zip(embs, attns, masks)):
            try:
                res = best_group_stats(emb, attn, mask, corpus[i]["image"])
            except Exception:
                continue
            if res is None:
                continue
            best_group, dp_kept, pm_kept = res
            all_results.append({
                "dataset": ds_short, "idx": i,
                "dp_kept": dp_kept, "pm_kept": pm_kept,
                **best_group,
            })

        print(f"  {len(embs)} docs scanned")

    all_results.sort(key=lambda r: -r["score"])

    print("\n" + "=" * 115)
    print(f"{'Rk':>3} {'Dataset':<32} {'idx':>5} {'DP':>4} {'PM':>4} "
          f"{'size':>4} {'attn‰':>7} {'coh':>5} {'sprd':>5} {'score':>7}")
    print("-" * 115)
    for rank, r in enumerate(all_results[:30], 1):
        print(f"{rank:>3} {r['dataset']:<32} {r['idx']:>5} {r['dp_kept']:>4} "
              f"{r['pm_kept']:>4} {r['size']:>4} {r['mean_attn']:>7.2f} "
              f"{r['coherence']:>5.2f} {r['spread']:>5.2f} {r['score']:>7.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
