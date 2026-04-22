#!/usr/bin/env python3
"""
Scan all ViDoRe-V2 docs using cached embeddings to find documents
with large, semantically meaningful merge groups.

Heuristic: rank by (largest cluster size × mean attention of absorbed patches).
This prioritizes docs where PostMerge merges big groups of attention-weighted
(== semantic) patches, not just whitespace.

Output: ranked list of (dataset, doc_idx, DP kept, PM kept, merge_gain,
largest cluster size, mean attn in largest cluster).
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

from pathlib import Path
import torch
from datasets import load_dataset

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


def score_doc(emb, attn, mask, k=K, thresh=THRESHOLD):
    """Run docpruner + dp_postmerge, return metrics."""
    dp_res = docpruner_compress(emb, attn, imgpad_mask=mask, k=k)
    pm_res = dp_postmerge_compress(emb, attn, imgpad_mask=mask,
                                    k=k, merge_threshold=thresh)

    dp_kept = dp_res.num_after
    pm_kept = pm_res.num_after
    merge_gain = dp_kept - pm_kept

    # Analyze clusters from pm_res.cluster_labels + kept_indices
    if pm_res.cluster_labels is None:
        return None

    labels = pm_res.cluster_labels
    # Only looking at the patches that got KEPT by DocPruner (before merging)
    # cluster_labels are over dp_res.kept_indices
    # Each unique label = a cluster; cluster_size > 1 means actual merging
    cluster_sizes = torch.bincount(labels)
    big_clusters = cluster_sizes[cluster_sizes > 1]
    n_merge_clusters = int((cluster_sizes > 1).sum().item())
    largest = int(cluster_sizes.max().item())

    # mean attention of patches in the largest cluster
    img_idx = mask.nonzero(as_tuple=True)[0]
    dp_kept_local = dp_res.kept_indices  # indices into image_pad subset
    dp_kept_global = img_idx[dp_kept_local]  # indices into full seq
    attn_on_kept = attn[dp_kept_global]

    if largest > 1:
        biggest_cluster_id = int(cluster_sizes.argmax().item())
        in_largest = (labels == biggest_cluster_id)
        mean_attn_largest = attn_on_kept[in_largest].mean().item()
    else:
        mean_attn_largest = 0.0

    # Overall mean absorbed attention (attention of non-representatives)
    rep_mask = pm_res.representative_mask.bool()
    absorbed_mask = ~rep_mask
    if absorbed_mask.any():
        mean_attn_absorbed = attn_on_kept[absorbed_mask].mean().item()
    else:
        mean_attn_absorbed = 0.0

    return {
        "n_img": dp_res.num_before,
        "dp_kept": dp_kept,
        "pm_kept": pm_kept,
        "merge_gain": merge_gain,
        "n_merge_clusters": n_merge_clusters,
        "largest_cluster": largest,
        "mean_attn_largest": mean_attn_largest,
        "mean_attn_absorbed": mean_attn_absorbed,
    }


def main():
    all_results = []

    for ds_short in DATASETS:
        emb_path = CACHE_DIR / f"{ds_short}_vidore_colqwen2.5-v0.2_emb.pt"
        attn_path = CACHE_DIR / f"{ds_short}_vidore_colqwen2.5-v0.2_attn.pt"
        mask_path = CACHE_DIR / f"{ds_short}_vidore_colqwen2.5-v0.2_imgpad_mask.pt"
        if not emb_path.exists():
            print(f"!! Skipping {ds_short} — no cache")
            continue

        print(f"\nScanning {ds_short} ...")
        embs = torch.load(emb_path, weights_only=False)
        attns = torch.load(attn_path, weights_only=False)
        masks = torch.load(mask_path, weights_only=False)

        for i, (emb, attn, mask) in enumerate(zip(embs, attns, masks)):
            try:
                s = score_doc(emb, attn, mask)
            except Exception as e:
                continue
            if s is None:
                continue
            s["dataset"] = ds_short
            s["doc_idx"] = i
            all_results.append(s)

        print(f"  done: {len(embs)} docs scanned")

    # Rank by: largest_cluster × mean_attn_largest  (big semantic cluster)
    def story_score(r):
        # Reward: large cluster size AND high attention of merged patches
        return r["largest_cluster"] * r["mean_attn_largest"] * 1000

    all_results.sort(key=story_score, reverse=True)

    print("\n\n" + "=" * 110)
    print(f"{'Rank':<4} {'Dataset':<35} {'idx':<5} {'DP':>4} {'PM':>4} {'gain':>5} "
          f"{'#clus':>5} {'MaxSz':>5} {'AttnMax':>9} {'AttnAbs':>9} {'Score':>7}")
    print("-" * 110)
    for rank, r in enumerate(all_results[:30], 1):
        print(f"{rank:<4} {r['dataset']:<35} {r['doc_idx']:<5} "
              f"{r['dp_kept']:>4} {r['pm_kept']:>4} {r['merge_gain']:>5} "
              f"{r['n_merge_clusters']:>5} {r['largest_cluster']:>5} "
              f"{r['mean_attn_largest']*1000:>8.3f}‰ "
              f"{r['mean_attn_absorbed']*1000:>8.3f}‰ "
              f"{story_score(r):>7.2f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
