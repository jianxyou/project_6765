"""
PTM + Post-hoc Merge (PTM-PostMerge).

Three-stage compression:
  Stage 1: Adaptive Pruning (remove low-info patches)
  Stage 2: Hierarchical Merging (cluster + centroid)
  Stage 3: Post-hoc Merge (merge near-duplicate centroids via attention-weighted avg)

Motivation: PTM's Stage 2 centroids from adjacent clusters can still be
very similar. Stage 3 finds and merges these residual near-duplicates,
squeezing out extra compression at minimal quality cost.
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import _robust_mu_sigma


def ptm_postmerge_compress(embeddings: torch.Tensor,
                           attention_scores: torch.Tensor,
                           imgpad_mask: Optional[torch.Tensor] = None,
                           *,
                           k: float = -0.75,
                           m: int = 4,
                           merge_threshold: float = 0.95,
                           **kwargs) -> PruneResult:
    from sklearn.cluster import AgglomerativeClustering

    n = embeddings.shape[0]
    if n == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=0, num_after=0)

    if imgpad_mask is None:
        imgpad_mask = torch.ones(n, dtype=torch.bool)

    text_indices = (~imgpad_mask).nonzero(as_tuple=True)[0]
    img_indices = imgpad_mask.nonzero(as_tuple=True)[0]
    n_img = img_indices.shape[0]

    if n_img == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=n, num_after=n)

    img_attn = attention_scores[img_indices]

    # ---- Stage 1: Adaptive Pruning ----
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k * sigma

    keep_mask = img_attn > tau
    if keep_mask.sum() == 0:
        keep_mask[img_attn.argmax()] = True

    pruned_img_indices = img_indices[keep_mask]
    pruned_img_attn = img_attn[keep_mask]
    n_pruned = pruned_img_indices.shape[0]

    if n_pruned <= 1:
        parts = [embeddings[text_indices], embeddings[pruned_img_indices]]
        result = torch.cat(parts, dim=0)
        return PruneResult(vectors=result,
                           pruning_ratio=1.0 - (1 / n_img) if n_img > 0 else 0.0,
                           num_before=n_img, num_after=1)

    # ---- Stage 2: Hierarchical Merging ----
    n_clusters = max(1, n_pruned // m)
    if n_clusters >= n_pruned:
        # No merging needed, skip to stage 3
        centroids = embeddings[pruned_img_indices]
        centroid_attn = pruned_img_attn
    else:
        pruned_emb = embeddings[pruned_img_indices].float()
        norms = pruned_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        pruned_emb_normed = pruned_emb / norms

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="euclidean", linkage="ward",
        )
        labels = clustering.fit_predict(pruned_emb_normed.numpy())

        centroid_list = []
        centroid_attn_list = []
        for c in range(n_clusters):
            cmask = torch.tensor(labels == c)
            if cmask.sum() == 0:
                continue
            # Centroid = mean of original embeddings
            centroid = embeddings[pruned_img_indices[cmask]].mean(dim=0)
            centroid_list.append(centroid)
            # Centroid attention = max attention in cluster (for Stage 3 ordering)
            centroid_attn_list.append(pruned_img_attn[cmask].max())

        centroids = torch.stack(centroid_list)
        centroid_attn = torch.stack(centroid_attn_list)

    n_after_stage2 = centroids.shape[0]

    if n_after_stage2 <= 1:
        parts = [embeddings[text_indices], centroids]
        result = torch.cat(parts, dim=0)
        return PruneResult(vectors=result,
                           pruning_ratio=1.0 - (n_after_stage2 / n_img),
                           num_before=n_img, num_after=n_after_stage2)

    # ---- Stage 3: Post-hoc Merge of near-duplicate centroids ----
    orig_dtype = centroids.dtype
    c_float = centroids.float()
    c_norms = c_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    c_normed = c_float / c_norms

    # Greedy merge: process centroids in descending attention order
    order = centroid_attn.argsort(descending=True)
    merged_vectors = []
    consumed = torch.zeros(n_after_stage2, dtype=torch.bool)

    for i in range(n_after_stage2):
        idx = order[i].item()
        if consumed[idx]:
            continue

        sim = c_normed[idx] @ c_normed.T
        similar_mask = (sim > merge_threshold) & (~consumed)
        similar_indices = similar_mask.nonzero(as_tuple=True)[0]

        if similar_indices.shape[0] == 1:
            merged_vectors.append(centroids[idx])
            consumed[idx] = True
        else:
            # Attention-weighted average
            group_emb = centroids[similar_indices].float()
            group_attn = centroid_attn[similar_indices].float()
            weights = group_attn / group_attn.sum()
            merged_vec = (group_emb * weights.unsqueeze(1)).sum(dim=0)
            merged_vectors.append(merged_vec.to(orig_dtype))
            consumed[similar_indices] = True

    # Assemble result
    merged_img = torch.stack(merged_vectors)
    text_emb = embeddings[text_indices]
    result = torch.cat([text_emb, merged_img], dim=0)

    n_final = merged_img.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_final / n_img),
        num_before=n_img,
        num_after=n_final,
    )
