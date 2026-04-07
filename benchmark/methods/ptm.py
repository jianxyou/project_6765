"""
Prune-then-Merge (Yan et al., 2026).

Stage 1: Adaptive Pruning (same as DocPruner, EOS attention threshold)
Stage 2: Hierarchical Merging (agglomerative clustering + centroid)
"""

import torch
from typing import Optional
from .base import PruneResult
def ptm_compress(embeddings: torch.Tensor,
                 attention_scores: torch.Tensor,
                 imgpad_mask: Optional[torch.Tensor] = None,
                 *,
                 k: float = -0.75,
                 m: int = 4,
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

    # Stage 1: Adaptive Pruning (DocPruner's IQR-clipped mean/std)
    # Paper's simple mean/std yields negative thresholds on ColQwen2.5 due to
    # attention distribution (μ≈0.001, σ≈0.02), making Stage 1 ineffective.
    # Using IQR-clipped stats (same as DocPruner) to get a meaningful threshold.
    from .docpruner import _robust_mu_sigma
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k * sigma

    keep_mask = img_attn > tau
    if keep_mask.sum() == 0:
        keep_mask[img_attn.argmax()] = True

    pruned_img_indices = img_indices[keep_mask]
    n_pruned = pruned_img_indices.shape[0]

    # Stage 2: Hierarchical Merging
    if n_pruned <= 1:
        parts = [embeddings[text_indices], embeddings[pruned_img_indices]]
        result = torch.cat(parts, dim=0)
        return PruneResult(vectors=result,
                           pruning_ratio=1.0 - (1 / n_img) if n_img > 0 else 0.0,
                           num_before=n_img, num_after=1)

    n_clusters = max(1, n_pruned // m)
    if n_clusters >= n_pruned:
        parts = [embeddings[text_indices], embeddings[pruned_img_indices]]
        result = torch.cat(parts, dim=0)
        return PruneResult(vectors=result,
                           pruning_ratio=1.0 - (n_pruned / n_img),
                           num_before=n_img, num_after=n_pruned)

    # L2 normalize -> Ward's linkage
    pruned_emb = embeddings[pruned_img_indices].float()
    norms = pruned_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    pruned_emb_normed = pruned_emb / norms

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="euclidean", linkage="ward",
    )
    labels = clustering.fit_predict(pruned_emb_normed.numpy())

    # Centroid per cluster (using original embeddings)
    centroids = []
    for c in range(n_clusters):
        cmask = torch.tensor(labels == c)
        if cmask.sum() == 0:
            continue
        centroid = embeddings[pruned_img_indices[cmask]].mean(dim=0)
        centroids.append(centroid)

    if not centroids:
        centroids = [embeddings[pruned_img_indices[0]]]

    merged = torch.stack(centroids)
    parts = [embeddings[text_indices], merged]
    result = torch.cat(parts, dim=0)

    n_after = merged.shape[0]
    return PruneResult(vectors=result,
                       pruning_ratio=1.0 - (n_after / n_img),
                       num_before=n_img, num_after=n_after)
