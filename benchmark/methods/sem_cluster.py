"""
Sem-Cluster — faithful reproduction of PTM paper Appendix D.1.1.

Steps per paper:
  1. L2-normalize all patch embeddings.
  2. Compute pairwise cosine distance (= 1 - cosine similarity).
  3. Hierarchical agglomerative clustering with Ward linkage.
     Target n_clusters = n_patches // merging_factor.
  4. Average the embeddings within each cluster to form the centroid.

Implementation note: sklearn's Ward linkage requires euclidean distance. For
L2-normalized vectors, ||a - b||^2 = 2(1 - cos_sim(a, b)), so euclidean Ward
on unit-norm vectors is equivalent to cosine Ward (up to a constant factor).
"""

import torch
import numpy as np
from typing import Optional
from .base import PruneResult


def sem_cluster_compress(embeddings: torch.Tensor,
                         attention_scores: Optional[torch.Tensor] = None,
                         imgpad_mask: Optional[torch.Tensor] = None,
                         *,
                         merging_factor: int = 4,
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

    n_clusters = max(1, n_img // merging_factor)

    if n_clusters >= n_img:
        parts = [embeddings[text_indices], embeddings[img_indices]]
        return PruneResult(vectors=torch.cat(parts, dim=0),
                           pruning_ratio=0.0,
                           num_before=n_img, num_after=n_img)

    img_emb = embeddings[img_indices].float()

    # Step 1: L2 normalize
    norms = img_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    img_emb_normed = img_emb / norms

    # Step 2 & 3: Agglomerative clustering with Ward linkage.
    # Ward on L2-normalized vectors ≡ cosine Ward up to constant.
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="euclidean", linkage="ward",
    )
    labels = clustering.fit_predict(img_emb_normed.numpy())

    # Step 4: centroid = mean of ORIGINAL (unnormalized) embeddings in each cluster
    centroids = []
    for c in range(n_clusters):
        cmask = torch.tensor(labels == c)
        if cmask.sum() == 0:
            continue
        centroid = embeddings[img_indices[cmask]].mean(dim=0)
        centroids.append(centroid)

    if not centroids:
        centroids = [embeddings[img_indices[0]]]

    merged = torch.stack(centroids)
    parts = [embeddings[text_indices], merged]
    result = torch.cat(parts, dim=0)

    n_after = merged.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_after / n_img),
        num_before=n_img,
        num_after=n_after,
    )
