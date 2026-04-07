"""
DocMerger: tri-level partitioning + attention-weighted clustering.

P_preserve = { d_j | I(d_j) > mu + k1*sigma }         -> keep as-is
P_merge    = { d_j | mu - k2*sigma < I(d_j) <= ... }   -> cluster + merge
P_discard  = { d_j | I(d_j) <= mu - k2*sigma }         -> remove
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import _robust_mu_sigma


def _docmerger_core(embeddings: torch.Tensor,
                    attention_scores: torch.Tensor,
                    imgpad_mask: Optional[torch.Tensor],
                    k1: float, k2: float, merge_ratio: float,
                    weighted: bool) -> PruneResult:
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
    mu, sigma = _robust_mu_sigma(img_attn)

    tau_high = mu + k1 * sigma
    tau_low = mu - k2 * sigma

    preserve_mask = img_attn > tau_high
    discard_mask = img_attn <= tau_low
    merge_mask = ~preserve_mask & ~discard_mask

    preserve_idx = img_indices[preserve_mask]
    merge_idx = img_indices[merge_mask]

    if preserve_idx.shape[0] == 0 and merge_idx.shape[0] == 0:
        preserve_idx = img_indices[img_attn.argmax().unsqueeze(0)]
        merge_idx = torch.tensor([], dtype=torch.long)

    merged_vectors = torch.empty(0, embeddings.shape[1])
    n_merge = merge_idx.shape[0]
    if n_merge > 0:
        n_clusters = max(1, int(n_merge * merge_ratio))
        if n_clusters >= n_merge:
            merged_vectors = embeddings[merge_idx]
        else:
            merge_emb = embeddings[merge_idx].float().numpy()
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters, metric="cosine", linkage="average"
            )
            labels = clustering.fit_predict(merge_emb)
            merge_attn = img_attn[merge_mask]

            centroids = []
            for c in range(n_clusters):
                cmask = torch.tensor(labels == c)
                c_emb = embeddings[merge_idx[cmask]]
                if weighted:
                    w = merge_attn[cmask].unsqueeze(1)
                    w = w / w.sum()
                    centroid = (c_emb * w).sum(dim=0)
                else:
                    centroid = c_emb.mean(dim=0)
                centroids.append(centroid)
            merged_vectors = torch.stack(centroids)

    parts = [embeddings[text_indices], embeddings[preserve_idx]]
    if merged_vectors.shape[0] > 0:
        parts.append(merged_vectors)
    result = torch.cat(parts, dim=0)

    n_img_after = preserve_idx.shape[0] + merged_vectors.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_img_after / n_img) if n_img > 0 else 0.0,
        num_before=n_img,
        num_after=int(n_img_after),
    )


def docmerger_compress(embeddings: torch.Tensor,
                       attention_scores: torch.Tensor,
                       imgpad_mask: Optional[torch.Tensor] = None,
                       *,
                       k1: float = 0.5, k2: float = 0.25,
                       merge_ratio: float = 0.25,
                       **kwargs) -> PruneResult:
    """DocMerger with attention-weighted centroids."""
    return _docmerger_core(embeddings, attention_scores, imgpad_mask,
                           k1, k2, merge_ratio, weighted=True)


def docmerger_avg_compress(embeddings: torch.Tensor,
                           attention_scores: torch.Tensor,
                           imgpad_mask: Optional[torch.Tensor] = None,
                           *,
                           k1: float = 0.5, k2: float = 0.25,
                           merge_ratio: float = 0.25,
                           **kwargs) -> PruneResult:
    """DocMerger with simple average centroids."""
    return _docmerger_core(embeddings, attention_scores, imgpad_mask,
                           k1, k2, merge_ratio, weighted=False)
