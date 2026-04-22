"""
Merge-then-Prune (ablation baseline for DP-PostMerge).

Reversed order: first merge near-duplicate patches on the FULL image patch set,
then apply DocPruner's attention threshold on the merged result.

Purpose: show that order matters — merging on noisy, unfiltered patches causes
feature dilution, leading to worse nDCG than prune-then-merge (DP-PostMerge).
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import _robust_mu_sigma


def merge_then_prune_compress(embeddings: torch.Tensor,
                              attention_scores: torch.Tensor,
                              imgpad_mask: Optional[torch.Tensor] = None,
                              *,
                              k: float = -0.25,
                              merge_threshold: float = 0.93,
                              **kwargs) -> PruneResult:
    """
    Step 1: Greedy cosine merge on ALL image patches (same merge logic as DP-PostMerge)
    Step 2: DocPruner adaptive threshold on the merged result
    """
    n = embeddings.shape[0]
    if n == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=0, num_after=0)

    if imgpad_mask is None:
        imgpad_mask = torch.ones(n, dtype=torch.bool)

    text_indices = (~imgpad_mask).nonzero(as_tuple=True)[0]
    img_indices = imgpad_mask.nonzero(as_tuple=True)[0]
    n_img = img_indices.shape[0]

    if n_img <= 2:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=n_img, num_after=n_img)

    # ================================================================
    # Step 1: Greedy merge on ALL image patches (before pruning)
    # Same logic as dp_postmerge but on the full unfiltered set
    # ================================================================
    img_emb = embeddings[img_indices]
    img_attn = attention_scores[img_indices]
    orig_dtype = img_emb.dtype

    # L2 normalize for cosine similarity
    img_float = img_emb.float()
    norms = img_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    img_normed = img_float / norms

    # Greedy merge: highest attention first absorbs near-duplicates
    order = img_attn.argsort(descending=True)
    merged_vectors = []
    merged_attns = []
    consumed = torch.zeros(n_img, dtype=torch.bool)

    for i in range(n_img):
        idx = order[i].item()
        if consumed[idx]:
            continue

        sim = img_normed[idx] @ img_normed.T
        similar_mask = (sim > merge_threshold) & (~consumed)
        similar_indices = similar_mask.nonzero(as_tuple=True)[0]

        if similar_indices.shape[0] == 1:
            merged_vectors.append(img_emb[idx])
            merged_attns.append(img_attn[idx])
            consumed[idx] = True
        else:
            group_emb = img_emb[similar_indices].float()
            group_attn = img_attn[similar_indices].float()
            weights = group_attn / group_attn.sum()
            centroid = (group_emb * weights.unsqueeze(1)).sum(dim=0).to(orig_dtype)
            merged_vectors.append(centroid)
            # Use max attention as the representative attention for pruning
            merged_attns.append(group_attn.max())
            consumed[similar_indices] = True

    merged_emb = torch.stack(merged_vectors)
    merged_attn = torch.stack(merged_attns)
    n_after_merge = merged_emb.shape[0]

    # ================================================================
    # Step 2: DocPruner adaptive threshold on merged vectors
    # ================================================================
    mu, sigma = _robust_mu_sigma(merged_attn)
    tau = mu + k * sigma

    keep_mask = merged_attn > tau
    if keep_mask.sum() == 0:
        keep_mask[merged_attn.argmax()] = True

    kept_merged = merged_emb[keep_mask]

    # Combine: text tokens + kept merged image patches
    parts = [embeddings[text_indices], kept_merged]
    result = torch.cat(parts, dim=0)

    n_after = kept_merged.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_after / n_img),
        num_before=n_img,
        num_after=n_after,
    )
