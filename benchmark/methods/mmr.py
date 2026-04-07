"""
MMR Selection: Maximal Marginal Relevance for patch selection.

Key idea: greedily select patches that balance importance (attention)
and diversity (dissimilarity to already-selected patches).

At each step, select:
  argmax_{d in remaining} [ lambda * attn(d) + (1-lambda) * min_dist(d, selected) ]

This avoids DocPruner's weakness of keeping redundant high-attention patches
from the same semantic region while dropping the only representative of
another region.
"""

import torch
from typing import Optional
from .base import PruneResult


def mmr_compress(embeddings: torch.Tensor,
                 attention_scores: torch.Tensor,
                 imgpad_mask: Optional[torch.Tensor] = None,
                 *,
                 target_ratio: float = 0.5,
                 lam: float = 0.7,
                 **kwargs) -> PruneResult:
    """
    MMR-based patch selection.

    Args:
        target_ratio: fraction of image patches to KEEP (e.g., 0.5 = keep 50%)
        lam: trade-off between attention (1.0) and diversity (0.0)
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

    n_keep = max(1, int(n_img * target_ratio))
    img_emb = embeddings[img_indices].float()
    img_attn = attention_scores[img_indices].float()

    # Normalize attention to [0, 1]
    attn_min = img_attn.min()
    attn_range = img_attn.max() - attn_min
    if attn_range > 0:
        attn_norm = (img_attn - attn_min) / attn_range
    else:
        attn_norm = torch.ones_like(img_attn)

    # Precompute pairwise cosine similarity
    img_emb_normed = img_emb / img_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # We'll compute distances on-the-fly to save memory for large n_img

    # Greedy MMR selection (vectorized with mask)
    available = torch.ones(n_img, dtype=torch.bool)
    selected_indices = torch.empty(n_keep, dtype=torch.long)

    # Start with highest attention patch
    first = img_attn.argmax().item()
    selected_indices[0] = first
    available[first] = False

    # Track min distance to selected set
    min_dist = 1.0 - (img_emb_normed @ img_emb_normed[first]).clamp(-1, 1)

    for step in range(1, n_keep):
        # Score all available patches at once
        div = min_dist.clone()
        div_max = div[available].max()
        if div_max > 0:
            div = div / div_max

        scores = lam * attn_norm + (1.0 - lam) * div
        scores[~available] = -float('inf')

        best = scores.argmax().item()
        selected_indices[step] = best
        available[best] = False

        # Update min distances
        dist_new = 1.0 - (img_emb_normed @ img_emb_normed[best]).clamp(-1, 1)
        min_dist = torch.minimum(min_dist, dist_new)

    selected_t = selected_indices.sort().values
    kept_img = img_indices[selected_t]
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    result = embeddings[all_kept]

    n_kept = selected_t.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )
