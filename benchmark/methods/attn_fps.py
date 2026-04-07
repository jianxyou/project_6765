"""
Attention-guided Farthest Point Sampling (Attn-FPS).

Combines attention importance with geometric diversity via
a modified farthest-point-sampling algorithm.

At each step, select:
  argmax_{d in remaining} [ attn(d)^alpha * dist(d, selected)^(1-alpha) ]

The multiplicative form ensures that both factors must be non-trivial:
a patch needs BOTH reasonable attention AND distance from selected set.
"""

import torch
from typing import Optional
from .base import PruneResult


def attn_fps_compress(embeddings: torch.Tensor,
                      attention_scores: torch.Tensor,
                      imgpad_mask: Optional[torch.Tensor] = None,
                      *,
                      target_ratio: float = 0.5,
                      alpha: float = 0.5,
                      **kwargs) -> PruneResult:
    """
    Attention-weighted Farthest Point Sampling.

    Args:
        target_ratio: fraction of image patches to KEEP
        alpha: power for attention term (higher = more attention-driven)
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

    # Normalize attention to [0, 1] with a floor to avoid zeroing out
    attn_min = img_attn.min()
    attn_range = img_attn.max() - attn_min
    if attn_range > 0:
        attn_norm = (img_attn - attn_min) / attn_range
    else:
        attn_norm = torch.ones_like(img_attn)
    # Floor to prevent zero scores
    attn_norm = attn_norm.clamp(min=0.01)

    # L2-normalize embeddings for cosine distance
    img_emb_normed = img_emb / img_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Greedy Attn-FPS selection (vectorized with mask)
    available = torch.ones(n_img, dtype=torch.bool)
    selected_indices = torch.empty(n_keep, dtype=torch.long)

    first = img_attn.argmax().item()
    selected_indices[0] = first
    available[first] = False

    min_dist = 1.0 - (img_emb_normed @ img_emb_normed[first]).clamp(-1, 1)

    for step in range(1, n_keep):
        # Multiplicative score: attn^alpha * dist^(1-alpha)
        scores = attn_norm.pow(alpha) * min_dist.clamp(min=1e-8).pow(1.0 - alpha)
        scores[~available] = -float('inf')

        best = scores.argmax().item()
        selected_indices[step] = best
        available[best] = False

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
