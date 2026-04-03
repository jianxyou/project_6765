"""
DocPruner + Diversity Rebalance.

Two-phase approach:
  Phase 1: Use a LOOSE DocPruner threshold (k_loose) to get a large candidate set
           (more permissive than the target, keeping ~70-80% of patches)
  Phase 2: From this candidate set, greedily select patches using MMR-style
           scoring to reach the target count, ensuring diversity.

This combines DocPruner's strength (attention-based filtering removes true noise)
with diversity-aware selection (avoids redundancy in high-attention regions).
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import _robust_mu_sigma


def dp_rebalance_compress(embeddings: torch.Tensor,
                          attention_scores: torch.Tensor,
                          imgpad_mask: Optional[torch.Tensor] = None,
                          *,
                          target_ratio: float = 0.5,
                          k_loose: float = -0.5,
                          lam: float = 0.6,
                          **kwargs) -> PruneResult:
    """
    DocPruner + Diversity Rebalance.

    Args:
        target_ratio: fraction of image patches to KEEP (final target)
        k_loose: loose DocPruner threshold for initial filtering (more negative = keep more)
        lam: MMR lambda for phase 2 (higher = more attention-driven)
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

    img_attn = attention_scores[img_indices].float()
    n_keep = max(1, int(n_img * target_ratio))

    # Phase 1: Loose DocPruner filtering
    mu, sigma = _robust_mu_sigma(img_attn)
    tau_loose = mu + k_loose * sigma
    candidate_mask = img_attn > tau_loose

    # Ensure we have enough candidates
    if candidate_mask.sum() < n_keep:
        # If loose threshold is too strict, take top n_keep*1.5 by attention
        n_candidates = min(n_img, int(n_keep * 1.5))
        topk_indices = img_attn.topk(n_candidates).indices
        candidate_mask = torch.zeros(n_img, dtype=torch.bool)
        candidate_mask[topk_indices] = True

    candidate_local = candidate_mask.nonzero(as_tuple=True)[0]
    n_candidates = candidate_local.shape[0]

    # If candidates already at or below target, just return them
    if n_candidates <= n_keep:
        kept_img = img_indices[candidate_local]
        all_kept = torch.cat([text_indices, kept_img]).sort().values
        result = embeddings[all_kept]
        return PruneResult(
            vectors=result,
            pruning_ratio=1.0 - (n_candidates / n_img),
            num_before=n_img,
            num_after=n_candidates,
        )

    # Phase 2: MMR-style diversity selection within candidates
    cand_emb = embeddings[img_indices[candidate_local]].float()
    cand_attn = img_attn[candidate_local]

    # Normalize
    cand_emb_normed = cand_emb / cand_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
    attn_min = cand_attn.min()
    attn_range = cand_attn.max() - attn_min
    if attn_range > 0:
        attn_norm = (cand_attn - attn_min) / attn_range
    else:
        attn_norm = torch.ones_like(cand_attn)

    # Greedy MMR (vectorized with mask)
    available = torch.ones(n_candidates, dtype=torch.bool)
    selected_indices = torch.empty(n_keep, dtype=torch.long)

    first = cand_attn.argmax().item()
    selected_indices[0] = first
    available[first] = False

    min_dist = 1.0 - (cand_emb_normed @ cand_emb_normed[first]).clamp(-1, 1)

    for step in range(1, n_keep):
        div = min_dist.clone()
        div_max = div[available].max()
        if div_max > 0:
            div = div / div_max

        scores = lam * attn_norm + (1.0 - lam) * div
        scores[~available] = -float('inf')

        best = scores.argmax().item()
        selected_indices[step] = best
        available[best] = False

        min_dist = torch.minimum(min_dist, 1.0 - (cand_emb_normed @ cand_emb_normed[best]).clamp(-1, 1))

    # Map back to global indices
    selected_local = candidate_local[selected_indices.sort().values]
    kept_img = img_indices[selected_local]
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    result = embeddings[all_kept]

    n_kept = selected_indices.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )
