"""
Pivot-Threshold (Zhang et al. 2025 / VisPruner, via DocPruner baselines).

Two-stage adaptive method:
  Stage 1 - Important set: attention adaptive threshold mu + k * sigma
            (same as DocPruner) selects important patches.
  Stage 2 - Pivot de-duplication: select top-`num_pivots` by attention
            from the important set as pivots; then for each non-pivot
            patch in the important set, if its max cosine similarity to
            any pivot exceeds a dynamic threshold mu_sim + k_dup * sigma_sim,
            drop it (too redundant with a pivot).

Parameters:
  k:         adaptation factor for Stage 1 (importance)
  k_dup:     adaptation factor for Stage 2 (de-duplication)
  num_pivots: number of pivots used as redundancy references
"""

import torch
from typing import Optional
from .base import PruneResult


def _robust_mu_sigma(scores: torch.Tensor):
    q75 = scores.quantile(0.75).item()
    q25 = scores.quantile(0.25).item()
    iqr = q75 - q25
    clipped = scores.clamp(max=q75 + 3.0 * iqr)
    mu = clipped.mean().item()
    sigma = clipped.std().item() if scores.shape[0] > 1 else 0.0
    return mu, sigma


def pivot_threshold_compress(embeddings: torch.Tensor,
                             attention_scores: torch.Tensor,
                             imgpad_mask: Optional[torch.Tensor] = None,
                             *,
                             k: float = -0.25,
                             k_dup: float = 0.0,
                             num_pivots: int = 10,
                             **kwargs) -> PruneResult:
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

    img_emb = embeddings[img_indices].float()
    img_attn = attention_scores[img_indices].float()

    # -------- Stage 1: Important set --------
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k * sigma
    important_mask = img_attn > tau
    if important_mask.sum() == 0:
        important_mask[img_attn.argmax()] = True

    important_idx = important_mask.nonzero(as_tuple=True)[0]  # indices into img subset
    if important_idx.shape[0] <= 1:
        kept_img = img_indices[important_idx]
        all_kept = torch.cat([text_indices, kept_img]).sort().values
        return PruneResult(
            vectors=embeddings[all_kept],
            pruning_ratio=1.0 - (important_idx.shape[0] / n_img),
            num_before=n_img,
            num_after=important_idx.shape[0],
        )

    # -------- Stage 2: Pivot de-duplication --------
    important_attn = img_attn[important_idx]
    important_emb = img_emb[important_idx]

    n_pivots = min(num_pivots, important_idx.shape[0])
    pivot_local = important_attn.topk(n_pivots).indices  # indices into `important_idx`
    pivot_emb = important_emb[pivot_local]  # [n_pivots, D]

    # L2 normalize for cosine
    important_norm = important_emb / important_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    pivot_norm = pivot_emb / pivot_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # max similarity to any pivot, for each patch in important set
    sim_to_pivots = important_norm @ pivot_norm.T  # [n_important, n_pivots]
    max_sim, _ = sim_to_pivots.max(dim=1)

    # Dynamic threshold on similarity
    # Exclude pivots themselves when computing threshold (they'd have sim=1.0)
    non_pivot_mask = torch.ones(important_idx.shape[0], dtype=torch.bool)
    non_pivot_mask[pivot_local] = False

    if non_pivot_mask.sum() > 1:
        non_pivot_sims = max_sim[non_pivot_mask]
        mu_s, sigma_s = _robust_mu_sigma(non_pivot_sims)
        tau_s = mu_s + k_dup * sigma_s

        # Keep pivots always; for non-pivots, keep if similarity <= threshold
        keep_in_important = torch.zeros(important_idx.shape[0], dtype=torch.bool)
        keep_in_important[pivot_local] = True
        non_pivot_keep = max_sim <= tau_s
        keep_in_important = keep_in_important | (non_pivot_keep & non_pivot_mask)
    else:
        keep_in_important = torch.ones(important_idx.shape[0], dtype=torch.bool)

    final_local = important_idx[keep_in_important]
    kept_img = img_indices[final_local]
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    pruned = embeddings[all_kept]

    n_kept = final_local.shape[0]
    return PruneResult(
        vectors=pruned,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )
