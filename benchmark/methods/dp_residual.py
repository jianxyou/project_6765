"""
DocPruner + Residual Injection.

Uses DocPruner's adaptive threshold to select patches, then injects
information from discarded patches back into their nearest kept neighbors.

For each discarded patch d_j:
  1. Find the kept patch d_k with highest cosine similarity
  2. Add a weighted residual: d_k' = d_k + beta * attn(d_j) * (d_j - d_k)

This preserves DocPruner's compression ratio and selection logic,
but enriches kept patches with information that would otherwise be lost.
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import _robust_mu_sigma


def dp_residual_compress(embeddings: torch.Tensor,
                         attention_scores: torch.Tensor,
                         imgpad_mask: Optional[torch.Tensor] = None,
                         *,
                         k: float = -0.25,
                         beta: float = 0.1,
                         **kwargs) -> PruneResult:
    """
    DocPruner + Residual Injection.

    Args:
        k: DocPruner threshold parameter (same semantics as docpruner)
        beta: residual injection strength (0 = pure DocPruner, higher = more injection)
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

    if n_img == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=n, num_after=n)

    img_attn = attention_scores[img_indices]
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k * sigma

    keep_mask = img_attn > tau
    if keep_mask.sum() == 0:
        keep_mask[img_attn.argmax()] = True

    discard_mask = ~keep_mask
    kept_local = keep_mask.nonzero(as_tuple=True)[0]
    discarded_local = discard_mask.nonzero(as_tuple=True)[0]

    # Start with kept patch embeddings (copy to avoid modifying originals)
    kept_emb = embeddings[img_indices[kept_local]].clone().float()

    # Inject residuals from discarded patches
    if discarded_local.shape[0] > 0 and kept_local.shape[0] > 0 and beta > 0:
        disc_emb = embeddings[img_indices[discarded_local]].float()
        disc_attn = img_attn[discarded_local].float()

        # Normalize attention of discarded patches to [0, 1]
        attn_max = disc_attn.max()
        attn_min = disc_attn.min()
        if attn_max > attn_min:
            disc_attn_norm = (disc_attn - attn_min) / (attn_max - attn_min)
        else:
            disc_attn_norm = torch.ones_like(disc_attn)

        # Find nearest kept patch for each discarded patch (cosine similarity)
        kept_normed = kept_emb / kept_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
        disc_normed = disc_emb / disc_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # [n_discarded, n_kept] similarity matrix
        sim = disc_normed @ kept_normed.T
        nearest = sim.argmax(dim=1)  # index into kept_local for each discarded

        # Inject: d_k' += beta * attn_norm(d_j) * (d_j - d_k)
        for i in range(discarded_local.shape[0]):
            ki = nearest[i].item()
            weight = beta * disc_attn_norm[i].item()
            kept_emb[ki] += weight * (disc_emb[i] - kept_emb[ki])

    # Cast back to original dtype
    kept_emb = kept_emb.to(embeddings.dtype)

    # Assemble: text tokens + modified kept image patches
    text_emb = embeddings[text_indices]
    result = torch.cat([text_emb, kept_emb], dim=0)

    n_kept = kept_local.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )
