"""
DocPruner (Section 3.2): EOS attention adaptive threshold pruning.

Formula:
  tau = mu + k * sigma   (over image_pad attention scores, IQR-clipped)
  keep patches with attention > tau
  at least keep one patch
"""

import torch
from typing import Optional
from .base import PruneResult


def _robust_mu_sigma(img_attn: torch.Tensor):
    """IQR-clipped mean and std to handle attention sinks."""
    q75 = img_attn.quantile(0.75).item()
    q25 = img_attn.quantile(0.25).item()
    iqr = q75 - q25
    clipped = img_attn.clamp(max=q75 + 3.0 * iqr)
    mu = clipped.mean().item()
    sigma = clipped.std().item() if img_attn.shape[0] > 1 else 0.0
    return mu, sigma


def docpruner_compress(embeddings: torch.Tensor,
                       attention_scores: torch.Tensor,
                       imgpad_mask: Optional[torch.Tensor] = None,
                       *,
                       k: float = -0.25,
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

    img_attn = attention_scores[img_indices]
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k * sigma

    keep_mask = img_attn > tau
    if keep_mask.sum() == 0:
        keep_mask[img_attn.argmax()] = True

    kept_img = img_indices[keep_mask]
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    pruned = embeddings[all_kept]

    n_kept = kept_img.shape[0]
    kept_local = keep_mask.nonzero(as_tuple=True)[0]  # indices into image_pad subset
    return PruneResult(
        vectors=pruned,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
        kept_indices=kept_local,
        cluster_labels=torch.arange(n_kept, dtype=torch.long),
        representative_mask=torch.ones(n_kept, dtype=torch.bool),
    )
