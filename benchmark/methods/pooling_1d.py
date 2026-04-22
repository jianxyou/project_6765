"""
1D-Pooling — faithful reproduction of PTM paper Appendix D.1.2.

Treats patch embeddings as a 1D sequence. Groups consecutive embeddings into
non-overlapping windows of size `merging_factor`. If the total patch count is
not divisible by the factor, the sequence is zero-padded to complete windows.
Each window's embeddings are averaged.

We apply mask-aware averaging so zero-padded positions do not dilute the final
averages (equivalent to treating padded positions as "non-contributing").
"""

import torch
from typing import Optional
from .base import PruneResult


def pool1d_compress(embeddings: torch.Tensor,
                    attention_scores: Optional[torch.Tensor] = None,
                    imgpad_mask: Optional[torch.Tensor] = None,
                    *,
                    merging_factor: int = 4,
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

    if n_img == 0 or merging_factor <= 1:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                           num_before=n, num_after=n)

    img_emb = embeddings[img_indices]  # [n_img, D]

    # Zero-pad to a multiple of merging_factor (paper: "padded with zero vectors")
    pad = (-n_img) % merging_factor
    if pad > 0:
        zero_pad = torch.zeros(pad, img_emb.shape[1], dtype=img_emb.dtype)
        img_emb = torch.cat([img_emb, zero_pad], dim=0)
        # Validity mask: 1 for real patch, 0 for zero-pad
        valid = torch.cat([
            torch.ones(n_img, dtype=torch.float32),
            torch.zeros(pad, dtype=torch.float32),
        ])
    else:
        valid = torch.ones(n_img, dtype=torch.float32)

    n_groups = img_emb.shape[0] // merging_factor
    # Reshape and mask-aware average
    grouped = img_emb.view(n_groups, merging_factor, -1)           # [G, mf, D]
    valid_g = valid.view(n_groups, merging_factor, 1)              # [G, mf, 1]
    # Sum of real patches / count of real patches (avoid dividing by 0)
    summed = (grouped * valid_g.to(grouped.dtype)).sum(dim=1)      # [G, D]
    count = valid_g.sum(dim=1).clamp(min=1.0).to(summed.dtype)     # [G, 1]
    pooled = summed / count                                        # [G, D]

    parts = [embeddings[text_indices], pooled]
    result = torch.cat(parts, dim=0)

    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_groups / n_img),
        num_before=n_img,
        num_after=n_groups,
    )
