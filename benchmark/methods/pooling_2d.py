"""
2D Spatial Pooling (Ma et al. 2025).

Arranges image patches into a 2D grid and applies 2D avg pooling with
kernel = sqrt(merging_factor). Requires knowing the (H, W) of the patch
grid for each document.

If grid_hw is not provided, falls back to assuming a square grid
(sqrt(n_img), sqrt(n_img)). For ColQwen2.5 dynamic resolution, pass
grid_hw extracted from the processor's image_grid_thw.

`merging_factor` must be a perfect square (4, 9, 16, 25, ...).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .base import PruneResult


def pool2d_compress(embeddings: torch.Tensor,
                    attention_scores: Optional[torch.Tensor] = None,
                    imgpad_mask: Optional[torch.Tensor] = None,
                    *,
                    merging_factor: int = 4,
                    grid_hw: Optional[Tuple[int, int]] = None,
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

    # merging_factor must be a perfect square
    k = int(round(math.sqrt(merging_factor)))
    if k * k != merging_factor:
        raise ValueError(f"merging_factor must be a perfect square, got {merging_factor}")

    # Determine grid shape
    if grid_hw is not None:
        H, W = grid_hw
        if H * W != n_img:
            # Mismatch: fall back to closest factorization
            grid_hw = None

    if grid_hw is None:
        # Fallback: assume square grid. May drop a few patches if not square.
        side = int(round(math.sqrt(n_img)))
        H = W = side
        if H * W > n_img:
            H = W = side - 1
        # Truncate to H*W
        n_img_used = H * W
        img_indices_used = img_indices[:n_img_used]
    else:
        n_img_used = n_img
        img_indices_used = img_indices

    img_emb = embeddings[img_indices_used].float()  # [H*W, D]
    D = img_emb.shape[-1]

    # Reshape to [1, D, H, W] for avg_pool2d
    grid = img_emb.view(H, W, D).permute(2, 0, 1).unsqueeze(0)

    # Pad so H and W are divisible by k (use ceil behavior)
    pad_h = (k - H % k) % k
    pad_w = (k - W % k) % k
    if pad_h > 0 or pad_w > 0:
        grid = F.pad(grid, (0, pad_w, 0, pad_h), mode="replicate")

    H_p = H + pad_h
    W_p = W + pad_w
    pooled = F.avg_pool2d(grid, kernel_size=k, stride=k)  # [1, D, H/k, W/k]
    H_out = H_p // k
    W_out = W_p // k

    # Flatten back to [N_out, D]
    pooled = pooled.squeeze(0).permute(1, 2, 0).reshape(H_out * W_out, D)
    pooled = pooled.to(embeddings.dtype)

    parts = [embeddings[text_indices], pooled]
    result = torch.cat(parts, dim=0)

    n_after = pooled.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_after / n_img),
        num_before=n_img,
        num_after=n_after,
    )
