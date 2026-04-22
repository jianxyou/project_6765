"""
Attention-plus-Similarity (Wen et al. 2025, via DocPruner baselines).

Adaptive pruning using a weighted combination of:
  - Importance: EOS attention score (like DocPruner)
  - Representativeness: cosine similarity to the EOS token's embedding

Composite score for each patch:
  score(d_j) = alpha * attention(d_j) + (1 - alpha) * cos_sim(d_j, e_eos)

Then prune with adaptive threshold tau = mu + k * sigma on composite scores
(both attention and similarity are normalized to [0, 1] before combining so
alpha is meaningful).
"""

import torch
from typing import Optional
from .base import PruneResult


def _robust_mu_sigma(scores: torch.Tensor):
    """IQR-clipped mean/std for robust threshold (handles attention-sink outliers)."""
    q75 = scores.quantile(0.75).item()
    q25 = scores.quantile(0.25).item()
    iqr = q75 - q25
    clipped = scores.clamp(max=q75 + 3.0 * iqr)
    mu = clipped.mean().item()
    sigma = clipped.std().item() if scores.shape[0] > 1 else 0.0
    return mu, sigma


def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
    lo = x.min()
    hi = x.max()
    rng = (hi - lo).clamp(min=1e-8)
    return (x - lo) / rng


def attn_similarity_compress(embeddings: torch.Tensor,
                             attention_scores: torch.Tensor,
                             imgpad_mask: Optional[torch.Tensor] = None,
                             *,
                             k: float = -0.25,
                             alpha: float = 0.5,
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

    # EOS token is the last position in the sequence
    eos_emb = embeddings[-1].float().unsqueeze(0)

    # Representativeness: cosine similarity to EOS embedding
    eos_sim = torch.nn.functional.cosine_similarity(img_emb, eos_emb)

    # Normalize each signal to [0, 1] then combine
    attn_n = _minmax_norm(img_attn)
    sim_n = _minmax_norm(eos_sim)
    score = alpha * attn_n + (1.0 - alpha) * sim_n

    mu, sigma = _robust_mu_sigma(score)
    tau = mu + k * sigma

    keep_mask = score > tau
    if keep_mask.sum() == 0:
        keep_mask[score.argmax()] = True

    kept_img = img_indices[keep_mask]
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    pruned = embeddings[all_kept]

    n_kept = kept_img.shape[0]
    return PruneResult(
        vectors=pruned,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )
