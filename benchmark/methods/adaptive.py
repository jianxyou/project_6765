"""
Adaptive Hybrid (Direction 1): per-document routing based on attention entropy.

Low entropy  -> DocPruner (concentrated attention, clear pruning signal)
High entropy -> DocMerger (diffuse attention, better to merge)
"""

import torch
from typing import List, Optional
from .base import PruneResult
from .docpruner import docpruner_compress
from .docmerger import docmerger_compress


def attention_entropy(attention_scores: torch.Tensor,
                      imgpad_mask: Optional[torch.Tensor] = None) -> float:
    if imgpad_mask is not None:
        attn = attention_scores[imgpad_mask]
    else:
        attn = attention_scores
    if attn.numel() == 0:
        return 0.0
    attn = attn.float().clamp(min=0)
    total = attn.sum()
    if total == 0:
        return 0.0
    p = attn / total
    return -(p * torch.log(p + 1e-10)).sum().item()


def compute_entropy_threshold(doc_attn_list: List[torch.Tensor],
                              doc_mask_list: List[torch.Tensor],
                              percentile: float = 50.0) -> tuple:
    """Returns (entropies, threshold)."""
    entropies = []
    for i in range(len(doc_attn_list)):
        mask = doc_mask_list[i] if i < len(doc_mask_list) else None
        entropies.append(attention_entropy(doc_attn_list[i], mask))
    threshold = torch.quantile(torch.tensor(entropies), percentile / 100.0).item()
    return entropies, threshold


def adaptive_compress(embeddings: torch.Tensor,
                      attention_scores: torch.Tensor,
                      imgpad_mask: Optional[torch.Tensor] = None,
                      *,
                      entropy: float = 0.0,
                      entropy_threshold: float = 0.0,
                      k_prune: float = -0.25,
                      k1: float = 1.0, k2: float = 0.0,
                      merge_ratio: float = 0.25,
                      **kwargs) -> PruneResult:
    if entropy < entropy_threshold:
        return docpruner_compress(embeddings, attention_scores, imgpad_mask,
                                  k=k_prune)
    else:
        return docmerger_compress(embeddings, attention_scores, imgpad_mask,
                                  k1=k1, k2=k2, merge_ratio=merge_ratio)
