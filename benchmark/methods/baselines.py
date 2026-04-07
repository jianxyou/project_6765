import torch
from typing import Optional
from .base import PruneResult


def identity_compress(embeddings: torch.Tensor,
                      attention_scores: Optional[torch.Tensor] = None,
                      imgpad_mask: Optional[torch.Tensor] = None,
                      **kwargs) -> PruneResult:
    n = embeddings.shape[0]
    return PruneResult(vectors=embeddings, pruning_ratio=0.0,
                       num_before=n, num_after=n)


def random_compress(embeddings: torch.Tensor,
                    attention_scores: Optional[torch.Tensor] = None,
                    imgpad_mask: Optional[torch.Tensor] = None,
                    *,
                    ratio: float = 0.5,
                    seed: int = 0,
                    **kwargs) -> PruneResult:
    n = embeddings.shape[0]
    num_keep = max(1, int(n * (1.0 - ratio)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)[:num_keep].sort().values
    pruned = embeddings[perm]
    return PruneResult(vectors=pruned, pruning_ratio=1.0 - (num_keep / n),
                       num_before=n, num_after=num_keep)
