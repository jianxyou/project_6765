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
    """Random baseline per PTM paper Appendix D.2.1.

    Discards a specified fraction of IMAGE patch embeddings uniformly at random.
    Text tokens are always kept (consistent with all other adaptive baselines).
    Keeps at least one image patch.
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

    num_keep = max(1, int(n_img * (1.0 - ratio)))
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n_img, generator=gen)[:num_keep].sort().values
    kept_img = img_indices[perm]

    all_kept = torch.cat([text_indices, kept_img]).sort().values
    pruned = embeddings[all_kept]

    return PruneResult(vectors=pruned,
                       pruning_ratio=1.0 - (num_keep / n_img),
                       num_before=n_img, num_after=num_keep)
