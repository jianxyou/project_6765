"""
DocPruner + Deduplication (DP-Dedup).

Observation: DocPruner selects patches purely by attention importance,
ignoring redundancy. High-attention patches that are near-duplicates
(cosine sim > threshold) waste storage budget.

Algorithm:
  1. Use a loose DocPruner threshold (k_loose) to create a larger candidate pool
  2. Sort candidates by attention (descending)
  3. Greedily accept patches: skip if too similar to any already-accepted patch
  4. Result: high-importance, low-redundancy patch set

This achieves better coverage of the document's semantic content at the
same compression ratio, or higher compression with the same quality.
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import _robust_mu_sigma


def dp_dedup_compress(embeddings: torch.Tensor,
                      attention_scores: torch.Tensor,
                      imgpad_mask: Optional[torch.Tensor] = None,
                      *,
                      k_loose: float = -0.5,
                      dedup_threshold: float = 0.95,
                      target_ratio: Optional[float] = None,
                      **kwargs) -> PruneResult:
    """
    DocPruner + Deduplication.

    Args:
        k_loose: DocPruner threshold for candidate selection (looser = more candidates)
        dedup_threshold: cosine similarity threshold for dedup (higher = less aggressive dedup)
        target_ratio: if set, stop accepting after keeping this fraction of image patches
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

    img_attn = attention_scores[img_indices]
    img_emb = embeddings[img_indices].float()

    # Step 1: Loose DocPruner threshold to get candidate pool
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k_loose * sigma

    candidate_mask = img_attn > tau
    if candidate_mask.sum() == 0:
        candidate_mask[img_attn.argmax()] = True

    candidate_local = candidate_mask.nonzero(as_tuple=True)[0]
    cand_attn = img_attn[candidate_local]
    cand_emb = img_emb[candidate_local]

    # Step 2: Sort by attention (descending)
    order = cand_attn.argsort(descending=True)
    candidate_local = candidate_local[order]
    cand_emb = cand_emb[order]

    # L2-normalize for cosine similarity
    cand_normed = cand_emb / cand_emb.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Step 3: Greedy dedup — accept if not too similar to any accepted patch
    n_candidates = candidate_local.shape[0]

    # Optional: compute target count from target_ratio
    max_keep = n_candidates
    if target_ratio is not None:
        max_keep = max(1, int(n_img * target_ratio))

    accepted = []  # indices into candidate_local
    accepted_normed = []  # normalized embeddings of accepted patches

    for i in range(n_candidates):
        if len(accepted) >= max_keep:
            break

        if len(accepted_normed) == 0:
            # First patch always accepted
            accepted.append(i)
            accepted_normed.append(cand_normed[i])
        else:
            # Check similarity to all accepted patches
            accepted_stack = torch.stack(accepted_normed)
            sim = (cand_normed[i] @ accepted_stack.T)
            max_sim = sim.max().item()

            if max_sim < dedup_threshold:
                accepted.append(i)
                accepted_normed.append(cand_normed[i])

    if len(accepted) == 0:
        accepted = [0]

    # Map back to global indices
    accepted_local = candidate_local[torch.tensor(accepted, dtype=torch.long)]
    kept_img = img_indices[accepted_local]

    # Assemble result: text tokens + accepted image patches
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    result = embeddings[all_kept]

    n_kept = len(accepted)
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )
