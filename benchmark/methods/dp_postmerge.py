"""
DocPruner + Post-hoc Merge (DP-PostMerge).

DocPruner's patch selection is unchanged. After pruning, find near-duplicate
patches among the *kept* set (cosine sim > threshold) and merge them via
attention-weighted average. This reduces patch count further without changing
which patches were originally selected.

Key difference from DP-Dedup:
  - DP-Dedup changes *which* patches are selected (greedy dedup during selection)
  - DP-PostMerge keeps DocPruner's selection intact, only merges afterwards

Claim: same nDCG@5 as DocPruner, but 5-10% extra compression for free.
"""

import torch
from typing import Optional
from .base import PruneResult
from .docpruner import docpruner_compress


def dp_postmerge_compress(embeddings: torch.Tensor,
                          attention_scores: torch.Tensor,
                          imgpad_mask: Optional[torch.Tensor] = None,
                          *,
                          k: float = -0.25,
                          merge_threshold: float = 0.95,
                          **kwargs) -> PruneResult:
    """
    DocPruner followed by post-hoc merging of near-duplicate kept patches.

    Steps:
      1. Run DocPruner(k) as normal -> get pruned embeddings
      2. Among the kept image patches, find pairs with cosine sim > threshold
      3. Merge them via attention-weighted average (greedy: highest-attn patch
         absorbs its near-duplicates)
      4. Return merged result

    Args:
        k: DocPruner adaptation factor
        merge_threshold: cosine similarity threshold for merging
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

    # Step 1: Run standard DocPruner
    dp_result = docpruner_compress(embeddings, attention_scores, imgpad_mask, k=k)

    # Separate text and image patches from DocPruner's output
    # We need to identify which vectors in dp_result are image patches
    # Re-derive: which image patches survived DocPruner?
    img_attn = attention_scores[img_indices]
    from .docpruner import _robust_mu_sigma
    mu, sigma = _robust_mu_sigma(img_attn)
    tau = mu + k * sigma

    keep_mask = img_attn > tau
    if keep_mask.sum() == 0:
        keep_mask[img_attn.argmax()] = True

    kept_img_indices = img_indices[keep_mask]
    n_kept = kept_img_indices.shape[0]

    if n_kept <= 1:
        return dp_result

    # Step 2: Find near-duplicates among kept image patches
    kept_emb = embeddings[kept_img_indices]
    kept_attn = attention_scores[kept_img_indices]
    orig_dtype = kept_emb.dtype

    # L2 normalize for cosine similarity (in float32 for precision)
    kept_float = kept_emb.float()
    norms = kept_float.norm(dim=1, keepdim=True).clamp(min=1e-8)
    kept_normed = kept_float / norms

    # Step 3: Greedy merge — process patches in descending attention order
    # Higher-attention patch absorbs its near-duplicates
    order = kept_attn.argsort(descending=True)

    merged_vectors = []
    merged = torch.zeros(n_kept, dtype=torch.bool)  # track which patches are consumed
    labels = torch.full((n_kept,), -1, dtype=torch.long)  # cluster id per kept patch
    is_rep = torch.zeros(n_kept, dtype=torch.bool)  # True = anchor of its cluster
    cluster_id = 0

    for i in range(n_kept):
        idx = order[i].item()
        if merged[idx]:
            continue

        # Find all unmerged patches similar to this one
        sim = kept_normed[idx] @ kept_normed.T
        similar_mask = (sim > merge_threshold) & (~merged)
        similar_indices = similar_mask.nonzero(as_tuple=True)[0]

        if similar_indices.shape[0] == 1:
            # No duplicates, keep as-is
            merged_vectors.append(kept_emb[idx])
            merged[idx] = True
            labels[idx] = cluster_id
            is_rep[idx] = True
        else:
            # Merge via attention-weighted average; anchor = idx (highest attention)
            group_emb = kept_emb[similar_indices].float()
            group_attn = kept_attn[similar_indices].float()
            weights = group_attn / group_attn.sum()
            merged_vec = (group_emb * weights.unsqueeze(1)).sum(dim=0)
            merged_vectors.append(merged_vec.to(orig_dtype))
            merged[similar_indices] = True
            labels[similar_indices] = cluster_id
            is_rep[idx] = True  # anchor only
        cluster_id += 1

    # Step 4: Assemble result
    merged_img = torch.stack(merged_vectors)
    text_emb = embeddings[text_indices]
    result = torch.cat([text_emb, merged_img], dim=0)

    # kept_indices are in image_pad-subset coordinates (position of each kept patch
    # among all image_pad patches, in original order)
    kept_local = keep_mask.nonzero(as_tuple=True)[0]

    n_after = merged_img.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_after / n_img),
        num_before=n_img,
        num_after=n_after,
        kept_indices=kept_local,
        cluster_labels=labels,
        representative_mask=is_rep,
    )
