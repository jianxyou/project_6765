"""
Coverage-Preserving Selection (CPS).

Three-step pipeline:
  1. Semantic Clustering (agglomerative)
  2. Representative Selection (attention-based or centrality-based)
  3. Redundancy Elimination (cosine dedup + optional dominance check)

Three variants:
  - cps_attn:    attention-based selection + dedup
  - cps_central: centrality-based selection + dedup
  - cps_domin:   attention selection + dedup + dominance check
"""

import torch
import numpy as np
from typing import Optional
from .base import PruneResult


# ---- Step 1: Clustering ----

def _cluster_patches(embeddings: torch.Tensor, n_clusters: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    n = embeddings.shape[0]
    if n <= n_clusters:
        return np.arange(n)

    emb_np = embeddings.float().numpy()
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average",
    )
    return clustering.fit_predict(emb_np)


# ---- Step 2: Selection ----

def _select_by_attention(embeddings: torch.Tensor, attention_scores: torch.Tensor,
                         labels: np.ndarray, n_clusters: int) -> torch.Tensor:
    selected = []
    labels_t = torch.tensor(labels, dtype=torch.long)
    for c in range(n_clusters):
        mask = (labels_t == c)
        if mask.sum() == 0:
            continue
        indices = mask.nonzero(as_tuple=True)[0]
        best = attention_scores[indices].argmax()
        selected.append(indices[best].item())
    if not selected:
        selected = [attention_scores.argmax().item()]
    return torch.tensor(selected, dtype=torch.long)


def _select_by_centrality(embeddings: torch.Tensor, labels: np.ndarray,
                          n_clusters: int) -> torch.Tensor:
    selected = []
    labels_t = torch.tensor(labels, dtype=torch.long)
    for c in range(n_clusters):
        mask = (labels_t == c)
        if mask.sum() == 0:
            continue
        indices = mask.nonzero(as_tuple=True)[0]
        cluster_emb = embeddings[indices].float()
        centroid = cluster_emb.mean(dim=0, keepdim=True)
        cos_sim = torch.nn.functional.cosine_similarity(cluster_emb, centroid)
        best = cos_sim.argmax()
        selected.append(indices[best].item())
    if not selected:
        selected = [0]
    return torch.tensor(selected, dtype=torch.long)


# ---- Step 3: Redundancy Elimination ----

def _eliminate_redundancy(embeddings: torch.Tensor, selected: torch.Tensor,
                          attention_scores: torch.Tensor,
                          threshold: float = 0.95) -> torch.Tensor:
    if selected.shape[0] <= 1:
        return selected

    sel_emb = embeddings[selected].float()
    sel_attn = attention_scores[selected]
    order = sel_attn.argsort(descending=True)

    kept = [order[0].item()]
    for i in range(1, order.shape[0]):
        idx = order[i].item()
        candidate = sel_emb[idx]
        kept_emb = sel_emb[torch.tensor(kept)]
        cos_sim = torch.nn.functional.cosine_similarity(
            candidate.unsqueeze(0), kept_emb
        )
        if cos_sim.max() < threshold:
            kept.append(idx)

    return selected[torch.tensor(kept, dtype=torch.long)]


def _dominance_check(embeddings: torch.Tensor, selected: torch.Tensor,
                     svd_energy: float = 0.9) -> torch.Tensor:
    if selected.shape[0] <= 2:
        return selected

    sel_emb = embeddings[selected].float()
    n = sel_emb.shape[0]

    U, S, Vh = torch.linalg.svd(sel_emb.T, full_matrices=False)
    cumsum = S.cumsum(0) / S.sum()
    k = max(2, (cumsum < svd_energy).sum().item() + 1)
    k = min(k, S.shape[0])

    reduced = sel_emb @ U[:, :k]
    sim = reduced @ reduced.T
    self_scores = sim.diag()
    max_others = sim.clone()
    max_others.fill_diagonal_(-float('inf'))
    max_other_scores = max_others.max(dim=1).values

    is_essential = self_scores >= max_other_scores
    if is_essential.sum() < n // 2:
        topk = max(n // 2, 1)
        _, topk_idx = self_scores.topk(topk)
        is_essential = torch.zeros(n, dtype=torch.bool)
        is_essential[topk_idx] = True

    return selected[is_essential]


# ---- Main function ----

def _cps_core(embeddings: torch.Tensor,
              attention_scores: torch.Tensor,
              imgpad_mask: Optional[torch.Tensor],
              cluster_ratio: float,
              dedup_threshold: float,
              selection: str,
              use_dominance: bool,
              svd_energy: float) -> PruneResult:
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

    img_emb = embeddings[img_indices]
    img_attn = attention_scores[img_indices]

    n_clusters = max(2, min(int(n_img * cluster_ratio), n_img))
    labels = _cluster_patches(img_emb, n_clusters)
    real_n_clusters = len(set(labels))

    if selection == "attention":
        selected = _select_by_attention(img_emb, img_attn, labels, real_n_clusters)
    else:
        selected = _select_by_centrality(img_emb, labels, real_n_clusters)

    if dedup_threshold < 1.0:
        selected = _eliminate_redundancy(img_emb, selected, img_attn, dedup_threshold)

    if use_dominance:
        selected = _dominance_check(img_emb, selected, svd_energy)

    if selected.shape[0] == 0:
        selected = torch.tensor([img_attn.argmax().item()], dtype=torch.long)

    kept_img = img_indices[selected]
    all_kept = torch.cat([text_indices, kept_img]).sort().values
    result = embeddings[all_kept]

    n_kept = selected.shape[0]
    return PruneResult(
        vectors=result,
        pruning_ratio=1.0 - (n_kept / n_img),
        num_before=n_img,
        num_after=n_kept,
    )


# ---- Public API ----

def cps_attn_compress(embeddings: torch.Tensor,
                      attention_scores: torch.Tensor,
                      imgpad_mask: Optional[torch.Tensor] = None,
                      *, cluster_ratio: float = 0.3,
                      dedup_threshold: float = 0.95,
                      **kwargs) -> PruneResult:
    return _cps_core(embeddings, attention_scores, imgpad_mask,
                     cluster_ratio, dedup_threshold,
                     selection="attention", use_dominance=False, svd_energy=0.9)


def cps_central_compress(embeddings: torch.Tensor,
                         attention_scores: torch.Tensor,
                         imgpad_mask: Optional[torch.Tensor] = None,
                         *, cluster_ratio: float = 0.3,
                         dedup_threshold: float = 0.95,
                         **kwargs) -> PruneResult:
    return _cps_core(embeddings, attention_scores, imgpad_mask,
                     cluster_ratio, dedup_threshold,
                     selection="centrality", use_dominance=False, svd_energy=0.9)


def cps_dominance_compress(embeddings: torch.Tensor,
                           attention_scores: torch.Tensor,
                           imgpad_mask: Optional[torch.Tensor] = None,
                           *, cluster_ratio: float = 0.4,
                           dedup_threshold: float = 0.95,
                           svd_energy: float = 0.9,
                           **kwargs) -> PruneResult:
    return _cps_core(embeddings, attention_scores, imgpad_mask,
                     cluster_ratio, dedup_threshold,
                     selection="attention", use_dominance=True, svd_energy=svd_energy)
