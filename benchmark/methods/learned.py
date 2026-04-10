"""
Learned Sparse Projection (Direction 3).

Train a linear projection W on merge-tier patches, then select top-k by norm.
Training uses MaxSim distillation loss.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from .base import PruneResult
from .docpruner import _robust_mu_sigma


class LearnedProjection(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _maxsim_score(q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    sim = q @ d.T
    return sim.max(dim=-1).values.sum()


def train_learned_projection(
    doc_emb_list: List[torch.Tensor],
    doc_attn_list: List[torch.Tensor],
    doc_mask_list: List[torch.Tensor],
    q_emb_list: List[torch.Tensor],
    qrels: Dict[str, Dict[str, int]],
    query_ids: List[str],
    corpus_ids: List[str],
    k1: float = 1.0, k2: float = 0.0, top_k_ratio: float = 0.25,
    lr: float = 1e-3, epochs: int = 30, n_neg: int = 4,
    device_str: str = "cpu",
) -> LearnedProjection:
    dim = doc_emb_list[0].shape[1]
    model = LearnedProjection(dim).to(device_str)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cid2idx = {cid: i for i, cid in enumerate(corpus_ids)}
    pairs = []
    for qi, qid in enumerate(query_ids):
        if qid in qrels:
            for cid, rel in qrels[qid].items():
                if rel > 0 and cid in cid2idx:
                    pairs.append((qi, cid2idx[cid]))

    # Precompute tri-level partitions
    partitions = []
    for i in range(len(doc_emb_list)):
        emb = doc_emb_list[i]
        attn = doc_attn_list[i]
        mask = doc_mask_list[i]
        img_idx = mask.nonzero(as_tuple=True)[0]
        txt_idx = (~mask).nonzero(as_tuple=True)[0]
        if img_idx.shape[0] == 0:
            partitions.append((txt_idx, img_idx, torch.tensor([], dtype=torch.long), img_idx))
            continue
        img_attn = attn[img_idx]
        mu, sigma = _robust_mu_sigma(img_attn)
        tau_high = mu + k1 * sigma
        tau_low = mu - k2 * sigma
        preserve = img_idx[img_attn > tau_high]
        merge = img_idx[(img_attn > tau_low) & (img_attn <= tau_high)]
        if preserve.shape[0] == 0 and merge.shape[0] == 0:
            preserve = img_idx[img_attn.argmax().unsqueeze(0)]
        partitions.append((txt_idx, preserve, merge, img_idx))

    n_docs = len(doc_emb_list)
    gen = torch.Generator()
    gen.manual_seed(42)

    print(f"  Training learned projection: {len(pairs)} pairs, {epochs} epochs")
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        for qi, di in pairs:
            q = q_emb_list[qi].float().to(device_str)
            neg_indices = torch.randint(0, n_docs, (n_neg,), generator=gen).tolist()
            doc_indices = [di] + neg_indices

            orig_scores, comp_scores = [], []
            for didx in doc_indices:
                emb = doc_emb_list[didx].float().to(device_str)
                txt_idx, preserve, merge, _ = partitions[didx]

                with torch.no_grad():
                    orig_scores.append(_maxsim_score(q, emb))

                parts = [emb[txt_idx], emb[preserve]]
                if merge.shape[0] > 0:
                    projected = model(emb[merge])
                    n_keep = max(1, int(merge.shape[0] * top_k_ratio))
                    norms = projected.norm(dim=-1)
                    topk_idx = norms.topk(min(n_keep, projected.shape[0])).indices
                    parts.append(projected[topk_idx])
                comp_scores.append(_maxsim_score(q, torch.cat(parts, dim=0)))

            loss = nn.functional.mse_loss(torch.stack(comp_scores),
                                          torch.stack(orig_scores).detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss / max(1, len(pairs)):.6f}")

    return model.cpu()


def learned_compress(embeddings: torch.Tensor,
                     attention_scores: torch.Tensor,
                     imgpad_mask: Optional[torch.Tensor] = None,
                     *,
                     proj_model: Optional[LearnedProjection] = None,
                     k1: float = 1.0, k2: float = 0.0,
                     top_k_ratio: float = 0.25,
                     **kwargs) -> PruneResult:
    n = embeddings.shape[0]
    if n == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0, num_before=0, num_after=0)
    if imgpad_mask is None:
        imgpad_mask = torch.ones(n, dtype=torch.bool)

    text_idx = (~imgpad_mask).nonzero(as_tuple=True)[0]
    img_idx = imgpad_mask.nonzero(as_tuple=True)[0]
    n_img = img_idx.shape[0]
    if n_img == 0:
        return PruneResult(vectors=embeddings, pruning_ratio=0.0, num_before=n, num_after=n)

    img_attn = attention_scores[img_idx]
    mu, sigma = _robust_mu_sigma(img_attn)

    preserve_mask = img_attn > mu + k1 * sigma
    discard_mask = img_attn <= mu - k2 * sigma
    merge_mask = ~preserve_mask & ~discard_mask

    preserve_idx = img_idx[preserve_mask]
    merge_idx = img_idx[merge_mask]

    if preserve_idx.shape[0] == 0 and merge_idx.shape[0] == 0:
        preserve_idx = img_idx[img_attn.argmax().unsqueeze(0)]
        merge_idx = torch.tensor([], dtype=torch.long)

    parts = [embeddings[text_idx], embeddings[preserve_idx]]
    n_merged = 0
    if merge_idx.shape[0] > 0 and proj_model is not None:
        with torch.no_grad():
            projected = proj_model(embeddings[merge_idx].float()).to(embeddings.dtype)
        n_keep = max(1, int(merge_idx.shape[0] * top_k_ratio))
        norms = projected.norm(dim=-1)
        topk = norms.topk(min(n_keep, projected.shape[0])).indices
        parts.append(projected[topk])
        n_merged = topk.shape[0]

    result = torch.cat(parts, dim=0)
    n_after = preserve_idx.shape[0] + n_merged
    return PruneResult(vectors=result,
                       pruning_ratio=1.0 - (n_after / n_img) if n_img > 0 else 0.0,
                       num_before=n_img, num_after=int(n_after))
