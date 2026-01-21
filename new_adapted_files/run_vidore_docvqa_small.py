#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a small ViDoRe benchmark (QA format) with ColPali multi-vector embeddings.

Default dataset:
  - vidore/docvqa_test_subsampled (500 samples)  :contentReference[oaicite:3]{index=3}

Embedding (ColPali engine):
  - processor.process_images / processor.process_queries  :contentReference[oaicite:4]{index=4}

Retrieval scoring:
  - Late interaction MaxSim (sum over query tokens of max dot over doc vectors)

Compressors:
  - identity
  - fixed_kmeans: per-page kmeans -> centroids
  - adaptive_kmeans: compute density proxy -> map to k -> kmeans -> centroids
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_pil(img) -> Image.Image:
    # HF datasets "image" feature usually returns PIL already,
    # but keep it robust.
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


# -------------------------
# Compressors
# -------------------------

@dataclass
class CompressResult:
    vectors: torch.Tensor          # [K, D]
    density_score: float
    target_k: int


class BaseCompressor:
    name: str = "base"
    def compress(self, vectors: torch.Tensor) -> CompressResult:
        raise NotImplementedError


class IdentityCompressor(BaseCompressor):
    name = "identity"
    def compress(self, vectors: torch.Tensor) -> CompressResult:
        k = vectors.shape[0]
        return CompressResult(vectors=vectors, density_score=float("nan"), target_k=k)


class FixedKMeansCompressor(BaseCompressor):
    name = "fixed_kmeans"

    def __init__(self, k: int, seed: int = 0, max_iter: int = 50):
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn not available; install scikit-learn to use kmeans compressors.")
        self.k = int(k)
        self.seed = seed
        self.max_iter = max_iter

    def compress(self, vectors: torch.Tensor) -> CompressResult:
        # vectors: [N, D] on CPU preferred for sklearn
        n, d = vectors.shape
        k = min(self.k, n)
        if k <= 0:
            return CompressResult(vectors=vectors[:0], density_score=float("nan"), target_k=0)
        if k == n:
            return CompressResult(vectors=vectors, density_score=float("nan"), target_k=k)

        x = vectors.detach().cpu().float().numpy()
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.seed,
            batch_size=min(2048, n),
            max_iter=self.max_iter,
            n_init="auto",
        )
        km.fit(x)
        centroids = torch.from_numpy(km.cluster_centers_).to(vectors.dtype)
        return CompressResult(vectors=centroids, density_score=float("nan"), target_k=k)


class AdaptiveKMeansCompressor(BaseCompressor):
    name = "adaptive_kmeans"

    def __init__(
        self,
        k_min: int,
        k_max: int,
        sample_for_density: int = 256,
        seed: int = 0,
        max_iter: int = 50,
    ):
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn not available; install scikit-learn to use kmeans compressors.")
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sample_for_density = int(sample_for_density)
        self.seed = seed
        self.max_iter = max_iter

    @staticmethod
    def _effective_rank_cov(v: torch.Tensor) -> float:
        """
        Cheap "diversity/density" proxy:
          - sample up to M vectors
          - compute covariance C (DxD)
          - effective rank approx: (tr(C)^2) / ||C||_F^2
        Range: ~[1, D]
        """
        # v: [m, d] float
        m, d = v.shape
        if m <= 1:
            return 1.0
        x = v - v.mean(dim=0, keepdim=True)
        # covariance (unbiased not important here)
        C = (x.T @ x) / float(m)
        tr = torch.trace(C).clamp_min(1e-12)
        frob2 = (C * C).sum().clamp_min(1e-12)
        erank = (tr * tr / frob2).item()
        # numerical guard
        return float(max(1.0, min(erank, float(d))))

    def _choose_k(self, density_score: float, d: int) -> int:
        # normalize effective-rank into [0,1]
        t = (density_score - 1.0) / max(1.0, (d - 1.0))
        t = max(0.0, min(1.0, t))
        k = int(round(self.k_min + t * (self.k_max - self.k_min)))
        return max(self.k_min, min(self.k_max, k))

    def compress(self, vectors: torch.Tensor) -> CompressResult:
        n, d = vectors.shape
        if n == 0:
            return CompressResult(vectors=vectors, density_score=1.0, target_k=0)

        # sample for density
        m = min(n, self.sample_for_density)
        if m < n:
            idx = torch.randperm(n)[:m]
            sample = vectors[idx].detach().cpu().float()
        else:
            sample = vectors.detach().cpu().float()

        density = self._effective_rank_cov(sample)  # ~[1, d]
        target_k = min(self._choose_k(density, d), n)
        if target_k <= 0:
            return CompressResult(vectors=vectors[:0], density_score=density, target_k=0)
        if target_k == n:
            return CompressResult(vectors=vectors, density_score=density, target_k=target_k)

        x = vectors.detach().cpu().float().numpy()
        km = MiniBatchKMeans(
            n_clusters=target_k,
            random_state=self.seed,
            batch_size=min(2048, n),
            max_iter=self.max_iter,
            n_init="auto",
        )
        km.fit(x)
        centroids = torch.from_numpy(km.cluster_centers_).to(vectors.dtype)
        return CompressResult(vectors=centroids, density_score=density, target_k=target_k)


def make_compressor(args) -> BaseCompressor:
    if args.compressor == "identity":
        return IdentityCompressor()
    if args.compressor == "fixed_kmeans":
        return FixedKMeansCompressor(k=args.k, seed=args.seed, max_iter=args.kmeans_max_iter)
    if args.compressor == "adaptive_kmeans":
        return AdaptiveKMeansCompressor(
            k_min=args.k_min,
            k_max=args.k_max,
            sample_for_density=args.density_sample,
            seed=args.seed,
            max_iter=args.kmeans_max_iter,
        )
    raise ValueError(f"Unknown compressor: {args.compressor}")


# -------------------------
# Embedding with ColPali Engine
# -------------------------

def load_colpali_engine(model_name: str, device: torch.device, dtype: str):
    """
    colpali_engine usage follows their published example:
      - ColPali.from_pretrained(..., device_map=...)
      - ColPaliProcessor.from_pretrained(...)
      - processor.process_images / process_queries
    :contentReference[oaicite:5]{index=5}
    """
    from colpali_engine.models import ColPali, ColPaliProcessor  # local import

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    # colpali_engine examples sometimes pass "cuda:0"/"mps" into device_map :contentReference[oaicite:6]{index=6}
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=str(device),
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def embed_images(model, processor, images: List[Image.Image], device: torch.device) -> torch.Tensor:
    batch = processor.process_images(images).to(device)
    embs = model(**batch)   # expected shape: [B, N, D]
    return embs


@torch.no_grad()
def embed_queries(model, processor, queries: List[str], device: torch.device) -> torch.Tensor:
    batch = processor.process_queries(queries).to(device)
    embs = model(**batch)   # expected shape: [B, L, D]
    return embs


# -------------------------
# Late Interaction (MaxSim sum) with chunking
# -------------------------

def pad_3d(list_2d: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    list_2d: list of [Li, D]
    returns:
      X: [B, Lmax, D]
      mask: [B, Lmax] (True for valid)
    """
    b = len(list_2d)
    d = list_2d[0].shape[1] if b > 0 else 0
    lmax = max(x.shape[0] for x in list_2d) if b > 0 else 0
    X = torch.full((b, lmax, d), pad_value, dtype=list_2d[0].dtype)
    mask = torch.zeros((b, lmax), dtype=torch.bool)
    for i, x in enumerate(list_2d):
        li = x.shape[0]
        if li == 0:
            continue
        X[i, :li] = x
        mask[i, :li] = True
    return X, mask


def maxsim_scores_chunked(
    Q_list: List[torch.Tensor],     # nq * [Lq, D]
    D_list: List[torch.Tensor],     # nd * [Ld, D]
    device: torch.device,
    batch_q: int = 8,
    batch_d: int = 32,
    topk: int = 10,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Compute topk retrieval for each query using late interaction:
      score(q,d) = sum_i max_j dot(q_i, d_j)

    Chunked to avoid building full nq*nd*Lq*Ld tensor.
    """
    assert topk >= 1
    nq = len(Q_list)
    nd = len(D_list)

    # Move nothing globally to GPU; we move per-chunk.
    all_top_idx: List[List[int]] = []
    all_top_score: List[List[float]] = []

    for qs in tqdm(range(0, nq, batch_q), desc="Scoring (queries)"):
        q_batch_list = Q_list[qs:qs + batch_q]
        Q, qmask = pad_3d(q_batch_list)
        Q = Q.to(device)
        qmask = qmask.to(device)
        bq, Lq, d = Q.shape

        # init topk buffers
        top_scores = torch.full((bq, topk), -1e9, device=device)
        top_indices = torch.full((bq, topk), -1, dtype=torch.long, device=device)

        for ds in range(0, nd, batch_d):
            d_batch_list = D_list[ds:ds + batch_d]
            D, dmask = pad_3d(d_batch_list)
            D = D.to(device)
            dmask = dmask.to(device)
            bd, Ld, _ = D.shape

            # sim: [bq, bd, Lq, Ld]
            # using einsum for dot: (qld)·(dmd) -> qblm
            sim = torch.einsum("qld,bmd->qblm", Q, D)

            # mask invalid doc tokens: set to very negative
            neg_inf = torch.tensor(-1e9, device=device, dtype=sim.dtype)
            sim = sim.masked_fill(~dmask.view(1, bd, 1, Ld), neg_inf)

            # max over doc tokens: [bq, bd, Lq]
            sim_max = sim.max(dim=-1).values

            # mask invalid query tokens: set to 0 contribution
            sim_max = sim_max * qmask.view(bq, 1, Lq).to(sim_max.dtype)

            # sum over query tokens: [bq, bd]
            scores = sim_max.sum(dim=-1)

            # update topk per query in this batch
            # concatenate old topk with new chunk
            cand_scores = torch.cat([top_scores, scores], dim=1)  # [bq, topk+bd]
            cand_indices = torch.cat(
                [top_indices, torch.arange(ds, ds + bd, device=device).view(1, bd).repeat(bq, 1)],
                dim=1,
            )
            new_top_scores, new_pos = cand_scores.topk(k=topk, dim=1)
            new_top_indices = torch.gather(cand_indices, 1, new_pos)

            top_scores, top_indices = new_top_scores, new_top_indices

        all_top_idx.extend(top_indices.detach().cpu().tolist())
        all_top_score.extend(top_scores.detach().cpu().tolist())

    return all_top_idx, all_top_score


# -------------------------
# Metrics for single-relevant QA retrieval
# -------------------------

def recall_at_k(ranks: List[int], k: int) -> float:
    # ranks: rank position (1-based); inf if not found
    hit = sum(1 for r in ranks if r <= k)
    return hit / max(1, len(ranks))


def ndcg_at_k_single_rel(ranks: List[int], k: int) -> float:
    """
    With a single relevant doc per query (gain=1):
      nDCG@k = 1/log2(rank+1) if rank<=k else 0
    """
    out = 0.0
    for r in ranks:
        if r <= k:
            out += 1.0 / math.log2(r + 1.0)
    return out / max(1, len(ranks))


# -------------------------
# Main experiment
# -------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    p.add_argument("--split", default="test")

    # ColPali engine model
    p.add_argument("--model-name", default="vidore/colpali-v1.2-merged")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    # batching
    p.add_argument("--batch-embed-img", type=int, default=4)
    p.add_argument("--batch-embed-q", type=int, default=16)
    p.add_argument("--batch-score-q", type=int, default=8)
    p.add_argument("--batch-score-d", type=int, default=32)

    # subset (debug / small run)
    p.add_argument("--max-samples", type=int, default=200, help="Use first N samples for a quick run (default 200). Set 500 for full dataset.")

    # compressor
    p.add_argument("--compressor", choices=["identity", "fixed_kmeans", "adaptive_kmeans"], default="identity")
    p.add_argument("--k", type=int, default=128, help="fixed_kmeans: target k")
    p.add_argument("--k-min", type=int, default=64, help="adaptive_kmeans: min k")
    p.add_argument("--k-max", type=int, default=256, help="adaptive_kmeans: max k")
    p.add_argument("--density-sample", type=int, default=256, help="adaptive_kmeans: sample size for density proxy")
    p.add_argument("--kmeans-max-iter", type=int, default=50)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--outdir", default="outputs_docvqa_small")

    args = p.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)

    # Load dataset (QA format)
    ds = load_dataset(args.dataset, split=args.split)
    n_total = len(ds)
    n = min(args.max_samples, n_total)
    ds = ds.select(range(n))

    # Schema reference (query/image/etc.) :contentReference[oaicite:7]{index=7}
    queries = [str(ds[i]["query"]) for i in range(n)]
    images = [to_pil(ds[i]["image"]) for i in range(n)]

    # In this QA subsampled setup: each query i's relevant doc is the same row i's image.
    relevant_doc = list(range(n))

    # Load ColPali engine model/processor :contentReference[oaicite:8]{index=8}
    model, processor = load_colpali_engine(args.model_name, device=device, dtype=args.dtype)

    # Embed corpus images
    doc_emb_list: List[torch.Tensor] = []
    for i in tqdm(range(0, n, args.batch_embed_img), desc="Embedding images"):
        batch_imgs = images[i:i + args.batch_embed_img]
        embs = embed_images(model, processor, batch_imgs, device=device)  # [B, Np, D]
        embs = embs.detach().cpu()
        for j in range(embs.shape[0]):
            doc_emb_list.append(embs[j])

    # Embed queries
    q_emb_list: List[torch.Tensor] = []
    for i in tqdm(range(0, n, args.batch_embed_q), desc="Embedding queries"):
        batch_q = queries[i:i + args.batch_embed_q]
        embs = embed_queries(model, processor, batch_q, device=device)    # [B, Lq, D]
        embs = embs.detach().cpu()
        for j in range(embs.shape[0]):
            q_emb_list.append(embs[j])

    assert len(doc_emb_list) == n and len(q_emb_list) == n

    # Compress corpus (per-page) + record stats
    compressor = make_compressor(args)
    compressed_docs: List[torch.Tensor] = []
    density_scores: List[float] = []
    target_ks: List[int] = []

    for i in tqdm(range(n), desc=f"Compressing docs ({compressor.name})"):
        res = compressor.compress(doc_emb_list[i])
        compressed_docs.append(res.vectors)
        density_scores.append(res.density_score)
        target_ks.append(res.target_k)

    # (Optional) You could also compress queries, but for a first benchmark keep queries as-is.
    compressed_queries = q_emb_list

    # Retrieval (topk per query)
    top_idx, top_scores = maxsim_scores_chunked(
        Q_list=compressed_queries,
        D_list=compressed_docs,
        device=device,
        batch_q=args.batch_score_q,
        batch_d=args.batch_score_d,
        topk=args.topk,
    )

    # Convert to ranks
    ranks: List[int] = []
    for qi in range(n):
        gt = relevant_doc[qi]
        retrieved = top_idx[qi]
        if gt in retrieved:
            rank = retrieved.index(gt) + 1  # 1-based
        else:
            rank = 10**9
        ranks.append(rank)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": n,
        "model_name": args.model_name,
        "compressor": compressor.name,
        "topk": args.topk,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "ndcg@5": ndcg_at_k_single_rel(ranks, 5),
        "avg_vectors_before": float(sum(x.shape[0] for x in doc_emb_list) / n),
        "avg_vectors_after": float(sum(x.shape[0] for x in compressed_docs) / n),
        "compression_ratio_vectors": float(
            (sum(x.shape[0] for x in compressed_docs) / max(1, sum(x.shape[0] for x in doc_emb_list)))
        ),
    }

    # Save results
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Save per-doc stats (useful for your adaptive plot / analysis)
    per_doc = [
        {
            "doc_index": i,
            "vectors_before": int(doc_emb_list[i].shape[0]),
            "vectors_after": int(compressed_docs[i].shape[0]),
            "density_score": None if math.isnan(density_scores[i]) else float(density_scores[i]),
            "target_k": int(target_ks[i]),
        }
        for i in range(n)
    ]
    (outdir / "per_doc_stats.json").write_text(json.dumps(per_doc, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run a small ViDoRe benchmark (QA format) with ColPali multi-vector embeddings.

Default dataset:
  - vidore/docvqa_test_subsampled (500 samples)  :contentReference[oaicite:3]{index=3}

Embedding (ColPali engine):
  - processor.process_images / processor.process_queries  :contentReference[oaicite:4]{index=4}

Retrieval scoring:
  - Late interaction MaxSim (sum over query tokens of max dot over doc vectors)

Compressors:
  - identity
  - fixed_kmeans: per-page kmeans -> centroids
  - adaptive_kmeans: compute density proxy -> map to k -> kmeans -> centroids
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None


# -------------------------
# Utils
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_pil(img) -> Image.Image:
    # HF datasets "image" feature usually returns PIL already,
    # but keep it robust.
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


# -------------------------
# Compressors
# -------------------------

@dataclass
class CompressResult:
    vectors: torch.Tensor          # [K, D]
    density_score: float
    target_k: int


class BaseCompressor:
    name: str = "base"
    def compress(self, vectors: torch.Tensor) -> CompressResult:
        raise NotImplementedError


class IdentityCompressor(BaseCompressor):
    name = "identity"
    def compress(self, vectors: torch.Tensor) -> CompressResult:
        k = vectors.shape[0]
        return CompressResult(vectors=vectors, density_score=float("nan"), target_k=k)


class FixedKMeansCompressor(BaseCompressor):
    name = "fixed_kmeans"

    def __init__(self, k: int, seed: int = 0, max_iter: int = 50):
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn not available; install scikit-learn to use kmeans compressors.")
        self.k = int(k)
        self.seed = seed
        self.max_iter = max_iter

    def compress(self, vectors: torch.Tensor) -> CompressResult:
        # vectors: [N, D] on CPU preferred for sklearn
        n, d = vectors.shape
        k = min(self.k, n)
        if k <= 0:
            return CompressResult(vectors=vectors[:0], density_score=float("nan"), target_k=0)
        if k == n:
            return CompressResult(vectors=vectors, density_score=float("nan"), target_k=k)

        x = vectors.detach().cpu().float().numpy()
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.seed,
            batch_size=min(2048, n),
            max_iter=self.max_iter,
            n_init="auto",
        )
        km.fit(x)
        centroids = torch.from_numpy(km.cluster_centers_).to(vectors.dtype)
        return CompressResult(vectors=centroids, density_score=float("nan"), target_k=k)


class AdaptiveKMeansCompressor(BaseCompressor):
    name = "adaptive_kmeans"

    def __init__(
        self,
        k_min: int,
        k_max: int,
        sample_for_density: int = 256,
        seed: int = 0,
        max_iter: int = 50,
    ):
        if MiniBatchKMeans is None:
            raise RuntimeError("scikit-learn not available; install scikit-learn to use kmeans compressors.")
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        self.sample_for_density = int(sample_for_density)
        self.seed = seed
        self.max_iter = max_iter

    @staticmethod
    def _effective_rank_cov(v: torch.Tensor) -> float:
        """
        Cheap "diversity/density" proxy:
          - sample up to M vectors
          - compute covariance C (DxD)
          - effective rank approx: (tr(C)^2) / ||C||_F^2
        Range: ~[1, D]
        """
        # v: [m, d] float
        m, d = v.shape
        if m <= 1:
            return 1.0
        x = v - v.mean(dim=0, keepdim=True)
        # covariance (unbiased not important here)
        C = (x.T @ x) / float(m)
        tr = torch.trace(C).clamp_min(1e-12)
        frob2 = (C * C).sum().clamp_min(1e-12)
        erank = (tr * tr / frob2).item()
        # numerical guard
        return float(max(1.0, min(erank, float(d))))

    def _choose_k(self, density_score: float, d: int) -> int:
        # normalize effective-rank into [0,1]
        t = (density_score - 1.0) / max(1.0, (d - 1.0))
        t = max(0.0, min(1.0, t))
        k = int(round(self.k_min + t * (self.k_max - self.k_min)))
        return max(self.k_min, min(self.k_max, k))

    def compress(self, vectors: torch.Tensor) -> CompressResult:
        n, d = vectors.shape
        if n == 0:
            return CompressResult(vectors=vectors, density_score=1.0, target_k=0)

        # sample for density
        m = min(n, self.sample_for_density)
        if m < n:
            idx = torch.randperm(n)[:m]
            sample = vectors[idx].detach().cpu().float()
        else:
            sample = vectors.detach().cpu().float()

        density = self._effective_rank_cov(sample)  # ~[1, d]
        target_k = min(self._choose_k(density, d), n)
        if target_k <= 0:
            return CompressResult(vectors=vectors[:0], density_score=density, target_k=0)
        if target_k == n:
            return CompressResult(vectors=vectors, density_score=density, target_k=target_k)

        x = vectors.detach().cpu().float().numpy()
        km = MiniBatchKMeans(
            n_clusters=target_k,
            random_state=self.seed,
            batch_size=min(2048, n),
            max_iter=self.max_iter,
            n_init="auto",
        )
        km.fit(x)
        centroids = torch.from_numpy(km.cluster_centers_).to(vectors.dtype)
        return CompressResult(vectors=centroids, density_score=density, target_k=target_k)


def make_compressor(args) -> BaseCompressor:
    if args.compressor == "identity":
        return IdentityCompressor()
    if args.compressor == "fixed_kmeans":
        return FixedKMeansCompressor(k=args.k, seed=args.seed, max_iter=args.kmeans_max_iter)
    if args.compressor == "adaptive_kmeans":
        return AdaptiveKMeansCompressor(
            k_min=args.k_min,
            k_max=args.k_max,
            sample_for_density=args.density_sample,
            seed=args.seed,
            max_iter=args.kmeans_max_iter,
        )
    raise ValueError(f"Unknown compressor: {args.compressor}")


# -------------------------
# Embedding with ColPali Engine
# -------------------------

def load_colpali_engine(model_name: str, device: torch.device, dtype: str):
    """
    colpali_engine usage follows their published example:
      - ColPali.from_pretrained(..., device_map=...)
      - ColPaliProcessor.from_pretrained(...)
      - processor.process_images / process_queries
    :contentReference[oaicite:5]{index=5}
    """
    from colpali_engine.models import ColPali, ColPaliProcessor  # local import

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    # colpali_engine examples sometimes pass "cuda:0"/"mps" into device_map :contentReference[oaicite:6]{index=6}
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=str(device),
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def embed_images(model, processor, images: List[Image.Image], device: torch.device) -> torch.Tensor:
    batch = processor.process_images(images).to(device)
    embs = model(**batch)   # expected shape: [B, N, D]
    return embs


@torch.no_grad()
def embed_queries(model, processor, queries: List[str], device: torch.device) -> torch.Tensor:
    batch = processor.process_queries(queries).to(device)
    embs = model(**batch)   # expected shape: [B, L, D]
    return embs


# -------------------------
# Late Interaction (MaxSim sum) with chunking
# -------------------------

def pad_3d(list_2d: List[torch.Tensor], pad_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    list_2d: list of [Li, D]
    returns:
      X: [B, Lmax, D]
      mask: [B, Lmax] (True for valid)
    """
    b = len(list_2d)
    d = list_2d[0].shape[1] if b > 0 else 0
    lmax = max(x.shape[0] for x in list_2d) if b > 0 else 0
    X = torch.full((b, lmax, d), pad_value, dtype=list_2d[0].dtype)
    mask = torch.zeros((b, lmax), dtype=torch.bool)
    for i, x in enumerate(list_2d):
        li = x.shape[0]
        if li == 0:
            continue
        X[i, :li] = x
        mask[i, :li] = True
    return X, mask


def maxsim_scores_chunked(
    Q_list: List[torch.Tensor],     # nq * [Lq, D]
    D_list: List[torch.Tensor],     # nd * [Ld, D]
    device: torch.device,
    batch_q: int = 8,
    batch_d: int = 32,
    topk: int = 10,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Compute topk retrieval for each query using late interaction:
      score(q,d) = sum_i max_j dot(q_i, d_j)

    Chunked to avoid building full nq*nd*Lq*Ld tensor.
    """
    assert topk >= 1
    nq = len(Q_list)
    nd = len(D_list)

    # Move nothing globally to GPU; we move per-chunk.
    all_top_idx: List[List[int]] = []
    all_top_score: List[List[float]] = []

    for qs in tqdm(range(0, nq, batch_q), desc="Scoring (queries)"):
        q_batch_list = Q_list[qs:qs + batch_q]
        Q, qmask = pad_3d(q_batch_list)
        Q = Q.to(device)
        qmask = qmask.to(device)
        bq, Lq, d = Q.shape

        # init topk buffers
        top_scores = torch.full((bq, topk), -1e9, device=device)
        top_indices = torch.full((bq, topk), -1, dtype=torch.long, device=device)

        for ds in range(0, nd, batch_d):
            d_batch_list = D_list[ds:ds + batch_d]
            D, dmask = pad_3d(d_batch_list)
            D = D.to(device)
            dmask = dmask.to(device)
            bd, Ld, _ = D.shape

            # sim: [bq, bd, Lq, Ld]
            # using einsum for dot: (qld)·(dmd) -> qblm
            sim = torch.einsum("qld,bmd->qblm", Q, D)

            # mask invalid doc tokens: set to very negative
            neg_inf = torch.tensor(-1e9, device=device, dtype=sim.dtype)
            sim = sim.masked_fill(~dmask.view(1, bd, 1, Ld), neg_inf)

            # max over doc tokens: [bq, bd, Lq]
            sim_max = sim.max(dim=-1).values

            # mask invalid query tokens: set to 0 contribution
            sim_max = sim_max * qmask.view(bq, 1, Lq).to(sim_max.dtype)

            # sum over query tokens: [bq, bd]
            scores = sim_max.sum(dim=-1)

            # update topk per query in this batch
            # concatenate old topk with new chunk
            cand_scores = torch.cat([top_scores, scores], dim=1)  # [bq, topk+bd]
            cand_indices = torch.cat(
                [top_indices, torch.arange(ds, ds + bd, device=device).view(1, bd).repeat(bq, 1)],
                dim=1,
            )
            new_top_scores, new_pos = cand_scores.topk(k=topk, dim=1)
            new_top_indices = torch.gather(cand_indices, 1, new_pos)

            top_scores, top_indices = new_top_scores, new_top_indices

        all_top_idx.extend(top_indices.detach().cpu().tolist())
        all_top_score.extend(top_scores.detach().cpu().tolist())

    return all_top_idx, all_top_score


# -------------------------
# Metrics for single-relevant QA retrieval
# -------------------------

def recall_at_k(ranks: List[int], k: int) -> float:
    # ranks: rank position (1-based); inf if not found
    hit = sum(1 for r in ranks if r <= k)
    return hit / max(1, len(ranks))


def ndcg_at_k_single_rel(ranks: List[int], k: int) -> float:
    """
    With a single relevant doc per query (gain=1):
      nDCG@k = 1/log2(rank+1) if rank<=k else 0
    """
    out = 0.0
    for r in ranks:
        if r <= k:
            out += 1.0 / math.log2(r + 1.0)
    return out / max(1, len(ranks))


# -------------------------
# Main experiment
# -------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    p.add_argument("--split", default="test")

    # ColPali engine model
    p.add_argument("--model-name", default="vidore/colpali-v1.2-merged")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")

    # batching
    p.add_argument("--batch-embed-img", type=int, default=4)
    p.add_argument("--batch-embed-q", type=int, default=16)
    p.add_argument("--batch-score-q", type=int, default=8)
    p.add_argument("--batch-score-d", type=int, default=32)

    # subset (debug / small run)
    p.add_argument("--max-samples", type=int, default=200, help="Use first N samples for a quick run (default 200). Set 500 for full dataset.")

    # compressor
    p.add_argument("--compressor", choices=["identity", "fixed_kmeans", "adaptive_kmeans"], default="identity")
    p.add_argument("--k", type=int, default=128, help="fixed_kmeans: target k")
    p.add_argument("--k-min", type=int, default=64, help="adaptive_kmeans: min k")
    p.add_argument("--k-max", type=int, default=256, help="adaptive_kmeans: max k")
    p.add_argument("--density-sample", type=int, default=256, help="adaptive_kmeans: sample size for density proxy")
    p.add_argument("--kmeans-max-iter", type=int, default=50)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--outdir", default="outputs_docvqa_small")

    args = p.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)

    # Load dataset (QA format)
    ds = load_dataset(args.dataset, split=args.split)
    n_total = len(ds)
    n = min(args.max_samples, n_total)
    ds = ds.select(range(n))

    # Schema reference (query/image/etc.) :contentReference[oaicite:7]{index=7}
    queries = [str(ds[i]["query"]) for i in range(n)]
    images = [to_pil(ds[i]["image"]) for i in range(n)]

    # In this QA subsampled setup: each query i's relevant doc is the same row i's image.
    relevant_doc = list(range(n))

    # Load ColPali engine model/processor :contentReference[oaicite:8]{index=8}
    model, processor = load_colpali_engine(args.model_name, device=device, dtype=args.dtype)

    # Embed corpus images
    doc_emb_list: List[torch.Tensor] = []
    for i in tqdm(range(0, n, args.batch_embed_img), desc="Embedding images"):
        batch_imgs = images[i:i + args.batch_embed_img]
        embs = embed_images(model, processor, batch_imgs, device=device)  # [B, Np, D]
        embs = embs.detach().cpu()
        for j in range(embs.shape[0]):
            doc_emb_list.append(embs[j])

    # Embed queries
    q_emb_list: List[torch.Tensor] = []
    for i in tqdm(range(0, n, args.batch_embed_q), desc="Embedding queries"):
        batch_q = queries[i:i + args.batch_embed_q]
        embs = embed_queries(model, processor, batch_q, device=device)    # [B, Lq, D]
        embs = embs.detach().cpu()
        for j in range(embs.shape[0]):
            q_emb_list.append(embs[j])

    assert len(doc_emb_list) == n and len(q_emb_list) == n

    # Compress corpus (per-page) + record stats
    compressor = make_compressor(args)
    compressed_docs: List[torch.Tensor] = []
    density_scores: List[float] = []
    target_ks: List[int] = []

    for i in tqdm(range(n), desc=f"Compressing docs ({compressor.name})"):
        res = compressor.compress(doc_emb_list[i])
        compressed_docs.append(res.vectors)
        density_scores.append(res.density_score)
        target_ks.append(res.target_k)

    # (Optional) You could also compress queries, but for a first benchmark keep queries as-is.
    compressed_queries = q_emb_list

    # Retrieval (topk per query)
    top_idx, top_scores = maxsim_scores_chunked(
        Q_list=compressed_queries,
        D_list=compressed_docs,
        device=device,
        batch_q=args.batch_score_q,
        batch_d=args.batch_score_d,
        topk=args.topk,
    )

    # Convert to ranks
    ranks: List[int] = []
    for qi in range(n):
        gt = relevant_doc[qi]
        retrieved = top_idx[qi]
        if gt in retrieved:
            rank = retrieved.index(gt) + 1  # 1-based
        else:
            rank = 10**9
        ranks.append(rank)

    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": n,
        "model_name": args.model_name,
        "compressor": compressor.name,
        "topk": args.topk,
        "recall@1": recall_at_k(ranks, 1),
        "recall@5": recall_at_k(ranks, 5),
        "recall@10": recall_at_k(ranks, 10),
        "ndcg@5": ndcg_at_k_single_rel(ranks, 5),
        "avg_vectors_before": float(sum(x.shape[0] for x in doc_emb_list) / n),
        "avg_vectors_after": float(sum(x.shape[0] for x in compressed_docs) / n),
        "compression_ratio_vectors": float(
            (sum(x.shape[0] for x in compressed_docs) / max(1, sum(x.shape[0] for x in doc_emb_list)))
        ),
    }

    # Save results
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Save per-doc stats (useful for your adaptive plot / analysis)
    per_doc = [
        {
            "doc_index": i,
            "vectors_before": int(doc_emb_list[i].shape[0]),
            "vectors_after": int(compressed_docs[i].shape[0]),
            "density_score": None if math.isnan(density_scores[i]) else float(density_scores[i]),
            "target_k": int(target_ks[i]),
        }
        for i in range(n)
    ]
    (outdir / "per_doc_stats.json").write_text(json.dumps(per_doc, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
