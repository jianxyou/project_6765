#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import List

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def pick_device(device: str | None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    return Image.fromarray(x)


@torch.no_grad()
def embed_images(model, processor, images: List[Image.Image], device):
    batch = processor.process_images(images).to(device)
    embs = model(**batch)  # [B, Npatch, D]
    return embs 


@torch.no_grad()
def embed_queries(model, processor, queries: List[str], device):
    batch = processor.process_queries(queries).to(device)
    embs = model(**batch)  # [B, Lq, D]
    return embs


def mean_pool(mv: torch.Tensor) -> torch.Tensor:
    """
    mv: [L, D] multi-vector -> [D] single vector
    """
    if mv.numel() == 0:
        return torch.zeros((mv.shape[-1],), dtype=mv.dtype)
    return mv.mean(dim=0)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2) + eps)


def recall_at_k(ranks: List[int], k: int) -> float:
    hit = sum(1 for r in ranks if r <= k)
    return hit / max(1, len(ranks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vidore/docvqa_test_subsampled")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=30)
    ap.add_argument("--model-name", default="vidore/colpali-v1.2-merged")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--batch-img", type=int, default=4)
    ap.add_argument("--batch-q", type=int, default=16)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = pick_device(args.device)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # 1) load small subset
    ds = load_dataset(args.dataset, split=args.split)
    n = min(args.max_samples, len(ds))
    ds = ds.select(range(n))
    queries = [str(ds[i]["query"]) for i in range(n)]
    images = [to_pil(ds[i]["image"]) for i in range(n)]

    # 2) load ColPali model/processor
    from colpali_engine.models import ColPali, ColPaliProcessor
    model = ColPali.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map=str(device),
    ).eval()
    processor = ColPaliProcessor.from_pretrained(args.model_name)

    # 3) embed docs (images) -> multi-vector
    doc_mvs: List[torch.Tensor] = []
    for i in tqdm(range(0, n, args.batch_img), desc="Embed images"):
        batch = images[i:i + args.batch_img]
        mv = embed_images(model, processor, batch, device).detach().cpu()  # [B, Np, D]
        for j in range(mv.shape[0]):
            doc_mvs.append(mv[j])

    # 4) embed queries -> multi-vector
    q_mvs: List[torch.Tensor] = []
    for i in tqdm(range(0, n, args.batch_q), desc="Embed queries"):
        batch = queries[i:i + args.batch_q]
        mv = embed_queries(model, processor, batch, device).detach().cpu()  # [B, Lq, D]
        for j in range(mv.shape[0]):
            q_mvs.append(mv[j])

    doc_vecs = torch.stack([l2_normalize(mean_pool(mv).float()) for mv in doc_mvs])  # [n, D]
    q_vecs = torch.stack([l2_normalize(mean_pool(mv).float()) for mv in q_mvs])      # [n, D]

    sims = q_vecs @ doc_vecs.T  # [n, n]

    # evaluate: assume gt doc for query i is doc i
    ranks = []
    for i in range(n):
        top = torch.topk(sims[i], k=min(args.topk, n), largest=True).indices.tolist()
        rank = top.index(i) + 1 if i in top else 10**9
        ranks.append(rank)

    print(f"n={n}, topk={args.topk}")
    print(f"Recall@1  = {recall_at_k(ranks, 1):.3f}")
    print(f"Recall@5  = {recall_at_k(ranks, 5):.3f}")
    print(f"Recall@10 = {recall_at_k(ranks, 10):.3f}")

    # quick sanity prints
    print("\nExample shapes:")
    print("doc multi-vector:", tuple(doc_mvs[0].shape), " -> pooled:", tuple(doc_vecs[0].shape))
    print("qry multi-vector:", tuple(q_mvs[0].shape), " -> pooled:", tuple(q_vecs[0].shape))


if __name__ == "__main__":
    main()
