#!/usr/bin/env python3
"""
Produce qualitative comparison figures for DocPruner (k=0) vs DP-PostMerge (k=0, t=0.93)
on sample ViDoRe-V2 documents.

Usage:
  python make_qualitative_figures.py [--n 6] [--dataset vidore/esg_reports_v2]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

import torch
from datasets import load_dataset

from benchmark.utils import pick_device, set_seed
from benchmark.model import load_colqwen25, embed_images_with_attention
from benchmark.methods.docpruner import docpruner_compress
from benchmark.methods.dp_postmerge import dp_postmerge_compress
from benchmark.visualize import compare_docpruner_vs_postmerge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="vidore/esg_reports_v2")
    parser.add_argument("--n", type=int, default=6, help="number of images to visualize")
    parser.add_argument("--k", type=float, default=0.0, help="DocPruner k")
    parser.add_argument("--threshold", type=float, default=0.93,
                        help="dp_postmerge cosine threshold")
    parser.add_argument("--outdir", default="results/figures/qualitative")
    parser.add_argument("--doc-indices", type=int, nargs="*", default=None,
                        help="specific corpus indices to use; otherwise sample")
    args = parser.parse_args()

    set_seed(0)
    device = pick_device()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset}/corpus ...")
    ds = load_dataset(args.dataset, "corpus", split="test")
    n_total = len(ds)

    if args.doc_indices:
        indices = args.doc_indices
    else:
        # Pick spread-out indices for visual variety
        step = max(1, n_total // args.n)
        indices = [i * step for i in range(args.n)]
    indices = [i for i in indices if i < n_total][: args.n]

    print(f"Loading ColQwen2.5 (eager, attention mode) ...")
    model, processor = load_colqwen25(
        "vidore/colqwen2.5-v0.2", device=device, need_attention=True
    )

    ds_short = args.dataset.split("/")[-1]

    for rank, idx in enumerate(indices):
        row = ds[int(idx)]
        img = row["image"]
        doc_id = row.get("corpus-id", str(idx))
        print(f"\n[{rank+1}/{len(indices)}] doc_id={doc_id} image size={img.size}")

        # Encode one image
        embs, attns, masks, _grids = embed_images_with_attention(
            model, processor, [img], device=device, extract_attention=True
        )
        emb = embs[0]
        attn = attns[0]
        mask = masks[0]
        n_img = mask.sum().item()
        print(f"  tokens={emb.shape[0]} image_pad={n_img}")

        # Run both methods
        dp_res = docpruner_compress(emb, attn, imgpad_mask=mask, k=args.k)
        pm_res = dp_postmerge_compress(
            emb, attn, imgpad_mask=mask, k=args.k, merge_threshold=args.threshold
        )
        print(f"  DocPruner:     kept {dp_res.num_after}/{dp_res.num_before} "
              f"({dp_res.pruning_ratio*100:.1f}% pruned)")
        print(f"  DP-PostMerge:  kept {pm_res.num_after}/{pm_res.num_before} "
              f"({pm_res.pruning_ratio*100:.1f}% pruned)")

        save_path = outdir / f"{ds_short}_doc{doc_id}_k{args.k}_t{args.threshold}.png"
        compare_docpruner_vs_postmerge(
            image=img,
            dp_result=dp_res,
            pm_result=pm_res,
            processor=processor,
            save_path=str(save_path),
            k=args.k,
            threshold=args.threshold,
            doc_label=f"{ds_short}  doc {doc_id}",
        )

    print(f"\nAll figures saved to {outdir}")


if __name__ == "__main__":
    main()
