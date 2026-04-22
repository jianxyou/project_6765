#!/usr/bin/env python3
"""
Detail inspection of what PostMerge actually merges for a given document:
  - For each merge group, show every absorbed patch's position in the grid,
    its attention score, and its pixel crop.
  - Dual view: page image with colored rectangles, plus a "zoom strip"
    showing the actual pixel content of each merged patch.

This is the raw-data view to verify: are the merges semantically meaningful?
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset

from benchmark.utils import pick_device, set_seed
from benchmark.model import load_colqwen25, embed_images_with_attention
from benchmark.methods.docpruner import docpruner_compress
from benchmark.methods.dp_postmerge import dp_postmerge_compress
from benchmark.visualize import get_patch_grid


GROUP_COLORS = [
    ("#E91E63", "pink"),    # group 1
    ("#00BCD4", "cyan"),    # group 2
    ("#FF9800", "orange"),  # group 3
    ("#4CAF50", "green"),   # group 4
    ("#9C27B0", "purple"),  # group 5
    ("#FFEB3B", "yellow"),  # group 6
]


def crop_patch(image: Image.Image, row: int, col: int,
               h_tokens: int, w_tokens: int, scale: int = 4):
    """Return a small PIL crop of the patch at (row, col)."""
    W, H = image.size
    cell_w = W / w_tokens
    cell_h = H / h_tokens
    x0 = int(col * cell_w)
    y0 = int(row * cell_h)
    x1 = int((col + 1) * cell_w)
    y1 = int((row + 1) * cell_h)
    return image.crop((x0, y0, x1, y1))


def inspect(dataset: str, doc_idx: int, k: float, threshold: float,
            top_groups: int, outdir: Path):
    set_seed(0)
    device = pick_device()

    print(f"Loading {dataset}/corpus ...")
    ds = load_dataset(dataset, "corpus", split="test")
    row = ds[int(doc_idx)]
    img = row["image"]
    doc_id = row.get("corpus-id", str(doc_idx))
    ds_short = dataset.split("/")[-1]

    print(f"Loading ColQwen2.5 ...")
    model, processor = load_colqwen25(
        "vidore/colqwen2.5-v0.2", device=device, need_attention=True)
    h_tokens, w_tokens = get_patch_grid(processor, img)
    print(f"Grid: {h_tokens} x {w_tokens} = {h_tokens * w_tokens} tokens")

    embs, attns, masks, _ = embed_images_with_attention(
        model, processor, [img], device=device, extract_attention=True)
    emb = embs[0]; attn = attns[0]; mask = masks[0]

    dp_res = docpruner_compress(emb, attn, imgpad_mask=mask, k=k)
    pm_res = dp_postmerge_compress(emb, attn, imgpad_mask=mask,
                                    k=k, merge_threshold=threshold)

    img_idx = mask.nonzero(as_tuple=True)[0]
    dp_kept_global = img_idx[dp_res.kept_indices]  # positions in full seq
    dp_kept_local = dp_res.kept_indices             # positions in image_pad subset

    labels = pm_res.cluster_labels
    rep_mask = pm_res.representative_mask.bool()
    cluster_sizes = torch.bincount(labels)

    print(f"\nDocPruner:   {dp_res.num_before} -> {dp_res.num_after}")
    print(f"DP-PostMerge: {dp_res.num_after} -> {pm_res.num_after}  "
          f"(merged {dp_res.num_after - pm_res.num_after} patches)")

    # Collect groups sorted by size
    multi_clusters = [(int(cid), int(size))
                      for cid, size in enumerate(cluster_sizes.tolist()) if size > 1]
    multi_clusters.sort(key=lambda x: -x[1])
    multi_clusters = multi_clusters[:top_groups]

    print(f"\nTop {len(multi_clusters)} merge groups:")
    for rank, (cid, size) in enumerate(multi_clusters):
        members = (labels == cid).nonzero(as_tuple=True)[0]  # indices into kept list
        anchor = members[rep_mask[members]]
        absorbed = members[~rep_mask[members]]
        member_local_idx = dp_kept_local[members]  # indices into image_pad
        attns_in_cluster = attn[img_idx[member_local_idx]]
        print(f"  Group {rank+1}: size={size}, attention range "
              f"{attns_in_cluster.min().item()*1000:.3f} - "
              f"{attns_in_cluster.max().item()*1000:.3f}‰  "
              f"mean {attns_in_cluster.mean().item()*1000:.3f}‰")

    # -------- Build figure --------
    n_groups = len(multi_clusters)
    fig = plt.figure(figsize=(16, 4 + n_groups * 1.2))
    gs = fig.add_gridspec(1 + n_groups, 10, width_ratios=[2.5] + [1]*9,
                          hspace=0.25, wspace=0.15)

    # Left column spanning all rows: original image with colored rectangles
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.set_title(f"{ds_short}  doc {doc_id}\n"
                     f"DocPruner: {dp_res.num_after}/{dp_res.num_before} "
                     f"→ +PostMerge: {pm_res.num_after}",
                     fontsize=11)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    W, H = img.size
    cell_w = W / w_tokens
    cell_h = H / h_tokens

    # Draw patch positions for each merge group
    for rank, (cid, size) in enumerate(multi_clusters):
        color = GROUP_COLORS[rank][0]
        members = (labels == cid).nonzero(as_tuple=True)[0]
        for m in members.tolist():
            local_idx = int(dp_kept_local[m].item())
            row_r = local_idx // w_tokens
            col_r = local_idx % w_tokens
            rect = mpatches.Rectangle(
                (col_r * cell_w, row_r * cell_h), cell_w, cell_h,
                linewidth=3, edgecolor=color, facecolor=color + "33",
                zorder=5,
            )
            ax_img.add_patch(rect)

    # Right side: per-group detail rows (top-right = group 1 etc.)
    for rank, (cid, size) in enumerate(multi_clusters):
        color = GROUP_COLORS[rank][0]
        members = (labels == cid).nonzero(as_tuple=True)[0]
        member_local_idx = dp_kept_local[members]  # indices into image_pad
        attns_in_cluster = attn[img_idx[member_local_idx]]

        row_idx = rank + 1  # skip header row

        # Label cell (leftmost of right area)
        ax_lbl = fig.add_subplot(gs[row_idx, 0])
        # Actually we want labels in first right column -- move to gs[row_idx, 0]?
        # Use sep axes in gs instead
        fig.delaxes(ax_lbl)

        # Show each member as a crop with attention annotation
        order = attns_in_cluster.argsort(descending=True).tolist()
        for j, ord_i in enumerate(order[:9]):
            ax = fig.add_subplot(gs[row_idx, j + 1])
            local_idx = int(member_local_idx[ord_i].item())
            r = local_idx // w_tokens
            c = local_idx % w_tokens
            patch_img = crop_patch(img, r, c, h_tokens, w_tokens)
            ax.imshow(patch_img)
            ax.set_xticks([]); ax.set_yticks([])
            # Color border matching the group
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)
            is_anchor = bool(rep_mask[members[ord_i]].item())
            role = "★ anchor" if is_anchor else "absorbed"
            attn_v = attns_in_cluster[ord_i].item() * 1000
            ax.set_title(f"r{r},c{c}\n{attn_v:.1f}‰  {role}",
                         fontsize=8, color=color if is_anchor else "black")

        # Header next to first crop: group label
        ax0 = fig.axes[-len(order[:9])] if order else None
        if ax0 is not None:
            ax0.text(-0.25, 0.5, f"Group {rank+1}\n{size} patches →1",
                     transform=ax0.transAxes, ha="right", va="center",
                     fontsize=10, fontweight="bold", color=color)

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"inspect_{ds_short}_doc{doc_id}_k{k}_t{threshold}.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="vidore/esg_reports_v2")
    p.add_argument("--doc-indices", type=int, nargs="+", required=True)
    p.add_argument("--k", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.93)
    p.add_argument("--top-groups", type=int, default=3)
    p.add_argument("--outdir", default="/active_work/environment/benchmark_outputs/figures/inspect")
    args = p.parse_args()

    for idx in args.doc_indices:
        inspect(args.dataset, idx, args.k, args.threshold,
                args.top_groups, Path(args.outdir))


if __name__ == "__main__":
    main()
