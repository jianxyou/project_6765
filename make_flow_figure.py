#!/usr/bin/env python3
"""
Clean 3-panel flow visualization:
  Panel 1: Original image (748 patches)
  Panel 2: After DocPruner (dropped patches DIMMED, kept patches visible)
  Panel 3: After DP-PostMerge (same as 2, PLUS colored outlines on merge groups)

Plus a side strip showing the pixel crops of the merged groups
with attention scores, so the reader can verify the merges are meaningful.
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


# Vivid distinct colors for merge groups
GROUP_COLORS = [
    "#FF1744",   # vivid red
    "#00E5FF",   # cyan
    "#FFEA00",   # bright yellow
    "#00E676",   # green
    "#FF9100",   # orange
    "#D500F9",   # purple
]


def dim_dropped(img: Image.Image, dropped_mask: np.ndarray,
                h_tokens: int, w_tokens: int,
                dim_factor: float = 0.25) -> Image.Image:
    """Return image with dropped patches dimmed (multiply RGB by dim_factor)."""
    arr = np.array(img.convert("RGB")).astype(np.float32)
    H, W = arr.shape[:2]
    cell_h = H / h_tokens
    cell_w = W / w_tokens
    for idx in np.where(dropped_mask)[0]:
        r = idx // w_tokens
        c = idx % w_tokens
        y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
        x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
        arr[y0:y1, x0:x1] = arr[y0:y1, x0:x1] * dim_factor
    return Image.fromarray(arr.astype(np.uint8))


def run(dataset: str, doc_idx: int, k: float, threshold: float,
        top_groups: int, outdir: Path,
        rank_by: str = "size", min_coherence: float = 0.85):
    set_seed(0)
    device = pick_device()

    ds = load_dataset(dataset, "corpus", split="test")
    row = ds[int(doc_idx)]
    img = row["image"]
    doc_id = row.get("corpus-id", str(doc_idx))
    ds_short = dataset.split("/")[-1]

    model, processor = load_colqwen25(
        "vidore/colqwen2.5-v0.2", device=device, need_attention=True)
    h_tokens, w_tokens = get_patch_grid(processor, img)

    embs, attns, masks, _ = embed_images_with_attention(
        model, processor, [img], device=device, extract_attention=True)
    emb = embs[0]; attn = attns[0]; mask = masks[0]

    dp_res = docpruner_compress(emb, attn, imgpad_mask=mask, k=k)
    pm_res = dp_postmerge_compress(emb, attn, imgpad_mask=mask,
                                    k=k, merge_threshold=threshold)

    n_img = int(mask.sum().item())
    # Boolean mask over image_pad subset
    kept_mask_local = np.zeros(n_img, dtype=bool)
    kept_mask_local[dp_res.kept_indices.numpy()] = True
    dropped_mask_local = ~kept_mask_local

    labels = pm_res.cluster_labels.numpy()
    rep_mask = pm_res.representative_mask.bool().numpy()
    cluster_sizes = np.bincount(labels)

    # Define positions needed by the coherence filter below
    img_idx = mask.nonzero(as_tuple=True)[0]
    dp_kept_local_t = dp_res.kept_indices
    img_idx_np = img_idx.numpy()
    dp_kept_local_np = dp_kept_local_t.numpy()

    multi_clusters = [(int(c), int(s)) for c, s in enumerate(cluster_sizes) if s > 1]

    # ---- Visual coherence filter: reject groups whose members look very different ----
    # Compute small thumbnails per patch and measure intra-group RGB variance.
    W0, H0 = img.size
    cell_w0 = W0 / w_tokens
    cell_h0 = H0 / h_tokens
    img_arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # H, W, 3

    def group_pixel_coherence(cid: int) -> float:
        """Return a coherence score in [0, 1]. Higher = patches look more alike."""
        members = np.where(labels == cid)[0]
        positions = dp_kept_local_np[members]
        # Downsampled 8x8 RGB means per patch
        thumbs = []
        for pos in positions:
            r = pos // w_tokens
            c = pos % w_tokens
            y0, y1 = int(r * cell_h0), int((r + 1) * cell_h0)
            x0, x1 = int(c * cell_w0), int((c + 1) * cell_w0)
            patch = img_arr[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            # 8x8 downsampled patch
            ph, pw = patch.shape[:2]
            bh, bw = max(1, ph // 8), max(1, pw // 8)
            patch_small = patch[: bh*8, : bw*8].reshape(
                8, bh, 8, bw, 3).mean(axis=(1, 3))  # 8,8,3
            thumbs.append(patch_small.flatten())
        if len(thumbs) < 2:
            return 0.0
        X = np.stack(thumbs)  # [n, 192]
        # Normalised correlation: 1 - mean pairwise L2 distance (clipped)
        # Use mean pairwise cosine similarity of the thumbnails.
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sim_matrix = X_norm @ X_norm.T
        n = sim_matrix.shape[0]
        # Exclude diagonal
        mean_sim = (sim_matrix.sum() - n) / (n * (n - 1))
        return float(mean_sim)

    # Rank groups: filter by min coherence then sort by user-chosen criterion
    scored = []
    for cid, size in multi_clusters:
        coh = group_pixel_coherence(cid)
        scored.append((cid, size, coh))
    scored = [(cid, size, coh) for cid, size, coh in scored if coh >= min_coherence]
    if rank_by == "coherence":
        scored.sort(key=lambda x: (-x[2], -x[1]))   # most coherent first, larger as tiebreaker
    else:
        scored.sort(key=lambda x: (-x[1], -x[2]))   # biggest first, more coherent as tiebreaker
    multi_clusters = [(cid, size) for cid, size, _ in scored[:top_groups]]

    # Build dim versions
    img_after_dp = dim_dropped(img, dropped_mask_local, h_tokens, w_tokens)

    # Merged-mask: indices in image_pad subset that are absorbed (not anchors, in multi-clusters)
    # Also anchors of multi-clusters get highlighted in a different color

    # ---------- Build figure ----------
    fig = plt.figure(figsize=(18, 8.5), facecolor="white")
    gs = fig.add_gridspec(2, 3,
                           width_ratios=[1, 1, 1],
                           height_ratios=[3.4, 1.0],
                           hspace=0.05, wspace=0.04,
                           left=0.02, right=0.98, top=0.94, bottom=0.03)

    # Panel 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    # Shared clean title styling
    TITLE_KW = dict(fontsize=16, fontweight="bold", pad=12,
                    color="#212121")

    ax1.set_title(f"① Original   {n_img}", **TITLE_KW)
    ax1.set_xticks([]); ax1.set_yticks([])

    # Panel 2: After DocPruner
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_after_dp)
    ax2.set_title(f"② DocPruner   {dp_res.num_after}",
                   color="#1565C0", **{k: v for k, v in TITLE_KW.items() if k != "color"})
    ax2.set_xticks([]); ax2.set_yticks([])

    # Panel 3: After DP-PostMerge
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_after_dp)
    ax3.set_title(f"③ + PostMerge   {pm_res.num_after}",
                   color="#E91E63", **{k: v for k, v in TITLE_KW.items() if k != "color"})
    ax3.set_xticks([]); ax3.set_yticks([])

    W, H = img.size
    cell_w = W / w_tokens
    cell_h = H / h_tokens

    # Overlay merge group outlines on Panel 3
    for rank, (cid, size) in enumerate(multi_clusters):
        color = GROUP_COLORS[rank]
        members = np.where(labels == cid)[0]  # indices into dp_kept list
        local_positions = dp_kept_local_np[members]
        for pos in local_positions:
            r = pos // w_tokens
            c = pos % w_tokens
            rect = mpatches.Rectangle(
                (c * cell_w, r * cell_h), cell_w, cell_h,
                linewidth=3.5, edgecolor=color, facecolor="none",
                zorder=10,
            )
            ax3.add_patch(rect)

    # ---------- Bottom: group detail strip ----------
    # For each of top N groups, show: colored label + all member crops + attn
    ax_strip = fig.add_subplot(gs[1, :])
    ax_strip.axis("off")

    # Build detail rows as a single long horizontal strip
    n_groups = len(multi_clusters)
    if n_groups == 0:
        ax_strip.text(0.5, 0.5, "(no merge groups)", ha="center", va="center")
    else:
        # Layout: each group is a self-contained sub-panel
        # [Group N label row, above]
        # [row of crops, below]
        group_gs = gs[1, :].subgridspec(1, n_groups, wspace=0.10)
        for rank, (cid, size) in enumerate(multi_clusters):
            color = GROUP_COLORS[rank]
            members = np.where(labels == cid)[0]
            member_pos = dp_kept_local_np[members]

            # Sub-grid: label ABOVE crops (2 rows)
            n_crops = min(len(members), 6)
            sub = group_gs[0, rank].subgridspec(
                2, n_crops,
                height_ratios=[0.25, 1], hspace=0.12, wspace=0.04)

            # Label row (spans all crop columns)
            ax_lbl = fig.add_subplot(sub[0, :])
            ax_lbl.axis("off")
            ax_lbl.text(0.0, 0.5,
                         f"Group {rank+1}",
                         color=color, fontsize=14, fontweight="bold",
                         ha="left", va="center")
            ax_lbl.text(1.0, 0.5,
                         f"{size} patches  →  1",
                         color="#555", fontsize=12,
                         ha="right", va="center")

            # Sort members: anchor first, then by attention
            attns_in_cluster = attn[img_idx[torch.tensor(member_pos, dtype=torch.long)]]
            order = attns_in_cluster.argsort(descending=True).numpy()
            for j in range(n_crops):
                ord_i = int(order[j])
                pos = int(member_pos[ord_i])
                r = pos // w_tokens
                c = pos % w_tokens
                x0 = int(c * cell_w)
                y0 = int(r * cell_h)
                x1 = int((c + 1) * cell_w)
                y1 = int((r + 1) * cell_h)
                crop = img.crop((x0, y0, x1, y1))
                ax_c = fig.add_subplot(sub[1, j])
                ax_c.imshow(crop)
                ax_c.set_xticks([]); ax_c.set_yticks([])
                for sp in ax_c.spines.values():
                    sp.set_edgecolor(color)
                    sp.set_linewidth(2.2)
                # Anchor star inside the crop, top-right corner
                is_anchor = bool(rep_mask[members[ord_i]])
                if is_anchor:
                    ax_c.text(0.92, 0.92, "★",
                               transform=ax_c.transAxes,
                               fontsize=14, color=color,
                               fontweight="bold",
                               ha="right", va="top",
                               path_effects=[
                                   __import__("matplotlib.patheffects",
                                              fromlist=["withStroke"])
                                   .withStroke(linewidth=2.5, foreground="white")
                               ])

    # No suptitle — numbers are already in each panel title

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"flow_{ds_short}_doc{doc_id}_k{k}_t{threshold}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="vidore/esg_reports_v2")
    p.add_argument("--doc-indices", type=int, nargs="+", required=True)
    p.add_argument("--k", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.93)
    p.add_argument("--top-groups", type=int, default=3)
    p.add_argument("--rank-by", choices=["size", "coherence"], default="size",
                   help="Rank merge groups by largest size or highest pixel coherence.")
    p.add_argument("--min-coherence", type=float, default=0.85,
                   help="Minimum pixel-cosine coherence required to display a group.")
    p.add_argument("--outdir",
                    default="/active_work/environment/benchmark_outputs/figures/flow")
    args = p.parse_args()

    for idx in args.doc_indices:
        run(args.dataset, idx, args.k, args.threshold,
            args.top_groups, Path(args.outdir),
            rank_by=args.rank_by, min_coherence=args.min_coherence)


if __name__ == "__main__":
    main()
