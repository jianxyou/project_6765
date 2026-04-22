"""
Qualitative visualization: overlay kept / merged patches on original document image.

Currently supports docpruner and dp_postmerge (and any method whose PruneResult
populates kept_indices + cluster_labels).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from .methods.base import PruneResult


def get_patch_grid(processor, image: Image.Image) -> Tuple[int, int]:
    """
    Return (h_tokens, w_tokens) — the 2x2-merged patch grid as seen by the LLM.
    Each image_pad token corresponds to one cell in this grid.
    """
    batch = processor.process_images([image])
    thw = batch["image_grid_thw"][0].tolist()  # (t, h, w) — h,w are pre-merge patches
    h_tokens = thw[1] // 2
    w_tokens = thw[2] // 2
    return h_tokens, w_tokens


def _make_cluster_colors(n_clusters: int, seed: int = 0) -> np.ndarray:
    """Generate n distinct RGB colors in [0, 1]."""
    rng = np.random.default_rng(seed)
    # Use tab20 + hsv for more colors if needed
    if n_clusters <= 20:
        cmap = plt.get_cmap("tab20")
        colors = np.array([cmap(i / 20)[:3] for i in range(n_clusters)])
    else:
        hues = rng.permutation(n_clusters) / n_clusters
        import colorsys
        colors = np.array([colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in hues])
    return colors


def render_pruning_overlay(
    image: Image.Image,
    result: PruneResult,
    h_tokens: int,
    w_tokens: int,
    mode: str = "prune",
    alpha_dropped: float = 0.75,
    alpha_anchor: float = 0.5,
    alpha_absorbed: float = 0.55,
    dropped_color: Tuple[float, float, float] = (0.1, 0.1, 0.1),
    anchor_color: Tuple[float, float, float] = (0.10, 0.75, 0.35),   # green
    absorbed_color: Tuple[float, float, float] = (1.0, 0.70, 0.12),  # amber
) -> Image.Image:
    """
    Render patch-level overlay on the document image.

    mode:
      - "prune":   dropped patches darkened; kept shown in original
      - "merged":  dropped darkened; anchors (cluster reps of a merged group)
                   tinted green; absorbed members tinted amber; singletons untouched
    """
    assert result.kept_indices is not None, "PruneResult missing kept_indices"

    img_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    H, W = img_rgb.shape[:2]
    cell_h = H / h_tokens
    cell_w = W / w_tokens

    kept = set(result.kept_indices.tolist())
    n_img = h_tokens * w_tokens

    anchor_set: set = set()
    absorbed_set: set = set()
    if mode == "merged":
        assert result.cluster_labels is not None and result.representative_mask is not None, \
            "merged mode needs cluster_labels and representative_mask"
        labels = result.cluster_labels.tolist()
        reps = result.representative_mask.tolist()
        kept_list = result.kept_indices.tolist()
        from collections import Counter
        cnt = Counter(labels)
        for idx, lab, is_rep in zip(kept_list, labels, reps):
            if cnt[lab] < 2:
                continue  # singleton: show original
            if is_rep:
                anchor_set.add(idx)
            else:
                absorbed_set.add(idx)

    overlay = img_rgb.copy()
    drop_c = np.array(dropped_color, dtype=np.float32)
    anc_c = np.array(anchor_color, dtype=np.float32)
    abs_c = np.array(absorbed_color, dtype=np.float32)

    for idx in range(n_img):
        r = idx // w_tokens
        c = idx % w_tokens
        y0 = int(round(r * cell_h))
        y1 = int(round((r + 1) * cell_h))
        x0 = int(round(c * cell_w))
        x1 = int(round((c + 1) * cell_w))

        if idx not in kept:
            overlay[y0:y1, x0:x1] = (
                (1 - alpha_dropped) * img_rgb[y0:y1, x0:x1]
                + alpha_dropped * drop_c
            )
        elif idx in anchor_set:
            overlay[y0:y1, x0:x1] = (
                (1 - alpha_anchor) * img_rgb[y0:y1, x0:x1]
                + alpha_anchor * anc_c
            )
        elif idx in absorbed_set:
            overlay[y0:y1, x0:x1] = (
                (1 - alpha_absorbed) * img_rgb[y0:y1, x0:x1]
                + alpha_absorbed * abs_c
            )

    out = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


def _extract_patch_tile(image: Image.Image, idx: int, h_tokens: int, w_tokens: int) -> np.ndarray:
    """Crop the image region corresponding to image_pad index `idx`."""
    img_rgb = np.array(image.convert("RGB"))
    H, W = img_rgb.shape[:2]
    cell_h = H / h_tokens
    cell_w = W / w_tokens
    r = idx // w_tokens
    c = idx % w_tokens
    y0 = int(round(r * cell_h))
    y1 = int(round((r + 1) * cell_h))
    x0 = int(round(c * cell_w))
    x1 = int(round((c + 1) * cell_w))
    return img_rgb[y0:y1, x0:x1]


def compare_docpruner_vs_postmerge(
    image: Image.Image,
    dp_result: PruneResult,
    pm_result: PruneResult,
    processor,
    save_path: str,
    k: float,
    threshold: float,
    doc_label: str = "",
    top_k_groups: int = 3,
):
    """
    Figure layout:
      Top row:     Original | DocPruner overlay | DP-PostMerge overlay
      Bottom row:  "evidence strip" — for the top-K largest merge groups,
                   display the pixel tiles of each member patch, showing that
                   they really are near-duplicates that collapse to one vector.
    """
    from collections import Counter
    import matplotlib.gridspec as gridspec

    h_tokens, w_tokens = get_patch_grid(processor, image)
    n_img = h_tokens * w_tokens
    n_dropped = dp_result.num_before - dp_result.num_after

    # Cluster info (list of (cluster_id, member_local_indices, anchor_local_idx))
    cnt = Counter(pm_result.cluster_labels.tolist())
    kept_list = pm_result.kept_indices.tolist()
    labels = pm_result.cluster_labels.tolist()
    reps = pm_result.representative_mask.tolist()
    groups: dict = {}  # cluster_id -> {"members":[img_pad_idx,...], "anchor": img_pad_idx}
    for img_pad_idx, lab, is_rep in zip(kept_list, labels, reps):
        g = groups.setdefault(int(lab), {"members": [], "anchor": None})
        g["members"].append(int(img_pad_idx))
        if is_rep:
            g["anchor"] = int(img_pad_idx)

    merge_groups = [g for g in groups.values() if len(g["members"]) >= 2]
    merge_groups.sort(key=lambda g: len(g["members"]), reverse=True)
    top_groups = merge_groups[:top_k_groups]

    n_merged_groups = len(merge_groups)
    n_absorbed = sum(len(g["members"]) - 1 for g in merge_groups)
    n_singletons = sum(1 for v in cnt.values() if v == 1)

    # --- Figure layout ---
    max_grp_size = max((len(g["members"]) for g in top_groups), default=1)
    top_h = 6.5
    strip_h = 1.15 * max(1, len(top_groups))
    fig = plt.figure(figsize=(18, top_h + strip_h))
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[top_h, strip_h], hspace=0.35)

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0], wspace=0.08)

    ax_orig = fig.add_subplot(gs_top[0])
    ax_dp = fig.add_subplot(gs_top[1])
    ax_pm = fig.add_subplot(gs_top[2])

    ax_orig.imshow(image)
    ax_orig.set_title(f"Original\n{h_tokens}×{w_tokens} = {n_img} patches", fontsize=11)
    ax_orig.axis("off")

    dp_overlay = render_pruning_overlay(image, dp_result, h_tokens, w_tokens, mode="prune")
    ax_dp.imshow(dp_overlay)
    ax_dp.set_title(
        f"DocPruner (k={k})\n"
        f"drop {n_dropped}  |  keep {dp_result.num_after} → {dp_result.num_after} vectors",
        fontsize=11,
    )
    ax_dp.axis("off")

    pm_overlay = render_pruning_overlay(image, pm_result, h_tokens, w_tokens, mode="merged")
    ax_pm.imshow(pm_overlay)
    ax_pm.set_title(
        f"DP-PostMerge (k={k}, t={threshold})\n"
        f"{n_singletons} singletons + {n_merged_groups} anchors "
        f"(absorbed {n_absorbed}) → {pm_result.num_after} vectors",
        fontsize=11,
    )
    ax_pm.axis("off")

    # --- Evidence strip: top-K merge groups ---
    # Each row: [group label on the left] [patch tile 1] [tile 2] ... [tile N] [→ 1 vector]
    # Use max_grp_size + 2 columns (1 for label, max_grp_size for tiles, 1 for arrow text)
    n_rows = max(1, len(top_groups))
    n_cols = max_grp_size + 2  # label | tiles | "→ 1 vector"
    gs_strip = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=gs_outer[1],
        width_ratios=[1.4] + [1.0] * max_grp_size + [1.2],
        wspace=0.12, hspace=0.25,
    )

    if not top_groups:
        ax_empty = fig.add_subplot(gs_outer[1])
        ax_empty.text(0.5, 0.5, "(no merge groups at this threshold)",
                      ha="center", va="center", fontsize=11, color="gray")
        ax_empty.axis("off")
    else:
        for row_idx, grp in enumerate(top_groups):
            members = grp["members"]
            anchor = grp["anchor"]
            # Put anchor first so eye catches it, others after
            ordered = [anchor] + [m for m in members if m != anchor]

            # Label cell
            ax_lab = fig.add_subplot(gs_strip[row_idx, 0])
            ax_lab.text(
                0.5, 0.55,
                f"Group {row_idx + 1}\n{len(members)} similar patches",
                ha="center", va="center", fontsize=10.5, fontweight="bold",
            )
            ax_lab.text(
                0.5, 0.15,
                "(anchor outlined green)",
                ha="center", va="center", fontsize=8.5, color="gray",
            )
            ax_lab.axis("off")

            # Patch tiles
            for col_idx in range(max_grp_size):
                ax_tile = fig.add_subplot(gs_strip[row_idx, col_idx + 1])
                if col_idx < len(ordered):
                    patch_idx = ordered[col_idx]
                    tile = _extract_patch_tile(image, patch_idx, h_tokens, w_tokens)
                    ax_tile.imshow(tile)
                    border_color = (0.10, 0.75, 0.35) if patch_idx == anchor else (0.85, 0.55, 0.10)
                    for spine in ax_tile.spines.values():
                        spine.set_edgecolor(border_color)
                        spine.set_linewidth(2.2)
                    ax_tile.set_xticks([])
                    ax_tile.set_yticks([])
                else:
                    ax_tile.axis("off")

            # Arrow + result cell
            ax_arrow = fig.add_subplot(gs_strip[row_idx, n_cols - 1])
            ax_arrow.text(
                0.5, 0.5,
                "→ 1 merged vector",
                ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=(0.10, 0.55, 0.25),
            )
            ax_arrow.axis("off")

    # Suptitle
    extra_compression = (dp_result.num_after - pm_result.num_after) / max(1, dp_result.num_before) * 100
    suptitle = (
        f"{doc_label}    —    DP-PostMerge collapses {n_absorbed} near-duplicate patches into "
        f"{n_merged_groups} anchors (weighted-avg).  Final: {pm_result.num_after} vectors  "
        f"({dp_result.num_after - pm_result.num_after} fewer, +{extra_compression:.1f}% compression)."
    )
    fig.suptitle(suptitle, fontsize=12, y=0.995)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {save_path}")


# Back-compat alias
compare_methods_figure = compare_docpruner_vs_postmerge
