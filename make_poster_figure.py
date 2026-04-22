#!/usr/bin/env python3
"""
Poster figure: Overall Performance Comparison on ColQwen2.5.

X-axis: Compression Ratio (%)
Y-axis: nDCG@5 (avg across 4 ViDoRe-V2 datasets)

Baselines: random, sem_cluster, pool1d, pool2d, attn_similarity,
           pivot_threshold, DocPruner
Winner: DP-PostMerge (θ=0.93)

Style:
  - solid line = adaptive (uses attention/content info)
  - dashed line = non-adaptive (fixed ratio/schedule)
  - circle = pruning-based
  - square = merging-based
  - DP-PostMerge highlighted (bold, star marker)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

OUTDIR = Path("/active_work/environment/benchmark_outputs")
V2 = {"esg_reports_v2", "biomedical_lectures_eng_v2",
      "economics_reports_v2", "esg_reports_human_labeled_v2"}


def parse_cfg_key(filepath: Path, d: dict) -> tuple:
    """Return (method_name, config_key) so we can group across 4 datasets."""
    p = d["pruner"]
    if p == "docpruner":
        return (p, f"k={d.get('k', '?')}")
    if p == "random":
        return (p, f"r={d.get('ratio', '?')}")
    if p in ("sem_cluster", "pool1d", "pool2d"):
        m = re.search(r"_mf(\d+)", filepath.name)
        return (p, f"mf={m.group(1)}" if m else (p, "?"))
    if p == "attn_similarity":
        return (p, f"k={d.get('k', '?')}")
    if p == "pivot_threshold":
        return (p, f"k={d.get('k', '?')}")
    if p == "dp_postmerge":
        m = re.search(r"_t(0\.\d+)", filepath.name)
        theta = m.group(1) if m else None
        return (p, f"k={d.get('k', '?')}_t{theta}")
    if p == "identity":
        return (p, "identity")
    return (p, "other")


def aggregate():
    """Load metrics, filter V2, aggregate across 4 datasets per (method, config)."""
    groups = defaultdict(list)
    for f in OUTDIR.glob("metrics_*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d["dataset_short"] not in V2:
            continue
        key = parse_cfg_key(f, d)
        groups[key].append(d)

    # Only keep configs with all 4 datasets
    points = defaultdict(list)  # method -> [(prune%, nDCG, label)]
    for (method, cfg), ms in groups.items():
        if len(ms) != 4:
            continue
        avg_ndcg = sum(m["ndcg@5"] for m in ms) / 4
        avg_prune = sum(m["avg_pruning_ratio"] for m in ms) / 4
        points[method].append((avg_prune * 100, avg_ndcg, cfg))

    # Sort by pruning%
    for m in points:
        points[m].sort()
    return points


# Matplotlib style per method
STYLES = {
    # name: (display_label, color, marker, linestyle, linewidth, zorder)
    "identity":        ("base",                        "#000000", None, "--", 1.2, 1),
    "random":          ("random",                      "#CCB44F", "o",  "--", 1.5, 2),
    "sem_cluster":     ("sem-cluster",                 "#7BA098", "s",  "--", 1.5, 2),
    "pool1d":          ("1d-pooling",                  "#8FCBEA", "s",  "--", 1.5, 2),
    "pool2d":          ("2d-pooling",                  "#4A9BD4", "s",  "--", 1.5, 2),
    "attn_similarity": ("attention-plus-similarity",   "#A89FC9", "o",  "-",  1.8, 3),
    "pivot_threshold": ("pivot-threshold",             "#D4A074", "o",  "-",  1.8, 3),
    "docpruner":       ("DocPruner",                   "#3678B4", "o",  "-",  2.0, 4),
    "dp_postmerge":    ("DP-PostMerge (θ=0.93)",       "#D62728", "*",  "-",  2.8, 5),
}


def plot_poster(points, save_path="poster_figure.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Identity: horizontal baseline
    if "identity" in points:
        base_ndcg = points["identity"][0][1]
        ax.axhline(base_ndcg, linestyle="--", color="black", alpha=0.5, linewidth=1.2,
                   label="base", zorder=1)

    # Plot each method (skip identity, handled above)
    plot_order = ["sem_cluster", "pool1d", "pool2d",
                  "attn_similarity", "docpruner", "dp_postmerge"]

    for method in plot_order:
        if method not in points:
            continue
        # For dp_postmerge, keep only θ=0.93 curve
        if method == "dp_postmerge":
            pts = [(x, y, lbl) for x, y, lbl in points[method] if "t0.93" in lbl]
        else:
            pts = points[method]
        if not pts:
            continue

        label, color, marker, linestyle, lw, zorder = STYLES[method]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # dp_postmerge gets extra highlighting
        markersize = 14 if method == "dp_postmerge" else 7
        markeredgewidth = 1.2 if method == "dp_postmerge" else 0.0
        markeredgecolor = "black" if method == "dp_postmerge" else color

        ax.plot(xs, ys, marker=marker, linestyle=linestyle, color=color,
                linewidth=lw, markersize=markersize,
                markeredgewidth=markeredgewidth, markeredgecolor=markeredgecolor,
                label=label, zorder=zorder)

    ax.set_xlabel("Compression Ratio (%)", fontsize=13)
    ax.set_ylabel("nDCG@5 (avg across 4 ViDoRe-V2 datasets)", fontsize=12)
    ax.set_title("Overall Performance Comparison on ColQwen2.5", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Custom legend order
    handles, labels = ax.get_legend_handles_labels()
    order = ["base", "sem-cluster", "1d-pooling", "2d-pooling",
             "attention-plus-similarity",
             "DocPruner", "DP-PostMerge (θ=0.93)"]
    by_label = dict(zip(labels, handles))
    ordered = [(lbl, by_label[lbl]) for lbl in order if lbl in by_label]
    ax.legend([h for _, h in ordered], [l for l, _ in ordered],
              fontsize=10, loc="lower left", framealpha=0.95)

    ax.set_xlim(0, 100)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.replace('.png', '.pdf')}")


def print_summary(points):
    print("\n" + "=" * 78)
    print(f"{'Method':<30} {'Config':<16} {'Prune%':>8} {'nDCG@5':>8}")
    print("-" * 78)
    plot_order = ["identity", "random", "sem_cluster", "pool1d", "pool2d",
                  "attn_similarity", "pivot_threshold", "docpruner", "dp_postmerge"]
    for method in plot_order:
        if method not in points:
            continue
        for pr, nd, cfg in points[method]:
            if method == "dp_postmerge" and "t0.93" not in cfg:
                continue
            label = STYLES[method][0]
            print(f"{label:<30} {cfg:<16} {pr:>7.1f}% {nd:>8.4f}")
    print("=" * 78)


def main():
    points = aggregate()
    print_summary(points)
    out_png = "/active_work/environment/benchmark_outputs/figures/poster_overall.png"
    plot_poster(points, save_path=out_png)


if __name__ == "__main__":
    main()
