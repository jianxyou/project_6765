"""
Plotting utilities for benchmark results.
"""

import json
from pathlib import Path
from typing import List, Optional
from collections import defaultdict

import matplotlib.pyplot as plt


def load_metrics(outdir: str = "outputs") -> List[dict]:
    """Load all metrics_*.json files from outdir."""
    results = []
    for f in sorted(Path(outdir).glob("metrics_*.json")):
        results.append(json.loads(f.read_text()))
    return results


def load_sweep(path: str) -> List[dict]:
    """Load a sweep_metrics.json file."""
    return json.loads(Path(path).read_text())


def plot_ndcg_vs_compression(
    metrics_list: List[dict],
    title: str = "nDCG@5 vs Compression Ratio",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot nDCG@5 vs compression ratio, one line per method.
    Averages across datasets for each method config.
    """
    # Group by method suffix -> list of (avg_prune%, avg_ndcg)
    method_points = defaultdict(list)

    # Group by (pruner, param_config) -> per-dataset metrics
    config_groups = defaultdict(list)
    for m in metrics_list:
        pruner = m["pruner"]
        # Build a config key that distinguishes parameter settings
        if pruner == "docpruner":
            key = (pruner, f"k={m.get('k', '?')}")
        elif pruner.startswith("cps"):
            key = (pruner, f"r={m.get('cluster_ratio', m.get('cps_cluster_ratio', '?'))}")
        elif pruner == "ptm":
            key = (pruner, f"k={m.get('ptm_k', '?')},m={m.get('ptm_m', '?')}")
        elif pruner in ("docmerger", "docmerger_avg"):
            key = (pruner, f"k1={m.get('k1', '?')},k2={m.get('k2', '?')}")
        else:
            key = (pruner, "")
        config_groups[key].append(m)

    # For each config, compute avg across datasets
    method_data = defaultdict(list)  # pruner -> [(prune%, ndcg)]
    for (pruner, params), ms in config_groups.items():
        avg_prune = sum(m["avg_pruning_ratio"] for m in ms) / len(ms)
        avg_ndcg = sum(m["ndcg@5"] for m in ms) / len(ms)
        method_data[pruner].append((avg_prune * 100, avg_ndcg))

    # Sort each method's points by compression
    for k in method_data:
        method_data[k].sort()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']

    for i, (method, points) in enumerate(sorted(method_data.items())):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(xs, ys, f'{marker}-', color=color, linewidth=2, markersize=7,
                label=method, zorder=3)

    ax.set_xlabel('Compression Ratio (%)', fontsize=12)
    ax.set_ylabel('nDCG@5 (avg across datasets)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig, ax


def plot_per_dataset(
    metrics_list: List[dict],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """
    Plot nDCG@5 vs compression for each dataset separately (2x2 grid).
    """
    ds_order = ["esg_reports_v2", "biomedical_lectures_eng_v2",
                "economics_reports_v2", "esg_reports_human_labeled_v2"]
    ds_titles = ["ESG Reports", "Biomedical Lectures", "Economics Reports", "ESG Human-Labeled"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

    # Group by dataset
    by_dataset = defaultdict(list)
    for m in metrics_list:
        by_dataset[m["dataset_short"]].append(m)

    for idx, (ds, ds_title) in enumerate(zip(ds_order, ds_titles)):
        ax = axes[idx]
        ds_metrics = by_dataset.get(ds, [])

        # Group by pruner
        by_method = defaultdict(list)
        for m in ds_metrics:
            by_method[m["pruner"]].append(m)

        for i, (method, ms) in enumerate(sorted(by_method.items())):
            points = sorted([(m["avg_pruning_ratio"] * 100, m["ndcg@5"]) for m in ms])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            ax.plot(xs, ys, f'{marker}-', color=color, linewidth=1.5, markersize=5,
                    label=method)

        ax.set_title(ds_title, fontsize=11)
        ax.set_xlabel('Compression %', fontsize=9)
        ax.set_ylabel('nDCG@5', fontsize=9)
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig


def plot_method_comparison_bar(
    metrics_list: List[dict],
    target_compression: float = 50.0,
    tolerance: float = 10.0,
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing methods at roughly the same compression level.
    """
    ds_order = ["esg_reports_v2", "biomedical_lectures_eng_v2",
                "economics_reports_v2", "esg_reports_human_labeled_v2"]
    ds_short = ["ESG", "Bio", "Econ", "ESG-H"]

    # Filter metrics near target compression
    near_target = [m for m in metrics_list
                   if abs(m["avg_pruning_ratio"] * 100 - target_compression) < tolerance]

    # Best config per method (closest to target compression)
    best_per_method = {}
    for m in near_target:
        pruner = m["pruner"]
        ds = m["dataset_short"]
        key = (pruner, ds)
        if key not in best_per_method:
            best_per_method[key] = m
        elif abs(m["avg_pruning_ratio"] * 100 - target_compression) < \
             abs(best_per_method[key]["avg_pruning_ratio"] * 100 - target_compression):
            best_per_method[key] = m

    # Aggregate
    methods = sorted(set(k[0] for k in best_per_method))

    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(ds_short))
    width = 0.8 / max(1, len(methods))
    colors = plt.cm.tab10.colors

    for i, method in enumerate(methods):
        scores = []
        for ds in ds_order:
            m = best_per_method.get((method, ds))
            scores.append(m["ndcg@5"] if m else 0)
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar([x + offset for x in x_pos], scores, width,
               label=method, color=colors[i % len(colors)])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_short)
    ax.set_ylabel('nDCG@5')
    ax.set_title(f'Method Comparison at ~{target_compression:.0f}% Compression')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
    return fig
