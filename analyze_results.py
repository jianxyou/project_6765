#!/usr/bin/env python3
"""
Analyze experiment results: compute average nDCG@5 across 4 datasets,
compare all methods against DocPruner k=-0.25, and find winners.

Usage:
    python analyze_results.py
    python analyze_results.py /path/to/results_dir
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "/active_work/environment/benchmark_outputs"

DS_ORDER = ["esg_reports_v2", "biomedical_lectures_eng_v2",
            "economics_reports_v2", "esg_reports_human_labeled_v2"]
DS_SHORT = ["ESG", "Bio", "Econ", "ESG-H"]


def load_all_metrics(outdir):
    """Load results from merged JSON or individual metric files."""
    results = []

    # Try merged file first
    merged = Path(outdir) / "all_results_merged.json"
    if merged.exists():
        data = json.loads(merged.read_text())
        results.extend([r for r in data if "error" not in r])

    # Also check per-GPU files
    for f in sorted(Path(outdir).glob("all_results_gpu*.json")):
        data = json.loads(f.read_text())
        results.extend([r for r in data if "error" not in r])

    # Also check individual metric files
    for f in sorted(Path(outdir).glob("metrics_*.json")):
        results.append(json.loads(f.read_text()))

    # Deduplicate by (dataset_short, pruner, param_key)
    seen = set()
    unique = []
    for r in results:
        key = (r.get("dataset_short", ""), r.get("pruner", ""), _param_key(r))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _param_key(m):
    """Build a string key from method-specific params."""
    pruner = m.get("pruner", "")
    parts = [pruner]
    for k in sorted(m.keys()):
        if k in ("dataset", "dataset_short", "model_name", "pruner",
                  "ndcg@5", "avg_pruning_ratio", "avg_patches_before",
                  "avg_patches_after", "num_corpus", "num_queries",
                  "language", "outdir", "device", "seed", "split",
                  "batch_doc", "batch_query", "batch_score_q", "batch_score_d",
                  "clear_cache"):
            continue
        v = m.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def _config_label(m):
    """Human-readable label for a method config."""
    pruner = m.get("pruner", "")
    if pruner == "docpruner":
        return f"docpruner(k={m.get('k', '?')})"
    elif pruner == "mmr":
        return f"mmr(λ={m.get('lam', '?')},r={m.get('target_ratio', '?')})"
    elif pruner == "attn_fps":
        return f"attn_fps(α={m.get('alpha', '?')},r={m.get('target_ratio', '?')})"
    elif pruner == "dp_rebalance":
        return f"dp_rebal(λ={m.get('lam', '?')},r={m.get('target_ratio', '?')},kl={m.get('k_loose', '?')})"
    elif pruner.startswith("cps"):
        return f"{pruner}(r={m.get('cluster_ratio', '?')})"
    elif pruner == "identity":
        return "identity(no compression)"
    else:
        return pruner


def main():
    results = load_all_metrics(OUTDIR)
    if not results:
        print(f"No results found in {OUTDIR}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment results\n")

    # Group by config (method + params) -> {dataset_short: metrics}
    config_groups = defaultdict(dict)
    for m in results:
        key = _param_key(m)
        ds = m.get("dataset_short", "unknown")
        config_groups[key][ds] = m

    # Compute averages across 4 datasets
    summaries = []
    for key, ds_dict in config_groups.items():
        if not ds_dict:
            continue
        sample = next(iter(ds_dict.values()))
        label = _config_label(sample)

        scores = [ds_dict[ds]["ndcg@5"] for ds in DS_ORDER if ds in ds_dict]
        prunes = [ds_dict[ds]["avg_pruning_ratio"] for ds in DS_ORDER if ds in ds_dict]

        if not scores:
            continue

        avg_ndcg = sum(scores) / len(scores)
        avg_prune = sum(prunes) / len(prunes)
        n_ds = len(scores)

        summaries.append({
            "label": label,
            "pruner": sample["pruner"],
            "avg_ndcg5": avg_ndcg,
            "avg_prune": avg_prune,
            "n_datasets": n_ds,
            "per_ds": {ds: ds_dict[ds]["ndcg@5"] for ds in DS_ORDER if ds in ds_dict},
            "per_ds_prune": {ds: ds_dict[ds]["avg_pruning_ratio"] for ds in DS_ORDER if ds in ds_dict},
        })

    # Sort by avg_ndcg descending
    summaries.sort(key=lambda x: -x["avg_ndcg5"])

    # Find DocPruner k=-0.25 baseline
    dp_baseline = None
    for s in summaries:
        if "docpruner(k=-0.25)" in s["label"]:
            dp_baseline = s
            break

    # Print results
    print(f"{'='*120}")
    print(f"{'Method':<55} {'#DS':>3} {'Avg nDCG@5':>10} {'Avg Prune%':>10} {'vs DP-0.25':>10}")
    print(f"{'-'*120}")

    for s in summaries:
        delta = ""
        if dp_baseline and s["label"] != dp_baseline["label"]:
            d = s["avg_ndcg5"] - dp_baseline["avg_ndcg5"]
            delta = f"{d:+.4f}"
            # Only highlight if at similar compression
            if abs(s["avg_prune"] - dp_baseline["avg_prune"]) < 0.1:
                if d > 0:
                    delta = f"✓ {d:+.4f}"
                else:
                    delta = f"  {d:+.4f}"
        elif dp_baseline and s["label"] == dp_baseline["label"]:
            delta = "BASELINE"

        print(f"{s['label']:<55} {s['n_datasets']:>3} "
              f"{s['avg_ndcg5']:>10.4f} {s['avg_prune']*100:>9.1f}% {delta:>10}")

    # Print winners (beat DocPruner at similar compression)
    if dp_baseline:
        print(f"\n{'='*120}")
        print(f"WINNERS (beat DocPruner k=-0.25 at similar compression ±10%):")
        print(f"DocPruner baseline: nDCG@5={dp_baseline['avg_ndcg5']:.4f}, prune={dp_baseline['avg_prune']*100:.1f}%")
        print(f"{'-'*120}")

        winners = []
        for s in summaries:
            if s["label"] == dp_baseline["label"]:
                continue
            # Similar compression (within 10 percentage points)
            if abs(s["avg_prune"] - dp_baseline["avg_prune"]) < 0.10:
                delta = s["avg_ndcg5"] - dp_baseline["avg_ndcg5"]
                if delta > 0:
                    winners.append((delta, s))

        if winners:
            winners.sort(key=lambda x: -x[0])
            for delta, s in winners:
                per_ds = " | ".join(f"{ds_s}: {s['per_ds'].get(ds, 0):.4f}" for ds, ds_s in zip(DS_ORDER, DS_SHORT))
                print(f"  ✓ {s['label']}")
                print(f"    nDCG@5={s['avg_ndcg5']:.4f} (+{delta:.4f}), prune={s['avg_prune']*100:.1f}%")
                print(f"    {per_ds}")
                print()
        else:
            print("  No method beat DocPruner at similar compression yet.")
            print("\n  Closest challengers:")
            close = [(s["avg_ndcg5"] - dp_baseline["avg_ndcg5"], s) for s in summaries
                     if abs(s["avg_prune"] - dp_baseline["avg_prune"]) < 0.10
                     and s["label"] != dp_baseline["label"]]
            close.sort(key=lambda x: -x[0])
            for delta, s in close[:5]:
                print(f"    {s['label']}: nDCG@5={s['avg_ndcg5']:.4f} ({delta:+.4f}), prune={s['avg_prune']*100:.1f}%")

    # Save summary
    summary_path = Path(OUTDIR) / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
