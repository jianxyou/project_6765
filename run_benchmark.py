#!/usr/bin/env python3
"""
Benchmark CLI entry point.

Usage:
  # Single experiment
  python run_benchmark.py run --pruner docpruner --k -0.25 --dataset vidore/esg_reports_v2

  # Sweep: DocPruner across all k values and datasets
  python run_benchmark.py sweep --preset docpruner

  # Sweep: all methods
  python run_benchmark.py sweep --preset all

  # Plot results
  python run_benchmark.py plot --outdir outputs

  # List available methods
  python run_benchmark.py methods
"""

import argparse
import json
import sys

from benchmark.data import VIDORE_V1_DATASETS, VIDORE_V2_DATASETS, VIDORE_V3_DATASETS, DOCPRUNER_K_VALUES
from benchmark.experiment import run_experiment, run_sweep
from benchmark.methods import list_methods


# ============================================================
# Preset sweep configs
# ============================================================

def _make_sweep_configs(preset: str, base_cfg: dict) -> list:
    configs = []

    def cfg(**overrides):
        c = dict(base_cfg)
        c.update(overrides)
        return c

    # Pick dataset list based on preset suffix
    if preset.endswith("_v1"):
        datasets = VIDORE_V1_DATASETS
        preset_base = preset[:-3]
    elif preset.endswith("_v3"):
        datasets = VIDORE_V3_DATASETS
        preset_base = preset[:-3]
    elif preset.endswith("_v2"):
        datasets = VIDORE_V2_DATASETS
        preset_base = preset[:-3]
    else:
        # Default: V2 for backward compat
        datasets = VIDORE_V2_DATASETS
        preset_base = preset

    if preset_base in ("docpruner", "all"):
        for ds in datasets:
            configs.append(cfg(dataset=ds, pruner="identity"))
        for k in DOCPRUNER_K_VALUES:
            for ds in datasets:
                configs.append(cfg(dataset=ds, pruner="docpruner", k=k))
        for ds in datasets:
            configs.append(cfg(dataset=ds, pruner="random", ratio=0.5))

    if preset_base in ("docmerger", "all"):
        for k1, k2, mr in [(0.5, 0.25, 0.25), (0.5, 0.5, 0.5), (1.0, 0, 0.25)]:
            for ds in datasets:
                configs.append(cfg(dataset=ds, pruner="docmerger", k1=k1, k2=k2, merge_ratio=mr))

    if preset_base in ("cps", "all"):
        for ratio in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]:
            for ds in datasets:
                configs.append(cfg(dataset=ds, pruner="cps_attn", cluster_ratio=ratio))
        for pruner in ["cps_central", "cps_domin"]:
            for ratio in [0.5, 0.3, 0.15]:
                for ds in datasets:
                    configs.append(cfg(dataset=ds, pruner=pruner, cluster_ratio=ratio))

    if preset_base in ("ptm", "all"):
        for k in [-1.0, -0.75, -0.5]:
            for m in [2, 4]:
                for ds in datasets:
                    configs.append(cfg(dataset=ds, pruner="ptm", k=k, m=m))

    if not configs:
        print(f"Unknown preset: {preset}")
        print(f"Available: docpruner, docmerger, cps, ptm, all")
        print(f"Append _v1, _v2, or _v3 to select benchmark (e.g. docpruner_v3)")
        sys.exit(1)

    return configs


# ============================================================
# CLI
# ============================================================

def cmd_run(args):
    config = {
        "dataset": args.dataset,
        "pruner": args.pruner,
        "model_name": args.model_name,
        "device": args.device,
        "seed": args.seed,
        "outdir": args.outdir,
        "clear_cache": args.clear_cache,
        "batch_doc": args.batch_doc,
        "batch_query": args.batch_query,
    }
    # Method-specific params
    if args.k is not None:
        config["k"] = args.k
    if args.k1 is not None:
        config["k1"] = args.k1
    if args.k2 is not None:
        config["k2"] = args.k2
    if args.merge_ratio is not None:
        config["merge_ratio"] = args.merge_ratio
    if args.cluster_ratio is not None:
        config["cluster_ratio"] = args.cluster_ratio
    if args.dedup_threshold is not None:
        config["dedup_threshold"] = args.dedup_threshold
    if args.ratio is not None:
        config["ratio"] = args.ratio
    if args.ptm_k is not None:
        config["k"] = args.ptm_k  # PTM uses 'k' param
    if args.ptm_m is not None:
        config["m"] = args.ptm_m
    if args.merging_factor is not None:
        config["merging_factor"] = args.merging_factor
    if args.alpha is not None:
        config["alpha"] = args.alpha
    if args.k_dup is not None:
        config["k_dup"] = args.k_dup
    if args.num_pivots is not None:
        config["num_pivots"] = args.num_pivots

    run_experiment(config)


def cmd_sweep(args):
    base_cfg = {
        "model_name": args.model_name,
        "device": args.device,
        "seed": args.seed,
        "outdir": args.outdir,
        "batch_doc": args.batch_doc,
        "batch_query": args.batch_query,
    }
    configs = _make_sweep_configs(args.preset, base_cfg)
    print(f"Sweep: {len(configs)} experiments with preset '{args.preset}'")
    run_sweep(configs, outdir=args.outdir)


def cmd_plot(args):
    from benchmark.plot import load_metrics, plot_ndcg_vs_compression, plot_per_dataset

    metrics = load_metrics(args.outdir)
    if not metrics:
        # Try sweep file
        sweep_path = f"{args.outdir}/sweep_metrics.json"
        from pathlib import Path
        if Path(sweep_path).exists():
            metrics = json.loads(Path(sweep_path).read_text())

    if not metrics:
        print(f"No metrics found in {args.outdir}/")
        sys.exit(1)

    print(f"Loaded {len(metrics)} experiment results")
    save_dir = f"{args.outdir}/figures"
    plot_ndcg_vs_compression(metrics, save_path=f"{save_dir}/ndcg_vs_compression.png")
    plot_per_dataset(metrics, save_path=f"{save_dir}/per_dataset.png")


def cmd_methods(args):
    print("Available methods:")
    for name in list_methods():
        print(f"  - {name}")


def main():
    parser = argparse.ArgumentParser(description="Document Compression Benchmark")
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Run a single experiment")
    p_run.add_argument("--dataset", default="vidore/esg_reports_v2")
    p_run.add_argument("--pruner", default="docpruner", choices=list_methods())
    p_run.add_argument("--model-name", default="vidore/colqwen2.5-v0.2")
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--outdir", default="/active_work/environment/benchmark_outputs")
    p_run.add_argument("--clear-cache", action="store_true")
    p_run.add_argument("--batch-doc", type=int, default=4)
    p_run.add_argument("--batch-query", type=int, default=8)
    # Method params
    p_run.add_argument("--k", type=float, default=None, help="DocPruner k")
    p_run.add_argument("--k1", type=float, default=None, help="DocMerger k1")
    p_run.add_argument("--k2", type=float, default=None, help="DocMerger k2")
    p_run.add_argument("--merge-ratio", type=float, default=None)
    p_run.add_argument("--cluster-ratio", type=float, default=None, help="CPS cluster ratio")
    p_run.add_argument("--dedup-threshold", type=float, default=None, help="CPS dedup threshold")
    p_run.add_argument("--ratio", type=float, default=None, help="Random prune ratio")
    p_run.add_argument("--ptm-k", type=float, default=None, help="PTM pruning k")
    p_run.add_argument("--ptm-m", type=int, default=None, help="PTM merging factor")
    p_run.add_argument("--merging-factor", type=int, default=None,
                       help="sem_cluster / pool1d / pool2d merging factor")
    p_run.add_argument("--alpha", type=float, default=None,
                       help="attn_similarity weighting factor")
    p_run.add_argument("--k-dup", type=float, default=None,
                       help="pivot_threshold dedup factor")
    p_run.add_argument("--num-pivots", type=int, default=None,
                       help="pivot_threshold number of pivots")

    # --- sweep ---
    p_sweep = sub.add_parser("sweep", help="Run a preset sweep of experiments")
    p_sweep.add_argument("--preset", required=True,
                         help="Preset: docpruner, docmerger, cps, ptm, all")
    p_sweep.add_argument("--model-name", default="vidore/colqwen2.5-v0.2")
    p_sweep.add_argument("--device", default=None)
    p_sweep.add_argument("--seed", type=int, default=0)
    p_sweep.add_argument("--outdir", default="/active_work/environment/benchmark_outputs")
    p_sweep.add_argument("--batch-doc", type=int, default=4)
    p_sweep.add_argument("--batch-query", type=int, default=8)

    # --- plot ---
    p_plot = sub.add_parser("plot", help="Plot results from output directory")
    p_plot.add_argument("--outdir", default="/active_work/environment/benchmark_outputs")

    # --- methods ---
    sub.add_parser("methods", help="List available compression methods")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "plot":
        cmd_plot(args)
    elif args.command == "methods":
        cmd_methods(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
