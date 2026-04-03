#!/usr/bin/env python3
"""
Run all experiments: DocPruner baseline + new methods on ViDoRe-V2.
Designed to be submitted via sbatch.
"""

import json
import os
import sys
from pathlib import Path

# Force HF cache to local disk (avoid thin client disconnect)
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/active_work/environment/.cache/huggingface")

import torch
# Limit CPU threads per worker to avoid contention when running 4 workers in parallel
n_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
threads_per_worker = max(1, 32 // n_gpus)  # 32 CPUs / 4 GPUs = 8 threads each
torch.set_num_threads(threads_per_worker)
os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)

from benchmark.data import VIDORE_V2_DATASETS
from benchmark.experiment import run_experiment

OUTDIR = "/active_work/environment/benchmark_outputs"


def build_configs():
    """Build all experiment configs."""
    configs = []

    for ds in VIDORE_V2_DATASETS:
        # === Baselines ===
        configs.append({"dataset": ds, "pruner": "identity"})
        configs.append({"dataset": ds, "pruner": "docpruner", "k": -0.25})
        configs.append({"dataset": ds, "pruner": "docpruner", "k": -0.5})
        configs.append({"dataset": ds, "pruner": "docpruner", "k": 0.0})

        # === MMR: sweep lambda and target_ratio ===
        for lam in [0.5, 0.6, 0.7, 0.8]:
            for tr in [0.35, 0.45, 0.5, 0.55]:
                configs.append({
                    "dataset": ds, "pruner": "mmr",
                    "target_ratio": tr, "lam": lam,
                })

        # === Attention-FPS: sweep alpha and target_ratio ===
        for alpha in [0.3, 0.5, 0.7]:
            for tr in [0.35, 0.45, 0.5, 0.55]:
                configs.append({
                    "dataset": ds, "pruner": "attn_fps",
                    "target_ratio": tr, "alpha": alpha,
                })

        # === DocPruner + Rebalance: sweep params ===
        for lam in [0.5, 0.6, 0.7]:
            for tr in [0.35, 0.45, 0.5, 0.55]:
                for k_loose in [-0.75, -0.5, -0.25]:
                    configs.append({
                        "dataset": ds, "pruner": "dp_rebalance",
                        "target_ratio": tr, "k_loose": k_loose, "lam": lam,
                    })

        # === CPS variants (re-run with matching compression) ===
        for cr in [0.35, 0.45, 0.5]:
            configs.append({
                "dataset": ds, "pruner": "cps_attn",
                "cluster_ratio": cr, "dedup_threshold": 0.95,
            })

    return configs


def main():
    configs = build_configs()
    print(f"Total experiments: {len(configs)}")

    # If a GPU index is passed, split work across GPUs
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Split configs across GPUs
    my_configs = [c for i, c in enumerate(configs) if i % n_gpus == gpu_id]
    print(f"GPU {gpu_id}/{n_gpus}: running {len(my_configs)} experiments")

    all_metrics = []
    for i, cfg in enumerate(my_configs):
        cfg["outdir"] = OUTDIR
        # When using CUDA_VISIBLE_DEVICES, the visible GPU is always cuda:0
        cfg["device"] = "cuda:0"
        print(f"\n[{i+1}/{len(my_configs)}] {cfg['pruner']} on {cfg['dataset'].split('/')[-1]}")
        try:
            m = run_experiment(cfg)
            all_metrics.append(m)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_metrics.append({"error": str(e), **cfg})

    # Save this GPU's results
    outpath = Path(OUTDIR) / f"all_results_gpu{gpu_id}.json"
    outpath.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"\nDone! Saved {len(all_metrics)} results to {outpath}")


if __name__ == "__main__":
    main()
