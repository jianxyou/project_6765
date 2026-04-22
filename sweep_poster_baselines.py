#!/usr/bin/env python3
"""
Sweep for poster figure: 5 baselines (random, sem_cluster, pool1d, pool2d,
attn_similarity, pivot_threshold) × 4 ViDoRe-V2 datasets × multiple
hyperparameter values.

Uses cached embeddings (no re-encoding).
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

from benchmark.experiment import run_experiment
from benchmark.data import VIDORE_V2_DATASETS


# Hyperparameter grids matching DocPruner paper's Appendix F
HPARAMS = {
    "random": [{"ratio": r} for r in [0.1, 0.3, 0.5, 0.7, 0.9]],
    "sem_cluster": [{"merging_factor": m} for m in [2, 4, 9, 16, 25]],
    "pool1d": [{"merging_factor": m} for m in [2, 4, 9, 16, 25]],
    # "pool2d": skipped — needs proper grid info from re-encoding
    "attn_similarity": [
        {"k": k, "alpha": 0.5}
        for k in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    ],
    "pivot_threshold": [
        {"k": k, "k_dup": 0.0, "num_pivots": 10}
        for k in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    ],
}


def main():
    all_configs = []
    for method, hp_list in HPARAMS.items():
        for hp in hp_list:
            for ds in VIDORE_V2_DATASETS:
                cfg = {"dataset": ds, "pruner": method, **hp}
                all_configs.append(cfg)

    print(f"Total experiments: {len(all_configs)}")
    for i, cfg in enumerate(all_configs, 1):
        print(f"\n[{i}/{len(all_configs)}] {cfg['pruner']} on {cfg['dataset'].split('/')[-1]} {hp}")
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"Sweep complete: {len(all_configs)} experiments")
    print("=" * 70)


if __name__ == "__main__":
    main()
