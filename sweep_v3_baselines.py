#!/usr/bin/env python3
"""
V3 sweep: same 5 baselines as V2 poster (random, sem_cluster, pool1d,
attn_similarity, pivot_threshold) + dp_postmerge theta sweep.

Uses cached V3 embeddings (no re-encoding).
Total: 273 experiments (~4.5 hours).
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

from benchmark.experiment import run_experiment
from benchmark.data import VIDORE_V3_DATASETS


HPARAMS = {
    "random": [{"ratio": r} for r in [0.1, 0.3, 0.5, 0.7, 0.9]],
    "sem_cluster": [{"merging_factor": m} for m in [2, 4, 9, 16, 25]],
    "pool1d": [{"merging_factor": m} for m in [2, 4, 9, 16, 25]],
    "attn_similarity": [
        {"k": k, "alpha": 0.5}
        for k in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    ],
    "pivot_threshold": [
        {"k": k, "k_dup": 0.0, "num_pivots": 10}
        for k in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    ],
    # dp_postmerge: theta sweep (already have theta=0.93 in v3_sweep_all.json)
    "dp_postmerge": [
        {"k": k, "merge_threshold": t}
        for k in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
        for t in [0.90, 0.95]
    ],
}


def main():
    all_configs = []
    for method, hp_list in HPARAMS.items():
        for hp in hp_list:
            for ds in VIDORE_V3_DATASETS:
                cfg = {"dataset": ds, "pruner": method, **hp}
                all_configs.append(cfg)

    print(f"Total experiments: {len(all_configs)}")
    print(f"Methods: {list(HPARAMS.keys())}")
    print(f"Datasets: {[d.split('/')[-1] for d in VIDORE_V3_DATASETS]}")

    for i, cfg in enumerate(all_configs, 1):
        ds_short = cfg['dataset'].split('/')[-1]
        extras = {k: v for k, v in cfg.items() if k not in ('dataset', 'pruner')}
        print(f"\n[{i}/{len(all_configs)}] {cfg['pruner']} on {ds_short} {extras}")
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"V3 sweep complete: {len(all_configs)} experiments")
    print("=" * 70)


if __name__ == "__main__":
    main()
