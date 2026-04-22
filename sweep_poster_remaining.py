#!/usr/bin/env python3
"""
Continuation sweep: runs experiments that were missing/failed in the previous run.

Missing:
  - attn_similarity k={0.5, 1.0} × 4 datasets = 8
  - pivot_threshold k={-0.5, -0.25, 0, 0.25, 0.5, 1.0} × 4 datasets = 24
  - pool2d mf={4, 9, 16, 25} × 4 datasets = 16
Total: 48 experiments
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

from benchmark.experiment import run_experiment
from benchmark.data import VIDORE_V2_DATASETS


HPARAMS = {
    "attn_similarity": [
        {"k": k, "alpha": 0.5}
        for k in [0.5, 1.0]  # k=-0.5,-0.25,0,0.25 already done
    ],
    "pivot_threshold": [
        {"k": k, "k_dup": 0.0, "num_pivots": 10}
        for k in [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    ],
    "pool2d": [
        {"merging_factor": mf}
        for mf in [4, 9, 16, 25]
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
        ds_short = cfg['dataset'].split('/')[-1]
        extra = {k: v for k, v in cfg.items() if k not in ('dataset', 'pruner')}
        print(f"\n[{i}/{len(all_configs)}] {cfg['pruner']} on {ds_short} {extra}")
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"Continuation sweep complete: {len(all_configs)} experiments")
    print("=" * 70)


if __name__ == "__main__":
    main()
