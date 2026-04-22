#!/usr/bin/env python3
"""
Re-run sweep with FIXED baselines per PTM paper Appendix D.

Target methods (all implementations updated 2026-04-15 to match Appendix D):
  - sem_cluster: L2 normalize + Ward linkage
  - pool1d:      zero padding + mask-aware averaging
  - random:      image-patches only (consistent with other adaptive methods)

Benchmark: ViDoRe-V2 (4 datasets) × ColQwen2.5
Hyperparameter ranges: exactly as in Appendix D (no search, full grid).

Uses cached embeddings on HPC (no re-encoding).
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

from benchmark.experiment import run_experiment
from benchmark.data import VIDORE_V2_DATASETS


# Appendix D hyperparameter ranges (full grid, no search)
HPARAMS = {
    "random":      [{"ratio": r}            for r in [0.1, 0.3, 0.5, 0.7, 0.9]],
    "sem_cluster": [{"merging_factor": mf}  for mf in [2, 4, 9, 16, 25]],
    "pool1d":      [{"merging_factor": mf}  for mf in [2, 4, 9, 16, 25]],
}


def main():
    configs = []
    for method, hp_list in HPARAMS.items():
        for hp in hp_list:
            for ds in VIDORE_V2_DATASETS:
                configs.append({"dataset": ds, "pruner": method, **hp})

    print(f"Total experiments: {len(configs)} "
          f"(3 methods × 5 hparams × 4 datasets)")

    for i, cfg in enumerate(configs, 1):
        params = {k: v for k, v in cfg.items() if k not in ("dataset", "pruner")}
        ds_short = cfg["dataset"].split("/")[-1]
        print(f"\n[{i}/{len(configs)}] {cfg['pruner']} {params} on {ds_short}")
        try:
            run_experiment(cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"Sweep complete: {len(configs)} experiments")
    print("=" * 70)


if __name__ == "__main__":
    main()
