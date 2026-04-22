#!/usr/bin/env python3
"""
Paired Wilcoxon + bootstrap 95% CI for DP-PostMerge vs DocPruner.

For each k in DOCPRUNER_K_VALUES:
  - Pool per-query nDCG@5 across 4 ViDoRe-V2 datasets
  - Paired Wilcoxon signed-rank
  - Bootstrap 95% CI of mean Δ
  - Holm-Bonferroni correction across 6 k values
"""

import os
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

from benchmark.data import VIDORE_V2_DATASETS, load_vidore_v2
from benchmark.methods.docpruner import docpruner_compress
from benchmark.methods.dp_postmerge import dp_postmerge_compress
from benchmark.model import load_colqwen25, encode_queries
from benchmark.eval import maxsim_retrieval
from benchmark.utils import pick_device, set_seed

try:
    import pytrec_eval
except ImportError:
    print("ERROR: pytrec_eval required"); exit(1)


CACHE = Path("/active_work/environment/benchmark_outputs/cache")
K_VALUES = [-0.5, -0.25, 0, 0.25, 0.5, 1.0]
THRESHOLD = 0.93


def per_query_ndcg5(run, qrels):
    """Return dict {qid: ndcg_cut_5}."""
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_5"})
    r = evaluator.evaluate(run)
    return {qid: v["ndcg_cut_5"] for qid, v in r.items()}


def run_one_config(emb_list, attn_list, mask_list, method_fn, method_kwargs,
                   q_emb_list, query_ids, corpus_ids, qrels, device):
    """Run a compression method + retrieval, return per-query nDCG dict."""
    pruned = []
    for i in range(len(emb_list)):
        r = method_fn(emb_list[i], attn_list[i],
                       imgpad_mask=mask_list[i], **method_kwargs)
        pruned.append(r.vectors)
    run = maxsim_retrieval(
        Q_list=q_emb_list, D_list=pruned,
        query_ids=query_ids, corpus_ids=corpus_ids,
        device=device, batch_q=4, batch_d=16)
    return per_query_ndcg5(run, qrels)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)


def holm_bonferroni(ps, alpha=0.05):
    """Return adjusted p-values."""
    p_arr = np.asarray(ps, dtype=float)
    order = p_arr.argsort()
    m = len(p_arr)
    adj = np.empty_like(p_arr)
    prev = 0.0
    for i, idx in enumerate(order):
        adj[idx] = max(prev, min(1.0, (m - i) * p_arr[idx]))
        prev = adj[idx]
    return adj


def main():
    set_seed(0)
    device = pick_device()
    print(f"Loading ColQwen2.5 for query encoding...")
    model, processor = load_colqwen25("vidore/colqwen2.5-v0.2",
                                       device=device, need_attention=False)

    # ---- Per-dataset per-query collection ----
    # pooled: {k: {method: [all per-query scores pooled across datasets]}}
    pooled = {k: {"docpruner": [], "postmerge": []} for k in K_VALUES}
    per_query_diffs = {k: [] for k in K_VALUES}

    for ds_name in VIDORE_V2_DATASETS:
        ds_short = ds_name.split("/")[-1]
        print(f"\n=== {ds_short} ===")
        pfx = f"{ds_short}_vidore_colqwen2.5-v0.2"
        embs = torch.load(CACHE / f"{pfx}_emb.pt", weights_only=False)
        attns = torch.load(CACHE / f"{pfx}_attn.pt", weights_only=False)
        masks = torch.load(CACHE / f"{pfx}_imgpad_mask.pt", weights_only=False)

        corpus_ids, _, query_ids, query_texts, qrels = load_vidore_v2(
            ds_name, split="test", language=None)
        print(f"  corpus={len(embs)} queries={len(query_ids)}")

        q_emb = encode_queries(model, processor, query_texts,
                                device=device, batch_size=16)

        for k in K_VALUES:
            dp_scores = run_one_config(embs, attns, masks,
                                        docpruner_compress, {"k": k},
                                        q_emb, query_ids, corpus_ids,
                                        qrels, device)
            pm_scores = run_one_config(embs, attns, masks,
                                        dp_postmerge_compress,
                                        {"k": k, "merge_threshold": THRESHOLD},
                                        q_emb, query_ids, corpus_ids,
                                        qrels, device)
            # Paired diffs — same qid order for both
            for qid in dp_scores:
                if qid in pm_scores:
                    per_query_diffs[k].append(pm_scores[qid] - dp_scores[qid])
                    pooled[k]["docpruner"].append(dp_scores[qid])
                    pooled[k]["postmerge"].append(pm_scores[qid])
            n_q = len(dp_scores)
            dp_mean = np.mean(list(dp_scores.values()))
            pm_mean = np.mean(list(pm_scores.values()))
            print(f"  k={k:>5}:  DP={dp_mean:.4f}  PM={pm_mean:.4f}  "
                  f"Δ={pm_mean-dp_mean:+.4f}  (n={n_q})")

    # ---- Aggregate statistics ----
    print("\n" + "=" * 95)
    print(f"{'k':>6} {'n':>5} {'DocPruner':>10} {'PostMerge':>10} {'mean Δ':>10} "
          f"{'95% CI':>22} {'p_Wilcoxon':>11} {'p_adj':>9} {'sig':>5}")
    print("-" * 95)

    raw_ps = []
    rows = []
    for k in K_VALUES:
        diffs = np.array(per_query_diffs[k])
        dp = np.array(pooled[k]["docpruner"])
        pm = np.array(pooled[k]["postmerge"])
        mean_diff = diffs.mean()
        ci_lo, ci_hi = bootstrap_ci(diffs)
        # Wilcoxon needs non-zero diffs, handle edge case
        try:
            stat, p = sstats.wilcoxon(pm, dp, alternative="two-sided",
                                       zero_method="wilcox")
        except ValueError:
            p = 1.0
        raw_ps.append(p)
        rows.append({
            "k": k, "n": len(diffs),
            "dp_mean": dp.mean(), "pm_mean": pm.mean(),
            "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_raw": p,
        })

    adj = holm_bonferroni(raw_ps)
    for r, p_adj in zip(rows, adj):
        r["p_adj"] = float(p_adj)
        if p_adj < 0.001:
            sig = "***"
        elif p_adj < 0.01:
            sig = "**"
        elif p_adj < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        r["sig"] = sig
        print(f"{r['k']:>6} {r['n']:>5} {r['dp_mean']:>10.4f} {r['pm_mean']:>10.4f} "
              f"{r['mean_diff']:>+10.4f} [{r['ci_lo']:>+.4f}, {r['ci_hi']:>+.4f}] "
              f"{r['p_raw']:>11.4f} {r['p_adj']:>9.4f} {sig:>5}")
    print("=" * 95)

    # Save JSON
    out = Path("/active_work/environment/benchmark_outputs/stats_dp_vs_pm.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
