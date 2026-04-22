#!/usr/bin/env python3
"""
Matched-compression paired comparison:
  At ~same compression rate, does DP-PostMerge have higher nDCG@5 than DocPruner?

We pair (DocPruner k=k_A) with (DP-PostMerge k=k_B) where k_A = k_B + 0.25
(empirically PostMerge at k=-0.5 matches DocPruner at k=-0.25 in compression).

Paired Wilcoxon (one-sided: PM > DP) + bootstrap 95% CI + Holm-Bonferroni.
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

import pytrec_eval


CACHE = Path("/active_work/environment/benchmark_outputs/cache")
THRESHOLD = 0.93

# Matched compression pairs: (DocPruner k, PostMerge k, approx_compression%)
PAIRS = [
    (-0.25, -0.5,   52),   # DP k=-0.25 ~52% ≈ PM k=-0.5 ~52%
    ( 0.0,  -0.25,  63),   # DP k=0    ~63% ≈ PM k=-0.25 ~63%
    ( 0.25,  0.0,   71),   # DP k=0.25 ~71% ≈ PM k=0    ~71%
    ( 0.5,   0.25,  77),   # DP k=0.5  ~77% ≈ PM k=0.25 ~77%
    ( 1.0,   0.5,   85),   # DP k=1.0  ~86% ≈ PM k=0.5  ~82%
]


def per_query_ndcg5(run, qrels):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_5"})
    r = evaluator.evaluate(run)
    return {qid: v["ndcg_cut_5"] for qid, v in r.items()}


def run_config(embs, attns, masks, fn, kwargs, q_emb, qids, cids, qrels, device):
    pruned = [fn(embs[i], attns[i], imgpad_mask=masks[i], **kwargs).vectors
              for i in range(len(embs))]
    run = maxsim_retrieval(Q_list=q_emb, D_list=pruned,
                            query_ids=qids, corpus_ids=cids,
                            device=device, batch_q=4, batch_d=16)
    return per_query_ndcg5(run, qrels)


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n_boot)])
    return float(np.quantile(boots, alpha/2)), float(np.quantile(boots, 1-alpha/2))


def holm_bonferroni(ps, alpha=0.05):
    p = np.asarray(ps, dtype=float)
    order = p.argsort()
    m = len(p)
    adj = np.empty_like(p)
    prev = 0.0
    for i, idx in enumerate(order):
        adj[idx] = max(prev, min(1.0, (m - i) * p[idx]))
        prev = adj[idx]
    return adj


def main():
    set_seed(0)
    device = pick_device()
    model, processor = load_colqwen25("vidore/colqwen2.5-v0.2",
                                       device=device, need_attention=False)

    # Collect per-pair diffs pooled across 4 datasets
    pair_data = {i: {"diffs": [], "dp": [], "pm": [],
                      "dp_prune_total": 0, "pm_prune_total": 0,
                      "n_docs": 0}
                  for i in range(len(PAIRS))}

    for ds_name in VIDORE_V2_DATASETS:
        ds_short = ds_name.split("/")[-1]
        print(f"\n=== {ds_short} ===")
        pfx = f"{ds_short}_vidore_colqwen2.5-v0.2"
        embs = torch.load(CACHE / f"{pfx}_emb.pt", weights_only=False)
        attns = torch.load(CACHE / f"{pfx}_attn.pt", weights_only=False)
        masks = torch.load(CACHE / f"{pfx}_imgpad_mask.pt", weights_only=False)

        corpus_ids, _, query_ids, query_texts, qrels = load_vidore_v2(
            ds_name, split="test", language=None)
        q_emb = encode_queries(model, processor, query_texts,
                                device=device, batch_size=16)

        for idx, (dp_k, pm_k, tgt) in enumerate(PAIRS):
            dp_scores = run_config(embs, attns, masks,
                                    docpruner_compress, {"k": dp_k},
                                    q_emb, query_ids, corpus_ids, qrels, device)
            pm_scores = run_config(embs, attns, masks,
                                    dp_postmerge_compress,
                                    {"k": pm_k, "merge_threshold": THRESHOLD},
                                    q_emb, query_ids, corpus_ids, qrels, device)

            # Also compute actual compression rates for verification
            dp_total_before = sum(int(masks[i].sum().item()) for i in range(len(embs)))
            dp_total_after = sum(
                docpruner_compress(embs[i], attns[i], imgpad_mask=masks[i], k=dp_k).num_after
                for i in range(len(embs)))
            pm_total_after = sum(
                dp_postmerge_compress(embs[i], attns[i], imgpad_mask=masks[i],
                                       k=pm_k, merge_threshold=THRESHOLD).num_after
                for i in range(len(embs)))

            pair_data[idx]["dp_prune_total"] += dp_total_before - dp_total_after
            pair_data[idx]["pm_prune_total"] += dp_total_before - pm_total_after
            pair_data[idx]["n_docs"] += dp_total_before

            for qid in dp_scores:
                if qid in pm_scores:
                    pair_data[idx]["diffs"].append(pm_scores[qid] - dp_scores[qid])
                    pair_data[idx]["dp"].append(dp_scores[qid])
                    pair_data[idx]["pm"].append(pm_scores[qid])

            dp_pr = (dp_total_before - dp_total_after) / dp_total_before * 100
            pm_pr = (dp_total_before - pm_total_after) / dp_total_before * 100
            dp_mean = np.mean(list(dp_scores.values()))
            pm_mean = np.mean(list(pm_scores.values()))
            print(f"  pair {idx} (tgt~{tgt}%): DP(k={dp_k}, prune={dp_pr:.1f}%, "
                  f"nDCG={dp_mean:.4f})  vs  "
                  f"PM(k={pm_k}, prune={pm_pr:.1f}%, nDCG={pm_mean:.4f})  "
                  f"Δ={pm_mean-dp_mean:+.4f}")

    # ---- Aggregate stats ----
    print("\n" + "=" * 110)
    print(f"{'Target%':>8} {'DP k':>6} {'PM k':>6} {'DP prune%':>10} {'PM prune%':>10} "
          f"{'DP':>8} {'PM':>8} {'Δ':>8} {'95% CI':>22} {'p(one-sided)':>13} "
          f"{'p_adj':>8} {'sig':>4}")
    print("-" * 110)

    raw_ps = []
    rows = []
    for idx, (dp_k, pm_k, tgt) in enumerate(PAIRS):
        d = pair_data[idx]
        diffs = np.array(d["diffs"])
        dp_arr = np.array(d["dp"])
        pm_arr = np.array(d["pm"])
        ci_lo, ci_hi = bootstrap_ci(diffs)
        # one-sided Wilcoxon: H1: PM > DP
        try:
            stat, p = sstats.wilcoxon(pm_arr, dp_arr, alternative="greater",
                                       zero_method="wilcox")
        except ValueError:
            p = 1.0
        raw_ps.append(p)
        dp_prune = d["dp_prune_total"] / d["n_docs"] * 100
        pm_prune = d["pm_prune_total"] / d["n_docs"] * 100
        rows.append({
            "target_pct": tgt, "dp_k": dp_k, "pm_k": pm_k,
            "dp_prune": dp_prune, "pm_prune": pm_prune,
            "dp_mean": float(dp_arr.mean()), "pm_mean": float(pm_arr.mean()),
            "mean_diff": float(diffs.mean()), "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_raw": p, "n": int(len(diffs)),
        })

    adj = holm_bonferroni(raw_ps)
    for r, p_adj in zip(rows, adj):
        r["p_adj"] = float(p_adj)
        r["sig"] = ("***" if p_adj < 0.001 else
                     "**"  if p_adj < 0.01  else
                     "*"   if p_adj < 0.05  else "n.s.")
        print(f"{r['target_pct']:>7}% {r['dp_k']:>6} {r['pm_k']:>6} "
              f"{r['dp_prune']:>9.1f}% {r['pm_prune']:>9.1f}% "
              f"{r['dp_mean']:>8.4f} {r['pm_mean']:>8.4f} "
              f"{r['mean_diff']:>+8.4f} [{r['ci_lo']:>+.4f},{r['ci_hi']:>+.4f}] "
              f"{r['p_raw']:>13.5f} {r['p_adj']:>8.5f} {r['sig']:>4}")
    print("=" * 110)

    out = Path("/active_work/environment/benchmark_outputs/stats_matched.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
