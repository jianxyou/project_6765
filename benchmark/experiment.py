"""
Core experiment pipeline: load data -> encode -> prune -> retrieve -> evaluate -> save.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# HuggingFace cache must be on HPC disk, not thin client
os.environ.setdefault("HF_HOME", "/active_work/environment/.cache/huggingface")

import torch
from tqdm import tqdm

from .utils import set_seed, pick_device
from .model import load_colqwen25, encode_corpus, encode_queries
from .data import load_vidore_v2, load_vidore_v1, VIDORE_V1_DATASETS
from .eval import maxsim_retrieval, evaluate_ndcg5
from .methods import get_method
from .methods.adaptive import compute_entropy_threshold
from .methods.learned import train_learned_projection


def run_experiment(config: dict) -> dict:
    """
    Run a single experiment with the given config.

    Required config keys:
        dataset, pruner
    Optional config keys (with defaults):
        model_name, device, seed, split, language,
        batch_doc, batch_query, batch_score_q, batch_score_d,
        outdir, clear_cache,
        + any method-specific params (k, k1, k2, merge_ratio, cluster_ratio, etc.)
    """
    # Defaults
    cfg = {
        "model_name": "vidore/colqwen2.5-v0.2",
        "device": None,
        "seed": 0,
        "split": "test",
        "language": None,  # use all languages (V2 has en/fr/de/es)
        "batch_doc": 4,
        "batch_query": 8,
        "batch_score_q": 4,
        "batch_score_d": 16,
        "outdir": "/active_work/environment/benchmark_outputs",
        "clear_cache": False,
    }
    cfg.update(config)

    set_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    compress_fn, needs_attention = get_method(cfg["pruner"])
    dataset_short = cfg["dataset"].split("/")[-1]

    print(f"\n{'='*70}")
    print(f"Experiment: {cfg['pruner']} on {dataset_short}")
    print(f"{'='*70}")

    # 1. Load dataset (auto-detect V1 vs V2/V3 format)
    print(f"\n[1/5] Loading dataset ({dataset_short})...")
    if cfg["dataset"] in VIDORE_V1_DATASETS:
        corpus_ids, corpus_images, query_ids, query_texts, qrels = \
            load_vidore_v1(cfg["dataset"], split=cfg["split"])
    else:
        corpus_ids, corpus_images, query_ids, query_texts, qrels = \
            load_vidore_v2(cfg["dataset"], split=cfg["split"], language=cfg["language"])

    n_corpus = len(corpus_ids)
    n_queries = len(query_ids)
    model = None
    processor = None

    # 2-3. Encode corpus (with caching)
    cache_dir = Path(cfg["outdir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_short = cfg["model_name"].replace("/", "_")
    cache_prefix = cache_dir / f"{dataset_short}_{model_short}"
    cache_emb = Path(f"{cache_prefix}_emb.pt")
    cache_attn = Path(f"{cache_prefix}_attn.pt")
    cache_mask = Path(f"{cache_prefix}_imgpad_mask.pt")
    cache_grid = Path(f"{cache_prefix}_grid.pt")

    if cfg["clear_cache"]:
        for p in [cache_emb, cache_attn, cache_mask, cache_grid]:
            if p.exists():
                p.unlink()

    if cache_emb.exists():
        print(f"\n[2-3/5] Loading corpus from cache...")
        doc_emb_list = torch.load(cache_emb, weights_only=False)
        doc_attn_list = torch.load(cache_attn, weights_only=False) if cache_attn.exists() else [None] * len(doc_emb_list)
        doc_mask_list = torch.load(cache_mask, weights_only=False) if cache_mask.exists() else [None] * len(doc_emb_list)
        doc_grid_list = torch.load(cache_grid, weights_only=False) if cache_grid.exists() else [None] * len(doc_emb_list)
        print(f"  Loaded {len(doc_emb_list)} docs from cache")
    else:
        print(f"\n[2/5] Loading ColQwen2.5 (eager mode)...")
        model, processor = load_colqwen25(cfg["model_name"], device=device, need_attention=True)

        print(f"\n[3/5] Encoding corpus ({n_corpus} docs)...")
        doc_emb_list, doc_attn_list, doc_mask_list, doc_grid_list = encode_corpus(
            model, processor, corpus_images, device, batch_size=cfg["batch_doc"])

        avg_patches = sum(e.shape[0] for e in doc_emb_list) / max(1, len(doc_emb_list))
        print(f"  Avg tokens: {avg_patches:.0f}")

        torch.save(doc_emb_list, cache_emb)
        torch.save(doc_attn_list, cache_attn)
        torch.save(doc_mask_list, cache_mask)
        torch.save(doc_grid_list, cache_grid)
        print(f"  Cache saved to {cache_dir}")

    # 4. Prune
    print(f"\n[4/5] Pruning ({cfg['pruner']})...")

    # Special prepare steps
    method_kwargs = {k: v for k, v in cfg.items()
                     if k not in ("dataset", "pruner", "model_name", "device", "seed",
                                  "split", "language", "batch_doc", "batch_query",
                                  "batch_score_q", "batch_score_d", "outdir", "clear_cache")}

    # Adaptive: pre-compute entropies
    if cfg["pruner"] == "adaptive":
        entropies, ent_threshold = compute_entropy_threshold(
            doc_attn_list, doc_mask_list,
            percentile=cfg.get("entropy_percentile", 50.0))
        n_prune = sum(1 for e in entropies if e < ent_threshold)
        print(f"  Entropy routing: {n_prune} -> DocPruner, {n_corpus - n_prune} -> DocMerger")

    # Learned: train projection
    proj_model = None
    if cfg["pruner"] == "learned":
        if model is None:
            model, processor = load_colqwen25(cfg["model_name"], device=device, need_attention=False)
        q_emb_train = encode_queries(model, processor, query_texts, device, batch_size=cfg["batch_query"])
        proj_model = train_learned_projection(
            doc_emb_list, doc_attn_list, doc_mask_list,
            q_emb_train, qrels, query_ids, corpus_ids,
            k1=cfg.get("k1", 1.0), k2=cfg.get("k2", 0.0),
            top_k_ratio=cfg.get("top_k_ratio", 0.25),
            lr=cfg.get("lp_lr", 1e-3), epochs=cfg.get("lp_epochs", 30),
        )

    pruned_docs: List[torch.Tensor] = []
    per_doc_stats = []

    for i in tqdm(range(n_corpus), desc="Pruning"):
        kwargs = dict(method_kwargs)
        mask_i = doc_mask_list[i] if i < len(doc_mask_list) else None

        # Inject special per-doc params
        if cfg["pruner"] == "adaptive":
            kwargs["entropy"] = entropies[i]
            kwargs["entropy_threshold"] = ent_threshold
        if cfg["pruner"] == "learned":
            kwargs["proj_model"] = proj_model
        if cfg["pruner"] == "pool2d":
            kwargs["grid_hw"] = doc_grid_list[i] if i < len(doc_grid_list) else None

        result = compress_fn(
            doc_emb_list[i],
            doc_attn_list[i] if doc_attn_list[i] is not None else torch.zeros(doc_emb_list[i].shape[0]),
            imgpad_mask=mask_i,
            **kwargs,
        )
        pruned_docs.append(result.vectors)
        per_doc_stats.append({
            "doc_id": corpus_ids[i],
            "num_before": result.num_before,
            "num_after": result.num_after,
            "pruning_ratio": round(result.pruning_ratio, 4),
        })

    avg_pruning = sum(s["pruning_ratio"] for s in per_doc_stats) / max(1, len(per_doc_stats))
    avg_after = sum(s["num_after"] for s in per_doc_stats) / max(1, len(per_doc_stats))
    avg_before = sum(s["num_before"] for s in per_doc_stats) / max(1, len(per_doc_stats))
    print(f"  Avg pruning: {avg_pruning*100:.1f}%, patches: {avg_before:.0f} -> {avg_after:.0f}")

    del doc_emb_list, doc_attn_list, doc_mask_list
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Encode queries + MaxSim retrieval + evaluate
    if model is None:
        print(f"\n  Loading ColQwen2.5 for query encoding...")
        model, processor = load_colqwen25(cfg["model_name"], device=device, need_attention=False)

    print(f"\n[5/5] Query encoding + MaxSim retrieval...")
    q_emb_list = encode_queries(model, processor, query_texts, device, batch_size=cfg["batch_query"])

    run = maxsim_retrieval(
        Q_list=q_emb_list, D_list=pruned_docs,
        query_ids=query_ids, corpus_ids=corpus_ids,
        device=device, batch_q=cfg["batch_score_q"], batch_d=cfg["batch_score_d"],
    )
    ndcg5 = evaluate_ndcg5(run, qrels)

    # Build metrics
    metrics = {
        "dataset": cfg["dataset"],
        "dataset_short": dataset_short,
        "model_name": cfg["model_name"],
        "pruner": cfg["pruner"],
        "ndcg@5": round(ndcg5, 4),
        "avg_pruning_ratio": round(avg_pruning, 4),
        "avg_patches_before": round(avg_before, 1),
        "avg_patches_after": round(avg_after, 1),
        "num_corpus": n_corpus,
        "num_queries": n_queries,
    }
    # Record method-specific params
    for key in ("k", "k1", "k2", "merge_ratio", "cluster_ratio", "dedup_threshold",
                "svd_energy", "ratio", "ptm_k", "ptm_m", "top_k_ratio",
                "entropy_percentile", "k_prune",
                "cps_cluster_ratio", "cps_dedup_thresh", "cps_svd_energy"):
        if key in cfg:
            metrics[key] = cfg[key]

    # Save
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    suffix = _make_suffix(cfg)
    fname = f"{dataset_short}_{suffix}"
    (outdir / f"metrics_{fname}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print(f"Result: nDCG@5 = {ndcg5:.4f} | Pruning = {avg_pruning*100:.1f}%")
    print(f"{'='*70}")

    return metrics


def _make_suffix(cfg: dict) -> str:
    pruner = cfg["pruner"]
    if pruner == "docpruner":
        return f"docpruner_k{cfg.get('k', -0.25)}"
    if pruner.startswith("cps"):
        return f"{pruner}_r{cfg.get('cluster_ratio', cfg.get('cps_cluster_ratio', 0.3))}"
    if pruner == "ptm":
        return f"ptm_k{cfg.get('k', cfg.get('ptm_k', -0.75))}_m{cfg.get('m', cfg.get('ptm_m', 4))}"
    if pruner == "docmerger" or pruner == "docmerger_avg":
        return f"{pruner}_k1{cfg.get('k1', 0.5)}_k2{cfg.get('k2', 0.25)}"
    if pruner == "dp_postmerge":
        t = cfg.get('merge_threshold', 0.95)
        return f"dp_postmerge_k{cfg.get('k', -0.25)}_t{t}"
    if pruner == "dp_dedup":
        return f"dp_dedup_k{cfg.get('k', -0.25)}"
    if pruner == "random":
        return f"random_r{cfg.get('ratio', 0.5)}"
    if pruner in ("sem_cluster", "pool1d", "pool2d"):
        return f"{pruner}_mf{cfg.get('merging_factor', 4)}"
    if pruner == "attn_similarity":
        return f"attn_similarity_k{cfg.get('k', -0.25)}_a{cfg.get('alpha', 0.5)}"
    if pruner == "pivot_threshold":
        return (f"pivot_threshold_k{cfg.get('k', -0.25)}"
                f"_kd{cfg.get('k_dup', 0.0)}"
                f"_np{cfg.get('num_pivots', 10)}")
    # For any method that uses k, include it in suffix
    if "k" in cfg:
        return f"{pruner}_k{cfg['k']}"
    return pruner


def run_sweep(configs: List[dict], outdir: str = "outputs") -> List[dict]:
    """Run multiple experiments and save aggregated results."""
    all_metrics = []
    for cfg in configs:
        cfg.setdefault("outdir", outdir)
        m = run_experiment(cfg)
        all_metrics.append(m)

    outpath = Path(outdir) / "sweep_metrics.json"
    outpath.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"\nSweep done! {len(all_metrics)} experiments saved to {outpath}")

    _print_summary_table(all_metrics)
    return all_metrics


def _print_summary_table(metrics_list: List[dict]):
    from collections import defaultdict

    ds_order = ["esg_reports_v2", "biomedical_lectures_eng_v2",
                "economics_reports_v2", "esg_reports_human_labeled_v2"]
    ds_short = ["ESG", "Bio", "Econ", "ESG-H"]

    print(f"\n{'='*100}")
    print(f"{'Method':<25} {'ESG':>8} {'Bio':>8} {'Econ':>8} {'ESG-H':>8} {'Avg':>8} {'Prune%':>8}")
    print("-" * 100)

    grouped = defaultdict(dict)
    for m in metrics_list:
        key = _make_suffix(m)
        grouped[key][m["dataset_short"]] = m

    for suffix, ds_dict in sorted(grouped.items()):
        scores = [ds_dict.get(ds, {}).get("ndcg@5", 0) for ds in ds_order]
        prunes = [ds_dict.get(ds, {}).get("avg_pruning_ratio", 0) for ds in ds_order]
        valid = [s for s in scores if s > 0]
        avg_score = sum(valid) / max(1, len(valid))
        avg_prune = sum(prunes) / max(1, len(prunes))
        print(f"{suffix:<25} "
              f"{scores[0]:>8.4f} {scores[1]:>8.4f} {scores[2]:>8.4f} {scores[3]:>8.4f} "
              f"{avg_score:>8.4f} {avg_prune*100:>7.1f}%")
    print(f"{'='*100}")
