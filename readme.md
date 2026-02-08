# DocMerger: Adaptive Hierarchical Patch Merging for Multi-Vector Visual Document Retrieval

## Overview

Multi-vector Visual Document Retrieval (VDR) systems represent each document page as hundreds of patch-level embeddings, achieving state-of-the-art retrieval quality but incurring prohibitive storage overhead. Recent work (DocPruner) addresses this via adaptive pruning, but permanently discards information, leading to sharp performance degradation at high compression ratios (>60%).

**DocMerger** introduces importance-aware tri-level partitioning that **preserves** high-importance patches, **merges** mid-importance patches via attention-weighted clustering, and **discards** low-importance ones. This hybrid approach retains partial information at high compression ratios where pure pruning fails.

## Experiment Scope

| | Choice | Rationale |
|---|--------|-----------|
| Benchmark | ViDoRe-V2 | DocPruner's primary benchmark, 4 English datasets |
| Model | ColQwen2.5 | Most widely used, best community support |
| Metric | nDCG@5 | Standard in VDR literature |
| Hardware | NVIDIA A100 80GB | |

## Project Structure

```
docmerger/
├── README.md
├── requirements.txt
├── src/
│   ├── encoder.py          # Load ColQwen2.5, extract embeddings + attentions
│   ├── importance.py       # EOS-attention importance scoring
│   ├── compress.py         # All compression methods (DocPruner, DocMerger, baselines)
│   ├── scoring.py          # MaxSim scoring + nDCG@5 evaluation
│   └── utils.py            # Data loading, caching, plotting
├── scripts/
│   ├── 01_extract_and_cache.py     # Extract embeddings + attentions (run once)
│   ├── 02_run_all_methods.py       # Run all compression methods + evaluate
│   └── 03_plot_results.py          # Generate figures for report
└── results/
    ├── embeddings/         # Cached embeddings (gitignored)
    ├── metrics/            # JSON results per method
    └── figures/            # Plots for report
```

## Experiment Plan

### Phase 1 — Setup & Embedding Extraction 

**Goal**: Pipeline works end-to-end, embeddings cached.

- [ ] Environment setup (colpali-engine, datasets, pytrec_eval, sklearn)
- [ ] Load ColQwen2.5, run on ViDoRe-V2, verify base nDCG@5 ≈ 0.5508
- [ ] Extract and cache: document embeddings, query embeddings, last-layer attentions
- [ ] Verify attention extraction works (inspect shapes, EOS token location)

**Division of work**: One person handles model + extraction, another handles data loading + metrics pipeline.

### Phase 2 — Baselines

**Goal**: Reproduce key baselines to compare against.

We only need **4 baselines** (not all 6 from DocPruner) — enough for a convincing comparison:

| Method | Why include it |
|--------|---------------|
| Base model (no compression) | Upper bound |
| Random pruning | Naive baseline, shows compression isn't free |
| Sem-Cluster (fixed merging) | Best non-adaptive merging method |
| **DocPruner** | State-of-the-art adaptive pruning — our main comparison target |

Each method is swept across multiple compression ratios to produce the nDCG@5 vs. compression curve.

**Division of work**: One person implements DocPruner, another implements random + sem-cluster. Scoring pipeline is shared.

### Phase 3 — DocMerger 

**Goal**: Implement and evaluate our method, show improvement at high compression.

**Core method**:
```
Given importance scores I(d_j) with mean μ and std σ:

P_preserve = { d_j | I(d_j) > μ + k1·σ }              → keep as-is
P_merge    = { d_j | μ - k2·σ < I(d_j) ≤ μ + k1·σ }   → cluster + weighted merge
P_discard  = { d_j | I(d_j) ≤ μ - k2·σ }              → remove

Final embeddings D' = P_preserve ∪ {merged centroids}
```

**Merging**: Agglomerative clustering (cosine distance) on P_merge, centroids weighted by importance.

**Hyperparameter sweep**:
- k1 ∈ {0, 0.25, 0.5, 1.0}
- k2 ∈ {0, 0.25, 0.5}
- Merge ratio for cluster count (e.g., reduce P_merge to 50%, 25%, 10% of its size)

**Ablation** (small but essential for novelty claim):
1. DocMerger vs. DocPruner at same compression ratio → tri-level helps?
2. Attention-weighted merge vs. simple-average merge → weighting helps?

**Division of work**: One person implements tri-level + merging, another runs the sweep + ablation. Third person (if 3) starts drafting report.

### Phase 4 — Report & Presentation (Week 7–8)

**Goal**: Write mini-paper-style report + prepare slides.

**Report structure** (NeurIPS workshop style, ~6 pages):
1. Introduction + motivation
2. Related work (DocPruner, ColPali, token merging)
3. Method (tri-level partitioning + attention-weighted merging)
4. Experiments
   - Main result: nDCG@5 vs. compression ratio (one figure, the money plot)
   - Table: nDCG@5 at specific compression ratios (50%, 60%, 70%, 80%)
   - Ablation table (2 rows: merge tier effect, weighting effect)
5. Analysis + discussion
6. Conclusion

**Figures needed** (just 2–3):
- Fig 1: Method diagram (tri-level partitioning illustration)
- Fig 2: nDCG@5 vs. compression ratio curve (all methods)
- Fig 3 (optional): Per-dataset breakdown or attention visualization

**Division of work**: Split report sections. One person owns figures, one owns writing, one owns presentation.

## Implementation Notes

### Why Not Use vidore-benchmark Directly

Our method needs attention weights from the model's forward pass, which the standard VisionRetriever interface doesn't expose. So we write a standalone pipeline:

```
colpali-engine (model) → extract embeddings + attentions → cache to disk
                              ↓
cached embeddings → compress (our methods) → MaxSim scoring → nDCG@5
```

The key advantage: **once embeddings are cached, all compression experiments are CPU-only and fast** (~seconds per method). This means the team can iterate quickly without GPU bottlenecks.

### Embedding Cache Format

```python
# Run once on GPU (~1 hour for full ViDoRe-V2):
torch.save({
    'doc_embeddings': List[Tensor],     # [num_docs] x [num_patches, embed_dim]
    'doc_attentions': List[Tensor],     # [num_docs] x [num_heads, seq_len, seq_len]  (last layer)
    'query_embeddings': List[Tensor],   # [num_queries] x [num_tokens, embed_dim]
    'qrels': dict,                      # query_id → {doc_id: relevance}
}, 'results/embeddings/colqwen2.5_vidore_v2.pt')
```

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Can't extract attention from ColQwen2.5 | Blocks everything | Test in week 1; fall back to ColPali if needed |
| Base nDCG@5 doesn't match DocPruner | Baselines not credible | Check model version, processor settings carefully |
| DocMerger doesn't beat DocPruner | No novelty | Even if marginal, the ablation showing merge tier helps is a valid finding |
| Agglomerative clustering too slow | Can't sweep hyperparams | Switch to KMeans or reduce cluster search space |

## Timeline Summary

```
Week 1-2:  Phase 1 — Setup, extraction, cache embeddings
Week 3-4:  Phase 2 — Implement baselines (random, sem-cluster, DocPruner)
Week 5-6:  Phase 3 — Implement DocMerger + ablation experiments
Week 7-8:  Phase 4 — Report writing + presentation prep
```

Buffer: 2 months = ~8 weeks. This plan uses all 8 with no slack, but phases overlap naturally (e.g., start DocMerger while still debugging baselines, start report while still running ablations).

## References

- [DocPruner](https://arxiv.org/abs/2509.23883) — Yan et al., 2025
- [ColPali](https://arxiv.org/abs/2407.01449) — Faysse et al., 2024
- [ViDoRe-V2](https://arxiv.org/abs/2505.17166) — Macé et al., 2025
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
- [colpali-engine](https://github.com/illuin-tech/colpali)

