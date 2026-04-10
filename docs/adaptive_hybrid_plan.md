# Direction 1: Adaptive Hybrid — Implementation & Experiment Plan

## Motivation

DocMerger outperforms DocPruner at >75% compression but underperforms at <60%. The reason: merging replaces patches with lossy centroids, which hurts when you could have just kept the originals. The key insight is that **which method is better depends on the document's attention distribution**:

- **Concentrated attention** (few dominant patches) → DocPruner wins. The important patches are clearly separable; pruning keeps them intact.
- **Diffuse attention** (many moderately important patches) → DocMerger wins. No single patch dominates; merging preserves distributed information better than discarding.

The Adaptive Hybrid selects the compression strategy per-document based on attention entropy, combining the strengths of both methods.

---

## Implementation Plan

### Step 1: Compute attention entropy per document

Add a function that computes the entropy of the normalized EOS attention distribution over image patches:

```python
def attention_entropy(attention_scores: torch.Tensor, imgpad_mask: torch.Tensor) -> float:
    """
    H = -Σ p(i) log p(i) where p(i) = softmax(attention over image patches).
    High H → diffuse attention → prefer merging.
    Low H → concentrated attention → prefer pruning.
    """
    img_attn = attention_scores[imgpad_mask]
    p = torch.softmax(img_attn, dim=0)
    log_p = torch.log(p + 1e-10)
    return -(p * log_p).sum().item()
```

Location: add after `docmerger_compress()` in `docpruner_replicate.py` (~line 470).

### Step 2: Implement adaptive_hybrid_compress

```python
def adaptive_hybrid_compress(
    embeddings, attention_scores, imgpad_mask,
    k_prune, k1_merge, k2_merge, merge_ratio,
    entropy_threshold=None,  # None = auto (median)
    corpus_entropies=None,   # pre-computed for auto threshold
) -> PruneResult:
    """
    Per-document selection:
    - If entropy < threshold → docpruner_prune(k=k_prune)
    - If entropy >= threshold → docmerger_compress(k1, k2, merge_ratio)
    """
```

Location: add after `attention_entropy()`.

### Step 3: Two-pass execution in run_experiment

The adaptive method needs a two-pass approach:

1. **Pass 1**: Compute attention entropy for all documents (fast, no compression).
2. **Compute threshold**: median entropy across corpus (or sweep percentiles).
3. **Pass 2**: For each document, route to DocPruner or DocMerger based on threshold.

Modify `run_experiment()` to handle `--pruner adaptive`:
- Before the pruning loop, compute all entropies.
- Set threshold (median by default, or from `--entropy-threshold` arg).
- In the loop, dispatch per document.

### Step 4: CLI integration

Add to argparse:
```
--pruner adaptive
--k-prune       (float, default=-0.25, DocPruner k for low-entropy docs)
--entropy-threshold  (float, optional, auto=median if not set)
--entropy-percentile (float, default=50, alternative to fixed threshold)
```

Reuse existing `--k1`, `--k2`, `--merge-ratio` for the DocMerger side.

---

## Experiment Plan

### Experiment 1: Entropy distribution analysis (no compression)

**Goal**: Understand the attention entropy landscape across datasets.

- Compute entropy for every document in all 4 datasets.
- Plot entropy histograms per dataset.
- Report mean, std, median, min, max.
- Visualize: do some datasets have more diffuse attention than others?

**Expected output**: Histogram figure + stats table. This motivates the adaptive approach — if entropy varies significantly within a dataset, per-document selection should help.

### Experiment 2: Oracle experiment (upper bound)

**Goal**: Establish the ceiling for adaptive selection.

- For each document, run both DocPruner and DocMerger at the same target compression.
- Pick whichever gives higher per-document MaxSim scores (oracle selection).
- Report nDCG@5 with oracle routing.

**Expected output**: Upper bound on adaptive hybrid performance. If the oracle gain is small, the adaptive approach has limited upside.

### Experiment 3: Entropy threshold sweep

**Goal**: Find the best threshold and validate the entropy heuristic.

Sweep entropy percentile thresholds: {25, 30, 40, 50, 60, 70, 75}

For each threshold, at each compression level:
- Documents below threshold → DocPruner
- Documents above threshold → DocMerger
- Report overall nDCG@5 and the fraction routed to each method.

Fixed configs for the two methods (matched compression):

| Target compression | DocPruner k | DocMerger (k1, k2, mr) |
|-------------------|-------------|------------------------|
| ~70% | k=0.25 | k1=0.5, k2=0, mr=0.5 |
| ~80% | k=0.50 | k1=1.0, k2=0, mr=0.25 |
| ~85% | k=1.00 | k1=1.0, k2=0, mr=0.1 |

**Expected output**: Table of nDCG@5 vs threshold percentile at each compression level. Identify the sweet spot.

### Experiment 4: Full comparison

**Goal**: Head-to-head comparison of all methods across the full compression range.

| Method | Configs |
|--------|---------|
| Identity | baseline |
| DocPruner | k ∈ {-0.5, -0.25, 0, 0.25, 0.5, 1.0} |
| DocMerger | best configs from Phase 3 |
| **Adaptive Hybrid** | best threshold from Exp 3, matched compression pairs |

Run on all 4 datasets. Generate the updated "money plot" (nDCG@5 vs compression) with three lines: DocPruner, DocMerger, Adaptive Hybrid.

**Expected output**: The adaptive line should be at or above the envelope of DocPruner and DocMerger at every compression level.

### Experiment 5: Analysis

- Report per-dataset routing statistics: what fraction of documents go to each method?
- Correlate entropy with document characteristics (page count, text density).
- Show example documents: one routed to pruning (concentrated attention) vs one routed to merging (diffuse attention).

---

## Success Criteria

1. Adaptive Hybrid matches or beats DocPruner at moderate compression (<60%).
2. Adaptive Hybrid matches or beats DocMerger at high compression (>75%).
3. The combined curve is at or above the envelope of both individual methods.
4. The entropy threshold is robust (±10 percentile doesn't crash performance).

## Timeline

| Task | Estimated time |
|------|---------------|
| Step 1–2: Implement entropy + adaptive function | 1 day |
| Step 3–4: CLI integration + two-pass execution | 1 day |
| Experiment 1: Entropy analysis | 0.5 day |
| Experiment 2: Oracle upper bound | 1 day |
| Experiment 3: Threshold sweep | 1 day |
| Experiment 4: Full comparison | 0.5 day (cached) |
| Experiment 5: Analysis + figures | 1 day |
| **Total** | **~6 days** |
