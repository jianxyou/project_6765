# Research Plan: Future Directions for Patch Compression in Multi-Vector VDR

## Context

DocMerger demonstrates that tri-level partitioning (preserve/merge/discard) with attention-weighted clustering outperforms pure pruning (DocPruner) at high compression ratios (>75%). However, it underperforms at moderate compression (<60%) because merging is inherently lossier than keeping original patches. The following directions address this limitation and push the compression–quality frontier further.

---

## Direction 1: Adaptive Hybrid (DocPruner + DocMerger)

### Idea

Instead of choosing one method globally, automatically select pruning or merging per-document based on its attention distribution. Documents with concentrated attention (few dominant patches) benefit from pruning; documents with diffuse attention (many moderately important patches) benefit from merging.

### Method

1. Compute attention entropy for each document: H = -Σ p(i) log p(i) where p(i) is the normalized EOS attention over image patches.
2. If H < threshold → use DocPruner (attention is concentrated, pruning keeps the important patches).
3. If H ≥ threshold → use DocMerger (attention is spread out, merging preserves more distributed information).
4. Threshold can be set via the median entropy across the corpus (no tuning needed).

### Why promising

- Zero training cost, simple to implement.
- Directly addresses DocMerger's weakness at low compression.
- Per-document adaptation should dominate any fixed global strategy.

### Estimated effort

1–2 weeks. Entropy computation is trivial from cached attention scores.

---

## Direction 2: Residual Merging

### Idea

Current merging replaces a cluster of patches with a single centroid, losing fine-grained information. Residual merging stores the centroid plus a low-rank residual that captures the dominant variation within the cluster.

### Method

1. For each cluster in P_merge, compute the centroid c = weighted_mean(patches).
2. Compute residuals: r_i = patch_i - c for each member.
3. Run rank-1 PCA on the residuals → get principal component v.
4. Store (c, v) as two vectors instead of one centroid.
5. During MaxSim scoring, compute max over both c and v for each cluster.

### Why promising

- Adds only ~1 extra vector per cluster (small storage overhead vs. the compression gained).
- Recovers the dominant axis of variation lost by averaging.
- Analogous to product quantization residuals — well-understood in IR.

### Estimated effort

2–3 weeks. Requires modifying the merge function and the MaxSim scoring to handle residual vectors.

---

## Direction 3: Learned Sparse Projection

### Idea

Instead of agglomerative clustering (which is heuristic and slow), learn a small projection layer that compresses the merge-tier patches into a fixed-size representation optimized for MaxSim retrieval.

### Method

1. Train a lightweight linear layer W ∈ R^(d × d) that projects merge-tier patches.
2. Loss function: MaxSim distillation — the compressed document representation should produce the same MaxSim scores as the original uncompressed representation for a set of training queries.
3. L = Σ_q ||MaxSim(q, D_original) - MaxSim(q, D_compressed)||²
4. Training data: use the ViDoRe-V2 queries or synthetic queries from the corpus.
5. At inference: apply W to merge-tier patches, then select top-k by norm as the compressed output.

### Why promising

- End-to-end optimized for the retrieval objective (unlike clustering which optimizes for embedding similarity).
- Fast at inference (single matrix multiply vs. iterative clustering).
- Could generalize across datasets if trained on diverse documents.

### Risks

- Requires training data and GPU time for optimization.
- May overfit to training query distribution.
- Adds a learned component to an otherwise training-free pipeline.

### Estimated effort

3–4 weeks. Needs a training loop, query sampling strategy, and evaluation across datasets.

---

## Priority & Sequencing

| Direction | Effort | Risk | Expected Impact | Priority |
|-----------|--------|------|-----------------|----------|
| 1. Adaptive Hybrid | Low | Low | Medium | **Start here** |
| 2. Residual Merging | Medium | Low | Medium–High | Second |
| 3. Learned Sparse Projection | High | Medium | High | Third (if time permits) |

Recommended approach: implement Direction 1 first as it's low-risk and directly plugs DocMerger's gap at moderate compression. If results are positive, layer Direction 2 on top for additional gains at high compression. Direction 3 is the most ambitious and should only be pursued if the simpler methods plateau.

---

## Evaluation: Extended Benchmarks (post-approach validation)

Once the best approach is identified from Directions 1–3, evaluate on the following benchmarks:

### Benchmark 1: ViDoRe-V2 (4 English datasets)

- **What:** The primary benchmark already implemented in this project (ESG reports + 3 other datasets).
- **Why:** Direct comparison against existing DocPruner and DocMerger baselines. Infrastructure and cached embeddings already in place.
- **Expected insight:** Whether the new approach improves the compression–quality frontier established by DocMerger.

### Benchmark 2: MP-DocVQA

- **What:** Multi-page document VQA with retrieval-style queries across page collections.
- **Why:** Tests multi-page reasoning, which is directly relevant to patch-level compression — the merge tier may behave differently on text-heavy vs visual-heavy pages.
- **Expected insight:** Whether compression gains hold when documents have heterogeneous page layouts within a single retrieval target.

### Benchmark 3: SlideVQA / InfographicsVQA

- **What:** Slide deck QA and infographic-heavy document QA, both with high visual information density.
- **Why:** These have very different visual density profiles than the ViDoRe-V2 datasets (e.g., ESG reports). Infographics and slides pack more information per patch, stress-testing whether attention-weighted clustering captures the right visual semantics.
- **Expected insight:** Whether DocMerger's advantage at high compression (>75%) generalizes across document types with varying visual complexity.
