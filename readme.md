# DocMerger: Adaptive Hierarchical Patch Compression for Multi-Vector Visual Document Retrieval

## Overview

Multi-vector Visual Document Retrieval (VDR) systems represent each document page as hundreds of patch-level embeddings, achieving state-of-the-art retrieval quality but incurring prohibitive storage overhead. Recent work (DocPruner) addresses this via adaptive pruning, but permanently discards information, leading to sharp performance degradation at high compression ratios (>60%).

**DocMerger** combines adaptive hierarchical patch compression with reinforcement learning to achieve superior performance–storage trade-offs, especially at high compression ratios (60–80%).

### Key Ideas

1. **Importance-Aware Tri-Level Partitioning**: Instead of binary keep/discard, patches are split into three tiers — *preserve* (high importance), *merge* (mid importance), and *discard* (low importance). The merge tier retains partial information via attention-weighted clustering rather than throwing it away entirely.

2. **RL-Based Adaptive Compression Policy**: A lightweight policy network trained via GRPO learns document-specific compression parameters (partition thresholds + cluster count), directly optimizing retrieval performance (nDCG@5) under storage budgets.

## Project Structure

```
docmerger/
├── README.md
├── requirements.txt
├── configs/                    # Experiment configs (model, dataset, hyperparams)
│   └── colqwen2.5_vidore_v2.yaml
├── src/
│   ├── models/                 # Model loading and embedding extraction
│   │   └── encoder.py          # Load VLM, extract embeddings + attention weights
│   ├── compression/            # Core compression methods
│   │   ├── importance.py       # Attention-based importance scoring
│   │   ├── docpruner.py        # DocPruner baseline (binary adaptive pruning)
│   │   ├── docmerger.py        # Our method (tri-level partition + weighted merging)
│   │   └── baselines.py        # Non-adaptive baselines (random, sem-cluster, pooling)
│   ├── rl/                     # RL policy for adaptive compression
│   │   ├── policy.py           # MLP policy network
│   │   ├── grpo.py             # GRPO training loop
│   │   └── reward.py           # Reward computation (nDCG@5 + compression ratio)
│   ├── retrieval/              # Scoring and evaluation
│   │   ├── maxsim.py           # MaxSim late-interaction scoring
│   │   └── metrics.py          # nDCG@5 and other IR metrics
│   └── utils/
│       ├── data.py             # ViDoRe-V2 data loading helpers
│       └── visualization.py    # Plotting (nDCG vs compression curves, etc.)
├── scripts/
│   ├── 00_verify_base_model.py       # Phase 0: verify base model nDCG@5
│   ├── 01_extract_embeddings.py      # Phase 0: extract & cache all embeddings + attentions
│   ├── 02_run_baselines.py           # Phase 1: run all baseline compression methods
│   ├── 03_run_docmerger_rule.py      # Phase 2: run rule-based DocMerger
│   ├── 04_train_rl_policy.py         # Phase 3: train GRPO policy
│   ├── 05_run_docmerger_rl.py        # Phase 3: evaluate RL-based DocMerger
│   └── 06_plot_results.py            # Phase 4: generate figures and tables
└── results/                    # Experiment outputs (metrics JSON, figures)
    ├── embeddings/             # Cached embeddings (gitignored)
    └── figures/
```

## Experiment Plan

### Setup

- **Benchmark**: ViDoRe-V2 (4 English datasets, BEIR format)
- **Primary Model**: ColQwen2.5 (expand to ColNomic, Jina V4 later)
- **Metric**: nDCG@5
- **Hardware**: NVIDIA A100 80GB

### Phase 0 — Environment Setup & Base Model Verification (Day 1–2)

**Goal**: Confirm the evaluation pipeline works end-to-end.

- [ ] Install dependencies (`colpali-engine`, `vidore-benchmark`, `datasets`, etc.)
- [ ] Load ColQwen2.5 and run inference on ViDoRe-V2
- [ ] Verify base model nDCG@5 matches DocPruner's reported number (~0.5508)
- [ ] Extract and cache all document embeddings + last-layer attention weights
- [ ] Profile: number of patches per document, embedding dimensions, attention shape

### Phase 1 — Reproduce DocPruner & Baselines (Day 3–7)

**Goal**: Reproduce DocPruner's Figure 2 (left panel) for ColQwen2.5 on ViDoRe-V2.

Methods to implement:

| Method | Type | Adaptive? | Key Hyperparameter |
|--------|------|-----------|--------------------|
| Random Pruning | Pruning | No | Pruning ratio |
| Sem-Cluster | Merging | No | Merging factor |
| 1D-Pooling | Merging | No | Window size |
| 2D-Pooling | Merging | No | Pool factor |
| Attention+Similarity | Pruning | Yes | k, alpha |
| Pivot-Threshold | Pruning | Yes | k, k_dup, num_pivots |
| **DocPruner** | Pruning | Yes | k ∈ {-0.5, -0.25, 0, 0.25, 0.5, 1.0} |

Deliverables:
- [ ] nDCG@5 vs. compression ratio curve for all methods
- [ ] Confirm DocPruner results align with paper (k=-0.25: ~51.5% pruning, nDCG@5 ≈ 0.547)

### Phase 2 — DocMerger: Rule-Based (Day 8–14)

**Goal**: Validate the core hypothesis that tri-level partitioning outperforms binary pruning at high compression.

**2a. Tri-Level Partitioning**
```
P_preserve = { d_j | I(d_j) > μ + k1·σ }       → keep original embeddings
P_merge    = { d_j | μ - k2·σ < I(d_j) ≤ μ + k1·σ }  → merge via clustering
P_discard  = { d_j | I(d_j) ≤ μ - k2·σ }       → discard
```

**2b. Attention-Weighted Merging**
- Agglomerative clustering (cosine distance) on P_merge
- Cluster centroids computed as importance-weighted averages (not simple mean)

**2c. Ablation Studies**
- [ ] Tri-level vs. binary (DocPruner): does the merge tier help?
- [ ] Attention-weighted vs. simple-average merging: does weighting matter?
- [ ] Sensitivity to k1, k2, and cluster count C

**Key Hypothesis**: At 60–80% compression, DocMerger should significantly outperform DocPruner because merged patches retain partial information that pure pruning discards.

### Phase 3 — DocMerger: RL Policy (Day 12–18)

**Goal**: Replace hand-tuned (k1, k2, C) with a learned, document-specific policy.

**MDP Formulation (contextual bandit)**:
- **State**: [μ_d, σ_d, skew_d, kurt_d, H(p_d), L_d] — statistics of each document's attention distribution
- **Action**: (k1, k2, C) — discretized compression parameters
- **Reward**: nDCG@5(d) + λ · (1 - |D'|/|D|)

**Training**:
- Policy: 3-layer MLP (128 hidden units)
- Algorithm: GRPO (Group Relative Policy Optimization)
- Warm start: supervised imitation of best rule-based parameters
- KL regularization toward rule-based policy to prevent collapse

Deliverables:
- [ ] RL policy converges and outperforms best fixed parameters
- [ ] Analysis of learned document-specific strategies (e.g., aggressive pruning on title pages vs. conservative merging on dense text)

### Phase 4 — Results & Extension (Day 18–21)

- [ ] Final nDCG@5 vs. compression ratio plot with all methods
- [ ] Ablation table
- [ ] Extend pipeline to ColNomic (Jina V4 deferred)
- [ ] Note: DocPruner observed that merging is unexpectedly strong on Jina V4 — our method may show even larger gains there

## Implementation Notes

### Architecture Decision: Standalone Evaluation Pipeline

We do **not** run experiments inside the `vidore-benchmark` framework. Instead, we write our own evaluation scripts that:

1. **Borrow from vidore-benchmark**: dataset loading (`datasets` library), metric computation (`pytrec_eval`)
2. **Use colpali-engine directly**: model loading, `output_attentions=True` for attention extraction
3. **Own the core loop**: embedding extraction → compression → MaxSim scoring → metrics

This follows the same approach as DocPruner ("evaluation codebase adapted from the official ViDoRe Benchmark repository"). The reason is that our compression methods need access to intermediate attention weights, which the standard `VisionRetriever` interface does not expose.

### Embedding Caching Strategy

Extracting embeddings + attentions is the most expensive step (~0.5s per document). We extract once and cache:

```
results/embeddings/
├── colqwen2.5/
│   ├── vidore_v2_corpus_embeddings.pt    # List[Tensor], one per document
│   ├── vidore_v2_corpus_attentions.pt    # List[Tensor], last-layer attention
│   ├── vidore_v2_query_embeddings.pt     # List[Tensor], one per query
│   └── vidore_v2_metadata.json           # patch counts, dims, etc.
```

All compression experiments then operate on cached embeddings — no GPU needed for Phase 1–2 experiments (except for initial extraction).

### Key Technical Questions to Resolve

1. **EOS token location**: How does ColQwen2.5 handle the EOS/global token? Need to verify the token index used for attention extraction.
2. **Attention shape**: Confirm `output_attentions=True` returns per-head attention of shape `[batch, heads, seq_len, seq_len]` for the last layer.
3. **Special tokens**: Are there special tokens (BOS, padding, image separator) that should be excluded from importance scoring?

## References

- [DocPruner](https://arxiv.org/abs/2509.23883) — Yan et al., 2025. Adaptive patch-level embedding pruning for VDR.
- [ColPali](https://arxiv.org/abs/2407.01449) — Faysse et al., 2024. Efficient document retrieval with VLMs.
- [ViDoRe-V2](https://arxiv.org/abs/2505.17166) — Macé et al., 2025. Visual document retrieval benchmark.
- [GRPO](https://arxiv.org/abs/2402.03300) — Shao et al., 2024. Group relative policy optimization.
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark) — Official evaluation codebase.
- [colpali-engine](https://github.com/illuin-tech/colpali) — Model implementations for ColPali, ColQwen2, etc.
