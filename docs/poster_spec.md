# DocMerger Poster Specification

## Poster Metadata

| Field | Value |
|-------|-------|
| Title | DocMerger: Adaptive Hierarchical Patch Merging for Multi-Vector Visual Document Retrieval |
| Authors | [Author names] |
| Affiliation | [Institution] |
| Conference | [Conference name, year] |
| Poster size | 80 × 40 inches (2032 × 1016 mm) — adjust per conference requirements |
| Tool | draw.io |
| Edge margin | 50pt from all edges |

---

## Color Scheme

| Role | Color | Usage |
|------|-------|-------|
| Primary | Deep blue (#1565C0) | Titles, block headers, key emphasis |
| Accent | Orange (#E65100) | DocMerger highlights, arrows, callout boxes |
| Secondary | Teal (#00897B) | DocPruner references, comparison elements |
| Background | White (#FFFFFF) | Poster background |
| Text | Dark gray (#212121) | Body text |
| Light fill | Light blue (#E3F2FD) | Block backgrounds for visual separation |

Rationale: Blue/orange contrast makes DocMerger results pop against DocPruner comparisons. Teal differentiates the baseline method.

---

## Header (Top Bar)

| Element | Position | Content |
|---------|----------|---------|
| Institution logo | Left | [University/lab logo] |
| Title + authors | Center | **DocMerger: Adaptive Hierarchical Patch Merging for Multi-Vector Visual Document Retrieval** — [Author 1], [Author 2], ... — [Affiliation] |
| Conference logo | Right | [Conference logo] |

---

## Column 1 (Left) — "What did you do?"

### Block 1A: Concept

**Visual**: Create a simple diagram showing the tri-level partitioning concept:
- Input: grid of patch embeddings (colored squares)
- Three arrows leading to:
  - 🟢 **Preserve** (green) — high-importance patches kept as-is
  - 🟡 **Merge** (orange) — mid-importance patches clustered into centroids
  - 🔴 **Discard** (red) — low-importance patches removed
- Output: compressed representation (fewer squares)

**Text**:
> Multi-vector visual document retrieval stores hundreds of patch embeddings per page — great quality, but prohibitive storage.
>
> **DocPruner** prunes low-importance patches but permanently loses information, degrading sharply at >75% compression.
>
> **DocMerger** introduces importance-aware tri-level partitioning: **preserve** the best patches, **merge** mid-tier patches via attention-weighted clustering, and **discard** only the least important. Merging retains partial information where pruning loses it entirely.

### Block 1B: Contributions

**Text** (bullet list):
- Tri-level partitioning framework (preserve / merge / discard) for patch compression
- Attention-weighted agglomerative clustering for the merge tier
- Outperforms DocPruner at high compression (>75%) across 5 benchmarks
- Consistent advantage in the hardest setting: global retrieval at high compression (+6.9%)
- Practical recommendation: DocPruner for ≤70%, DocMerger for >75% — complementary, not competing

### Block 1C: Implementation

**Text**:
> Code available at: [repo URL]

**Visual**: QR code linking to the code repository (generate via qrcode-monkey.com)

---

## Column 2 (Center) — "How did you do it?"

### Block 2A: Method

**Visual**: Method diagram showing the full pipeline. Create a new figure for the poster:

```
Document page → ColQwen2.5 encoder → Patch embeddings {d₁, ..., dₙ}
                                          ↓
                              EOS attention scores I(dⱼ)
                                          ↓
                              ┌─────────────────────────┐
                              │   Tri-Level Partitioning │
                              │                         │
                              │  I(dⱼ) > μ + k₁·σ      │ → P_preserve (keep as-is)
                              │  μ - k₂·σ < I(dⱼ) ≤ μ + k₁·σ │ → P_merge
                              │  I(dⱼ) ≤ μ - k₂·σ      │ → P_discard (remove)
                              └─────────────────────────┘
                                          ↓
                              P_merge → Agglomerative clustering
                                        (cosine distance)
                                          ↓
                              Attention-weighted centroids
                                          ↓
                              D' = P_preserve ∪ {centroids}
```

**Key formula** (use LaTeX typesetting in draw.io):

$$P_{\text{preserve}} = \{ d_j \mid I(d_j) > \mu + k_1 \cdot \sigma \}$$
$$P_{\text{merge}} = \{ d_j \mid \mu - k_2 \cdot \sigma < I(d_j) \leq \mu + k_1 \cdot \sigma \}$$
$$P_{\text{discard}} = \{ d_j \mid I(d_j) \leq \mu - k_2 \cdot \sigma \}$$

**Text below diagram**:
> Importance scores from EOS attention partition patches into three tiers. Merge-tier patches are clustered via agglomerative clustering (cosine distance), with centroids weighted by attention importance. The final compressed representation combines preserved patches with merged centroids.

### Block 2B: Why Merging Wins at High Compression (small inset)

**Visual**: Simple 2-panel comparison diagram:
- Panel A (moderate compression): DocPruner keeps top patches → good. DocMerger replaces some with centroids → slightly worse.
- Panel B (high compression): DocPruner must discard many patches → information lost. DocMerger routes them through merge tier → partial information retained.

**Text**:
> Merging is lossier than keeping the original patch, but better than discarding it. This tradeoff pays off only when aggressive compression forces heavy discarding.

---

## Column 3 (Right) — "What did you get?"

### Block 3A: Quantitative Results — ViDoRe-V2

**Visual**: Use `docmerger_vs_docpruner.png` (from `results/figures/`)

**Highlight with colored callout box** (orange):
> At ~86% compression: DocMerger **+0.012 nDCG@5** over DocPruner

**Summary table** (simplified for poster):

| Compression | DocPruner | DocMerger | Δ |
|-------------|-----------|-----------|---|
| ~70% | 0.587 | **0.590** | +0.003 |
| ~86% | 0.517 | **0.529** | **+0.012** |

Emphasize the +0.012 row with an orange rectangle and arrow.

### Block 3B: Quantitative Results — MP-DocVQA

**Visual**: Use `three_benchmarks.png` (from `results/figures/`)

**Highlight with colored callout box** (orange):
> Global retrieval at high compression: DocMerger **+0.022 nDCG@5** (+6.9%)

**Summary table**:

| Setting | Compression | DocPruner | DocMerger | Δ |
|---------|-------------|-----------|-----------|---|
| Per-question | ~81–86% | 0.848 | **0.863** | +0.015 |
| Global | ~81–86% | 0.319 | **0.341** | **+0.022** |

Emphasize the global retrieval row — this is the strongest result.

### Block 3C: Ablation — Attention-Weighted Merging

**Text** (small block):
> Attention-weighted centroids provide +0.002 avg nDCG@5 over simple averaging. Effect is consistent across datasets.

| Merge type | Avg nDCG@5 |
|------------|------------|
| Simple average | 0.554 |
| **Attention-weighted** | **0.555** |

### Block 3D: Diversity Selection & Residual Injection (Phase 4)

**Text**:
> Can we improve patch selection with diversity or enrich kept patches with residual information from discarded ones? (~52% compression)

**Summary table**:

| Method | Avg nDCG@5 | vs DocPruner |
|--------|------------|-------------|
| DocPruner k=−0.25 | 0.603 | BASE |
| MMR (diversity) | 0.548 | −0.054 |
| Attn-FPS (diversity) | 0.539 | −0.064 |
| CPS-Attn (cluster) | 0.574 | −0.029 |
| DP-Rebalance | 0.595 | −0.008 |
| **DP-Residual (β=0.02)** | **0.604** | **+0.001** |

**Key insight**: Diversity-based selection hurts — attention importance dominates. Only residual injection provides marginal gain. At ~50% compression, DocPruner's threshold is near-optimal for selection.

---

## Footer

**References**:
- DocPruner — Yan et al., 2025
- ColPali — Faysse et al., 2024
- ViDoRe-V2 — Macé et al., 2025

---

## Figure Assignments

| Poster block | Figure source | Notes |
|--------------|---------------|-------|
| Concept (1A) | **NEW** — create tri-level partitioning diagram | Simple colored-squares diagram showing preserve/merge/discard |
| Method (2A) | **NEW** — create pipeline diagram | Full method flow from encoder to compressed output |
| Why merging wins (2B) | **NEW** — create 2-panel comparison | Simple conceptual diagram |
| ViDoRe-V2 results (3A) | `results/figures/docmerger_vs_docpruner.png` | Existing plot — may need to enlarge key region |
| MP-DocVQA results (3B) | `results/figures/three_benchmarks.png` | Existing plot — shows all 3 benchmark settings |
| Implementation (1C) | **NEW** — generate QR code | Use qrcode-monkey.com |

## Recommended Emphasis (Visual Callouts)

1. **+0.012 nDCG@5 at 86% compression** (ViDoRe-V2) — orange callout box with arrow
2. **+6.9% improvement in global retrieval** (MP-DocVQA) — orange callout box with arrow
3. **Tri-level partitioning formula** — centered in method block with blue background
4. **"Merging is worse than keeping, but better than discarding"** — key insight, bold text in the "why it works" inset

## New Figures to Create

1. **Tri-level partitioning concept diagram** — for the Concept block. Show a grid of colored patches being split into three groups with different fates.
2. **Method pipeline diagram** — for the Method block. End-to-end flow from document page through encoding, partitioning, clustering, to compressed output.
3. **Compression tradeoff diagram** — for the "why merging wins" inset. Two scenarios showing what happens at moderate vs high compression.

## Poster Creation Checklist

- [ ] Set poster size in draw.io (Custom paper size per conference requirements)
- [ ] Place header: logos + title + authors
- [ ] Create tri-level partitioning concept figure
- [ ] Write Concept block text
- [ ] Write Contributions bullet list
- [ ] Generate QR code for repo
- [ ] Create method pipeline diagram
- [ ] Add LaTeX formulas (enable Mathematical Typesetting)
- [ ] Create "why merging wins" inset diagram
- [ ] Place `docmerger_vs_docpruner.png` in results block
- [ ] Place `three_benchmarks.png` in results block
- [ ] Add summary tables with emphasis callouts
- [ ] Add ablation mini-block
- [ ] Add footer references
- [ ] Verify alignment, spacing, font consistency
- [ ] Proofread all text
- [ ] Export PDF (for printing) and PNG (for online platform)
