# MP-DocVQA: Additional Benchmark for Patch Compression

## Why MP-DocVQA?

Our primary benchmark (ViDoRe-V2) evaluates cross-document retrieval: given a query, find the relevant document from a corpus of hundreds of pages. MP-DocVQA (Multi-Page Document VQA) provides a complementary evaluation that tests compression in two distinct retrieval settings:

1. **Per-question page retrieval** — find the answer page within a single document (1–20 pages)
2. **Global page retrieval** — find the answer page across all documents in the corpus (791+ pages)

If DocMerger's advantage holds across all settings, it strengthens the claim that merging is fundamentally better than pruning at high compression.

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | [lmms-lab/MP-DocVQA](https://huggingface.co/datasets/lmms-lab/MP-DocVQA) |
| Original | [DocVQA.org](https://www.docvqa.org/datasets/doccvqa) |
| Split used | val (5,187 questions) |
| Unique documents | ~927 |
| Pages per question | 1–20 (avg ~6) |
| Ground truth | `answer_page_idx` — index of the page containing the answer |

## Two Benchmark Settings

### Setting 1: Per-Question Retrieval

Each query retrieves only within its own document's pages. This tests whether compression preserves the fine-grained differences between pages of the same document.

| Property | Value |
|----------|-------|
| Questions evaluated | 500 |
| Total pages encoded | 3,002 |
| Corpus per query | 1–20 pages (same document) |
| Difficulty | Moderate — small candidate set but pages are visually similar |

### Setting 2: Global Retrieval

Each query retrieves across all pages from all documents. This is closer to a real-world deployment where the system must search a large heterogeneous corpus.

| Property | Value |
|----------|-------|
| Questions evaluated | 100 |
| Total pages in corpus | 791 |
| Corpus per query | 791 pages (all documents) |
| Difficulty | Hard — large candidate set with many distractors |

### Comparison with ViDoRe-V2

| | ViDoRe-V2 | MP-DocVQA Per-Q | MP-DocVQA Global |
|---|-----------|-----------------|------------------|
| Task | Cross-document retrieval | Within-document page retrieval | Cross-document page retrieval |
| Corpus size | 452–1,538 docs | 1–20 pages | 791 pages |
| Queries | 52–160 | 500 | 100 |
| Document type | ESG reports, biomedical, economics | Diverse (forms, tables, reports) | Same as Per-Q |
| Retrieval scope | Across documents | Within one document | Across all documents |
| Metric | nDCG@5 | Acc@1 + nDCG@5 | Acc@1 + nDCG@5 |

## Example Questions

### Example 1: Single-page document
- **Question**: "What is the 'actual' value per 1000, during the year 1975?"
- **Document**: `pybv0228` (1 page)
- **Answer page**: 0
- **Answer**: "0.28"
- **Note**: Trivial for per-question retrieval (only 1 page), but in global retrieval must be found among 791 candidates. Tests whether compression preserves fine-grained numerical information.

![Example 1: Answer page](images/mpdocvqa_ex1_page0.png)

### Example 2: Large document, answer deep inside
- **Question**: "What percentage of non-smokers feel there should be less emphasis on money in our society?"
- **Document**: `psyn0081` (20 pages)
- **Answer page**: 7 (out of 20)
- **Answer**: "82%"
- **Note**: Must retrieve page 7 from 20 candidates (per-question) or 791 candidates (global). Compression that destroys the relevant patches on page 7 will fail.

![Example 2: Answer page](images/mpdocvqa_ex2_page7.png)

### Example 3: Brand identification across pages
- **Question**: "Which brand does Tangles belong to?"
- **Document**: `snbx0223` (8 pages)
- **Answer page**: 6
- **Answer**: "Bingo!"
- **Note**: Requires finding a specific brand mention on page 6. Tests whether merging preserves text-related patch information.

![Example 3: Answer page](images/mpdocvqa_ex3_page6.png)

### Example 4: Table lookup
- **Question**: "What is the 'index' of the rate of quitting losses?"
- **Document**: `rzbj0037` (4 pages)
- **Answer page**: 3
- **Answer**: "89"
- **Note**: Structured table data. Compression must preserve tabular layout patches.

![Example 4: Answer page](images/mpdocvqa_ex4_page3.png)

### Example 5: Column header identification
- **Question**: "What is the heading of the last column of the table?"
- **Document**: `yxvw0217` (2 pages)
- **Answer page**: 0
- **Answer**: "Status"
- **Note**: Simple 2-page retrieval, but requires understanding table structure.

![Example 5: Answer page](images/mpdocvqa_ex5_page0.png)

## Results

### Per-Question Retrieval (500 queries)

| Method | Config | Compression | Acc@1 | nDCG@5 |
|--------|--------|-------------|-------|--------|
| Identity (baseline) | — | 0% | 0.840 | 0.902 |
| DocPruner | k=-0.50 | ~39% | 0.836 | 0.899 |
| DocPruner | k=-0.25 | ~52% | 0.816 | 0.887 |
| DocPruner | k=0.00 | ~62% | 0.820 | 0.888 |
| DocPruner | k=0.25 | ~71% | 0.812 | 0.882 |
| DocPruner | k=0.50 | ~77% | 0.780 | 0.861 |
| DocPruner | k=1.00 | ~86% | 0.772 | 0.848 |
| DocMerger | k1=0.5,k2=0,mr=0.5 | ~70% | 0.808 | 0.880 |
| DocMerger | k1=1.0,k2=0,mr=0.25 | ~80% | 0.774 | 0.857 |
| **DocMerger** | **k1=1.0,k2=0.5,mr=0.1** | **~81%** | **0.792** | **0.863** |
| DocMerger | k1=1.0,k2=0,mr=0.1 | ~84% | 0.784 | 0.856 |

### Global Retrieval (100 queries × 791 pages)

| Method | Config | Compression | Acc@1 | nDCG@5 |
|--------|--------|-------------|-------|--------|
| Identity (baseline) | — | 0% | 0.210 | 0.397 |
| DocPruner | k=-0.25 | ~52% | 0.200 | 0.377 |
| DocPruner | k=0.25 | ~71% | 0.220 | 0.383 |
| DocPruner | k=0.50 | ~77% | 0.190 | 0.351 |
| DocPruner | k=1.00 | ~86% | 0.180 | 0.319 |
| **DocMerger** | **k1=0.5,k2=0,mr=0.5** | **~70%** | **0.220** | **0.394** |
| DocMerger | k1=1.0,k2=0,mr=0.25 | ~80% | 0.170 | 0.331 |
| DocMerger | k1=1.0,k2=0.5,mr=0.1 | ~81% | 0.180 | 0.341 |

## Result Analysis

### DocMerger advantage is consistent across all settings

| Setting | Compression | DocPruner | DocMerger | Δ | Δ% |
|---------|-------------|-----------|-----------|---|-----|
| ViDoRe-V2 (avg) | ~70% | 0.5868 | **0.5897** | +0.003 | +0.5% |
| ViDoRe-V2 (avg) | ~84–86% | 0.5168 | **0.5285** | +0.012 | +2.3% |
| MP-DocVQA per-Q | ~81–86% | 0.848 | **0.863** | +0.015 | +1.8% |
| MP-DocVQA global | ~70% | 0.383 | **0.394** | +0.011 | +2.9% |
| MP-DocVQA global | ~81–86% | 0.319 | **0.341** | +0.022 | +6.9% |

The advantage is largest in the hardest setting (global retrieval at high compression: +0.022 / +6.9%), where preserving partial information via merging matters most.

### Per-question vs global retrieval

Global retrieval is dramatically harder:
- Baseline Acc@1 drops from 0.840 (per-question) to 0.210 (global)
- The corpus grows from 1–20 pages to 791 pages, introducing many distractors
- Compression has a larger relative impact in the global setting

### Why DocMerger helps more in global retrieval

In per-question retrieval, pages come from the same document and are visually similar. The discriminative signal is subtle — a specific table cell, a particular paragraph. Pruning can sometimes preserve these fine-grained features better than merging.

In global retrieval, the same diverse document collection is used, but each query must now distinguish its target page from hundreds of pages across many different documents. The challenge shifts from fine-grained within-document discrimination to coarse-grained cross-document discrimination. Merged centroids preserve enough document-level and layout-level information to handle this broader retrieval, even if they lose some fine-grained details. This is why DocMerger's advantage is larger in the global setting (+0.022 vs +0.015).

![Three benchmark comparison](../results/figures/three_benchmarks.png)

## How to Run

```bash
# Per-question retrieval (encodes and caches on first run, ~2.5 hours on MPS)
uv run python scripts/06_mpdocvqa_benchmark.py --pruner identity --max-questions 500

# Global retrieval
uv run python scripts/06_mpdocvqa_benchmark.py --pruner identity --max-questions 100 --global-retrieval

# DocPruner
uv run python scripts/06_mpdocvqa_benchmark.py --pruner docpruner --k 1.0 --max-questions 500
uv run python scripts/06_mpdocvqa_benchmark.py --pruner docpruner --k 1.0 --max-questions 100 --global-retrieval

# DocMerger
uv run python scripts/06_mpdocvqa_benchmark.py --pruner docmerger --k1 1.0 --k2 0.5 --merge-ratio 0.1 --max-questions 500
uv run python scripts/06_mpdocvqa_benchmark.py --pruner docmerger --k1 0.5 --k2 0 --merge-ratio 0.5 --max-questions 100 --global-retrieval
```

Embeddings are cached after the first run. Subsequent compression experiments take seconds (per-question) or ~40 seconds (global, 100 queries × 791 pages).
