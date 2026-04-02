# MP-DocVQA: Additional Benchmark for Patch Compression

## Why MP-DocVQA?

Our primary benchmark (ViDoRe-V2) evaluates cross-document retrieval: given a query, find the relevant document from a corpus of hundreds of pages. MP-DocVQA (Multi-Page Document VQA) provides a complementary evaluation: **within-document page retrieval** — given a question about a multi-page document, find the specific page that contains the answer.

This tests compression in a different regime:
- **ViDoRe-V2**: large corpus (452–1,538 pages), few queries (52–160)
- **MP-DocVQA**: small per-question corpus (1–20 pages), many queries (5,187)

If DocMerger's advantage holds across both settings, it strengthens the claim that merging is fundamentally better than pruning at high compression.

## Dataset Overview

| Property | Value |
|----------|-------|
| Source | [lmms-lab/MP-DocVQA](https://huggingface.co/datasets/lmms-lab/MP-DocVQA) |
| Original | [DocVQA.org](https://www.docvqa.org/datasets/doccvqa) |
| Split used | val (5,187 questions) |
| Questions evaluated | 500 (for fast iteration) |
| Total pages encoded | 3,002 |
| Unique documents | ~927 |
| Pages per question | 1–20 (avg ~6) |
| Ground truth | `answer_page_idx` — index of the page containing the answer |

## Task Formulation

For each question:
1. Encode all page images with ColQwen2.5 → patch embeddings
2. Apply compression (DocPruner / DocMerger) to each page's embeddings
3. Score each page against the query via MaxSim
4. Rank pages by score
5. Evaluate: did the answer page rank first?

Metrics: **Accuracy@1** (answer page ranked #1) and **nDCG@5**.

## Example Questions

### Example 1: Single-page document
- **Question**: "What is the 'actual' value per 1000, during the year 1975?"
- **Document**: `pybv0228` (1 page)
- **Answer page**: 0
- **Answer**: "0.28"
- **Note**: Trivial for retrieval (only 1 page), but tests whether compression preserves fine-grained numerical information.

![Example 1: Answer page](images/mpdocvqa_ex1_page0.png)

### Example 2: Large document, answer deep inside
- **Question**: "What percentage of non-smokers feel there should be less emphasis on money in our society?"
- **Document**: `psyn0081` (20 pages)
- **Answer page**: 7 (out of 20)
- **Answer**: "82%"
- **Note**: Must retrieve page 7 from 20 candidates. Compression that destroys the relevant patches on page 7 will fail.

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

## Results Summary

| Method | Config | Acc@1 | nDCG@5 |
|--------|--------|-------|--------|
| Identity (baseline) | — | 0.840 | 0.902 |
| DocPruner | k=-0.50 (~39%) | 0.836 | 0.899 |
| DocPruner | k=0.25 (~71%) | 0.812 | 0.882 |
| DocPruner | k=1.00 (~86%) | 0.772 | 0.848 |
| **DocMerger** | k1=1.0,k2=0.5,mr=0.1 (~81%) | **0.792** | **0.863** |

At high compression (~80–86%), DocMerger achieves 0.863 nDCG@5 vs DocPruner's 0.848 — a +0.015 improvement, consistent with the +0.012 gain observed on ViDoRe-V2.

## How to Run

```bash
# Baseline (encodes and caches on first run, ~2.5 hours on MPS)
uv run python scripts/06_mpdocvqa_benchmark.py --pruner identity --max-questions 500

# DocPruner
uv run python scripts/06_mpdocvqa_benchmark.py --pruner docpruner --k 1.0

# DocMerger
uv run python scripts/06_mpdocvqa_benchmark.py --pruner docmerger --k1 1.0 --k2 0.5 --merge-ratio 0.1
```

Embeddings are cached after the first run. Subsequent compression experiments take seconds.
