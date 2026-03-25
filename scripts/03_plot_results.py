#!/usr/bin/env python3
"""Plot nDCG@5 vs compression ratio: DocPruner vs DocMerger (avg across 4 datasets)."""

import matplotlib.pyplot as plt

# DocPruner baselines (avg across 4 datasets)
dp_prune = [0, 39, 52, 62, 71, 77, 86]
dp_ndcg  = [0.6205, 0.6076, 0.6061, 0.5895, 0.5868, 0.5608, 0.5168]

# DocMerger (avg across 4 datasets), sorted by compression
dm_data = [
    (55, 0.5779, "k1=0.25,k2=0.5,mr=0.5"),
    (57, 0.5744, "k1=0,k2=0.5,mr=0.25"),
    (57, 0.5793, "k1=0,k2=0.25,mr=0.5"),
    (58, 0.5985, "k1=0.5,k2=0.5,mr=0.5"),
    (60, 0.5915, "k1=0,k2=0.5,mr=0.1"),
    (65, 0.6005, "k1=0.5,k2=0.25,mr=0.5"),
    (68, 0.5902, "k1=0.5,k2=0.5,mr=0.25"),
    (70, 0.5897, "k1=0.5,k2=0,mr=0.5"),
    (80, 0.5551, "k1=1.0,k2=0,mr=0.25"),
    (81, 0.5433, "k1=1.0,k2=0.5,mr=0.1"),
    (84, 0.5285, "k1=1.0,k2=0,mr=0.1"),
]
dm_prune = [d[0] for d in dm_data]
dm_ndcg  = [d[1] for d in dm_data]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(dp_prune, dp_ndcg, 'o-', color='#2196F3', linewidth=2, markersize=7, label='DocPruner', zorder=3)
ax.plot(dm_prune, dm_ndcg, 's--', color='#E91E63', linewidth=2, markersize=7, label='DocMerger', zorder=3)

ax.axvspan(75, 90, alpha=0.08, color='green', label='DocMerger advantage zone')

ax.set_xlabel('Compression Ratio (%)', fontsize=12)
ax.set_ylabel('nDCG@5 (avg across 4 datasets)', fontsize=12)
ax.set_title('DocPruner vs DocMerger: nDCG@5 vs Compression', fontsize=13)
ax.legend(fontsize=10, loc='lower left')
ax.set_xlim(-2, 92)
ax.set_ylim(0.50, 0.63)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

fig.text(0.5, 0.01,
         'Figure 2: nDCG@5 vs compression ratio averaged across 4 ViDoRe-V2 datasets. '
         'DocMerger retains partial information\n'
         'via attention-weighted clustering, degrading more gracefully than DocPruner at high compression (>75%).',
         ha='center', fontsize=9, style='italic')

plt.subplots_adjust(bottom=0.18)
plt.savefig('results/figures/docmerger_vs_docpruner.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/docmerger_vs_docpruner.pdf', bbox_inches='tight')
print("Saved to results/figures/")
