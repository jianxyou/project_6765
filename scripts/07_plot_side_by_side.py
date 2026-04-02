#!/usr/bin/env python3
"""Side-by-side plot: ViDoRe-V2 (avg) vs MP-DocVQA."""

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: ViDoRe-V2 (avg across 4 datasets) ---
dp_prune = [0, 39, 52, 62, 71, 77, 86]
dp_ndcg  = [0.6205, 0.6076, 0.6061, 0.5895, 0.5868, 0.5608, 0.5168]

dm_data = [
    (55, 0.5779), (57, 0.5793), (58, 0.5985), (60, 0.5915),
    (65, 0.6005), (68, 0.5902), (70, 0.5897),
    (80, 0.5551), (81, 0.5433), (84, 0.5285),
]
dm_prune = [d[0] for d in dm_data]
dm_ndcg  = [d[1] for d in dm_data]

ax1.plot(dp_prune, dp_ndcg, 'o-', color='#2196F3', linewidth=2, markersize=7, label='DocPruner')
ax1.plot(dm_prune, dm_ndcg, 's--', color='#E91E63', linewidth=2, markersize=7, label='DocMerger')
ax1.axvspan(75, 90, alpha=0.08, color='green')
ax1.set_xlabel('Compression Ratio (%)', fontsize=11)
ax1.set_ylabel('nDCG@5', fontsize=11)
ax1.set_title('ViDoRe-V2 (avg across 4 datasets)', fontsize=12)
ax1.legend(fontsize=9, loc='lower left')
ax1.set_xlim(-2, 92)
ax1.set_ylim(0.50, 0.63)
ax1.grid(True, alpha=0.3)

# --- Right: MP-DocVQA ---
# DocPruner: k -> (approx prune%, nDCG@5)
# Using same ~prune% as ViDoRe-V2 since same k values
mp_dp_prune = [0, 39, 52, 62, 71, 77, 86]
mp_dp_ndcg  = [0.902, 0.899, 0.887, 0.888, 0.882, 0.861, 0.848]

# DocMerger configs
mp_dm_data = [
    (70, 0.880),   # k1=0.5,k2=0,mr=0.5
    (80, 0.857),   # k1=1.0,k2=0,mr=0.25
    (81, 0.863),   # k1=1.0,k2=0.5,mr=0.1
    (84, 0.856),   # k1=1.0,k2=0,mr=0.1
]
mp_dm_prune = [d[0] for d in mp_dm_data]
mp_dm_ndcg  = [d[1] for d in mp_dm_data]

ax2.plot(mp_dp_prune, mp_dp_ndcg, 'o-', color='#2196F3', linewidth=2, markersize=7, label='DocPruner')
ax2.plot(mp_dm_prune, mp_dm_ndcg, 's--', color='#E91E63', linewidth=2, markersize=7, label='DocMerger')
ax2.axvspan(75, 90, alpha=0.08, color='green')
ax2.set_xlabel('Compression Ratio (%)', fontsize=11)
ax2.set_ylabel('nDCG@5', fontsize=11)
ax2.set_title('MP-DocVQA (500 questions)', fontsize=12)
ax2.legend(fontsize=9, loc='lower left')
ax2.set_xlim(-2, 92)
ax2.set_ylim(0.83, 0.91)
ax2.grid(True, alpha=0.3)

fig.text(0.5, -0.01,
         'Figure 3: DocPruner vs DocMerger on two benchmarks. DocMerger (dashed) outperforms DocPruner (solid)\n'
         'at high compression (>75%, green zone) on both ViDoRe-V2 cross-document retrieval and MP-DocVQA page retrieval.',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
plt.savefig('results/figures/side_by_side_benchmarks.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/side_by_side_benchmarks.pdf', bbox_inches='tight')
print("Saved to results/figures/side_by_side_benchmarks.{png,pdf}")
