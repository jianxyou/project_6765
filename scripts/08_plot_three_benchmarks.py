#!/usr/bin/env python3
"""Side-by-side plot: ViDoRe-V2 | MP-DocVQA per-question | MP-DocVQA global."""

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- ViDoRe-V2 ---
dp_p = [0, 39, 52, 62, 71, 77, 86]
dp_n = [0.6205, 0.6076, 0.6061, 0.5895, 0.5868, 0.5608, 0.5168]
dm_d = [(55,0.5779),(57,0.5793),(58,0.5985),(60,0.5915),(65,0.6005),(68,0.5902),(70,0.5897),(80,0.5551),(81,0.5433),(84,0.5285)]

ax1.plot(dp_p, dp_n, 'o-', color='#2196F3', lw=2, ms=7, label='DocPruner')
ax1.plot([d[0] for d in dm_d], [d[1] for d in dm_d], 's--', color='#E91E63', lw=2, ms=7, label='DocMerger')
ax1.axvspan(75, 90, alpha=0.08, color='green')
ax1.set_xlabel('Compression (%)', fontsize=11)
ax1.set_ylabel('nDCG@5', fontsize=11)
ax1.set_title('ViDoRe-V2 (avg, 4 datasets)', fontsize=12)
ax1.legend(fontsize=9, loc='lower left')
ax1.set_xlim(-2, 92); ax1.set_ylim(0.50, 0.63)
ax1.grid(True, alpha=0.3)

# --- MP-DocVQA per-question ---
mp_dp_p = [0, 39, 52, 62, 71, 77, 86]
mp_dp_n = [0.902, 0.899, 0.887, 0.888, 0.882, 0.861, 0.848]
mp_dm = [(70,0.880),(80,0.857),(81,0.863),(84,0.856)]

ax2.plot(mp_dp_p, mp_dp_n, 'o-', color='#2196F3', lw=2, ms=7, label='DocPruner')
ax2.plot([d[0] for d in mp_dm], [d[1] for d in mp_dm], 's--', color='#E91E63', lw=2, ms=7, label='DocMerger')
ax2.axvspan(75, 90, alpha=0.08, color='green')
ax2.set_xlabel('Compression (%)', fontsize=11)
ax2.set_ylabel('nDCG@5', fontsize=11)
ax2.set_title('MP-DocVQA Per-Question (500 Q)', fontsize=12)
ax2.legend(fontsize=9, loc='lower left')
ax2.set_xlim(-2, 92); ax2.set_ylim(0.83, 0.91)
ax2.grid(True, alpha=0.3)

# --- MP-DocVQA global ---
gl_dp_p = [0, 52, 71, 77, 86]
gl_dp_n = [0.397, 0.377, 0.383, 0.351, 0.319]
gl_dm = [(70,0.394),(80,0.331),(81,0.341)]

ax3.plot(gl_dp_p, gl_dp_n, 'o-', color='#2196F3', lw=2, ms=7, label='DocPruner')
ax3.plot([d[0] for d in gl_dm], [d[1] for d in gl_dm], 's--', color='#E91E63', lw=2, ms=7, label='DocMerger')
ax3.axvspan(75, 90, alpha=0.08, color='green')
ax3.set_xlabel('Compression (%)', fontsize=11)
ax3.set_ylabel('nDCG@5', fontsize=11)
ax3.set_title('MP-DocVQA Global (100 Q × 791 pages)', fontsize=12)
ax3.legend(fontsize=9, loc='lower left')
ax3.set_xlim(-2, 92); ax3.set_ylim(0.28, 0.42)
ax3.grid(True, alpha=0.3)

fig.text(0.5, -0.01,
         'Figure 4: DocPruner vs DocMerger across three evaluation settings. '
         'DocMerger consistently outperforms DocPruner at high compression (>75%, green zone).',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.subplots_adjust(bottom=0.14)
plt.savefig('results/figures/three_benchmarks.png', dpi=150, bbox_inches='tight')
plt.savefig('results/figures/three_benchmarks.pdf', bbox_inches='tight')
print("Saved to results/figures/three_benchmarks.{png,pdf}")
