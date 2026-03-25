#!/usr/bin/env python3
"""Interactive Plotly plot of all experiments: DocPruner vs DocMerger across all datasets."""

import plotly.graph_objects as go

datasets = ["ESG", "Bio", "Econ", "ESG-H"]
colors_ds = {"ESG": "#2196F3", "Bio": "#4CAF50", "Econ": "#FF9800", "ESG-H": "#9C27B0"}

identity = {"ESG": 0.5996, "Bio": 0.6228, "Econ": 0.6226, "ESG-H": 0.6369}

# DocPruner: k -> {dataset: (prune%, ndcg)}
docpruner = {
    -0.50: {"ESG": (39.1, 0.5997), "Bio": (38.1, 0.6135), "Econ": (39.7, 0.6145), "ESG-H": (39.1, 0.6026)},
    -0.25: {"ESG": (52.1, 0.6049), "Bio": (51.0, 0.6083), "Econ": (53.2, 0.6244), "ESG-H": (52.1, 0.5869)},
     0.00: {"ESG": (62.3, 0.5825), "Bio": (61.5, 0.6004), "Econ": (63.5, 0.6226), "ESG-H": (62.3, 0.5526)},
     0.25: {"ESG": (70.5, 0.5919), "Bio": (69.9, 0.5984), "Econ": (71.5, 0.6173), "ESG-H": (70.5, 0.5395)},
     0.50: {"ESG": (76.9, 0.5684), "Bio": (76.5, 0.5763), "Econ": (77.6, 0.5765), "ESG-H": (76.9, 0.5220)},
     1.00: {"ESG": (85.9, 0.5614), "Bio": (85.9, 0.5097), "Econ": (85.8, 0.5179), "ESG-H": (85.9, 0.4783)},
}

# DocMerger: config -> {dataset: (prune%, ndcg)}
docmerger = {
    "k1=0.25,k2=0.5,mr=0.5":   {"ESG": (54.8, 0.5889), "Bio": (54.0, 0.6195), "Econ": (55.6, 0.6286), "ESG-H": (54.8, 0.5746)},
    "k1=0,k2=0.5,mr=0.25":     {"ESG": (56.6, 0.5887), "Bio": (55.7, 0.5986), "Econ": (57.6, 0.6410), "ESG-H": (56.6, 0.5692)},
    "k1=0,k2=0.25,mr=0.5":     {"ESG": (57.2, 0.5894), "Bio": (56.3, 0.6022), "Econ": (58.4, 0.6377), "ESG-H": (57.3, 0.5877)},
    "k1=0.5,k2=0.5,mr=0.5":    {"ESG": (58.0, 0.6037), "Bio": (57.4, 0.6192), "Econ": (58.7, 0.6199), "ESG-H": (58.1, 0.5713)},
    "k1=0,k2=0.5,mr=0.1":      {"ESG": (60.1, 0.5998), "Bio": (59.2, 0.5952), "Econ": (61.2, 0.6247), "ESG-H": (60.1, 0.5462)},
    "k1=0.5,k2=0.25,mr=0.5":   {"ESG": (64.5, 0.6021), "Bio": (63.8, 0.6098), "Econ": (65.5, 0.6079), "ESG-H": (64.6, 0.5820)},
    "k1=0.5,k2=0.5,mr=0.25":   {"ESG": (67.5, 0.5932), "Bio": (67.0, 0.6038), "Econ": (68.2, 0.6043), "ESG-H": (67.5, 0.5596)},
    "k1=0.5,k2=0,mr=0.5":      {"ESG": (69.6, 0.6049), "Bio": (69.0, 0.5984), "Econ": (70.6, 0.6006), "ESG-H": (69.6, 0.5547)},
    "k1=1.0,k2=0,mr=0.25":     {"ESG": (80.1, 0.5957), "Bio": (79.8, 0.5766), "Econ": (80.3, 0.5520), "ESG-H": (80.1, 0.4962)},
    "k1=1.0,k2=0.5,mr=0.1":    {"ESG": (81.3, 0.5905), "Bio": (81.2, 0.5534), "Econ": (81.3, 0.5380), "ESG-H": (81.3, 0.4913)},
    "k1=1.0,k2=0,mr=0.1":      {"ESG": (83.6, 0.5866), "Bio": (83.5, 0.5352), "Econ": (83.7, 0.5207), "ESG-H": (83.6, 0.4715)},
}

# DocMerger_avg ablation at k1=1.0,k2=0,mr=0.25
docmerger_avg = {"ESG": (80.1, 0.5909), "Bio": (79.8, 0.5786), "Econ": (80.3, 0.5482), "ESG-H": (80.1, 0.4962)}

fig = go.Figure()

# DocPruner per dataset
for ds in datasets:
    prunes = [0] + [docpruner[k][ds][0] for k in sorted(docpruner)]
    ndcgs  = [identity[ds]] + [docpruner[k][ds][1] for k in sorted(docpruner)]
    k_labels = ["identity"] + [f"k={k}" for k in sorted(docpruner)]
    fig.add_trace(go.Scatter(
        x=prunes, y=ndcgs, mode='lines+markers', name=f'DocPruner – {ds}',
        line=dict(color=colors_ds[ds], width=2),
        marker=dict(size=7, symbol='circle'),
        text=k_labels,
        hovertemplate=f'<b>DocPruner {ds}</b><br>%{{text}}<br>Compression: %{{x:.1f}}%<br>nDCG@5: %{{y:.4f}}<extra></extra>',
    ))

# DocMerger per dataset (sorted by compression)
sorted_cfgs = sorted(docmerger.keys(), key=lambda c: docmerger[c]["ESG"][0])
for ds in datasets:
    prunes = [docmerger[c][ds][0] for c in sorted_cfgs]
    ndcgs  = [docmerger[c][ds][1] for c in sorted_cfgs]
    fig.add_trace(go.Scatter(
        x=prunes, y=ndcgs, mode='lines+markers', name=f'DocMerger – {ds}',
        line=dict(color=colors_ds[ds], width=2, dash='dash'),
        marker=dict(size=9, symbol='square', line=dict(width=1.5, color='black')),
        text=sorted_cfgs,
        hovertemplate=f'<b>DocMerger {ds}</b><br>%{{text}}<br>Compression: %{{x:.1f}}%<br>nDCG@5: %{{y:.4f}}<extra></extra>',
    ))

# Ablation points
for ds in datasets:
    p, n = docmerger_avg[ds]
    fig.add_trace(go.Scatter(
        x=[p], y=[n], mode='markers', name=f'DocMerger(avg) – {ds}',
        marker=dict(size=10, symbol='diamond', color=colors_ds[ds], line=dict(width=2, color='gray')),
        hovertemplate=f'<b>DocMerger(avg) {ds}</b><br>k1=1.0,k2=0,mr=0.25<br>Compression: %{{x:.1f}}%<br>nDCG@5: %{{y:.4f}}<extra></extra>',
        visible='legendonly',
    ))

fig.update_layout(
    title='DocPruner vs DocMerger: nDCG@5 vs Compression (All Datasets)',
    xaxis_title='Compression Ratio (%)',
    yaxis_title='nDCG@5',
    hovermode='closest',
    legend=dict(font=dict(size=10)),
    width=1000, height=600,
)

fig.write_html('results/figures/all_experiments_interactive.html')
print("Saved to results/figures/all_experiments_interactive.html")
