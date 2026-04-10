"""
Method registry.

To add a new method:
  1. Create a file in benchmark/methods/
  2. Implement compress(embeddings, attention_scores, imgpad_mask, **params) -> PruneResult
  3. Import and register it here
"""

from .base import PruneResult
from .baselines import identity_compress, random_compress
from .docpruner import docpruner_compress
from .docmerger import docmerger_compress, docmerger_avg_compress
from .cps import cps_attn_compress, cps_central_compress, cps_dominance_compress
from .ptm import ptm_compress
from .learned import learned_compress
from .adaptive import adaptive_compress
from .mmr import mmr_compress
from .attn_fps import attn_fps_compress
from .dp_rebalance import dp_rebalance_compress
from .dp_residual import dp_residual_compress
from .dp_dedup import dp_dedup_compress
from .dp_postmerge import dp_postmerge_compress

# name -> (compress_fn, needs_attention)
METHODS = {
    "identity":      (identity_compress,      False),
    "random":        (random_compress,         False),
    "docpruner":     (docpruner_compress,      True),
    "docmerger":     (docmerger_compress,      True),
    "docmerger_avg": (docmerger_avg_compress,  True),
    "cps_attn":      (cps_attn_compress,       True),
    "cps_central":   (cps_central_compress,    True),
    "cps_domin":     (cps_dominance_compress,  True),
    "ptm":           (ptm_compress,            True),
    "learned":       (learned_compress,        True),
    "adaptive":      (adaptive_compress,       True),
    "mmr":           (mmr_compress,            True),
    "attn_fps":      (attn_fps_compress,       True),
    "dp_rebalance":  (dp_rebalance_compress,   True),
    "dp_residual":   (dp_residual_compress,    True),
    "dp_dedup":      (dp_dedup_compress,       True),
    "dp_postmerge":  (dp_postmerge_compress,   True),
}


def get_method(name: str):
    """Returns (compress_fn, needs_attention) for a registered method."""
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Available: {list(METHODS.keys())}")
    return METHODS[name]


def list_methods():
    return list(METHODS.keys())
